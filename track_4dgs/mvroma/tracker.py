import os
from argparse import Namespace
from collections.abc import Sequence
from typing import List, Tuple

import cv2
import torch
import torch.nn.functional as F
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement

from mvroma import MVRoMa, ModelConfig
from mvroma.models.pipeline import normalize_xy
from mvroma.romatch.models.transformer import vit_large
from mvroma.utils.grids import build_token_center_grid
from track_4dgs.tracker import AbstractViewPointTracker, Track

DEFAULT_CHECKPOINT = "checkpoints/outdoor_final.pth"
# submodules/MV-RoMa/demo.py#L105-L112
DEFAULT_COARSE_RES_HW = (560, 560)
DEFAULT_TARGET_RES_HW = (560, 840)
# submodules/MV-RoMa/src/run_model.py#L74-L77
UFM_MATCH_HW = (420, 560)


def load_mvroma(
        checkpoint: str = DEFAULT_CHECKPOINT,
        num_cluster: int = 512) -> MVRoMa:
    """Load MV-RoMa the way ``demo.py`` builds the dense matcher."""
    # submodules/MV-RoMa/demo.py#L15-L27
    # submodules/MV-RoMa/src/build_model.py#L4-L13
    args = Namespace(
        use_dinov2=True,
        train_until_16x=False,
        train_refiner=False,
        train_all_model=False,
        num_cluster=num_cluster,
    )
    cfg = ModelConfig()
    cfg.num_cluster = num_cluster
    backbone = vit_large(patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0, img_size=518)
    cfg.img_size = 518
    cfg.patch_size = 14
    cfg.backbone_name = "v2"
    cfg.coarse_feature_dim = 512
    model = MVRoMa(
        cfg,
        backbone=backbone,
        train_until_16x=args.train_until_16x,
        train_refiner=args.train_refiner,
        train_all_model=args.train_all_model,
    )
    # submodules/MV-RoMa/demo.py#L30-L33
    if checkpoint and os.path.exists(checkpoint):
        weight = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(weight, strict=False)
    else:
        raise ValueError(f"No weight path available from {checkpoint}.")
    model.eval()
    return model


def load_prematch(device=None):
    """Load the UFM prematcher used to seed MV-RoMa tracks.

    The demo stores ``[None, model]`` and later indexes ``[1]``.
    """
    # submodules/MV-RoMa/src/matchers/run_matcher_path.py#L10-L17
    # submodules/MV-RoMa/demo.py#L35-L37
    model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
    model.eval()
    if device is not None:
        model = model.to(device)
    return [None, model]


def _resize_uint8(frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Stretch one RGB ``[3, H, W]`` frame in ``[0, 1]`` to uint8 ``[3, height, width]``.

    ``run_ufm_multi`` resizes with ``cv2.resize`` before matching. ``dsize`` is ``(width, height)``.
    """
    # submodules/MV-RoMa/src/matchers/run_matcher_path.py#L136-L137
    image = frame.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    image = cv2.resize(image, dsize=(width, height))
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


# submodules/MV-RoMa/src/matchers/run_matcher_path.py#L135-L183
def run_ufm_multi(matcher_model, query_img, ref_imgs, device="cuda:0", batched=False):
    """Match one query against every reference with UFM.

    ``query_img`` is uint8 ``[1, 3, H, W]`` and ``ref_imgs`` is uint8 ``[R, 3, H, W]``.
    The path loads in the source are replaced by these already-resized tensors.
    """
    matcher = matcher_model[1]
    if batched:
        num_refview = ref_imgs.shape[0]
        with torch.inference_mode():
            result = matcher.predict_correspondences_batched(
                source_image=query_img.repeat(num_refview, 1, 1, 1).to(device),
                target_image=ref_imgs.to(device),
            )
        flow_output = result.flow.flow_output
        covisibility = result.covisibility.mask
    else:
        query_img = query_img.to(device)
        flow_outputs = []
        covis_outputs = []
        for ref_index in range(ref_imgs.shape[0]):
            ref_img = ref_imgs[ref_index:ref_index + 1].to(device)
            with torch.inference_mode():
                result = matcher.predict_correspondences_batched(
                    source_image=query_img,
                    target_image=ref_img,
                )
            flow_outputs.append(result.flow.flow_output)
            covis_outputs.append(result.covisibility.mask)
        flow_output = torch.cat(flow_outputs, dim=0)
        covisibility = torch.cat(covis_outputs, dim=0)

    height, width = flow_output.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords_q = torch.stack((xx, yy), dim=-1)
    coords_refs = coords_q.unsqueeze(0) + flow_output.permute(0, 2, 3, 1)
    return coords_q, coords_refs, covisibility


def _torch_kmeans_gpu(x: torch.Tensor, k: int, iters: int = 12, seed: int = 0) -> torch.Tensor:
    """Lloyd's k-means entirely on GPU.

    Copied from ``submodules/MV-RoMa/src/track_cluster.py``.
    ``x`` is ``[M, D]`` float32 CUDA. Returns centers ``[k, D]``.
    """
    assert x.is_cuda and x.dtype == torch.float32 and x.dim() == 2
    rows, dim = x.shape
    if k <= 0 or rows == 0:
        return x.new_empty((0, dim))
    if k == 1:
        return x.mean(dim=0, keepdim=True)

    generator = torch.Generator(device=x.device).manual_seed(seed)
    if k <= rows:
        perm = torch.randperm(rows, generator=generator, device=x.device)
        centers = x[perm[:k]].clone()
    else:
        idx = torch.randint(low=0, high=rows, size=(k,), generator=generator, device=x.device)
        centers = x[idx].clone()

    x_sq = (x * x).sum(dim=1, keepdim=True)

    for _ in range(max(1, iters)):
        c_sq = (centers * centers).sum(dim=1).unsqueeze(0)
        dist = x_sq + c_sq - 2.0 * (x @ centers.t())
        labels = dist.argmin(dim=1)

        new_centers = torch.zeros_like(centers)
        assign = labels.unsqueeze(1).expand(-1, dim)
        new_centers.scatter_add_(0, assign, x)

        counts = torch.bincount(labels, minlength=k)
        has_pts = counts > 0
        if has_pts.any():
            new_centers[has_pts] = new_centers[has_pts] / counts[has_pts].unsqueeze(1).to(x.dtype)

        if (~has_pts).any():
            empty_idx = (~has_pts).nonzero(as_tuple=False).squeeze(1)
            n_empty = int(empty_idx.numel())
            point_dists = dist[torch.arange(rows, device=x.device), labels]
            if rows >= n_empty:
                repl = torch.topk(point_dists, k=n_empty, largest=True, sorted=False).indices
            else:
                head = torch.topk(point_dists, k=rows, largest=True, sorted=False).indices
                tail = torch.randint(low=0, high=rows, size=(n_empty - rows,), generator=generator, device=x.device)
                repl = torch.cat((head, tail), dim=0)
            new_centers[empty_idx] = x[repl]

        if torch.allclose(new_centers, centers, rtol=0.0, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    c_sq = (centers * centers).sum(dim=1).unsqueeze(0)
    dist = x_sq + c_sq - 2.0 * (x @ centers.t())
    labels = dist.argmin(dim=1)

    closest = torch.empty_like(centers)
    point_indices = torch.arange(rows, device=x.device)
    for cluster in range(k):
        cluster_mask = labels == cluster
        if cluster_mask.any():
            cluster_dist = dist[cluster_mask, cluster]
            cluster_indices = point_indices[cluster_mask]
            min_pos = cluster_dist.argmin()
            closest[cluster] = x[cluster_indices[min_pos]]
        else:
            closest[cluster] = centers[cluster]
    return closest


# submodules/MV-RoMa/src/track_cluster.py#L6-L177
def extract_tracks_match(
        coords_q: torch.Tensor,
        coords_refs: torch.Tensor,
        covis,
        N: int,
        covisibility_threshold: float = 0.3,
        downsample_stride: int = 8,
        kmeans_iters: int = 12,
        kmeans_seed: int = 0,
        cluster_mode: str = "global",
        device=None,
) -> torch.Tensor:
    """Cluster UFM matches into multi-view tracks.

    Returns ``[R, N_eff, 4]`` float32. Each row is ``[x_q, y_q, x_r, y_r]``.
    References that do not see a track stay at ``-1000``. The source wraps this
    in ``inference_mode``; the tracker already runs under ``no_grad``.
    """
    if device is None:
        device = coords_q.device

    if cluster_mode not in ("per-group", "global"):
        raise ValueError(f"Unsupported cluster_mode: {cluster_mode}")

    height, width = coords_q.shape[:2]
    n_refs = coords_refs.shape[0]
    # The copied grouping packs one bit per reference into an int32 code.
    if n_refs > 31:
        raise ValueError("extract_tracks_match supports at most 31 reference frames")
    vis = covis > covisibility_threshold

    if not vis.any():
        return torch.empty((n_refs, 0, 4), device=device, dtype=torch.float32)

    codes = torch.zeros((height, width), dtype=torch.int32, device=device)
    for ref_index in range(n_refs):
        codes |= (vis[ref_index].to(torch.int32) << ref_index)

    present_codes = torch.unique(codes)
    present_codes = present_codes[present_codes != 0]

    pattern_list: List[Tuple[int, Tuple[int, ...], int]] = []
    for code in present_codes.tolist():
        mask = (codes == code)
        count = int(mask.sum().item())
        if count > 0:
            ref_idxs = tuple(index for index in range(n_refs) if (code >> index) & 1)
            pattern_list.append((code, ref_idxs, count))
    pattern_list.sort(key=lambda item: (-len(item[1]), -item[2], item[0]))

    if not pattern_list:
        return torch.empty((n_refs, 0, 4), device=device, dtype=torch.float32)

    caps = [max(1, (count + downsample_stride - 1) // downsample_stride) for (_, _, count) in pattern_list]
    total_cap = int(sum(caps))
    if total_cap == 0:
        return torch.empty((n_refs, 0, 4), device=device, dtype=torch.float32)

    n_eff = min(int(N), total_cap)
    raw_fracs = [n_eff * (cap / total_cap) for cap in caps]
    ks = [min(caps[index], int(raw_fracs[index])) for index in range(len(caps))]
    assigned = sum(ks)
    frac_parts = sorted(
        [(raw_fracs[index] - int(raw_fracs[index]), index) for index in range(len(caps))],
        key=lambda item: -item[0],
    )
    remaining = n_eff - assigned
    for _, index in frac_parts:
        if remaining == 0:
            break
        if ks[index] < caps[index]:
            ks[index] += 1
            remaining -= 1
    if remaining > 0:
        for index in range(len(caps)):
            if remaining == 0:
                break
            free = caps[index] - ks[index]
            if free > 0:
                give = min(free, remaining)
                ks[index] += give
                remaining -= give

    stride = max(1, int(downsample_stride))
    k_use_list: List[int] = []
    samples_idx: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for (code, _, _), k_i in zip(pattern_list, ks):
        mask = (codes == code)
        yi, xi = mask.nonzero(as_tuple=True)
        yi = yi[::stride]
        xi = xi[::stride]
        samples_idx.append((yi, xi))
        k_use_list.append(min(k_i, int(yi.numel())))

    n_eff = int(sum(k_use_list))
    if n_eff == 0:
        return torch.empty((n_refs, 0, 4), device=device, dtype=torch.float32)

    out = torch.full((n_refs, n_eff, 4), float("-1000."), device=device, dtype=torch.float32)
    cursor = 0

    if cluster_mode == "per-group":
        for ((_, ref_idxs, _), k_i, (yi, xi)) in zip(pattern_list, k_use_list, samples_idx):
            k_use = int(k_i)
            if k_use <= 0:
                continue
            feats = [coords_q[yi, xi]]
            for ref_index in ref_idxs:
                feats.append(coords_refs[ref_index, yi, xi])
            group = torch.cat(feats, dim=1).to(torch.float32)
            centers = _torch_kmeans_gpu(group, k=k_use, iters=kmeans_iters, seed=kmeans_seed)
            k_actual = int(centers.shape[0])
            if k_actual <= 0:
                continue
            out[:, cursor:cursor + k_actual, 0] = centers[:, 0].unsqueeze(0)
            out[:, cursor:cursor + k_actual, 1] = centers[:, 1].unsqueeze(0)
            for ref_index in ref_idxs:
                pos = ref_idxs.index(ref_index)
                out[ref_index, cursor:cursor + k_actual, 2] = centers[:, 2 + 2 * pos + 0]
                out[ref_index, cursor:cursor + k_actual, 3] = centers[:, 2 + 2 * pos + 1]
            cursor += k_actual
    else:
        feature_dim = 2 + 2 * n_refs
        missing_sentinel = -100000.0
        row_count = int(sum(int(yi.numel()) for (k_i, (yi, _)) in zip(k_use_list, samples_idx) if int(k_i) > 0))
        if row_count <= 0:
            return torch.empty((n_refs, 0, 4), device=device, dtype=torch.float32)

        x_global = torch.full((row_count, feature_dim), missing_sentinel, device=device, dtype=torch.float32)
        row = 0
        for ((_, ref_idxs, _), k_i, (yi, xi)) in zip(pattern_list, k_use_list, samples_idx):
            if int(k_i) <= 0:
                continue
            count = int(yi.numel())
            if count <= 0:
                continue
            span = slice(row, row + count)
            x_global[span, 0:2] = coords_q[yi, xi]
            for ref_index in ref_idxs:
                start = 2 + 2 * ref_index
                x_global[span, start:start + 2] = coords_refs[ref_index, yi, xi]
            row += count
        x_global = x_global[:row]

        centers = _torch_kmeans_gpu(x_global, k=n_eff, iters=kmeans_iters, seed=kmeans_seed)
        k_actual = int(centers.shape[0])
        out = torch.full((n_refs, k_actual, 4), float("-1000."), device=device, dtype=torch.float32)
        if k_actual > 0:
            out[:, :, 0] = centers[:, 0].unsqueeze(0)
            out[:, :, 1] = centers[:, 1].unsqueeze(0)
            for ref_index in range(n_refs):
                start = 2 + 2 * ref_index
                xr = centers[:, start + 0]
                yr = centers[:, start + 1]
                valid = (xr != missing_sentinel) & (yr != missing_sentinel)
                if valid.any():
                    out[ref_index, valid, 2] = xr[valid]
                    out[ref_index, valid, 3] = yr[valid]
        cursor = k_actual

    return out[:, :cursor, :]


# submodules/MV-RoMa/src/run_model.py#L41-L60
def match_and_cluster(
        query_img,
        ref_imgs,
        match_w,
        match_h,
        img_w,
        img_h,
        prematch_model,
        num_cluster,
        covisibility_threshold,
        device,
        batched=False):
    query_coord, ref_coord, ref_covis = run_ufm_multi(
        prematch_model, query_img, ref_imgs, device=device, batched=batched,
    )
    if img_h / match_h != 1. or img_w / match_w != 1.:
        query_coord[..., 1] = query_coord[..., 1] * (img_h / match_h)
        query_coord[..., 0] = query_coord[..., 0] * (img_w / match_w)
        ref_coord[..., 1] = ref_coord[..., 1] * (img_h / match_h)
        ref_coord[..., 0] = ref_coord[..., 0] * (img_w / match_w)
    return extract_tracks_match(
        query_coord, ref_coord, ref_covis,
        N=num_cluster, downsample_stride=4,
        covisibility_threshold=covisibility_threshold, device=device,
    )


def _resize_float(frame: torch.Tensor, height: int, width: int, device) -> torch.Tensor:
    """``transforms.Resize`` + ``ToTensor`` for one frame already in ``[0, 1]``."""
    # submodules/MV-RoMa/src/run_model.py#L100-L116
    image = _resize_uint8(frame, height, width)
    return image.float().div(255).to(device)


# submodules/MV-RoMa/src/run_model.py#L63-L213
def run_model(
        model: MVRoMa,
        frames: Sequence[torch.Tensor],
        coarse_res_hw: tuple[int, int],
        target_res_hw: tuple[int, int],
        prematch_model,
        num_cluster: int,
        upsample_preds: bool,
        covisibility_threshold: float,
        device,
        batched: bool = False):
    """Run one MV-RoMa match.

    ``frames[0]`` is the query and the rest are references, matching
    ``query_img_path`` / ``ref_img_paths`` in the demo. Square padding stays
    off, which is ``run_model_test``'s default.
    """
    coarse_h, coarse_w = coarse_res_hw
    assert coarse_h % model.patch_size == 0
    assert coarse_w % model.patch_size == 0
    # submodules/MV-RoMa/src/run_model.py#L74-L77
    match_h, match_w = UFM_MATCH_HW
    n_view = len(frames) - 1

    query_match = _resize_uint8(frames[0], match_h, match_w).unsqueeze(0)
    ref_match = torch.stack([_resize_uint8(frame, match_h, match_w) for frame in frames[1:]])
    with torch.inference_mode():
        batch_track = match_and_cluster(
            query_match, ref_match, match_w=match_w, match_h=match_h,
            img_w=coarse_w, img_h=coarse_h,
            prematch_model=prematch_model, num_cluster=num_cluster,
            covisibility_threshold=covisibility_threshold, device=device, batched=batched,
        )
        model_track_input = batch_track.unsqueeze(0)
    if model_track_input.shape[2] == 0:
        raise RuntimeError("MV-RoMa prematch produced no covisible tracks")

    # submodules/MV-RoMa/src/run_model.py#L94-L98
    feature_grid_coords = build_token_center_grid(
        B=1, T=n_view + 1,
        H=coarse_h, W=coarse_w,
        patch=model.patch_size, device=device,
    )
    coarse_img_tensors = torch.stack([
        _resize_float(frame, coarse_h, coarse_w, device) for frame in frames
    ]).unsqueeze(0)

    upsample_img_tensors = None
    if upsample_preds:
        # submodules/MV-RoMa/src/run_model.py#L174-L199
        target_h, target_w = target_res_hw
        upsample_img_tensors = torch.stack([
            _resize_float(frame, target_h, target_w, device) for frame in frames
        ]).unsqueeze(0)

    # submodules/MV-RoMa/src/run_model.py#L201-L208
    with torch.inference_mode():
        corresps = model.match(
            multi_view_frames=coarse_img_tensors,
            multi_view_frames_org=upsample_img_tensors,
            point_tracks=model_track_input,
            feature_grid_coords=feature_grid_coords,
            upsample_preds=upsample_preds,
        )
        # submodules/MV-RoMa/demo.py#L123-L126
        finest_scale = min(corresps.keys())
        flow = corresps[finest_scale]["flow"][0]
        certainty = corresps[finest_scale]["certainty"][0, :, 0].sigmoid()
    # Leave inference mode with ordinary tensors so query sampling can run under no_grad.
    return _as_normal(flow), _as_normal(certainty)


def _as_normal(tensor: torch.Tensor) -> torch.Tensor:
    values = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    values.copy_(tensor)
    return values


def sample_correspondences(
        flow: torch.Tensor,
        certainty: torch.Tensor,
        query_points: torch.Tensor,
        query_size: tuple[int, int],
        target_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample dense query-to-target flow at original-image query pixels.

    ``flow`` is ``[R, 2, H, W]`` normalized target coordinates. ``certainty``
    is ``[R, H, W]`` after sigmoid. ``target_sizes`` is ``[R, 2]`` as
    ``(height, width)``.
    """
    query_h, query_w = query_size
    # Inverse of submodules/MV-RoMa/eval_hpatches.py#L55-L64, align_corners=False.
    grid = normalize_xy(query_points, query_h, query_w).view(1, 1, -1, 2)
    grid = grid.expand(flow.shape[0], 1, -1, 2)
    sampled_flow = F.grid_sample(
        flow.float(), grid, mode="bilinear", padding_mode="border", align_corners=False,
    )
    sampled_flow = sampled_flow.permute(3, 0, 1, 2).squeeze(-1)
    # submodules/MV-RoMa/eval_hpatches.py#L66-L75
    target_h = target_sizes[:, 0].to(dtype=sampled_flow.dtype).view(1, -1)
    target_w = target_sizes[:, 1].to(dtype=sampled_flow.dtype).view(1, -1)
    pixels = torch.stack([
        target_w * (sampled_flow[..., 0] + 1) / 2 - 0.5,
        target_h * (sampled_flow[..., 1] + 1) / 2 - 0.5,
    ], dim=-1)
    sampled_certainty = F.grid_sample(
        certainty.float().unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=False,
    )
    return pixels, sampled_certainty[:, 0, 0].transpose(0, 1)


class MVRoMaPointTracker(AbstractViewPointTracker):
    """Track queried points with MV-RoMa dense multi-view correspondences.

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]``. Query points are
    ``[x, y]`` pixels in their source frames. Each distinct source frame is
    the demo's query image; every other frame is a reference. UFM proposes
    the sparse tracks MV-RoMa conditions on, then the finest flow map is
    sampled at the query pixels.

    The source frame keeps the query coordinate. ``visibility`` and
    ``confidence`` are the sampled sigmoid certainty in ``[0, 1]``, and 1 on
    the source frame. The copied track clustering kernel requires CUDA and at
    most 31 reference frames.
    """

    def __init__(
            self,
            checkpoint: str = DEFAULT_CHECKPOINT,
            num_cluster: int = 512,
            coarse_res_hw: tuple[int, int] = DEFAULT_COARSE_RES_HW,
            target_res_hw: tuple[int, int] = DEFAULT_TARGET_RES_HW,
            upsample_preds: bool = True,
            covisibility_threshold: float = 0.3,
            prematch_batched: bool = False):
        coarse_h, coarse_w = coarse_res_hw
        if coarse_h <= 0 or coarse_w <= 0 or coarse_h % 14 or coarse_w % 14:
            raise ValueError("coarse_res_hw must be positive and divisible by patch size 14")
        if num_cluster <= 0:
            raise ValueError("num_cluster must be a positive integer")
        if upsample_preds:
            target_h, target_w = target_res_hw
            if target_h <= 0 or target_w <= 0:
                raise ValueError("target_res_hw must be positive when upsample_preds is enabled")
        self.model = load_mvroma(checkpoint=checkpoint, num_cluster=num_cluster)
        self.prematch = load_prematch()
        self.num_cluster = num_cluster
        self.coarse_res_hw = (int(coarse_res_hw[0]), int(coarse_res_hw[1]))
        self.target_res_hw = (int(target_res_hw[0]), int(target_res_hw[1]))
        self.upsample_preds = upsample_preds
        self.covisibility_threshold = covisibility_threshold
        self.prematch_batched = prematch_batched
        self.device = torch.device("cpu")
        self.model.eval()

    def to(self, device) -> 'MVRoMaPointTracker':
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.prematch[1] = self.prematch[1].to(self.device)
        self.model.eval()
        self.prematch[1].eval()
        return self

    @torch.no_grad()
    def track_batch(
            self,
            points: torch.Tensor,
            frame_indices: torch.Tensor,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[torch.Tensor | None]) -> Track:
        del frame_masks
        if any(frame.shape[0] != 3 for frame in frames):
            raise ValueError("MVRoMaPointTracker expects RGB frames with shape [3, H, W]")
        if points.shape[0] == 0:
            raise ValueError("MVRoMaPointTracker expects at least one query point")
        if len(frames) < 2:
            raise ValueError("MVRoMaPointTracker expects a query frame and at least one reference frame")
        if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= len(frames):
            raise ValueError("frame_indices must point at frames in the sequence")
        if self.device.type != "cuda":
            raise RuntimeError("MVRoMaPointTracker requires a CUDA device for track clustering")

        query_points = points.to(device=self.device, dtype=torch.float32)
        frame_indices = frame_indices.to(device=self.device, dtype=torch.long)
        n_frames = len(frames)
        pred = query_points.new_zeros((n_frames, query_points.shape[0], 2))
        visibility = query_points.new_zeros((n_frames, query_points.shape[0]))
        confidence = query_points.new_zeros((n_frames, query_points.shape[0]))
        orig_sizes = [(int(frame.shape[1]), int(frame.shape[2])) for frame in frames]

        for source_index in torch.unique(frame_indices).tolist():
            source_index = int(source_index)
            selected = frame_indices == source_index
            ordered = [source_index] + [index for index in range(n_frames) if index != source_index]
            flow, certainty = run_model(
                self.model,
                [frames[index] for index in ordered],
                coarse_res_hw=self.coarse_res_hw,
                target_res_hw=self.target_res_hw,
                prematch_model=self.prematch,
                num_cluster=self.num_cluster,
                upsample_preds=self.upsample_preds,
                covisibility_threshold=self.covisibility_threshold,
                device=self.device,
                batched=self.prematch_batched,
            )
            ref_ids = ordered[1:]
            target_sizes = query_points.new_tensor(
                [[orig_sizes[index][0], orig_sizes[index][1]] for index in ref_ids]
            )
            sampled_points, sampled_certainty = sample_correspondences(
                flow,
                certainty,
                query_points[selected],
                orig_sizes[source_index],
                target_sizes,
            )
            selected_index = selected.nonzero(as_tuple=False).flatten()
            ref_index = torch.tensor(ref_ids, device=self.device, dtype=torch.long)
            pred[ref_index[:, None], selected_index[None, :]] = sampled_points.permute(1, 0, 2)
            visibility[ref_index[:, None], selected_index[None, :]] = sampled_certainty.transpose(0, 1)
            confidence[ref_index[:, None], selected_index[None, :]] = sampled_certainty.transpose(0, 1)
            pred[source_index, selected] = query_points[selected]
            visibility[source_index, selected] = 1
            confidence[source_index, selected] = 1

        pred = pred.to(device=points.device, dtype=points.dtype)
        visibility = visibility.to(device=points.device, dtype=points.dtype).clamp(0, 1)
        confidence = confidence.to(device=points.device, dtype=points.dtype).clamp(0, 1)
        return Track(
            points=pred,
            visibility=visibility,
            confidence=confidence,
            mask=torch.ones(visibility.shape, dtype=torch.bool, device=pred.device),
        )
