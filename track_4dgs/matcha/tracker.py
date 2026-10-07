import os
from collections.abc import Sequence

import cv2
import torch
import torch.nn.functional as F

from track_4dgs.tracker import AbstractViewPointTracker, Track

from .geoaware_sc.flow_matching import get_flow
from .matcha import Matcha

DEFAULT_CHECKPOINT = "checkpoints/matcha_pretrained.pth"


def load_matcha(
        checkpoint: str = DEFAULT_CHECKPOINT,
        image_size: int = 512,
        ensemble_size: int = 8) -> Matcha:
    """Load MATCHA fusion weights.

    DIFT and DINOv2 weights are created inside :class:`Matcha`.
    """
    model = Matcha(image_size=image_size, ensemble_size=ensemble_size)
    # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/benchmark/run_benchmarks.py#L50-L54
    # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/feature/matcha_feature.py#L30-L32
    if checkpoint and os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint), strict=False)
    else:
        raise ValueError(f"No weight path available from {checkpoint}.")
    model.eval()
    return model


# https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/benchmark/temporal/tapvid.py#L50-L61
def post_process(feat_g, feat_s, norm=True):
    if feat_g.shape[2] != feat_s.shape[2] or feat_g.shape[3] != feat_s.shape[3]:
        feat_s = F.interpolate(
            feat_s, size=(feat_g.shape[2], feat_g.shape[3]), mode="bilinear"
        )
        feat_s = F.normalize(feat_s, dim=1)

    stride = feat_s.shape[1] // feat_g.shape[1]
    feat = torch.concat([feat_g, feat_s[:, ::stride, :, :]], dim=1)
    if norm:
        feat = F.normalize(feat, dim=1)
    return feat


# https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/utils/semantic_matching.py#L7-L50
def compute_semantic_matches(
        src_ft: torch.Tensor,
        trg_ft: torch.Tensor,
        src_kps: torch.Tensor,
        soft_eval: bool = False,
        soft_eval_window: int = 7,
):
    """
    Compute semantic matches between two feature maps using a similarity matrix.

    ``src_ft`` is ``[Q, C, H, W]`` and ``trg_ft`` is ``[D, C, H, W]``. ``src_kps``
    is ``[N, 3]`` integer ``(query, x, y)``. The source function is one pair and
    ``[N, 2]`` keypoints; the query and target axes are the pair batch.
    """
    _, C, H, W = src_ft.shape
    # Calculate similarity matrix
    # sim_1_to_2 = torch.matmul(img1_desc, img2_desc.permute(0, 2, 1))[
    #     0
    # ]  # [3600, 3600]
    with torch.no_grad():
        sim_1to2 = torch.matmul(
            src_ft.reshape(src_ft.shape[0], C, H * W).transpose(1, 2).unsqueeze(1),
            trg_ft.reshape(trg_ft.shape[0], C, -1).unsqueeze(0),
        )  # [Q, D, H*W, H*W]
    if soft_eval:
        # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/utils/semantic_matching.py#L33-L34
        flow = torch.stack([
            get_flow(sim, soft_eval_window, num_patches=H)[0]
            for sim in sim_1to2.flatten(0, 1)
        ]).view(src_ft.shape[0], trg_ft.shape[0], H, W, 2)
        flow_idxed = flow[src_kps[:, 0].long(), :, src_kps[:, 2].long(), src_kps[:, 1].long()]
        nn_y, nn_x = (
            flow_idxed[..., 1].clamp(0, H - 1),
            flow_idxed[..., 0].clamp(0, W - 1),
        )
    else:
        # Find nearest neighbors if soft evaluation is not enabled
        sim_1to2_idxed = sim_1to2[src_kps[:, 0].long(), :, src_kps[:, 2].long(), src_kps[:, 1].long()]
        _, nn_1_to_2 = torch.max(sim_1to2_idxed, dim=-1)
        nn_y, nn_x = nn_1_to_2 // W, nn_1_to_2 % W

    # Stack the transformed keypoints
    kps_1_to_2 = torch.stack([nn_x, nn_y], dim=-1)
    return kps_1_to_2, sim_1to2


class MatchaPointTracker(AbstractViewPointTracker):
    """Track queried points by pairwise MATCHA feature matching.

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]``. Query points are
    ``[x, y]`` pixels in their source frames. Each frame is stretched to
    ``image_size`` and described once. All query frames and target frames are
    then matched together with one similarity tensor, using the TAPVID
    protocol (``semantic_mode=False``, ``soft_eval=True`` by default).

    The source frame keeps the query coordinate. MATCHA has no occlusion
    head, so ``visibility`` is 1. ``confidence`` is the nearest-neighbor
    cosine similarity mapped with ``(cosine + 1) / 2`` into ``[0, 1]``, and
    1 on the source frame.
    """

    def __init__(
            self,
            checkpoint: str = DEFAULT_CHECKPOINT,
            image_size: int = 512,
            ensemble_size: int = 8,
            semantic_mode: bool = False,
            soft_eval: bool = True,
            soft_eval_window: int = 7,
            post_process: bool = False):
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be a positive integer")
        self.model = load_matcha(
            checkpoint=checkpoint,
            image_size=image_size,
            ensemble_size=ensemble_size,
        )
        self.image_size = image_size
        self.semantic_mode = semantic_mode
        self.soft_eval = soft_eval
        self.soft_eval_window = soft_eval_window
        self.post_process = post_process
        self.device = torch.device("cpu")
        self.model.eval()

    def to(self, device) -> 'MatchaPointTracker':
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
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
            raise ValueError("MatchaPointTracker expects RGB frames with shape [3, H, W]")
        if points.shape[0] == 0:
            raise ValueError("MatchaPointTracker expects at least one query point")
        if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= len(frames):
            raise ValueError("frame_indices must point at frames in the sequence")

        features = []
        orig_sizes = []
        for frame in frames:
            # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/benchmark/temporal/tapvid.py#L101-L109
            _, orig_h, orig_w = frame.shape
            image = frame.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            image = cv2.resize(image, dsize=(self.image_size, self.image_size))
            image = (
                torch.from_numpy(image.astype(float) / 255.0)
                .permute(2, 0, 1)
                .float()
                .to(self.device)[None]
            )
            orig_sizes.append((orig_h, orig_w))

            # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/benchmark/temporal/tapvid.py#L118-L129
            # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/feature/matcha_feature.py#L34-L41
            feat = self.model(
                img=image,
                semantic_mode=self.semantic_mode,
            )
            if self.post_process:
                feat = post_process(feat_g=feat, feat_s=feat)
            features.append(feat[0])

        features = torch.stack(features)
        _, _, feat_h, feat_w = features.shape
        frame_indices = frame_indices.to(device=features.device, dtype=torch.long)
        orig_h = features.new_tensor([height for height, _ in orig_sizes])
        orig_w = features.new_tensor([width for _, width in orig_sizes])
        query_points = points.to(device=features.device, dtype=features.dtype)

        # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/benchmark/temporal/tapvid.py#L131
        ds_scale = self.image_size / feat_w
        # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/benchmark/temporal/tapvid.py#L149-L150
        ref_points = torch.stack([
            query_points[:, 0] * (self.image_size / orig_w[frame_indices]),
            query_points[:, 1] * (self.image_size / orig_h[frame_indices]),
        ], dim=-1)
        # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/benchmark/temporal/tapvid.py#L190
        query_frames, inverse = torch.unique(frame_indices, sorted=True, return_inverse=True)
        src_kps = torch.stack([
            inverse,
            (ref_points[:, 0] / ds_scale).long().clamp(0, feat_w - 1),
            (ref_points[:, 1] / ds_scale).long().clamp(0, feat_h - 1),
        ], dim=-1)
        kps_1_to_2, sim_1to2 = compute_semantic_matches(
            src_ft=features[query_frames],
            trg_ft=features,
            src_kps=src_kps,
            soft_eval=self.soft_eval,
            soft_eval_window=self.soft_eval_window,
        )

        # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/benchmark/temporal/tapvid.py#L200
        pred_resized = kps_1_to_2 * ds_scale
        # Inverse of tapvid.py#L149-L150, which scaled original pixels into the resized image.
        pred = torch.stack([
            pred_resized[..., 0] * (orig_w / self.image_size),
            pred_resized[..., 1] * (orig_h / self.image_size),
        ], dim=-1).permute(1, 0, 2)
        # https://github.com/nv-dvl/matcha/blob/4277633d0e147b667b7b852cecb7fb335a7db379/matcha/matcher/base_matcher.py#L106
        sim_nn = sim_1to2[
            src_kps[:, 0], :, src_kps[:, 2] * feat_w + src_kps[:, 1],
        ].max(dim=-1).values
        confidence = (sim_nn + 1) / 2

        is_source = torch.arange(len(frames), device=features.device)[:, None] == frame_indices[None, :]
        pred = pred.to(device=points.device, dtype=points.dtype)
        confidence = confidence.transpose(0, 1).to(device=points.device, dtype=points.dtype).clamp(0, 1)
        is_source = is_source.to(device=points.device)
        pred = torch.where(is_source.unsqueeze(-1), points.unsqueeze(0), pred)
        confidence = torch.where(is_source, torch.ones_like(confidence), confidence)
        return Track(
            points=pred,
            visibility=points.new_ones((len(frames), points.shape[0])),
            confidence=confidence,
            mask=torch.ones(confidence.shape, dtype=torch.bool, device=pred.device),
        )
