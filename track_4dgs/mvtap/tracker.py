from collections.abc import Sequence

import torch
import torch.nn.functional as F
from gaussian_splatting.dataset import CameraDataset

from track_4dgs.tracker import AbstractBatchPointTracker, Query, Track

from .models.mvtap import MVTAP


def load_mvtap(checkpoint: str = "checkpoints/mvtap.ckpt", **model_kwargs) -> MVTAP:
    model = MVTAP(**model_kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    return model


class MVTAPPointTracker(AbstractBatchPointTracker):
    """Track queried points with MV-TAP.

    Frames are expected to be RGB ``[3, H, W]`` tensors in ``[0, 1]`` and query
    points use pixel coordinates ``[x, y]`` in the original frame resolution.
    Camera ``K`` / ``R`` / ``T`` are read from the frame datasets.

    ``visibility`` and ``confidence`` are CoTracker3-style sigmoid scores in
    ``(0, 1)``. Confidence is trained as ``P(error <= 12 px)``.

    ``batch_size`` chunks query points after ``fnet``. ``None`` tracks every
    point in one call.
    """

    def __init__(
            self,
            checkpoint: str = "checkpoints/mvtap.ckpt",
            iters: int = 4,
            view_att: bool = True,
            use_cam_embed: bool = True,
            bilinear_mode: str = "border",
            use_checkpoint: bool = False,
            batch_size: int | None = None):
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.model = load_mvtap(
            checkpoint,
            view_att=view_att,
            use_cam_embed=use_cam_embed,
            bilinear_mode=bilinear_mode,
            use_checkpoint=use_checkpoint,
        )
        self.model.eval()
        self.iters = iters
        self.device = torch.device("cpu")

    def to(self, device) -> 'MVTAPPointTracker':
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    @torch.no_grad()
    def track_batch(
            self,
            query: Query,
            frames: Sequence[CameraDataset]) -> Sequence[Track]:
        height_out = self.model.model_resolution_H
        width_out = self.model.model_resolution_W

        videos, query_tensors, intrinsics, extrinsics, scales = [], [], [], [], []
        for view_idx in range(query.points.shape[0]):
            images, Ks, w2cs, view_scales = [], [], [], []
            for dataset in frames:
                camera = dataset[view_idx]
                _, orig_h, orig_w = camera.ground_truth_image.shape
                query_scale = query.points.new_tensor([width_out / orig_w, height_out / orig_h])
                view_scales.append(query_scale)
                images.append(F.interpolate(
                    camera.ground_truth_image.unsqueeze(0),
                    size=(height_out, width_out),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0) * 255.0)

                K = camera.K.float().clone()
                K[0] *= width_out / orig_w
                K[1] *= height_out / orig_h
                Ks.append(K)

                w2c = torch.eye(4, device=camera.R.device, dtype=torch.float32)
                w2c[:3, :3] = camera.R.float()
                w2c[:3, 3] = camera.T.float()
                w2cs.append(w2c)

            scale = torch.stack(view_scales)
            scales.append(scale)
            videos.append(torch.stack(images))
            intrinsics.append(torch.stack(Ks))
            extrinsics.append(torch.stack(w2cs))
            frame_indices = query.frame_indices[view_idx]
            query_tensors.append(torch.cat([
                frame_indices.to(dtype=query.points.dtype).unsqueeze(-1),
                query.points[view_idx] * scale[frame_indices],
            ], dim=-1))

        video = torch.stack(videos).unsqueeze(0).to(self.device)
        queries = torch.stack(query_tensors).unsqueeze(0).to(self.device)
        intrinsic = torch.stack(intrinsics).unsqueeze(0).to(self.device)
        extrinsic = torch.stack(extrinsics).unsqueeze(0).to(self.device)
        scale = torch.stack(scales).to(self.device)

        n_frames = video.shape[2]
        window_len = self.model.window_len
        queries[..., 0].masked_fill_(
            ~query.in_image.unsqueeze(0).to(self.device),
            n_frames + window_len,
        )
        if n_frames < window_len:
            pad = window_len - n_frames
            video = torch.cat([video, video[:, :, -1:].expand(-1, -1, pad, -1, -1, -1)], dim=2)
            intrinsic = torch.cat([intrinsic, intrinsic[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
            extrinsic = torch.cat([extrinsic, extrinsic[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)

        queries = queries.clone()
        n_points = queries.shape[2]
        if self.batch_size is None or n_points <= self.batch_size:
            coords, vis, conf, _ = self.model(
                video, queries, intrinsic, extrinsic,
                iters=self.iters, is_train=False, fmaps_chunk_size=1,
            )
        else:
            # fnet reads only the video, but MVTAP calls it from inside forward.
            # The first point chunk runs fnet and stores every output. Each later
            # chunk hands those tensors back in the same order.
            fnet = self.model.fnet
            fnet_forward = fnet.forward
            encoded = []

            def fnet_record(*args, **kwargs):
                features = fnet_forward(*args, **kwargs)
                encoded.append(features)
                return features

            replay_index = 0

            def fnet_replay(*args, **kwargs):
                nonlocal replay_index
                features = encoded[replay_index]
                replay_index += 1
                return features

            coords_parts, vis_parts, conf_parts = [], [], []
            try:
                fnet.forward = fnet_record
                for start in range(0, n_points, self.batch_size):
                    end = min(start + self.batch_size, n_points)
                    coords, vis, conf, _ = self.model(
                        video, queries[:, :, start:end], intrinsic, extrinsic,
                        iters=self.iters, is_train=False, fmaps_chunk_size=1,
                    )
                    coords_parts.append(coords)
                    vis_parts.append(vis)
                    conf_parts.append(conf)
                    replay_index = 0
                    fnet.forward = fnet_replay
            finally:
                fnet.forward = fnet_forward
            coords = torch.cat(coords_parts, dim=3)
            vis = torch.cat(vis_parts, dim=3)
            conf = torch.cat(conf_parts, dim=3)
        coords = coords[:, :, :n_frames]
        vis = vis[:, :, :n_frames]
        conf = conf[:, :, :n_frames]
        points = coords[0] / scale[:, :, None, :]
        mask = torch.ones(vis[0].shape, dtype=torch.bool, device=points.device)

        return [
            Track(points=p, visibility=visibility, confidence=confidence, mask=m)
            for p, visibility, confidence, m in zip(points, vis[0], conf[0], mask)
        ]
