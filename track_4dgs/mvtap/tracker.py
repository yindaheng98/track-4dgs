from collections.abc import Sequence

import torch
import torch.nn.functional as F
from gaussian_splatting.dataset import CameraDataset

from track_4dgs.tracker import AbstractMultiViewPointTracker, Query, Track

from .models.mvtap import MVTAP


def load_mvtap(checkpoint: str = "checkpoints/mvtap.ckpt", **model_kwargs) -> MVTAP:
    model = MVTAP(**model_kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    return model


class MVTAPPointTracker(AbstractMultiViewPointTracker):
    """Track queried points with MV-TAP.

    Frames are expected to be RGB ``[3, H, W]`` tensors in ``[0, 1]`` and query
    points use pixel coordinates ``[x, y]`` in the original frame resolution.
    Camera ``K`` / ``R`` / ``T`` are read from the frame datasets.
    """

    def __init__(
            self,
            checkpoint: str = "checkpoints/mvtap.ckpt",
            iters: int = 4,
            view_att: bool = True,
            use_cam_embed: bool = True,
            bilinear_mode: str = "border",
            use_checkpoint: bool = False):
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
            view_queries: Sequence[Query],
            frame_datasets: Sequence[CameraDataset]) -> Sequence[Track]:
        height_out = self.model.model_resolution_H
        width_out = self.model.model_resolution_W

        videos, query_tensors, intrinsics, extrinsics, scales = [], [], [], [], []
        for view_idx, query in enumerate(view_queries):
            frames, Ks, w2cs, view_scales = [], [], [], []
            for dataset in frame_datasets:
                camera = dataset[view_idx]
                _, orig_h, orig_w = camera.ground_truth_image.shape
                query_scale = query.points.new_tensor([width_out / orig_w, height_out / orig_h])
                view_scales.append(query_scale)
                frames.append(F.interpolate(
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
            videos.append(torch.stack(frames))
            intrinsics.append(torch.stack(Ks))
            extrinsics.append(torch.stack(w2cs))
            query_tensors.append(torch.cat([
                query.frame_indices.to(dtype=query.points.dtype).unsqueeze(-1),
                query.points * scale[query.frame_indices],
            ], dim=-1))

        video = torch.stack(videos).unsqueeze(0).to(self.device)
        queries = torch.stack(query_tensors).unsqueeze(0).to(self.device)
        intrinsic = torch.stack(intrinsics).unsqueeze(0).to(self.device)
        extrinsic = torch.stack(extrinsics).unsqueeze(0).to(self.device)
        scale = torch.stack(scales).to(self.device)

        n_frames = video.shape[2]
        window_len = self.model.window_len
        if n_frames < window_len:
            pad = window_len - n_frames
            video = torch.cat([video, video[:, :, -1:].expand(-1, -1, pad, -1, -1, -1)], dim=2)
            intrinsic = torch.cat([intrinsic, intrinsic[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
            extrinsic = torch.cat([extrinsic, extrinsic[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)

        coords, vis, conf, _ = self.model(
            video, queries.clone(), intrinsic, extrinsic,
            iters=self.iters, is_train=False, fmaps_chunk_size=1,
        )
        coords = coords[:, :, :n_frames]
        vis = vis[:, :, :n_frames]
        conf = conf[:, :, :n_frames]

        return [
            Track(points=points, visibility=visibility, confidence=confidence)
            for points, visibility, confidence in zip(
                coords[0] / scale[:, :, None, :], vis[0], conf[0],
            )
        ]
