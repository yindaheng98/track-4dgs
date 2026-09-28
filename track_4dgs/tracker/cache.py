import hashlib
import os
from collections.abc import Sequence
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from .dataset import CameraDatasetTracker, TrackedCameraDataset
from .tracker import Query, Track


def hash_tensor(hasher: "hashlib._Hash", tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    hasher.update(str(cpu.dtype).encode("ascii") + b"\0")
    hasher.update(repr(tuple(cpu.shape)).encode("ascii") + b"\0")
    raw = cpu.numpy().tobytes()
    hasher.update(len(raw).to_bytes(8, "little"))
    hasher.update(raw)


class CachedCameraDatasetTracker(CameraDatasetTracker):
    """Reuse tracks when ``query`` and ``frames`` match a previous call.

    A hit loads the saved view tracks and attaches them to the ``frames``
    passed in this call. A miss calls :meth:`CameraDatasetTracker.__call__`
    and writes the result under ``cache_dir``. ``batch_size`` is not part of
    the key. Use a separate ``cache_dir`` for each tracker configuration; the
    filename also includes the wrapped tracker class name.
    """

    def __init__(self, tracker, cache_dir: str):
        super().__init__(tracker)
        self.cache_dir = cache_dir

    @staticmethod
    def hash_query_frames(query: Query, frames: Sequence[CameraDataset]) -> str:
        """SHA-256 of the tensors ``AbstractPointTracker.__call__`` actually reads.

        That is ``query`` plus each camera's ``ground_truth_image`` and
        ``ground_truth_image_mask``, in frame-major then view order. Camera poses
        are not included. Equal tensor values hash the same regardless of device.
        """
        hasher = hashlib.sha256()
        for tensor in (query.points, query.frame_indices, query.in_image):
            hash_tensor(hasher, tensor)
        hasher.update(len(frames).to_bytes(8, "little"))
        for dataset in frames:
            hasher.update(len(dataset).to_bytes(8, "little"))
            for view_idx in range(len(dataset)):
                camera = dataset[view_idx]
                hash_tensor(hasher, camera.ground_truth_image)
                mask = camera.ground_truth_image_mask
                if mask is None:
                    hasher.update(b"\0")
                else:
                    hasher.update(b"\1")
                    hash_tensor(hasher, mask)
        return hasher.hexdigest()

    def __call__(
            self,
            query: Query,
            frames: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> list[TrackedCameraDataset]:
        frames = list(frames)
        if len(frames) == 0:
            return []

        hash = self.hash_query_frames(query, frames)
        view_tracks = self.load(hash)
        if view_tracks is None:
            tracked = super().__call__(query, frames, batch_size=batch_size)
            self.save(hash, tracked)
            return tracked

        device = frames[0][0].ground_truth_image.device
        return self.attach_view_tracks(frames, [track.to(device) for track in view_tracks])

    def cache_path(self, hash: str) -> str:
        return os.path.join(self.cache_dir, f"{type(self.tracker).__name__}-{hash}.pt")

    def load(self, hash: str) -> Optional[list[Track]]:
        path = self.cache_path(hash)
        if not os.path.isfile(path):
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return [
            Track(
                points=item["points"],
                visibility=item["visibility"],
                confidence=item["confidence"],
                mask=item["mask"],
            ) for item in payload
        ]

    def save(self, hash: str, tracked: Sequence[TrackedCameraDataset]) -> None:
        n_views = len(tracked[0].camera_tracks)
        view_tracks: list[Track] = []
        for view_idx in range(n_views):
            camera_tracks = [dataset.camera_tracks[view_idx] for dataset in tracked]
            view_tracks.append(Track(
                points=torch.stack([track.points for track in camera_tracks]),
                visibility=torch.stack([track.visibility for track in camera_tracks]),
                confidence=torch.stack([track.confidence for track in camera_tracks]),
                mask=torch.stack([track.mask for track in camera_tracks]),
            ))
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self.cache_path(hash)
        payload = [{
            "points": track.points.detach().cpu(),
            "visibility": track.visibility.detach().cpu(),
            "confidence": track.confidence.detach().cpu(),
            "mask": track.mask.detach().cpu(),
        } for track in view_tracks]
        temporary = path + ".tmp"
        torch.save(payload, temporary)
        os.replace(temporary, path)
