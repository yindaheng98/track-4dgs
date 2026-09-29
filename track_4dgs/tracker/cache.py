import hashlib
import os
from collections.abc import Sequence
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from .dataset import CameraDatasetTracker, TrackedCameraDataset
from .tracker import Query, Track


class QueryBank:
    """Row-major cache of per-point tracks loaded from ``path``.

    ``keys[i]`` names row ``i``. ``points`` is ``[M, V, D, 2]`` and
    ``visibility`` / ``confidence`` / ``mask`` are ``[M, V, D]``.
    ``bank[keys]`` returns one :class:`Track` per view for those query points.
    :meth:`add` appends rows and writes the bank back to ``path``.
    """

    def __init__(self, path: str):
        self.path = path
        self.keys: list[str] = []
        self.points = torch.empty((0, 0, 0, 2))
        self.visibility = torch.empty((0, 0, 0))
        self.confidence = torch.empty((0, 0, 0))
        self.mask = torch.empty((0, 0, 0), dtype=torch.bool)
        if os.path.isfile(path):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.keys = list(payload["keys"])
            self.points = payload["points"]
            self.visibility = payload["visibility"]
            self.confidence = payload["confidence"]
            self.mask = payload["mask"]
        self.key_index = {key: index for index, key in enumerate(self.keys)}

    def add(self, keys: Sequence[str], tracked: Sequence[TrackedCameraDataset]) -> None:
        view_points = []
        view_visibility = []
        view_confidence = []
        view_mask = []
        for view_idx in range(len(tracked[0].camera_tracks)):
            camera_tracks = [dataset.camera_tracks[view_idx] for dataset in tracked]
            view_points.append(torch.stack([track.points for track in camera_tracks]).detach().cpu())
            view_visibility.append(torch.stack([track.visibility for track in camera_tracks]).detach().cpu())
            view_confidence.append(torch.stack([track.confidence for track in camera_tracks]).detach().cpu())
            view_mask.append(torch.stack([track.mask for track in camera_tracks]).detach().cpu())
        points = torch.stack(view_points).permute(2, 0, 1, 3).contiguous()
        visibility = torch.stack(view_visibility).permute(2, 0, 1).contiguous()
        confidence = torch.stack(view_confidence).permute(2, 0, 1).contiguous()
        mask = torch.stack(view_mask).permute(2, 0, 1).contiguous()
        if points.shape[0] != len(keys):
            raise ValueError("cache bank rows must match the number of new keys")
        if len(self.keys) == 0:
            self.points = points
            self.visibility = visibility
            self.confidence = confidence
            self.mask = mask
        else:
            self.points = torch.cat((self.points, points), dim=0)
            self.visibility = torch.cat((self.visibility, visibility), dim=0)
            self.confidence = torch.cat((self.confidence, confidence), dim=0)
            self.mask = torch.cat((self.mask, mask), dim=0)
        for key in keys:
            self.key_index[key] = len(self.keys)
            self.keys.append(key)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = self.path + ".tmp"
        torch.save({
            "keys": list(self.keys),
            "points": self.points,
            "visibility": self.visibility,
            "confidence": self.confidence,
            "mask": self.mask,
        }, temporary)
        os.replace(temporary, self.path)

    def missing(self, keys: Sequence[str]) -> list[int]:
        missing_index: list[int] = []
        seen: set[str] = set()
        for index, key in enumerate(keys):
            if key in self.key_index or key in seen:
                continue
            seen.add(key)
            missing_index.append(index)
        return missing_index

    def __getitem__(self, keys: Sequence[str]) -> list[Track]:
        indices = torch.tensor([self.key_index[key] for key in keys], dtype=torch.long)
        points = self.points[indices].permute(1, 2, 0, 3).contiguous()
        visibility = self.visibility[indices].permute(1, 2, 0).contiguous()
        confidence = self.confidence[indices].permute(1, 2, 0).contiguous()
        mask = self.mask[indices].permute(1, 2, 0).contiguous()
        return [
            Track(
                points=points[view_idx],
                visibility=visibility[view_idx],
                confidence=confidence[view_idx],
                mask=mask[view_idx],
            )
            for view_idx in range(points.shape[0])
        ]


def hash_tensor(hasher: "hashlib._Hash", tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    hasher.update(str(cpu.dtype).encode("ascii") + b"\0")
    hasher.update(repr(tuple(cpu.shape)).encode("ascii") + b"\0")
    raw = cpu.numpy().tobytes()
    hasher.update(len(raw).to_bytes(8, "little"))
    hasher.update(raw)


class CachedCameraDatasetTracker(CameraDatasetTracker):
    """Reuse per-point tracks from one cache-bank file per frames combination.

    Each bank is ``{tracker class}-{hash_frames}.pt`` in ``cache_dir``. Rows are
    query-point tracks, and ``bank[keys]`` is that query's view tracks. The
    point hash covers coordinates, frame index, and ``in_image`` only; frames
    identity is the filename. Saved points are loaded. Only points missing from
    the bank are passed to :meth:`CameraDatasetTracker.__call__`. Their results are
    added to the bank and placed back at the original query positions.
    ``batch_size`` is not part of the key. Use a separate ``cache_dir`` for
    each tracker configuration.
    """

    def __init__(self, tracker, cache_dir: str):
        super().__init__(tracker)
        self.cache_dir = cache_dir

    @staticmethod
    def hash_frames(frames: Sequence[CameraDataset]) -> str:
        """SHA-256 of the frame tensors ``AbstractPointTracker.__call__`` reads.

        That is each camera's ``ground_truth_image`` and
        ``ground_truth_image_mask``, in frame-major then view order. Camera
        poses are not included. Equal tensor values hash the same regardless
        of device.
        """
        hasher = hashlib.sha256()
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

    @staticmethod
    def hash_query(points: torch.Tensor, frame_indices: torch.Tensor, in_image: torch.Tensor) -> str:
        """SHA-256 of one query point column.

        ``points`` is ``[V, 2]``, ``frame_indices`` and ``in_image`` are ``[V]``.
        Frames are not included; they select the bank file.
        """
        hasher = hashlib.sha256()
        hash_tensor(hasher, points)
        hash_tensor(hasher, frame_indices)
        hash_tensor(hasher, in_image)
        return hasher.hexdigest()

    @staticmethod
    def subset_query(query: Query, indices: Sequence[int]) -> Query:
        return Query(
            points=query.points[:, indices],
            frame_indices=query.frame_indices[:, indices],
            in_image=query.in_image[:, indices],
        )

    def point_keys(self, query: Query) -> list[str]:
        points = query.points.detach().cpu()
        frame_indices = query.frame_indices.detach().cpu()
        in_image = query.in_image.detach().cpu()
        return [
            self.hash_query(
                points[:, point_idx],
                frame_indices[:, point_idx],
                in_image[:, point_idx],
            )
            for point_idx in range(points.shape[1])
        ]

    def __call__(
            self,
            query: Query,
            frames: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> list[TrackedCameraDataset]:
        frames = list(frames)
        if len(frames) == 0:
            return []
        if query.points.shape[1] == 0:
            return super().__call__(query, frames, batch_size=batch_size)

        path = self.bank_path(self.hash_frames(frames))
        bank = QueryBank(path)
        keys = self.point_keys(query)
        missing_index = bank.missing(keys)
        if missing_index:
            tracked = super().__call__(
                self.subset_query(query, missing_index),
                frames,
                batch_size=batch_size,
            )
            bank.add([keys[index] for index in missing_index], tracked)

        device = frames[0][0].ground_truth_image.device
        return self.attach_view_tracks(frames, [track.to(device) for track in bank[keys]])

    def bank_path(self, frames_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{type(self.tracker).__name__}-{frames_hash}.pt")
