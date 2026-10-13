import hashlib
import os
from collections.abc import Sequence
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from .dataset import CameraDatasetTracker, TrackedCameraDataset
from .tracker import Query, Track


class QueryBank:
    """Per-point tracks stored under ``path`` as ``{key}/{view}.pt``.

    Each file is an ``N=1`` :class:`Track` written with :meth:`Track.save`.
    :meth:`add` writes points from one :class:`Track` per view.
    :meth:`missing` returns indices of keys that do not yet have every view.
    ``bank[keys]`` loads them back as one :class:`Track` per view.
    """

    def __init__(self, path: str, n_views: int):
        self.path = path
        self.n_views = n_views

    def file(self, key: str, view_idx: int) -> str:
        return os.path.join(self.path, key, f"{view_idx}.pt")

    def add(self, keys: Sequence[str], view_tracks: Sequence[Track]) -> None:
        if len(view_tracks) != self.n_views or any(track.points.shape[1] != len(keys) for track in view_tracks):
            raise ValueError("cache bank rows must match the number of new keys")
        for point_idx, key in enumerate(keys):
            item = slice(point_idx, point_idx + 1)
            for view_idx, track in enumerate(view_tracks):
                path = self.file(key, view_idx)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                temporary = path + ".tmp"
                Track(
                    points=track.points[:, item],
                    visibility=track.visibility[:, item],
                    confidence=track.confidence[:, item],
                    mask=track.mask[:, item],
                ).save(temporary)
                os.replace(temporary, path)

    def missing(self, keys: Sequence[str]) -> list[int]:
        missing_index: list[int] = []
        seen: set[str] = set()
        for index, key in enumerate(keys):
            if key in seen:
                continue
            seen.add(key)
            if any(not os.path.isfile(self.file(key, view_idx)) for view_idx in range(self.n_views)):
                missing_index.append(index)
        return missing_index

    def __getitem__(self, keys: Sequence[str]) -> list[Track]:
        loaded = []
        for view_idx in range(self.n_views):
            tracks = [Track.load(self.file(key, view_idx)) for key in keys]
            loaded.append(Track(
                points=torch.cat([track.points for track in tracks], dim=1),
                visibility=torch.cat([track.visibility for track in tracks], dim=1),
                confidence=torch.cat([track.confidence for track in tracks], dim=1),
                mask=torch.cat([track.mask for track in tracks], dim=1),
            ))
        return loaded


def hash_tensor(hasher: "hashlib._Hash", tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    hasher.update(str(cpu.dtype).encode("ascii") + b"\0")
    hasher.update(repr(tuple(cpu.shape)).encode("ascii") + b"\0")
    raw = cpu.numpy().tobytes()
    hasher.update(len(raw).to_bytes(8, "little"))
    hasher.update(raw)


class CachedCameraDatasetTracker(CameraDatasetTracker):
    """Reuse per-point tracks from one cache directory per frames combination.

    Each directory is ``{tracker class}-{hash_frames}`` in ``cache_dir``. A point
    is ``{point hash}/{view}.pt``, an ``N=1`` :class:`Track` written with
    :meth:`Track.save`. The point hash covers coordinates, frame index, and
    ``in_image`` only; frames identity is the directory name. Saved points are
    loaded with :meth:`Track.load`. Only points missing a view file are passed
    to the tracker. Their results are saved and placed back at the original
    query positions. ``batch_size`` is not part of the key. Use a separate
    ``cache_dir`` for each tracker configuration.
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
        Frames are not included; they select the cache directory.
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
        bank = QueryBank(path, query.points.shape[0])
        keys = self.point_keys(query)
        missing_index = bank.missing(keys)
        if missing_index:
            tracked = self.tracker(
                self.subset_query(query, missing_index),
                frames,
                batch_size=batch_size,
            )
            bank.add([keys[index] for index in missing_index], tracked)

        device = frames[0][0].ground_truth_image.device
        return self.attach_view_tracks(frames, [track.to(device) for track in bank[keys]])

    def bank_path(self, frames_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{type(self.tracker).__name__}-{frames_hash}")
