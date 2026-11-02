from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .singleview import AbstractViewPointTracker
from .tracker import Track


def sample_dense(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample ``field`` at pixel coordinates ``points``.

    ``field`` is ``[C, H, W]`` or ``[H, W]``. ``points`` is ``[N, 2]`` as
    ``(x, y)``. The result is ``[N, C]`` (``C`` is 1 for a ``[H, W]`` map).
    """
    if field.ndim == 2:
        field = field.unsqueeze(0)
    _, height, width = field.shape
    points = points.to(device=field.device, dtype=torch.float32)
    grid = torch.stack([
        (points[:, 0] + 0.5) / width,
        (points[:, 1] + 0.5) / height,
    ], dim=-1)
    grid = grid.mul(2).sub(1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        field.float().unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, 0].transpose(0, 1)


@dataclass(frozen=True)
class DenseTrack:
    """Dense correspondence from one source frame to one target frame.

    ``flow`` is ``[2, H, W]`` with channels ``(dx, dy)`` in the source image's
    pixel frame. ``H`` and ``W`` match the source frame. ``visibility`` and
    ``confidence`` are optional ``[H, W]`` maps. A missing map is 1 at every
    query point.
    """

    flow: torch.Tensor
    visibility: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.flow.ndim != 3 or self.flow.shape[0] != 2:
            raise ValueError("DenseTrack.flow must have shape [2, H, W]")
        height, width = self.flow.shape[-2:]
        for name, field in (("visibility", self.visibility), ("confidence", self.confidence)):
            if field is None:
                continue
            if field.ndim == 2 and tuple(field.shape) == (height, width):
                continue
            raise ValueError(f"DenseTrack.{name} must have shape [{height}, {width}]")

    def sample(self, points: torch.Tensor) -> Track:
        """Sample this dense correspondence at query ``points`` ``[N, 2]`` (x, y).

        Returns a one-frame :class:`Track`. Predicted points are the query
        plus the sampled flow. A missing visibility or confidence map is 1.
        """
        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError("points must have shape [N, 2]")
        displacement = sample_dense(self.flow, points)
        pred = points.to(device=displacement.device, dtype=torch.float32) + displacement
        ones = torch.ones(points.shape[0], device=points.device, dtype=points.dtype)
        visibility = ones if self.visibility is None else sample_dense(self.visibility, points).squeeze(-1).to(ones)
        confidence = ones if self.confidence is None else sample_dense(self.confidence, points).squeeze(-1).to(ones)
        return Track(
            points=pred.to(device=points.device, dtype=points.dtype).unsqueeze(0),
            visibility=visibility.unsqueeze(0),
            confidence=confidence.unsqueeze(0),
            mask=torch.ones((1, points.shape[0]), dtype=torch.bool, device=points.device),
        )


class AbstractDensePointTracker(AbstractViewPointTracker):
    """Point tracker that samples a dense source-to-target correspondence.

    :meth:`track_dense` predicts one dense map from a source frame to a
    target frame. :meth:`track_batch` calls it for each queried source and
    every other frame, then samples that map with :meth:`DenseTrack.sample`.
    The source frame keeps the query coordinate.
    """

    @torch.no_grad()
    def track_batch(
            self,
            points: torch.Tensor,
            frame_indices: torch.Tensor,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]]) -> Track:
        del frame_masks
        name = type(self).__name__
        if any(frame.ndim != 3 or frame.shape[0] != 3 for frame in frames):
            raise ValueError(f"{name} expects RGB frames with shape [3, H, W]")
        if points.shape[0] == 0:
            raise ValueError(f"{name} expects at least one query point")
        if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= len(frames):
            raise ValueError("frame_indices must point at frames in the sequence")

        frame_indices = frame_indices.to(device=points.device, dtype=torch.long)
        n_frames = len(frames)
        pred = points.new_zeros((n_frames, points.shape[0], 2))
        visibility = points.new_ones((n_frames, points.shape[0]))
        confidence = points.new_ones((n_frames, points.shape[0]))

        for source_index in torch.unique(frame_indices).tolist():
            selected = frame_indices == source_index
            source_points = points[selected]
            for target_index in range(n_frames):
                if target_index == source_index:
                    pred[target_index, selected] = source_points
                    continue
                sampled = self.track_dense(frames[source_index], frames[target_index]).sample(source_points)
                pred[target_index, selected] = sampled.points[0]
                visibility[target_index, selected] = sampled.visibility[0]
                confidence[target_index, selected] = sampled.confidence[0]

        return Track(
            points=pred,
            visibility=visibility,
            confidence=confidence,
            mask=torch.ones(visibility.shape, dtype=torch.bool, device=points.device),
        )

    @abstractmethod
    def track_dense(self, source: torch.Tensor, target: torch.Tensor) -> DenseTrack:
        """Predict a dense correspondence from ``source`` to ``target``.

        Both arguments are RGB ``[3, H, W]`` frames. The returned flow uses
        the source frame's resolution and pixel coordinates.
        """
        raise NotImplementedError
