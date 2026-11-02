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
    """Dense correspondence from one source frame to every frame.

    ``flow`` is ``[T, 2, H, W]`` with channels ``(dx, dy)`` in the source
    image's pixel frame. ``H`` and ``W`` match the source frame. ``visibility``
    and ``confidence`` are optional ``[T, H, W]`` maps. A missing map is 1 at
    every query point. The source frame itself is included.
    """

    flow: torch.Tensor
    visibility: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.flow.ndim != 4 or self.flow.shape[1] != 2:
            raise ValueError("DenseTrack.flow must have shape [T, 2, H, W]")
        time, _, height, width = self.flow.shape
        for name, field in (("visibility", self.visibility), ("confidence", self.confidence)):
            if field is None:
                continue
            if field.ndim == 3 and tuple(field.shape) == (time, height, width):
                continue
            raise ValueError(f"DenseTrack.{name} must have shape [{time}, {height}, {width}]")

    def sample(self, points: torch.Tensor) -> Track:
        """Sample this dense correspondence at query ``points`` ``[N, 2]`` (x, y).

        Returns a :class:`Track` over every frame. Predicted points are the
        query plus the sampled flow. A missing visibility or confidence map
        is 1.
        """
        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError("points must have shape [N, 2]")
        displacement = torch.stack([sample_dense(flow, points) for flow in self.flow])
        pred = points.to(device=displacement.device, dtype=torch.float32) + displacement
        ones = torch.ones(self.flow.shape[0], points.shape[0], device=points.device, dtype=points.dtype)
        visibility = ones if self.visibility is None else torch.stack([
            sample_dense(frame, points).squeeze(-1) for frame in self.visibility
        ]).to(ones)
        confidence = ones if self.confidence is None else torch.stack([
            sample_dense(frame, points).squeeze(-1) for frame in self.confidence
        ]).to(ones)
        return Track(
            points=pred.to(device=points.device, dtype=points.dtype),
            visibility=visibility,
            confidence=confidence,
            mask=torch.ones(ones.shape, dtype=torch.bool, device=points.device),
        )


class AbstractDensePointTracker(AbstractViewPointTracker):
    """Point tracker that samples a dense correspondence from each source frame.

    :meth:`track_dense` predicts one dense map from a source frame to every
    frame. :meth:`track_view` calls it once per queried source, then samples
    that map with :meth:`DenseTrack.sample`.
    """

    @torch.no_grad()
    def track_view(
            self,
            points: torch.Tensor,
            frame_indices: torch.Tensor,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]]) -> Track:
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
            sampled = self.track_dense(frames, frame_masks, int(source_index)).sample(points[selected])
            pred[:, selected] = sampled.points
            visibility[:, selected] = sampled.visibility
            confidence[:, selected] = sampled.confidence

        return Track(
            points=pred,
            visibility=visibility,
            confidence=confidence,
            mask=torch.ones(visibility.shape, dtype=torch.bool, device=points.device),
        )

    @abstractmethod
    def track_dense(
            self,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]],
            source_index: int) -> DenseTrack:
        """Predict a dense correspondence from ``frames[source_index]`` to every frame.

        Frames are RGB ``[3, H, W]``. The returned flow includes the source
        frame and uses that frame's resolution and pixel coordinates.
        """
        raise NotImplementedError


class AbstractPairDensePointTracker(AbstractDensePointTracker):
    """Dense tracker that matches one source frame to each other frame.

    :meth:`track_pair_dense` predicts one pair. :meth:`track_dense` calls it for
    every other frame and stacks the maps. The source frame is zero flow.
    """

    def track_dense(
            self,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]],
            source_index: int) -> DenseTrack:
        del frame_masks
        source = frames[source_index]
        pairs = [
            None if index == source_index else self.track_pair_dense(source, frame)
            for index, frame in enumerate(frames)
        ]
        known = next((pair for pair in pairs if pair is not None), None)
        if known is None:
            return DenseTrack(flow=source.new_zeros(1, 2, *source.shape[-2:]))
        zero = torch.zeros_like(known.flow[0])
        flow = torch.stack([zero if pair is None else pair.flow[0] for pair in pairs])
        visibility = None
        if known.visibility is not None:
            one = torch.ones_like(known.visibility[0])
            visibility = torch.stack([one if pair is None else pair.visibility[0] for pair in pairs])
        confidence = None
        if known.confidence is not None:
            one = torch.ones_like(known.confidence[0])
            confidence = torch.stack([one if pair is None else pair.confidence[0] for pair in pairs])
        return DenseTrack(flow=flow, visibility=visibility, confidence=confidence)

    @abstractmethod
    def track_pair_dense(self, source: torch.Tensor, target: torch.Tensor) -> DenseTrack:
        """Predict a one-frame dense correspondence from ``source`` to ``target``.

        The returned flow is ``[1, 2, H, W]`` in the source frame's pixels.
        """
        raise NotImplementedError
