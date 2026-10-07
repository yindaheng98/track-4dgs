from collections.abc import Sequence

import torch
import torch.nn.functional as F
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence

from track_4dgs.tracker import AbstractViewPointTracker, Track

# submodules/UFM/example_inference.py#L110-L117
MODEL_REPOS = {
    "base": "infinity1096/UFM-Base",
    "refine": "infinity1096/UFM-Refine",
    "base-980": "infinity1096/UFM-Base-980",
    "refine-980": "infinity1096/UFM-Refine-980",
    "base-dinov2l-init": "infinity1096/UFM-Base-DINOv2L-init",
    "base-dinov2g-init": "infinity1096/UFM-Base-DINOv2G-init",
}


def load_uniflowmatch(model: str = "base"):
    """Load a pretrained UniFlowMatch checkpoint.

    ``model`` uses the same names as ``example_inference.py``.
    """
    # submodules/UFM/example_inference.py#L119-L126
    if model not in MODEL_REPOS:
        raise ValueError(
            "Please choose from [base, refine, base-980, refine-980, "
            "base-dinov2l-init, base-dinov2g-init]"
        )
    if "base" in model:
        loaded = UniFlowMatchConfidence.from_pretrained(MODEL_REPOS[model])
    elif "refine" in model:
        loaded = UniFlowMatchClassificationRefinement.from_pretrained(MODEL_REPOS[model])
    else:
        raise ValueError(
            "Please choose from [base, refine, base-980, refine-980, "
            "base-dinov2l-init, base-dinov2g-init]"
        )
    loaded.eval()
    return loaded


# submodules/UFM/example_inference.py#L31-L42
def predict_correspondences(model, source_image, target_image):
    """Predict correspondences between source and target images.

    Images follow the example: RGB uint8, ``HWC`` or ``CHW``. Flow stays
    ``[2, H, W]`` and covisibility stays ``[H, W]`` so query points can be
    sampled on device. Flow channels are ``(dx, dy)`` in the source image's
    original pixel frame.
    """
    with torch.no_grad():
        result = model.predict_correspondences_batched(
            source_image=source_image,
            target_image=target_image,
        )

        flow_output = result.flow.flow_output[0]
        covisibility = result.covisibility.mask[0]

    return flow_output, covisibility


def sample_at_points(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample ``field`` ``[C, H, W]`` at pixel coordinates ``[N, 2]`` (x, y)."""
    _, height, width = field.shape
    # Pixel-center mapping used by UFM's own grid_sample.
    # submodules/UFM/uniflowmatch/models/ufm.py#L1178-L1183
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


class UniFlowMatchPointTracker(AbstractViewPointTracker):
    """Track queried points with UniFlowMatch dense correspondences.

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]``. Query points are
    ``[x, y]`` pixels in their source frames. Each source frame is converted
    to uint8 RGB and matched to every other frame with
    ``predict_correspondences_batched``, the example inference path.

    The source frame keeps the query coordinate. Predicted points are the
    query plus the sampled flow. ``visibility`` and ``confidence`` are the
    sampled covisibility in ``[0, 1]``, and 1 on the source frame.
    """

    def __init__(self, model: str = "base"):
        self.model = load_uniflowmatch(model)
        self.device = torch.device("cpu")
        self.model.eval()

    def to(self, device) -> 'UniFlowMatchPointTracker':
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
            raise ValueError("UniFlowMatchPointTracker expects RGB frames with shape [3, H, W]")
        if points.shape[0] == 0:
            raise ValueError("UniFlowMatchPointTracker expects at least one query point")
        if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= len(frames):
            raise ValueError("frame_indices must point at frames in the sequence")

        # example_inference.py loads RGB uint8 HWC. predict_correspondences_batched
        # normalizes uint8 itself, so float frames in [0, 1] are converted here.
        images = [
            frame.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous().to(self.device)
            for frame in frames
        ]
        query_points = points.to(device=self.device, dtype=torch.float32)
        frame_indices = frame_indices.to(device=self.device, dtype=torch.long)
        n_frames = len(images)
        pred = query_points.new_zeros((n_frames, query_points.shape[0], 2))
        visibility = query_points.new_zeros((n_frames, query_points.shape[0]))
        confidence = query_points.new_zeros((n_frames, query_points.shape[0]))

        for source_index in torch.unique(frame_indices).tolist():
            selected = frame_indices == source_index
            source_points = query_points[selected]
            source_image = images[source_index]
            for target_index, target_image in enumerate(images):
                if target_index == source_index:
                    pred[target_index, selected] = source_points
                    visibility[target_index, selected] = 1
                    confidence[target_index, selected] = 1
                    continue
                # submodules/UFM/uniflowmatch/utils/viz.py#L39-L42
                # target pixel = source pixel + (flow_x, flow_y)
                flow_output, covisibility = predict_correspondences(
                    self.model, source_image, target_image,
                )
                displacement = sample_at_points(flow_output, source_points)
                covisible = sample_at_points(covisibility.unsqueeze(0), source_points).squeeze(-1)
                pred[target_index, selected] = source_points + displacement.to(source_points.dtype)
                visibility[target_index, selected] = covisible
                confidence[target_index, selected] = covisible

        pred = pred.to(device=points.device, dtype=points.dtype)
        visibility = visibility.to(device=points.device, dtype=points.dtype).clamp(0, 1)
        confidence = confidence.to(device=points.device, dtype=points.dtype).clamp(0, 1)
        return Track(
            points=pred,
            visibility=visibility,
            confidence=confidence,
            mask=torch.ones(visibility.shape, dtype=torch.bool, device=pred.device),
        )
