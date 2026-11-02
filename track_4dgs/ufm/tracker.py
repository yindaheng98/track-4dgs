import torch
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence

from track_4dgs.tracker import AbstractDensePointTracker, DenseTrack

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


class UniFlowMatchPointTracker(AbstractDensePointTracker):
    """Track queried points with UniFlowMatch dense correspondences.

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]``. Query points are
    ``[x, y]`` pixels in their source frames. ``track_dense`` converts the
    pair to uint8 RGB and calls ``predict_correspondences_batched``, the
    example inference path. The match does not depend on the query points.

    Flow channels are ``(dx, dy)`` in the source image's pixel frame.
    ``visibility`` and ``confidence`` are the covisibility map in ``[0, 1]``.
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
    def track_dense(self, source: torch.Tensor, target: torch.Tensor) -> DenseTrack:
        # example_inference.py loads RGB uint8 HWC. predict_correspondences_batched
        # normalizes uint8 itself, so float frames in [0, 1] are converted here.
        source_image = source.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous().to(self.device)
        target_image = target.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous().to(self.device)
        # submodules/UFM/uniflowmatch/utils/viz.py#L39-L42
        # target pixel = source pixel + (flow_x, flow_y)
        flow_output, covisibility = predict_correspondences(self.model, source_image, target_image)
        return DenseTrack(flow=flow_output, visibility=covisibility, confidence=covisibility)
