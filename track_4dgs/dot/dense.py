from collections.abc import Sequence

import torch
import torch.nn.functional as F

from track_4dgs.tracker import AbstractDensePointTracker, DenseTrack

from .tracker import NUM_TRACKS, SIM_TRACKS, load_dot, resize_model


class DOTDensePointTracker(AbstractDensePointTracker):
    """Dense tracks with DOT from one source frame to every other frame.

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]`` and share one
    resolution. ``model`` selects the sparse initializer: ``dot`` (CoTracker2),
    ``cotracker3``, ``tapir``, or ``bootstapir``. The working resolution
    follows the frame, rounded down to a multiple of 8.

    ``track_dense`` rolls the video so ``source_index`` is first, calls
    ``get_tracks_from_first_to_every_other_frame``, and rolls the maps back.
    Flow is ``(dx, dy)`` in the source image's pixels. ``visibility`` and
    ``confidence`` are the refiner alpha.
    """

    def __init__(self, model: str = "dot"):
        self.model = load_dot(model)
        self.device = torch.device("cpu")
        self.model.eval()

    def to(self, device) -> "DOTDensePointTracker":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    @torch.no_grad()
    def track_dense(
            self,
            frames: Sequence[torch.Tensor],
            source_index: int) -> DenseTrack:
        if any(frame.shape[-2:] != frames[0].shape[-2:] for frame in frames):
            raise ValueError("DOTDensePointTracker expects all frames in a view to have the same resolution")
        _, height, width = frames[0].shape
        self.model = resize_model(self.model, height, width, self.device)
        video = torch.stack([
            frame.detach().clamp(0, 1).to(device=self.device, dtype=torch.float32)
            for frame in frames
        ], dim=0).unsqueeze(0)
        video = torch.roll(video, shifts=-source_index, dims=1)
        prediction = self.model.get_tracks_from_first_to_every_other_frame(
            {"video": video},
            num_tracks=NUM_TRACKS,
            sim_tracks=SIM_TRACKS,
        )
        # [T, H, W, 3] is [x, y, alpha] on the model grid. x, y are target pixels.
        tracks = torch.roll(prediction["tracks"][0], shifts=source_index, dims=0)
        model_h, model_w = (int(v) for v in self.model.resolution)
        ys = torch.linspace(0, model_h - 1, model_h, device=tracks.device, dtype=tracks.dtype)
        xs = torch.linspace(0, model_w - 1, model_w, device=tracks.device, dtype=tracks.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        disp = tracks[..., :2] - torch.stack([grid_x, grid_y], dim=-1)
        disp = disp * disp.new_tensor([(width - 1) / (model_w - 1), (height - 1) / (model_h - 1)])
        disp = disp.permute(0, 3, 1, 2).contiguous()
        alpha = tracks[..., 2]
        if disp.shape[-2:] != (height, width):
            disp = F.interpolate(disp, size=(height, width), mode="bilinear", align_corners=True)
            alpha = F.interpolate(
                alpha.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True,
            ).squeeze(1)
        return DenseTrack(flow=disp, visibility=alpha, confidence=alpha)
