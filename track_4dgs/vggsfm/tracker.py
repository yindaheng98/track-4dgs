import os
from collections.abc import Sequence

import torch
from vggt.dependency.vggsfm_tracker import TrackerPredictor

from track_4dgs.tracker import AbstractViewPointTracker, Track
from track_4dgs.vggt.tracker import padding_square, points_from_square, points_to_square


def calculate_index_mappings(query_index: int, frame_num: int, device=None) -> torch.Tensor:
    """Swap ``query_index`` with 0 so the query frame is first."""
    order = torch.arange(frame_num)
    order[0] = query_index
    order[query_index] = 0
    if device is not None:
        order = order.to(device)
    return order


def switch_tensor_order(tensors: Sequence[torch.Tensor | None], order: torch.Tensor, dim: int = 1):
    """Reorder tensors along ``dim``. ``None`` entries stay ``None``."""
    return [torch.index_select(tensor, dim, order) if tensor is not None else None for tensor in tensors]


def predict_tracks_in_chunks(
        track_predictor: TrackerPredictor,
        images_feed: torch.Tensor,
        query_points_list,
        fmaps_feed: torch.Tensor,
        fine_tracking: bool,
        coarse_iters: int = 6,
        fine_chunk: int = 40960):
    """Run ``TrackerPredictor`` on chunks of query points and concatenate."""
    if not isinstance(query_points_list, (list, tuple)):
        query_points_list = [query_points_list]

    fine_pred_track_list = []
    pred_vis_list = []
    pred_score_list = []
    for split_points in query_points_list:
        fine_pred_track, _, pred_vis, pred_score = track_predictor(
            images_feed,
            split_points,
            fmaps=fmaps_feed,
            coarse_iters=coarse_iters,
            fine_tracking=fine_tracking,
            fine_chunk=fine_chunk,
        )
        fine_pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)

    fine_pred_track = torch.cat(fine_pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)
    if pred_score_list[0] is None:
        pred_score = None
    else:
        pred_score = torch.cat(pred_score_list, dim=2)
    return fine_pred_track, pred_vis, pred_score


def load_vggsfm(checkpoint: str = "checkpoints/vggsfm_v2_tracker.pt") -> TrackerPredictor:
    model = TrackerPredictor()
    if os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt",
            map_location="cpu",
        ))
    return model


class VGGSfMPointTracker(AbstractViewPointTracker):
    """Track queried points with the VGGSfM ``TrackerPredictor``.

    Frames are expected to be RGB ``[3, H, W]`` tensors in ``[0, 1]`` and query
    points use pixel coordinates ``[x, y]`` in the source frame. Images are
    center-padded to a square and resized with the same helpers as
    :class:`~track_4dgs.vggt.tracker.VGGTPointTracker`. The tracker reads query
    features from frame 0, so each distinct query frame is swapped to the front
    before prediction and swapped back afterwards.

    ``visibility`` is the coarse predictor's sigmoid score in ``(0, 1)``.
    ``confidence`` is 1: fine refinement is run with ``compute_score=False``,
    so the released tracker does not emit a localization score.
    """

    def __init__(
            self,
            checkpoint: str = "checkpoints/vggsfm_v2_tracker.pt",
            img_load_resolution: int = 1024,
            coarse_iters: int = 6,
            fine_tracking: bool = True,
            max_points_num: int = 163840,
            fine_chunk: int = 40960):
        self.model = load_vggsfm(checkpoint)
        self.model.eval()
        self.img_load_resolution = img_load_resolution
        self.coarse_iters = coarse_iters
        self.fine_tracking = fine_tracking
        self.max_points_num = max_points_num
        self.fine_chunk = fine_chunk
        self.device = torch.device("cpu")

    def to(self, device) -> 'VGGSfMPointTracker':
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
            raise ValueError("VGGSfMPointTracker expects RGB frames with shape [3, H, W]")
        if points.shape[0] == 0:
            raise ValueError("VGGSfMPointTracker expects at least one query point")
        frame_num = len(frames)
        if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= frame_num:
            raise ValueError("frame_indices must point at frames in the sequence")

        # 1. Preprocess each image: center-pad + bicubic to img_load_resolution
        square_frames = []
        orig_sizes = []
        for img in frames:
            square_frames.append(padding_square(img, self.img_load_resolution))
            orig_sizes.append(img.shape[-2:])
        images = torch.stack(square_frames)

        # 2. Feature maps once for the whole sequence
        fmaps = self.model.process_images_to_fmaps(images)

        track_points = points.new_empty((frame_num, points.shape[0], 2))
        visibility = points.new_empty((frame_num, points.shape[0]))
        confidence = points.new_empty((frame_num, points.shape[0]))

        # 3. Track each query frame after moving it to index 0
        for query_index in frame_indices.unique().tolist():
            selected = frame_indices == query_index
            query_h, query_w = orig_sizes[query_index]
            query_points = points_to_square(
                points[selected], query_h, query_w, square_size=self.img_load_resolution,
            ).unsqueeze(0)

            reorder_index = calculate_index_mappings(query_index, frame_num, device=images.device)
            images_feed, fmaps_feed = switch_tensor_order([images, fmaps], reorder_index, dim=0)
            images_feed = images_feed.unsqueeze(0)
            fmaps_feed = fmaps_feed.unsqueeze(0)

            point_count = images_feed.shape[1] * query_points.shape[1]
            if point_count > self.max_points_num:
                num_splits = (point_count + self.max_points_num - 1) // self.max_points_num
                query_points_list = list(torch.chunk(query_points, num_splits, dim=1))
            else:
                query_points_list = [query_points]

            pred_track, pred_vis, pred_score = predict_tracks_in_chunks(
                self.model,
                images_feed,
                query_points_list,
                fmaps_feed,
                fine_tracking=self.fine_tracking,
                coarse_iters=self.coarse_iters,
                fine_chunk=self.fine_chunk,
            )
            pred_track, pred_vis, pred_score = switch_tensor_order(
                [pred_track, pred_vis, pred_score], reorder_index, dim=1,
            )
            pred_track = pred_track.squeeze(0)
            pred_vis = pred_vis.squeeze(0)
            if pred_score is None:
                pred_score = torch.ones_like(pred_vis)
            else:
                pred_score = pred_score.squeeze(0)

            # 4. Restore original pixel coordinates for each image
            restored = []
            for frame_idx, (height, width) in enumerate(orig_sizes):
                restored.append(points_from_square(
                    pred_track[frame_idx], height, width, square_size=self.img_load_resolution,
                ))
            track_points[:, selected] = torch.stack(restored)
            visibility[:, selected] = pred_vis.to(dtype=points.dtype)
            confidence[:, selected] = pred_score.to(dtype=points.dtype)

        return Track(
            points=track_points,
            visibility=visibility,
            confidence=confidence,
            mask=torch.ones(visibility.shape, dtype=torch.bool, device=track_points.device),
        )
