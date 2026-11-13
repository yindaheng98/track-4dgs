"""Select frames from a sequence."""


def selected_frame_indices(frame_start: int, frame_end: int | None, frame_step: int, num_frames: int) -> list[int]:
    """Frames ``range(frame_start, frame_end + 1, frame_step)``.

    ``frame_end`` defaults to the last frame. Every index must lie in
    ``0 .. num_frames - 1``.
    """

    if num_frames <= 0:
        raise ValueError("sequence has no frames")
    end = num_frames - 1 if frame_end is None else frame_end
    indices = list(range(frame_start, end + 1, frame_step))
    if not indices or any(index < 0 or index >= num_frames for index in indices):
        raise ValueError(
            f"frame range {frame_start}..{end} step {frame_step} is outside 0..{num_frames - 1}"
        )
    return indices
