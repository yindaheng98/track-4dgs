"""Object votes from first-hit ray casts."""

import torch


def ray_votes(object_ids: torch.Tensor) -> tuple[int, int, int, int]:
    """Count which object received the most ray hits.

    ``object_ids`` is ``(R,)`` long, and ``-1`` is a miss.
    Returns the winning object, its hit count, the runner-up count, and the miss count.
    The winning object is ``-1`` when every ray missed.
    """
    misses = int((object_ids < 0).sum())
    hits = object_ids[object_ids >= 0]
    if hits.numel() == 0:
        return -1, 0, 0, misses
    ids, counts = torch.unique(hits, return_counts=True)
    order = torch.argsort(counts, descending=True)
    count = int(counts[order[0]])
    winner = int(ids[order[0]])
    second = int(counts[order[1]]) if counts.numel() > 1 else 0
    return winner, count, second, misses
