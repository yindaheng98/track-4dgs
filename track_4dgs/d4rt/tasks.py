# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/src/eval/tasks.py
"""Minimal model-query helpers used by WorldTrack evaluation."""

from __future__ import annotations

from typing import Any

import torch


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/src/eval/tasks.py#L98-L145
def _run_model_for_queries(
    model: torch.nn.Module,
    video_b: torch.Tensor,
    aspect_b: torch.Tensor | None,
    query: dict[str, torch.Tensor],
    chunk_size: int,
    memory_b: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    num_queries = int(query["u"].numel())
    if num_queries == 0:
        return {
            "xyz_3d": torch.empty((0, 3), dtype=video_b.dtype),
            "uv_2d": torch.empty((0, 2), dtype=video_b.dtype),
            "visibility": torch.empty((0,), dtype=video_b.dtype),
            "displacement": torch.empty((0, 3), dtype=video_b.dtype),
            "normal": torch.empty((0, 3), dtype=video_b.dtype),
            "confidence": torch.empty((0,), dtype=video_b.dtype),
        }

    out_chunks: dict[str, list[torch.Tensor]] = {}
    step = max(1, int(chunk_size))
    cached_model = getattr(model, "module", model)
    decode_queries = getattr(cached_model, "decode_queries", None)

    for start in range(0, num_queries, step):
        end = min(num_queries, start + step)
        query_chunk = {
            "u": query["u"][start:end].view(1, -1),
            "v": query["v"][start:end].view(1, -1),
            "t_src": query["t_src"][start:end].view(1, -1),
            "t_tgt": query["t_tgt"][start:end].view(1, -1),
            "t_cam": query["t_cam"][start:end].view(1, -1),
        }
        if memory_b is not None and callable(decode_queries):
            pred = decode_queries(video=video_b, query=query_chunk, memory=memory_b)
        else:
            batch: dict[str, Any] = {"video": video_b, "query": query_chunk}
            if aspect_b is not None:
                batch["aspect_ratio"] = aspect_b
            pred = model(batch)

        for key, value in pred.items():
            chunk_value = value[0].detach()
            if torch.is_floating_point(chunk_value):
                chunk_value = chunk_value.to(dtype=torch.float32)
            out_chunks.setdefault(key, []).append(chunk_value.cpu())

    return {key: torch.cat(chunks, dim=0) for key, chunks in out_chunks.items()}


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/src/eval/tasks.py#L148-L157
def _encode_model_memory(
    model: torch.nn.Module,
    video_b: torch.Tensor,
    aspect_b: torch.Tensor | None,
) -> torch.Tensor | None:
    cached_model = getattr(model, "module", model)
    encode_video = getattr(cached_model, "encode_video", None)
    if not callable(encode_video):
        return None
    return encode_video(video=video_b, aspect_ratio=aspect_b)


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/src/eval/tasks.py#L160-L169
def _model_clip_frames(model: torch.nn.Module | None, default: int = 48) -> int:
    if model is None:
        return int(default)
    cached_model = getattr(model, "module", model)
    query_embedder = getattr(cached_model, "query_embedder", None)
    max_frames = getattr(query_embedder, "max_frames", None)
    try:
        return max(1, int(max_frames))
    except (TypeError, ValueError):
        return int(default)
