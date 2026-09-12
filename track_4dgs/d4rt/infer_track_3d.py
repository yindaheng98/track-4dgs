# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/infer_track_3d.py
"""Minimal D4RT inference helpers used by WorldTrack evaluation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from .tasks import _encode_model_memory, _model_clip_frames, _run_model_for_queries


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/infer_track_3d.py#L26-L34
def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    return {}


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/infer_track_3d.py#L37-L43
def _resize_video(video_rgb: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    resized = [
        cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if frame.shape[0] >= h else cv2.INTER_LINEAR)
        for frame in video_rgb
    ]
    return np.stack(resized, axis=0)


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/infer_track_3d.py#L89-L136
def _make_anchor_clip_indices(num_frames: int, clip_frames: int, target_idx: int, source_idx: int = 0) -> np.ndarray:
    num_frames = int(num_frames)
    clip_frames = max(1, int(clip_frames))
    target_idx = int(np.clip(int(target_idx), 0, max(0, num_frames - 1)))
    source_idx = int(np.clip(int(source_idx), 0, max(0, num_frames - 1)))
    if num_frames <= clip_frames:
        return np.arange(num_frames, dtype=np.int64)
    if clip_frames == 1:
        return np.asarray([target_idx], dtype=np.int64)
    if source_idx != 0:
        window = [int(source_idx)]
        tail_len = clip_frames - 1
        seg_end = target_idx + 1
        seg_start = max(0, seg_end - tail_len)
        seg_end = min(num_frames, seg_start + tail_len)
        seg_start = max(0, seg_end - tail_len)
        for frame_idx in range(seg_start, seg_end):
            if frame_idx != source_idx:
                window.append(int(frame_idx))
        if target_idx not in window:
            window.append(int(target_idx))
        window = sorted(set(window))
        if len(window) > clip_frames:
            mandatory = {int(source_idx), int(target_idx)}
            ranked = sorted(
                [idx for idx in window if idx not in mandatory],
                key=lambda idx: (min(abs(idx - target_idx), abs(idx - source_idx)), idx),
            )
            keep = set(ranked[: max(0, clip_frames - len(mandatory))]) | mandatory
            window = sorted(keep)
        return np.asarray(window, dtype=np.int64)

    tail_len = clip_frames - 1
    seg_end = target_idx + 1
    seg_start = max(1, seg_end - tail_len)
    seg_end = min(num_frames, seg_start + tail_len)
    seg_start = max(1, seg_end - tail_len)
    tail = np.arange(seg_start, seg_end, dtype=np.int64)
    if target_idx not in tail:
        tail[-1] = target_idx
        tail = np.unique(tail)
        while tail.shape[0] < tail_len:
            cand = max(1, int(tail[0]) - 1)
            if cand in tail:
                break
            tail = np.concatenate([np.asarray([cand], dtype=np.int64), tail], axis=0)
        tail = np.sort(tail)[-tail_len:]
    return np.concatenate([np.asarray([0], dtype=np.int64), tail], axis=0)


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/infer_track_3d.py#L219-L232
def _build_query_for_targets(
    query_uv_norm: np.ndarray,
    t_src: np.ndarray,
    t_tgt: np.ndarray,
    t_cam: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "u": torch.from_numpy(query_uv_norm[:, 0]).to(device=device, dtype=torch.float32),
        "v": torch.from_numpy(query_uv_norm[:, 1]).to(device=device, dtype=torch.float32),
        "t_src": torch.from_numpy(t_src).to(device=device, dtype=torch.long),
        "t_tgt": torch.from_numpy(t_tgt).to(device=device, dtype=torch.long),
        "t_cam": torch.from_numpy(t_cam).to(device=device, dtype=torch.long),
    }


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/infer_track_3d.py#L235-L298
def _run_full_clip_queries(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    query_src_indices: np.ndarray | None,
    query_chunk_size: int,
    num_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    num_queries = int(query_uv_norm.shape[0])
    repeated_uv = np.repeat(query_uv_norm, num_frames, axis=0)
    if query_src_indices is None:
        query_src = np.zeros((num_queries,), dtype=np.int64)
    else:
        query_src = np.asarray(query_src_indices, dtype=np.int64).reshape(-1)
        if query_src.shape[0] != num_queries:
            raise ValueError(f"query_src_indices must have shape [{num_queries}], got {query_src.shape}")

    t_src = np.repeat(query_src, num_frames, axis=0)
    t_tgt = np.tile(np.arange(num_frames, dtype=np.int64), num_queries)
    memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_ratio)
    query_local = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=t_tgt.copy(),
        device=video_clip.device,
    )
    query_ref = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=np.zeros_like(t_tgt),
        device=video_clip.device,
    )
    pred_local = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_local,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    pred_ref = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_ref,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )

    def _reshape(pred: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in pred.items():
            arr = value.numpy()
            if arr.ndim == 1:
                out[key] = arr.reshape(num_queries, num_frames)
            else:
                out[key] = arr.reshape(num_queries, num_frames, *arr.shape[1:])
        return out

    return _reshape(pred_local), _reshape(pred_ref)


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/infer_track_3d.py#L301-L369
def _run_clip_queries_for_target_indices(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    memory: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    query_src_indices: np.ndarray | None,
    local_target_indices: np.ndarray,
    query_chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    target_ids = np.asarray(local_target_indices, dtype=np.int64).reshape(-1)
    num_queries = int(query_uv_norm.shape[0])
    num_targets = int(target_ids.shape[0])
    if num_targets <= 0:
        return {}, {}

    repeated_uv = np.repeat(query_uv_norm, num_targets, axis=0)
    if query_src_indices is None:
        query_src = np.zeros((num_queries,), dtype=np.int64)
    else:
        query_src = np.asarray(query_src_indices, dtype=np.int64).reshape(-1)
        if query_src.shape[0] != num_queries:
            raise ValueError(f"query_src_indices must have shape [{num_queries}], got {query_src.shape}")

    t_src = np.repeat(query_src, num_targets)
    t_tgt = np.tile(target_ids, num_queries)
    query_local = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=t_tgt.copy(),
        device=video_clip.device,
    )
    query_ref = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=np.zeros_like(t_tgt),
        device=video_clip.device,
    )
    pred_local = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_local,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    pred_ref = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_ref,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )

    def _reshape(pred: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in pred.items():
            arr = value.numpy()
            if arr.ndim == 1:
                out[key] = arr.reshape(num_queries, num_targets)
            else:
                out[key] = arr.reshape(num_queries, num_targets, *arr.shape[1:])
        return out

    return _reshape(pred_local), _reshape(pred_ref)


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/infer_track_3d.py#L372-L500
def _infer_tracks(
    *,
    model: torch.nn.Module,
    video_model_rgb: np.ndarray,
    native_aspect_ratio: float,    
    query_uv_norm: np.ndarray,
    query_chunk_size: int,
    query_src_indices_global: np.ndarray | None = None,
    umeyama_slide_window: bool = False,
    umeyama_slide_window_dense: bool = False,
    dense_grid_size: int = 32,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    num_frames = int(video_model_rgb.shape[0])
    num_queries = int(query_uv_norm.shape[0])
    clip_frames = _model_clip_frames(model)
    aspect_value = np.asarray([[native_aspect_ratio]], dtype=np.float32)
    aspect_tensor = torch.from_numpy(aspect_value).to(device=device, dtype=torch.float32)

    tracks_xyz_local = np.full((num_queries, num_frames, 3), np.nan, dtype=np.float32)
    tracks_xyz_ref0 = np.full_like(tracks_xyz_local, np.nan)
    tracks_uv = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    tracks_visibility = np.zeros((num_queries, num_frames), dtype=bool)
    tracks_visibility_logits = np.full((num_queries, num_frames), np.nan, dtype=np.float32)
    tracks_confidence = np.full((num_queries, num_frames), np.nan, dtype=np.float32)
    dense_mode = bool(umeyama_slide_window_dense)
    slide_window_enabled = bool(umeyama_slide_window) or dense_mode
    stitch_diagnostics: dict[str, Any] = {
        "mode": "umeyama_slide_window_dense" if dense_mode else ("umeyama_slide_window" if slide_window_enabled else "anchor_clip"),
        "clip_frames": int(clip_frames),
        "dense_grid_size": int(dense_grid_size) if dense_mode else 0,
        "keep_ratio": 0.85,
        "chunks": [],
    }
    if query_src_indices_global is None:
        query_src_indices_global = np.zeros((num_queries,), dtype=np.int64)
    else:
        query_src_indices_global = np.asarray(query_src_indices_global, dtype=np.int64).reshape(num_queries)

    video_tensor = (
        torch.from_numpy(video_model_rgb)
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        / 255.0
    )

    with torch.no_grad():
        if num_frames <= clip_frames:
            pred_local, pred_ref = _run_full_clip_queries(
                model=model,
                video_clip=video_tensor,
                aspect_ratio=aspect_tensor,
                query_uv_norm=query_uv_norm,
                query_src_indices=query_src_indices_global,
                query_chunk_size=query_chunk_size,
                num_frames=num_frames,
            )
            tracks_xyz_local[:] = pred_local["xyz_3d"].astype(np.float32)
            tracks_xyz_ref0[:] = pred_ref["xyz_3d"].astype(np.float32)
            tracks_uv[:] = pred_local["uv_2d"].astype(np.float32)
            tracks_visibility_logits[:] = pred_local["visibility"].astype(np.float32)
            tracks_visibility[:] = 1.0 / (1.0 + np.exp(-tracks_visibility_logits)) > 0.5
            tracks_confidence[:] = pred_local["confidence"].astype(np.float32)
        else:
            clip_groups: dict[tuple[int, ...], list[tuple[int, int, int, int]]] = {}
            for frame_idx in range(num_frames):
                for src_idx in sorted(set(int(v) for v in query_src_indices_global.tolist())):
                    clip_indices = _make_anchor_clip_indices(
                        num_frames=num_frames,
                        clip_frames=clip_frames,
                        target_idx=frame_idx,
                        source_idx=src_idx,
                    )
                    local_tgt_idx = int(np.flatnonzero(clip_indices == frame_idx)[0])
                    local_src_idx = int(np.flatnonzero(clip_indices == int(src_idx))[0])
                    clip_groups.setdefault(tuple(int(v) for v in clip_indices.tolist()), []).append(
                        (frame_idx, local_tgt_idx, int(src_idx), local_src_idx)
                    )

            for clip_key, assignments in clip_groups.items():
                clip_indices = np.asarray(clip_key, dtype=np.int64)
                video_clip = video_tensor[:, clip_indices]
                memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_tensor)

                by_src: dict[int, list[tuple[int, int, int]]] = {}
                for frame_idx, local_tgt_idx, src_idx, local_src_idx in assignments:
                    by_src.setdefault(int(src_idx), []).append(
                        (int(frame_idx), int(local_tgt_idx), int(local_src_idx))
                    )
                for src_idx, src_assignments in by_src.items():
                    global_frame_ids = np.asarray([item[0] for item in src_assignments], dtype=np.int64)
                    local_target_ids = np.asarray([item[1] for item in src_assignments], dtype=np.int64)
                    local_src_idx = int(src_assignments[0][2])
                    valid_idx = np.flatnonzero(query_src_indices_global == int(src_idx))
                    if valid_idx.size <= 0:
                        continue
                    local_src_ids = np.full((valid_idx.shape[0],), local_src_idx, dtype=np.int64)
                    pred_local, pred_ref = _run_clip_queries_for_target_indices(
                        model=model,
                        video_clip=video_clip,
                        aspect_ratio=aspect_tensor,
                        memory=memory,
                        query_uv_norm=query_uv_norm[valid_idx],
                        query_src_indices=local_src_ids,
                        local_target_indices=local_target_ids,
                        query_chunk_size=query_chunk_size,
                    )

                    tracks_xyz_local[valid_idx[:, None], global_frame_ids[None, :]] = pred_local["xyz_3d"].astype(np.float32)
                    tracks_xyz_ref0[valid_idx[:, None], global_frame_ids[None, :]] = pred_ref["xyz_3d"].astype(np.float32)
                    tracks_uv[valid_idx[:, None], global_frame_ids[None, :]] = pred_local["uv_2d"].astype(np.float32)
                    pred_vis_logits = pred_local["visibility"].astype(np.float32)
                    tracks_visibility_logits[valid_idx[:, None], global_frame_ids[None, :]] = pred_vis_logits
                    tracks_visibility[valid_idx[:, None], global_frame_ids[None, :]] = (
                        1.0 / (1.0 + np.exp(-pred_vis_logits)) > 0.5
                    )
                    tracks_confidence[valid_idx[:, None], global_frame_ids[None, :]] = pred_local["confidence"].astype(np.float32)

    return {
        "tracks_xyz_local": tracks_xyz_local,
        "tracks_xyz_ref0": tracks_xyz_ref0,
        "tracks_uv_norm": tracks_uv,
        "tracks_visibility": tracks_visibility,
        "tracks_visibility_logits": tracks_visibility_logits,
        "tracks_confidence": tracks_confidence,
        "clip_frames": np.asarray(clip_frames, dtype=np.int32),
        "stitch_diagnostics": stitch_diagnostics,
    }
