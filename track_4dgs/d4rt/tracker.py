from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from track_4dgs.tracker import AbstractViewPointTracker, Track

from .checkpoint import load_checkpoint
from .config import load_yaml_config
from .infer_track_3d import _infer_tracks, _resize_video, _unwrap_state_dict
from .model.d4rt import D4RTModel

DEFAULT_CHECKPOINT = "checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt"


def load_d4rt(
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        model_config: str | Path | Mapping[str, Any] | None = None) -> tuple[D4RTModel, dict[str, Any]]:
    """Load an OpenD4RT model and its model configuration.

    ``checkpoint`` may name either ``opend4rt.ckpt`` or its containing
    directory. When ``model_config`` is omitted, ``model.yaml`` is read beside
    the checkpoint.
    """
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "opend4rt.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"D4RT checkpoint not found: {checkpoint_path}")

    if model_config is None:
        model_config = checkpoint_path.with_name("model.yaml")
    if isinstance(model_config, Mapping):
        payload = dict(model_config)
    else:
        payload = load_yaml_config(model_config)
    model_cfg = payload["model"] if "model" in payload else payload
    if not isinstance(model_cfg, Mapping):
        raise ValueError("D4RT model config must contain a mapping under 'model'")
    model_cfg = dict(model_cfg)
    pretrained = model_cfg.get("encoder", {}).get("pretrained")
    if isinstance(pretrained, dict):
        pretrained["enabled"] = False

    # https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/eval_track3d_in_worldtrack.py#L447-L452
    model = D4RTModel(model_cfg).eval()
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=False)
    return model, model_cfg


class D4RTPointTracker(AbstractViewPointTracker):
    """Track queried points with OpenD4RT's 2D correspondence head.

    Frames must be RGB ``[3, H, W]`` tensors in ``[0, 1]`` and all frames in a
    view must have the same resolution. Query points are ``[x, y]`` pixels in
    their respective source frames. Visibility is a sigmoid probability in
    ``(0, 1)``. Confidence is the model's sigmoid localization score; unlike
    CoTracker/VGGT confidence, it is not trained against a fixed pixel-error
    threshold.

    Videos longer than the model's configured clip length are evaluated with
    source-preserving anchor clips, matching OpenD4RT's inference protocol.
    """

    def __init__(
            self,
            checkpoint: str | Path = DEFAULT_CHECKPOINT,
            model_config: str | Path | Mapping[str, Any] | None = None,
            query_chunk_size: int = 4096):
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be a positive integer")
        self.model, self.model_config = load_d4rt(
            checkpoint=checkpoint,
            model_config=model_config,
        )
        input_config = self.model_config.get("input", {})
        image_size = input_config.get("image_size", [256, 256])
        if not isinstance(image_size, Sequence) or len(image_size) != 2:
            raise ValueError("D4RT model.input.image_size must contain [height, width]")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.query_chunk_size = query_chunk_size
        self.device = torch.device("cpu")
        self.model.eval()

    def to(self, device) -> 'D4RTPointTracker':
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
        del frame_masks  # D4RT predicts visibility directly.
        if any(frame.shape[0] != 3 for frame in frames):
            raise ValueError("D4RTPointTracker expects RGB frames with shape [3, H, W]")
        if any(frame.shape[-2:] != frames[0].shape[-2:] for frame in frames):
            raise ValueError("D4RTPointTracker expects all frames in a view to have the same resolution")

        original_h, original_w = frames[0].shape[-2:]
        video_rgb = (
            torch.stack(list(frames), dim=0)
            .detach()
            .clamp(0, 1)
            .mul(255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        # https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/eval_track3d_in_worldtrack.py#L501
        video_model_rgb = _resize_video(video_rgb, image_hw=self.image_size)

        # https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/eval_track3d_in_worldtrack.py#L518-L521
        query_uv_norm = points.detach().cpu().numpy().astype(np.float32)
        query_uv_norm[:, 0] /= float(max(original_w - 1, 1))
        query_uv_norm[:, 1] /= float(max(original_h - 1, 1))
        query_uv_norm = np.clip(query_uv_norm, 0.0, 1.0)

        pred = _infer_tracks(
            model=self.model,
            video_model_rgb=video_model_rgb,
            native_aspect_ratio=float(original_w) / float(max(original_h, 1)),
            query_uv_norm=query_uv_norm,
            query_chunk_size=int(self.query_chunk_size),
            query_src_indices_global=frame_indices.detach().cpu().numpy(),
        )
        scale = np.asarray(
            [float(max(original_w - 1, 1)), float(max(original_h - 1, 1))],
            dtype=np.float32,
        )
        track_points = pred["tracks_uv_norm"] * scale
        vis_logits = pred["tracks_visibility_logits"]
        visibility = 1.0 / (1.0 + np.exp(-vis_logits))
        confidence = 1.0 / (1.0 + np.exp(-pred["tracks_confidence"]))
        mask = (
            np.isfinite(track_points).all(axis=-1)
            & np.isfinite(visibility)
            & np.isfinite(confidence)
        )

        track_points = torch.from_numpy(np.transpose(track_points, (1, 0, 2))).to(
            device=points.device, dtype=points.dtype)
        visibility = torch.from_numpy(np.transpose(visibility, (1, 0))).to(
            device=points.device, dtype=points.dtype)
        confidence = torch.from_numpy(np.transpose(confidence, (1, 0))).to(
            device=points.device, dtype=points.dtype)
        mask = torch.from_numpy(np.transpose(mask, (1, 0))).to(device=points.device)
        return Track(
            points=track_points,
            visibility=visibility,
            confidence=confidence,
            mask=mask,
        )
