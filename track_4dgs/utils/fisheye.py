"""Fisheye-to-pinhole remap tables."""

import cv2
import numpy as np
import torch


def fisheye_undistort_grid(
        width: int, height: int, fx: float, fy: float, cx: float, cy: float,
        model: str, distortion: tuple[float, ...],
) -> torch.Tensor:
    """``(H, W, 2)`` grid for :func:`torch.nn.functional.grid_sample`.

    Each entry is the fisheye-image location of one pinhole pixel, scaled to ``[-1, 1]``
    with ``align_corners=True``. The pinhole matrix is ``fx, fy, cx, cy``.
    """

    if model != "OPENCV_FISHEYE":
        raise ValueError(f"unsupported camera model {model}")
    if len(distortion) != 4:
        raise ValueError(f"OPENCV_FISHEYE distortion has length {len(distortion)}, expected 4")
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    coeffs = np.asarray(distortion, dtype=np.float64).reshape(4, 1)
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        matrix, coeffs, np.eye(3, dtype=np.float64), matrix, (width, height), cv2.CV_32FC1,
    )
    xs = torch.tensor(map_x).mul(2.0 / (width - 1)).sub(1.0)
    ys = torch.tensor(map_y).mul(2.0 / (height - 1)).sub(1.0)
    return torch.stack((xs, ys), dim=-1)
