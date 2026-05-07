from typing import List, Optional

from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset

from .registry import build_point_tracker
from .tracker import CameraDatasetTracker, ReorderedCameraDataset


def prepare_datasets(
        sources: List[str], device: str,
        trainable_camera: bool = False, load_cameras: Optional[List[str]] = None, load_mask=True, load_depth=True,
        reorder_reference_idx: Optional[int] = None,
) -> List[CameraDataset]:
    """Load cameras for all sources and optionally align them to a reference.

    When reorder_reference_idx is provided, every returned dataset is reordered
    to match that dataset. Otherwise datasets keep their original order.
    """
    assert len(sources) > 0, "sources must not be empty"
    if reorder_reference_idx is not None:
        assert 0 <= reorder_reference_idx < len(sources), "reorder_reference_idx must point to one of the sources"

    load_cameras = load_cameras if load_cameras is not None else [None] * len(sources)
    assert len(load_cameras) == len(sources), "len(load_cameras) must equal len(sources)"
    datasets = [
        prepare_dataset(
            source=source, device=device,
            trainable_camera=trainable_camera, load_camera=load_camera,
            load_mask=load_mask, load_depth=load_depth,
        )
        for source, load_camera in zip(sources, load_cameras)
    ]
    if reorder_reference_idx is None:
        return datasets

    reorder_reference = datasets[reorder_reference_idx]
    reference_source = sources[reorder_reference_idx]
    datasets = [
        ReorderedCameraDataset(
            dataset=dataset,
            dataset_source=source,
            reference=reorder_reference,
            reference_source=reference_source,
        )
        for dataset, source in zip(datasets, sources)
    ]
    return datasets


def prepare_tracker(tracker_name: str, device: str, tracker_configs: dict = {}) -> CameraDatasetTracker:
    tracker = build_point_tracker(tracker_name, **tracker_configs)
    return CameraDatasetTracker(tracker).to(device)
