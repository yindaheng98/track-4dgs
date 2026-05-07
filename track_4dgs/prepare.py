from typing import List, Optional

from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset

from .tracker import ReorderedCameraDataset


def prepare_datasets(
        sources: List[str], device: str,
        trainable_camera: bool = False, load_cameras: List[str] = None, load_mask=True, load_depth=True,
        reorder_reference_source: Optional[str] = None,
) -> List[CameraDataset]:
    """Load cameras for all sources and optionally align them to a reference.

    When reorder_reference_source is provided, it is loaded as a plain Gaussian
    Splatting dataset and every returned dataset is reordered to match it.
    Otherwise datasets keep their original order.
    """
    assert len(sources) > 0, "sources must not be empty"

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
    if reorder_reference_source is None:
        return datasets

    reorder_reference = prepare_dataset(
        source=reorder_reference_source, device=device,
        trainable_camera=trainable_camera,
        load_mask=load_mask, load_depth=load_depth,
    )
    datasets = [
        ReorderedCameraDataset(
            dataset=dataset,
            dataset_source=source,
            reference=reorder_reference,
            reference_source=reorder_reference_source,
        )
        for dataset, source in zip(datasets, sources)
    ]
    del reorder_reference
    return datasets
