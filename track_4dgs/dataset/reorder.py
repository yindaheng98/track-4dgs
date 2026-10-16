import os
from typing import List, Optional

from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset


def relative_path(path: str, root: str) -> str:
    try:
        rel_path = os.path.relpath(path, root)
    except ValueError:
        rel_path = path
    return os.path.normcase(os.path.normpath(rel_path))


class ReorderedCameraDataset(CameraDataset):
    @staticmethod
    def find_matching_index(reference_camera: Camera, reference_source: str, dataset: CameraDataset, dataset_source: str) -> int:
        """Find the index of the best-matching camera in *dataset*."""
        rel_reference = relative_path(reference_camera.ground_truth_image_path, reference_source)
        for idx in range(len(dataset)):
            rel_dataset = relative_path(dataset[idx].ground_truth_image_path, dataset_source)
            if rel_dataset == rel_reference:
                return idx
        raise ValueError(f"No camera in dataset matches relative path: {rel_reference}")

    def __init__(
        self,
        dataset: CameraDataset,
        dataset_source: str,
        reference: CameraDataset,
        reference_source: str,
    ):
        self.dataset = dataset
        assert len(dataset) == len(reference), \
            "ReorderedCameraDataset requires dataset and reference to have the same length."
        self.index_map = [
            self.find_matching_index(reference[idx], reference_source, dataset, dataset_source)
            for idx in range(len(reference))
        ]
        assert sorted(self.index_map) == list(range(len(dataset))), \
            "ReorderedCameraDataset requires dataset and reference to be in one-to-one correspondence."

    @property
    def __class__(self):
        return self.dataset.__class__

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def to(self, device) -> "ReorderedCameraDataset":
        self.dataset = self.dataset.to(device)
        return self

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx):
        return self.dataset[self.index_map[idx]]


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
    return [
        ReorderedCameraDataset(
            dataset=dataset,
            dataset_source=source,
            reference=reorder_reference,
            reference_source=reference_source,
        )
        for dataset, source in zip(datasets, sources)
    ]
