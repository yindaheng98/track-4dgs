from .dataset import AbstractDataset


REGISTRY: dict[str, type[AbstractDataset]] = {}


def register_supervision_dataset(name: str, dataset_cls: type[AbstractDataset]) -> None:
    """Register a supervision dataset class under *name*."""
    if name in REGISTRY:
        raise ValueError(f"Supervision dataset '{name}' is already registered.")
    REGISTRY[name] = dataset_cls
