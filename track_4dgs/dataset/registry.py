from .dataset import AbstractDataset


REGISTRY: dict[str, type[AbstractDataset]] = {}


def register_supervision_dataset(name: str, dataset_cls: type[AbstractDataset]) -> None:
    """Register a supervision dataset class under *name*."""
    if name in REGISTRY:
        raise ValueError(f"Supervision dataset '{name}' is already registered.")
    REGISTRY[name] = dataset_cls


def get_available_supervision_datasets() -> list[str]:
    """Return the names of all registered supervision datasets."""
    return list(REGISTRY.keys())


def get_supervision_dataset(name: str) -> type[AbstractDataset]:
    """Return the supervision dataset class registered under *name*."""
    if name not in REGISTRY:
        raise KeyError(
            f"Supervision dataset '{name}' not found. "
            f"Available: {get_available_supervision_datasets()}"
        )
    return REGISTRY[name]
