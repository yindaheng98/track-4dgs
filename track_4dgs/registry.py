from .tracker import AbstractPointTracker


REGISTRY: dict[str, type[AbstractPointTracker]] = {}


def register_point_tracker(name: str, tracker_cls: type[AbstractPointTracker]) -> None:
    """Register a point tracker class under *name*."""
    if name in REGISTRY:
        raise ValueError(f"Point tracker '{name}' is already registered.")
    REGISTRY[name] = tracker_cls


def get_available_point_trackers() -> list[str]:
    """Return the names of all registered point trackers."""
    return list(REGISTRY.keys())


def build_point_tracker(name: str, **configs: object) -> AbstractPointTracker:
    """Build a point tracker by name."""
    if name not in REGISTRY:
        raise KeyError(
            f"Point tracker '{name}' not found. "
            f"Available: {get_available_point_trackers()}"
        )
    return REGISTRY[name](**configs)
