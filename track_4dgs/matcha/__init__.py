import os

from track_4dgs.registry import register_point_tracker

# DIFT imports the PyTorch Stable Diffusion pipeline. diffusers then imports
# transformers, which loads TensorFlow when that package is installed. The
# installed TensorFlow build does not run on NumPy 2.
os.environ.setdefault("USE_TF", "0")

from .tracker import MatchaPointTracker

register_point_tracker("matcha", MatchaPointTracker)

__all__ = [
    "MatchaPointTracker",
]
