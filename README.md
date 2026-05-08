# Point Tracking for 4DGS

This repo is the **point tracking Python extension for 4D Gaussian Splatting**. It wraps sequence point trackers such as [CoTracker3](https://github.com/facebookresearch/co-tracker) and [VGGT](https://github.com/facebookresearch/vggt) behind one small registry, then applies them either to plain image sequences or to multi-timestep Gaussian Splatting camera datasets.

The package provides two common workflows:

* track sampled 2D points through one ordered image sequence
* project 3D Gaussians into a reference timestep, track those projected points across all timesteps, and attach the resulting tracks back to Gaussian Splatting camera datasets

## Features

* [x] Organised as a standard Python package with `pip install` support
* [x] Shared point tracker registry with `cotracker3` and `vggt` implementations
* [x] Single-view image sequence tracking and rendering
* [x] Multi-timestep Gaussian Splatting camera dataset tracking
* [x] Camera dataset reordering against a selected reference timestep
* [ ] Port the motion estimation workflow from [TrackerSplat](https://github.com/yindaheng98/TrackerSplat) for point-tracker-driven motion synthesis
* [ ] Regularize 3DGS training with point tracker trajectories

## Install

### Prerequisites

* [PyTorch](https://pytorch.org/) (CUDA build recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) matching your PyTorch installation
* [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting)
* [`CoTracker`](https://github.com/facebookresearch/co-tracker)
* [`VGGT`](https://github.com/facebookresearch/vggt) for the optional `vggt` tracker

Install the core Gaussian Splatting dependency used by the 4DGS dataset utilities:

```shell
pip install wheel setuptools
pip install --upgrade gaussian-splatting
```

If you have trouble with `gaussian-splatting`, try installing it from source:

```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
```

Install tracker dependencies used by this package:

```shell
pip install --upgrade git+https://github.com/facebookresearch/co-tracker.git@main
pip install --upgrade Pillow hydra-core omegaconf
pip install --upgrade git+https://github.com/facebookresearch/vggt.git@main
pip install --upgrade git+https://github.com/jytime/LightGlue.git#egg=lightglue
```

## PyPI Install

```shell
pip install --upgrade track-4dgs
```

or build latest from source:

```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/track-4dgs.git@master --no-build-isolation
```

### Development Install

```shell
git clone --recursive https://github.com/yindaheng98/track-4dgs.git
cd track-4dgs
pip install --target . --upgrade . --no-deps
```

### Download Checkpoints

CoTracker3 offline checkpoint:

```shell
mkdir -p checkpoints
wget -P checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

VGGT commercial checkpoint:

```shell
mkdir -p checkpoints
wget -P checkpoints https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/vggt_1B_commercial.pt --header="Authorization: Bearer $HF_TOKEN"
```

If the VGGT checkpoint is not present, `VGGTPointTracker` falls back to `VGGT.from_pretrained("facebook/VGGT-1B")`.

## Command-Line Usage

### List Registered Point Trackers

Verify that `track_4dgs` can import and register trackers:

```shell
python -c "import track_4dgs; print(track_4dgs.get_available_point_trackers())"
```

The built-in trackers are:

* `cotracker3`: CoTracker3 offline point tracker
* `vggt`: VGGT TrackHead point tracker

### Track One Image Sequence

```shell
python -m track_4dgs.track1v \
    -s data/frame_000.png data/frame_001.png data/frame_002.png \
    -d output/track1v-cotracker3 \
    --tracker cotracker3 \
    --num-points 256
```

The command samples random query points on the first image, tracks them through all source images, and saves a rendered track overlay sequence under the destination directory.

Tracker-specific options can be passed with repeated `-m/--option_tracker` values:

```shell
python -m track_4dgs.track1v \
    -s data/frame_000.png data/frame_001.png data/frame_002.png \
    -d output/track1v-vggt \
    --tracker vggt \
    -m checkpoint="'checkpoints/vggt_1B_commercial.pt'" \
    -m iters=4
```

### Track 4DGS Camera Datasets

```shell
python -m track_4dgs.track2d \
    -s data/sequence/frame_000 data/sequence/frame_001 data/sequence/frame_002 \
    -d output/sequence \
    -i 30000 \
    --tracker cotracker3 \
    --init-dataset-index 0 \
    --num-points 256
```

`track2d` loads a trained Gaussian point cloud from:

```text
<destination>/point_cloud/iteration_<iteration>/point_cloud.ply
```

It projects Gaussian centers into every camera of the reference timestep, tracks those projected points across the same camera views in all timesteps, and saves visualisations under:

```text
<destination>/ours_<iteration>/track2d-<tracker>/
```

If cameras were saved outside the source scene folders, pass one camera path per source:

```shell
python -m track_4dgs.track2d \
    -s data/sequence/frame_000 data/sequence/frame_001 \
    -d output/sequence \
    -i 30000 \
    --load_cameras output/frame_000/cameras.json output/frame_001/cameras.json \
    --mode camera
```

## API Usage

### Build a Tracker

```python
from track_4dgs.registry import build_point_tracker, get_available_point_trackers

print(get_available_point_trackers())

tracker = build_point_tracker(
    "cotracker3",
    checkpoint="checkpoints/scaled_offline.pth",
).to("cuda")
```

### Track Image Tensors

```python
import torch

from track_4dgs.track1v import load_image, sample_query

frames = [
    load_image("data/frame_000.png", "cuda"),
    load_image("data/frame_001.png", "cuda"),
]
query = sample_query(frames[0], num_points=256)

with torch.no_grad():
    track = tracker(query, frames, [None] * len(frames))

print(track.points.shape)      # [num_frames, num_points, 2]
print(track.visibility.shape)  # [num_frames, num_points]
```

### Track Gaussian Splatting Camera Datasets

```python
from gaussian_splatting.prepare import prepare_gaussians

from track_4dgs.prepare import prepare_datasets, prepare_tracker
from track_4dgs.track2d import query_views_from_gaussians

sources = [
    "data/sequence/frame_000",
    "data/sequence/frame_001",
]
datasets = prepare_datasets(
    sources=sources,
    device="cuda",
    reorder_reference_idx=0,
)
gaussians = prepare_gaussians(
    sh_degree=3,
    source=sources[0],
    device="cuda",
    load_ply="output/sequence/point_cloud/iteration_30000/point_cloud.ply",
)
dataset_tracker = prepare_tracker(
    tracker_name="cotracker3",
    device="cuda",
    tracker_configs={"checkpoint": "checkpoints/scaled_offline.pth"},
)

queries = query_views_from_gaussians(
    datasets=datasets,
    gaussians=gaussians,
    init_dataset_index=0,
    num_points=256,
)
tracked_datasets = dataset_tracker(queries, datasets)

camera_track = tracked_datasets[0][0].custom_data["track"]
print(camera_track.points.shape)
```

## Design: Point Tracker Registry

The core abstraction is `AbstractPointTracker`. A tracker receives one `Query`, a frame-major list of images for one camera view, and optional masks:

```text
Query points + frame sequence -> Point Tracker -> tracked points + visibility
```

`CameraDatasetTracker` lifts the same tracker to Gaussian Splatting datasets by iterating over views. Input datasets are frame-major, while queries are view-major:

```text
Frame 0 cameras --\
Frame 1 cameras ----> per-view tracking ----> tracked camera datasets
Frame 2 cameras --/
```

This keeps model-specific code isolated in tracker implementations while the 4DGS workflow can use any registered tracker by name.

## Extending: Adding a New Point Tracker

Create a tracker class that returns `Track(points=[D, N, 2], visibility=[D, N])`:

```python
from collections.abc import Sequence

import torch

from track_4dgs.tracker import AbstractPointTracker, Query, Track


class MyPointTracker(AbstractPointTracker):
    def to(self, device):
        self.device = torch.device(device)
        return self

    def track(
        self,
        query: Query,
        frames: Sequence[torch.Tensor],
        frame_masks: Sequence[torch.Tensor | None],
    ) -> Track:
        ...
```

Register it at import time:

```python
from track_4dgs.registry import register_point_tracker

register_point_tracker("mytracker", MyPointTracker)
```

After registration, it is available everywhere:

```shell
python -m track_4dgs.track1v --tracker mytracker -s frame0.png frame1.png -d output/mytracker
```

## Acknowledgement

This repo is developed based on [CoTracker](https://github.com/facebookresearch/co-tracker), [VGGT](https://github.com/facebookresearch/vggt), [LightGlue](https://github.com/jytime/LightGlue), and [gaussian-splatting (packaged)](https://github.com/yindaheng98/gaussian-splatting). Many thanks to the authors for open-sourcing their codebases.
