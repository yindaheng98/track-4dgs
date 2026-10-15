# Point Tracking for 4DGS

This repo is the **point tracking Python extension for 4D Gaussian Splatting**. It wraps sequence point trackers such as [CoTracker3](https://github.com/facebookresearch/co-tracker), [VGGT](https://github.com/facebookresearch/vggt), [MV-TAP](https://github.com/cvlab-kaist/MV-TAP), [OpenD4RT](https://github.com/Lijiaxin0111/Open-d4rt), [MATCHA](https://github.com/nv-dvl/matcha), [MV-RoMa](https://github.com/IceTea-CV/MV-RoMa), and [UniFlowMatch](https://github.com/UniFlowMatch/UFM) behind one small registry, then applies them either to plain image sequences or to multi-timestep Gaussian Splatting camera datasets.

The package provides three common workflows:

- track sampled 2D points through one ordered image sequence
- project 3D Gaussians into a reference timestep, track those projected points across all timesteps, and attach the resulting tracks back to Gaussian Splatting camera datasets
- build supervision samples (query, frames, and per-view tracks) from camera datasets, a trained 3DGS point cloud, or a Blender scene, and save them for later training

## Features

- [x] Organised as a standard Python package with `pip install` support
- [x] Shared point tracker registry with `cotracker3`, `vggt`, `vggsfm`, `mvtap`, `d4rt`, `matcha`, `mvroma`, `uniflowmatch`, and `blender`
- [x] Single-view image sequence tracking and rendering
- [x] Multi-timestep Gaussian Splatting camera dataset tracking
- [x] Point batching through `--batch-size` for long tracks
- [x] Camera dataset reordering against a selected reference timestep
- [x] Blender rigid-mesh tracking for scenes rendered by [NVS2COLMAP](https://github.com/yindaheng98/NVS2COLMAP)
- [x] Supervision datasets (`tracker`, `tracker_gs`, `blender`) that save query, frames, and tracks
- [ ] Port the motion estimation workflow from [TrackerSplat](https://github.com/yindaheng98/TrackerSplat) for point-tracker-driven motion synthesis
- [ ] Regularize 3DGS training with point tracker trajectories

## Install

### Prerequisites

- [PyTorch](https://pytorch.org/) (CUDA build recommended)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) matching your PyTorch installation
- [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting) `>= 2.3.8`

`pip install track-4dgs` also installs `Pillow`, `einops`, `timm`, `PyYAML`, `numpy`, `opencv-python`, `warp-lang`, and `diffusers>=0.30,<0.36`. Keep `diffusers` below `0.36` so it does not pull `huggingface-hub` 1.x, which breaks `transformers<5`.

The published wheel leaves out Git dependencies. Install them for the trackers that need them:

```shell
pip install wheel setuptools
pip install --upgrade "gaussian-splatting>=2.3.8"

# cotracker3
pip install --upgrade git+https://github.com/facebookresearch/co-tracker.git@main

# vggt and vggsfm
pip install --upgrade hydra-core omegaconf
pip install --upgrade git+https://github.com/facebookresearch/vggt.git@main
pip install --upgrade "git+https://github.com/jytime/LightGlue.git#egg=lightglue"

# uniflowmatch and mvroma
pip install --upgrade "git+https://github.com/castacks/UniCeption.git@ee7fa0b50d8295b3f0485e42c1d133fb534bfce7"
```

`matcha`, `mvroma`, and `uniflowmatch` ship inside this package. Their source trees are the `submodules/matcha`, `submodules/MV-RoMa`, and `submodules/UFM` submodules, so a source install needs `git clone --recursive`.

If you have trouble with `gaussian-splatting`, try installing it from source:

```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
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

`--no-deps` skips everything in [Prerequisites](#prerequisites). Install those packages first.

### Download Checkpoints

CoTracker3 offline checkpoint (`cotracker3`, default `checkpoints/scaled_offline.pth`):

```shell
mkdir -p checkpoints
wget -P checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

VGGT commercial checkpoint (`vggt`, default `checkpoints/vggt_1B_commercial.pt`):

```shell
mkdir -p checkpoints
wget -P checkpoints https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/vggt_1B_commercial.pt --header="Authorization: Bearer $HF_TOKEN"
```

If the VGGT checkpoint is not present, `VGGTPointTracker` falls back to `VGGT.from_pretrained("facebook/VGGT-1B")`. VGGT reads query features from frame 0, so every query frame index must be 0.

VGGSfM tracker checkpoint (`vggsfm`, default `checkpoints/vggsfm_v2_tracker.pt`):

```shell
mkdir -p checkpoints
wget -P checkpoints https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt
```

If the checkpoint is not present, `load_vggsfm` downloads it from that URL.

MATCHA checkpoint (`matcha`, default `checkpoints/matcha_pretrained.pth`, from the [MATCHA release](https://drive.google.com/file/d/17p0-ne4B9H60cC_AF6G284SetuImDUDO/view?usp=sharing)):

```shell
mkdir -p checkpoints
# save the downloaded file as checkpoints/matcha_pretrained.pth
```

MV-TAP checkpoint (`mvtap`, default `checkpoints/mvtap.ckpt`, from the [MV-TAP release](https://drive.google.com/file/d/1sCml0BL6VQGy-MGgpidz2-BdymAJhboU/view?usp=sharing)):

```shell
mkdir -p checkpoints
# save the downloaded file as checkpoints/mvtap.ckpt
```

MV-RoMa checkpoint (`mvroma`, default `checkpoints/outdoor_final.pth`, from the [MV-RoMa release](https://drive.google.com/file/d/19lrmYdZLD8nsiSLr-c5KFqENqCAv4W1D/view?usp=sharing)). Indoor weights are [here](https://drive.google.com/file/d/1L1GrTDy9N5u4CXR09fWOoAdCaQzgkeLV/view?usp=sharing). The tracker also downloads `infinity1096/UFM-Refine` as its prematcher. It runs on CUDA and accepts at most 31 reference frames.

```shell
mkdir -p checkpoints
# save the downloaded file as checkpoints/outdoor_final.pth
```

UniFlowMatch (`uniflowmatch`) downloads a Hugging Face repo on first use. `-m model="'base'"` selects `infinity1096/UFM-Base`. Other names are `refine`, `base-980`, `refine-980`, `base-dinov2l-init`, and `base-dinov2g-init`.

OpenD4RT checkpoint and model config (`d4rt`, default `checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt`):

```shell
# 48-frame model
mkdir -p checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG
wget -P checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG https://huggingface.co/Lijiaxin0111/OpenD4RT/resolve/main/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
wget -P checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG https://huggingface.co/Lijiaxin0111/OpenD4RT/resolve/main/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml

# 32-frame model
mkdir -p checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG
wget -P checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG https://huggingface.co/Lijiaxin0111/OpenD4RT/resolve/main/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
wget -P checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG https://huggingface.co/Lijiaxin0111/OpenD4RT/resolve/main/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml
```

Each model requires both `opend4rt.ckpt` and its adjacent `model.yaml`. Pass either the checkpoint file or its containing directory through the `checkpoint` tracker option.

## Command-Line Usage

### List Registered Point Trackers

Verify that `track_4dgs` can import and register trackers:

```shell
python -c "import track_4dgs; print(track_4dgs.get_available_point_trackers())"
```

| Name | Class | Kind | `track1v` |
| --- | --- | --- | --- |
| `cotracker3` | `Cotracker3PointTracker` | single-view | yes |
| `vggt` | `VGGTPointTracker` | single-view | yes |
| `vggsfm` | `VGGSfMPointTracker` | single-view | yes |
| `d4rt` | `D4RTPointTracker` | single-view | yes |
| `matcha` | `MatchaPointTracker` | single-view | yes |
| `mvroma` | `MVRoMaPointTracker` | single-view | yes |
| `uniflowmatch` | `UniFlowMatchPointTracker` | single-view | yes |
| `mvtap` | `MVTAPPointTracker` | joint multi-view | no |
| `blender` | `BlenderPointTracker` | joint rigid-mesh | no |

`track1v` accepts an `AbstractViewPointTracker`. Joint trackers run through `build_dataset`. `blender` is documented in [Blender mesh tracking](#blender-mesh-tracking).

### Track One Image Sequence

```shell
python -m track_4dgs.track1v \
    -s data/frame_000.png data/frame_001.png data/frame_002.png \
    -d output/track1v-cotracker3 \
    --tracker cotracker3 \
    --num-points 256
```

The command samples random query points on the first image, tracks them through all source images, and saves a rendered track overlay sequence under the destination directory. Overlay color follows confidence: red is low, green is high.

Tracker-specific options can be passed with repeated `-m/--option_tracker` values. `--batch-size` forwards chunks of query points to the tracker; omit it to track every point in one call.

```shell
python -m track_4dgs.track1v \
    -s data/frame_000.png data/frame_001.png data/frame_002.png \
    -d output/track1v-vggt \
    --tracker vggt \
    --batch-size 128 \
    -m checkpoint="'checkpoints/vggt_1B_commercial.pt'" \
    -m iters=4
```

### Build a Supervision Dataset

`python -m track_4dgs.build_dataset` runs a registered supervision dataset. `--save-folder` and `--vis-folder` are both optional, and at least one is required. With `--save-folder`, it writes one folder per sample:

```text
<save-folder>/supervision<index>/
    query.pt
    frame1/ ...          # COLMAP images and sparse/0
    0.pt 1.pt ...        # one Track per view
```

`--save-start` is the first index and defaults to 1, so the first saved sample is `supervision1`. `--vis-folder` writes rainbow overlays. If `--save-folder` is set, the overlays are drawn from the saved samples. If it is omitted, `build_dataset` draws the in-memory dataset directly and does not write it to disk. Omit `--vis-folder` to save without overlays.

List the dataset names:

```shell
python -c "import track_4dgs; from track_4dgs.dataset.registry import get_available_supervision_datasets; print(get_available_supervision_datasets())"
```

- `tracker`: sample points uniformly inside the scene cube, keep those seen by more cameras than `--visible-cameras` (default 2), project them, and track
- `tracker_gs`: sample a random subset of a trained 3DGS point cloud, then track those points
- `blender`: render a `.blend` file from sampled cameras and track the query on its rigid meshes. See [Blender mesh tracking](#blender-mesh-tracking)

Uniform scene points:

```shell
python -m track_4dgs.build_dataset \
    --dataset tracker \
    --tracker cotracker3 \
    -s data/sequence/frame_000 data/sequence/frame_001 data/sequence/frame_002 \
    --num-points 256 \
    --visible-cameras 2 \
    --save-folder output/sequence/dataset-cotracker3 \
    --vis-folder output/sequence/dataset-cotracker3-vis
```

Gaussian point cloud. `-d` and `-i` name the run whose checkpoint is sampled:

```text
<destination>/point_cloud/iteration_<iteration>/point_cloud.ply
```

```shell
python -m track_4dgs.build_dataset \
    --dataset tracker_gs \
    --tracker mvtap \
    -s data/sequence/frame_000 data/sequence/frame_001 data/sequence/frame_002 \
    -d output/sequence \
    -i 30000 \
    --num-points 256 \
    --batch-size 32 \
    --save-folder output/sequence/dataset-mvtap \
    --vis-folder output/sequence/dataset-mvtap-vis
```

Repeat `-s/--sources` to add another sample. `--load_cameras` and `--init-dataset-index` attach to the preceding `--sources` and may each appear once in that group. `--init-dataset-index` defaults to 0. For `tracker_gs`, repeat `-d` and `-i` once per `--sources`.

### Blender Mesh Tracking

`track_4dgs.blender` (`--tracker blender`) binds query pixels to rigid meshes in a `.blend` file and carries them with each mesh `matrix_world`. Use it together with the Blender tools in [NVS2COLMAP](https://github.com/yindaheng98/NVS2COLMAP).

[NVS2COLMAP](https://github.com/yindaheng98/NVS2COLMAP) renders that blend file into a Neural 3D Video scene (`poses_bounds.npy` and one video per camera), then `nvs2colmap.n3dv` writes one COLMAP camera dataset per frame. This tracker reads the same `.blend` file and those frame datasets. Cameras stay in the blend world, look along +Z, and treat image +Y as downward, which is the OpenCV camera NVS2COLMAP stores in `poses_bounds.npy`. Dataset frame `i` in `--sources` is Blender frame `frame_start + i * frame_step`, and that frame is at most `frame_end`. Pass the same `--frame_start`, `--frame_end`, and `--frame_step` as `nvs2colmap.blender`. `frame_start` defaults to 1. `frame_end` and `frame_step` default to the blend scene. Install NVS2COLMAP and keep a Blender executable on `PATH` as `blender`, or pass another path with `-m blender="'path/to/blender'"`. `warp-lang` is already installed with this package.

`nvs2colmap-sample-cameras` can place the cameras and write `poses_bounds.npy`. Render the blend file from those cameras:

```shell
python -m nvs2colmap.blender \
    --path data/scene \
    --blend data/scene.blend
```

Decode the rendered videos into per-frame COLMAP datasets:

```shell
python -m nvs2colmap.n3dv \
    --path data/scene \
    --ffmpeg ffmpeg \
    --ffprobe ffprobe
```

Track those frames with the same blend file. This only writes overlays:

```shell
python -m track_4dgs.build_dataset \
    --dataset tracker_gs \
    --tracker blender \
    -s data/scene/frame1 data/scene/frame2 data/scene/frame3 \
    -d output/scene \
    -i 30000 \
    -m blend="'data/scene.blend'" \
    --vis-folder output/scene/track-blender-vis
```

Copy `--frame_start`, `--frame_end`, and `--frame_step` from the render command onto the tracker. `nvs2colmap.blender --frame_step` already drops frames inside each camera video, and `nvs2colmap.n3dv` still writes consecutive `frame*` folders. Leave `nvs2colmap.n3dv --start-number` at its default of 1: that flag counts frames inside the rendered video, so `frame1` is the first frame the Blender render wrote. `--sources` then lists `frame1`, `frame2`, ... in order.

```shell
python -m nvs2colmap.blender \
    --path data/scene \
    --blend data/scene.blend \
    --frame_start 10 \
    --frame_end 14 \
    --frame_step 2

python -m track_4dgs.build_dataset \
    --dataset tracker_gs \
    --tracker blender \
    -s data/scene/frame1 data/scene/frame2 data/scene/frame3 \
    -d output/scene \
    -i 30000 \
    -m blend="'data/scene.blend'" \
    -m frame_start=10 \
    -m frame_end=14 \
    -m frame_step=2 \
    --vis-folder output/scene/track-blender-vis
```

`build_dataset --dataset blender` can skip the COLMAP export and render supervision directly. It samples OpenCV cameras on a spherical cap, samples the query near the mesh at `--init-dataset-index`, renders every view, and binds the tracks to the rigid meshes. `--frame_start`, `--frame_end`, `--frame_step`, and `--init-dataset-index` attach to the preceding `--blend`. Repeat `--blend` for another sample.

```shell
python -m track_4dgs.build_dataset \
    --dataset blender \
    --blend data/scene.blend \
    --n-cameras 8 \
    --height 512 \
    --width 512 \
    --focal 500 \
    --num-points 256 \
    --save-folder output/scene/dataset-blender \
    --vis-folder output/scene/dataset-blender-vis
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

`sample_query` returns pixel coordinates and a frame index of 0 for every point. `track_view` is the single-view entry used by `track1v`.

```python
import torch

from track_4dgs.track1v import load_image, sample_query

frames = [
    load_image("data/frame_000.png", "cuda"),
    load_image("data/frame_001.png", "cuda"),
]
points, frame_indices = sample_query(frames[0], num_points=256)

with torch.no_grad():
    track = tracker.track_view(
        points,
        frame_indices,
        frames,
        [None] * len(frames),
    )

print(track.points.shape)      # [num_frames, num_points, 2]
print(track.visibility.shape)  # [num_frames, num_points]
print(track.confidence.shape)  # [num_frames, num_points]
```

### Track Gaussian Splatting Camera Datasets

`build_camera_dataset_tracker` and `CameraDatasetTracker` are reserved. They attach each view's track onto the cameras as `camera.custom_data["track"]`. `build_dataset` does not call them; it keeps the view-major `Supervision` tracks.

```python
from gaussian_splatting.prepare import prepare_gaussians

from track_4dgs.dataset import build_camera_dataset_tracker, prepare_datasets, query_views_from_gaussians

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
dataset_tracker = build_camera_dataset_tracker(
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
tracked_datasets = dataset_tracker(queries, datasets, batch_size=128)

camera_track = tracked_datasets[0][0].custom_data["track"]
print(camera_track.points.shape)  # [num_points, 2] for this camera in this frame
```

### Load Saved Supervision

```python
from track_4dgs import SavedDataset

dataset = SavedDataset("output/sequence/dataset-mvtap", start=1)
sample = dataset[0]

print(sample.query.points.shape)       # [num_views, num_points, 2]
print(len(sample.frames))               # num_frames
print(sample.tracks[0].points.shape)    # [num_frames, num_points, 2]
```

`Supervision` holds a view-major `Query`, one camera dataset per frame, and one `Track` per view. `AbstractDataset.dump` writes that layout; `SavedDataset` reads it back.

## Design: Point Tracker Registry

`AbstractPointTracker.__call__` validates inputs, then calls `track`. It takes a view-major `Query` (`points` `[V, N, 2]`, `frame_indices` `[V, N]`, `in_image` `[V, N]`) and frame-major camera datasets, and returns one `Track` per view. `batch_size` splits the point dimension; `None` tracks every point in one call. `build_dataset` keeps that view-major `Supervision` layout.

`CameraDatasetTracker` and `build_camera_dataset_tracker` are reserved for a later step that hangs those tracks back on the cameras. `build_camera_dataset_tracker(tracker_name, device, tracker_configs)` builds a registered point tracker and wraps it. Calling the wrapper with a view-major `Query` and frame-major camera datasets returns one `TrackedCameraDataset` per frame. Each camera then carries its `CameraTrack` on `camera.custom_data["track"]`. `build_dataset` does not call this path:

```text
query + frames -> validate + track -> view_tracks -> tracked camera datasets
```

Single-view trackers (`AbstractViewPointTracker`, e.g. CoTracker3 / VGGT / MATCHA) split the query by view and implement `track_batch(points, frame_indices, frames, masks)` for one camera sequence. Joint trackers (`AbstractBatchPointTracker`, e.g. MV-TAP / Blender) consume all views at once through `track_batch(query, frames)` and batch along the point dimension.

```text
Frame 0 cameras --\
Frame 1 cameras ----> single-view loop, or one joint multi-view call ----> view_tracks
Frame 2 cameras --/
```

`in_image` marks query pixels that project inside the frame (and in front of the camera, when the query comes from `Query.from_projection`). Single-view trackers skip points that fail this test. Joint trackers fold it into `Track.mask`.

This keeps model-specific code isolated in tracker implementations while the 4DGS workflow can use any registered tracker by name.

## Extending: Adding a New Point Tracker

Single-view trackers return `Track(points=[D, N, 2], visibility=[D, N], confidence=[D, N], mask=[D, N])` from `track_batch`:

```python
from collections.abc import Sequence

import torch

from track_4dgs.tracker import AbstractViewPointTracker, Track


class MyPointTracker(AbstractViewPointTracker):
    def to(self, device):
        self.device = torch.device(device)
        return self

    def track_batch(
        self,
        points: torch.Tensor,
        frame_indices: torch.Tensor,
        frames: Sequence[torch.Tensor],
        frame_masks: Sequence[torch.Tensor | None],
    ) -> Track:
        ...
```

A joint tracker subclasses `AbstractBatchPointTracker` and implements `track_batch(self, query, frames) -> Sequence[Track]`, one `Track` per view.

Register it at import time:

```python
from track_4dgs.registry import register_point_tracker

register_point_tracker("mytracker", MyPointTracker)
```

After registration, it is available everywhere a single-view tracker is accepted:

```shell
python -m track_4dgs.track1v --tracker mytracker -s frame0.png frame1.png -d output/mytracker
```

Import the module that calls `register_point_tracker` before `build_point_tracker`. The built-in trackers register themselves from `track_4dgs/__init__.py`.

## Acknowledgement

This repo is developed based on [CoTracker](https://github.com/facebookresearch/co-tracker), [VGGT](https://github.com/facebookresearch/vggt), [MV-TAP](https://github.com/cvlab-kaist/MV-TAP), [OpenD4RT](https://github.com/Lijiaxin0111/Open-d4rt), [MATCHA](https://github.com/nv-dvl/matcha), [MV-RoMa](https://github.com/IceTea-CV/MV-RoMa), [UniFlowMatch](https://github.com/UniFlowMatch/UFM), [UniCeption](https://github.com/castacks/UniCeption), [LightGlue](https://github.com/jytime/LightGlue), [gaussian-splatting (packaged)](https://github.com/yindaheng98/gaussian-splatting), and [NVS2COLMAP](https://github.com/yindaheng98/NVS2COLMAP). Many thanks to the authors for open-sourcing their codebases.
