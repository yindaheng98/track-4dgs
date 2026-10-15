import os

import torch

from track_4dgs.dataset import AbstractDataset, SavedDataset
from track_4dgs.dataset.registry import get_available_supervision_datasets, get_supervision_dataset
from track_4dgs.track1v import draw_rainbow_tracks


def rendering(dataset: AbstractDataset, save: str) -> None:
    start = getattr(dataset, "start", 0)
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_dir = os.path.join(save, f"supervision{start + index}")
        print(f"Saving {sample.tracks[0].points.shape[1]} tracked points to {sample_dir}")
        for frame_idx, frame in enumerate(sample.frames, start=1):
            for view_idx, track in enumerate(sample.tracks):
                canvas = draw_rainbow_tracks(
                    image=frame[view_idx].ground_truth_image,
                    track=track,
                    frame_idx=frame_idx - 1,
                )
                path = os.path.join(sample_dir, f"frame{frame_idx}", f"{view_idx}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                canvas.save(path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, choices=get_available_supervision_datasets())
    parser.add_argument("--save-folder", default=None, type=str)
    parser.add_argument("--save-start", default=1, type=int)
    parser.add_argument("--vis-folder", default=None, type=str)
    args, rest = parser.parse_known_args()
    assert args.save_folder is not None or args.vis_folder is not None, "--save-folder and --vis-folder cannot both be None"
    dataset, unknown = get_supervision_dataset(args.dataset).from_argv(rest)
    if unknown:
        parser.error("unrecognized arguments: " + " ".join(unknown))

    with torch.no_grad():
        if args.save_folder is not None:
            print(f"Saving {len(dataset)} samples to {args.save_folder} starting at supervision{args.save_start}")
            dataset.dump(args.save_folder, args.save_start)
            if args.vis_folder is not None:
                rendering(SavedDataset(args.save_folder, args.save_start), args.vis_folder)
        else:
            rendering(dataset, args.vis_folder)
