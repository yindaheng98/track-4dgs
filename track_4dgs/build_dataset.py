import torch

from track_4dgs.dataset.registry import get_available_supervision_datasets, get_supervision_dataset


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, choices=get_available_supervision_datasets())
    args, rest = parser.parse_known_args()
    dataset, unknown = get_supervision_dataset(args.dataset).from_argv(rest)

    parser = ArgumentParser()
    parser.add_argument("--save-folder", required=True, type=str)
    parser.add_argument("--save-start", default=0, type=int)
    args = parser.parse_args(unknown)

    with torch.no_grad():
        print(f"Saving {len(dataset)} samples to {args.save_folder} starting at supervision{args.save_start}")
        dataset.dump(args.save_folder, args.save_start)
