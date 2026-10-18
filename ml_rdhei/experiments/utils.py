import shutil
import random
import argparse
from pathlib import Path


def setup_datasets(
    BOSSBase_path: Path | str,
    BOWS2_path: Path | str,
    CAR_path: Path | str,
    TCIA_path: Path | str) -> None:
    """Copies the datasets into the correct folder structure for training and testing.

    Args:
        BOSSBase_path (Path | str): local path to the BOSSBase dataset
        BOWS2_path (Path | str): local path to the BOWS2 dataset
        CAR_TCIA_path (Path | str): local path to the CAR+TCIA dataset
    """
    rng = random.Random(0)  # Set a fixed seed for repro

    def split_and_copy(dataset_path: Path | str, output_type: str, prefix: str) -> None:
        dataset_path = Path(dataset_path)
        dataset = sorted(p for p in dataset_path.iterdir() if p.is_file())
        
        rng.shuffle(dataset)

        size = len(dataset)
        train_size = int(size * 0.70)
        val_size = int(size * 0.15)

        splits = {
            "train": dataset[:train_size],
            "val": dataset[train_size:train_size + val_size],
            "test": dataset[train_size + val_size:],
        }

        for split_name, files in splits.items():
            destination = Path("../datasets") / output_type / split_name
            destination.mkdir(parents=True, exist_ok=True)

            for file in files:
                shutil.copy2(file, destination / f"{prefix}_{file.name}")

    split_and_copy(BOSSBase_path, "pgm", "boss")
    split_and_copy(BOWS2_path, "pgm", "bows")
    split_and_copy(CAR_path, "dcm", "car")
    split_and_copy(TCIA_path, "dcm", "tcia")


def main():
    parser = argparse.ArgumentParser(description="Split datasets into train, val, and test.")

    parser.add_argument("--bossbase", type=Path, required=True, help="Path to BOSSBase")
    parser.add_argument("--bows2", type=Path, required=True, help="Path to BOWS2")
    parser.add_argument("--car", type=Path, required=True, help="Path to CAR")
    parser.add_argument("--tcia", type=Path, required=True, help="Path to TCIA")

    args = parser.parse_args()

    setup_datasets(
        BOSSBase_path=args.bossbase,
        BOWS2_path=args.bows2,
        CAR_path=args.car,
        TCIA_path=args.tcia
    )


if __name__ == "__main__":
    main()