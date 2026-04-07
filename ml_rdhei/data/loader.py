import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import re


class ImageDataset(Dataset):
    
    def __init__(self, img_dir: str | Path, regex: re.Pattern | None = None) -> None:
        # Dataset holds file paths
        all_files = list(Path(img_dir).glob("*.pgm"))
        if regex is None:
            self.files = all_files

        else:
            self.files = [f for f in all_files if regex.match(f.name)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.files[idx]), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise OSError(f"Could not load {self.files[idx]}")
        
        return torch.from_numpy(img).float() # torch requires floats rn so we can avoid double storing...
    
        # if we do a dummy run to save images with torch.save()
        # we can do torch.load(map_location="cuda") to skip CPU & openCV completely - good idea?


def get_loader(dataset_dir: str | Path, regex: re.Pattern | None = None) -> tuple[DataLoader, int]:
    # if Ur on Windows, and this runs slow switch 'num_workers' to 0 in the DataLoaders
    # apparently this is a known headache for Windows machines - bruh
    
    dataset = ImageDataset(dataset_dir, regex)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return loader, len(dataset)
