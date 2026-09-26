import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
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
        
        return torch.from_numpy(img)
    
        # if we do a dummy run to save images with torch.save()
        # we can do torch.load(map_location="cuda") to skip CPU & openCV completely


class DicomDataset(Dataset):
    def __init__(self, img_dir: str | Path, regex: re.Pattern | None = None) -> None:
        all_files = sorted(list(Path(img_dir).glob("*.dcm")))
        if regex is None:
            self.files = all_files
        else:
            self.files = [f for f in all_files if regex.match(f.name)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> torch.Tensor:
        ds = pydicom.dcmread(str(self.files[idx]))
        img = ds.pixel_array.astype("int16")

        if img is None:
            raise OSError(f"Could not load {self.files[idx]}")

        return torch.from_numpy(img).clamp(min=0) # DICOM images loaded with pydicom can hold negative values
                                                  # which are 99% just pure black, so we can clamp to 0

def get_loader(dataset_dir: str | Path, regex: re.Pattern | None = None,
               batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, int]:
    # if Ur on Windows, and this runs slow switch 'num_workers' to 0 in the DataLoaders
    # apparently this is a known headache for Windows machines - bruh
    
    dataset = ImageDataset(dataset_dir, regex)
    loader = DataLoader(
        dataset,
        pin_memory=True,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return loader, len(dataset)

def get_dicom_loader(dataset_dir: str | Path, regex: re.Pattern | None = None,
                     num_workers: int = 4) -> tuple[DataLoader, int]:
    
    dataset = DicomDataset(dataset_dir, regex)
    loader = DataLoader(
        dataset,
        pin_memory=True,
        shuffle=True,
        batch_size=1, # batch_size = 1 because DICOM images can be different shapes
        num_workers=num_workers
    )

    return loader, len(dataset)
