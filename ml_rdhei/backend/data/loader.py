import torch
import cv2
import pydicom
import re

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from backend.predictor.features import lr_decompose


class ImageDataset(Dataset):

    def __init__(self, img_dir: str | Path, regex: re.Pattern | None = None) -> None:
        all_files = sorted(Path(img_dir))
        if regex is None:
            self.files = all_files
        else:
            self.files = [f for f in all_files if regex.match(f.name)]

    def __len__(self) -> int:
        return len(self.files)

    # def __getitem__(self, idx: int) -> torch.Tensor:
    # if we do a dummy run to save images with torch.save()
    # we can do torch.load(map_location="cuda") to skip CPU & openCV completely


class GraymapDataset(ImageDataset):

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise OSError(f"Could not load {path}")

        return torch.from_numpy(img)


class DicomDataset(ImageDataset):

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        ds = pydicom.dcmread(str(path)) # raises already
        img = ds.pixel_array.astype("int16")

        return torch.from_numpy(img).clamp(min=0) # DICOM images loaded with pydicom can hold negative values
                                                  # which are 99% just pure black, so we can clamp to 0

class DicomTrainDataset(ImageDataset):

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        ds = pydicom.dcmread(str(path)) # raises already
        img = ds.pixel_array.astype("int16")
        img = torch.from_numpy(img).clamp(min=0)

        _, img2 = lr_decompose(img) # Give the img2 only for training

        return img2


def get_loader(dataset_dir: str | Path, regex: re.Pattern | None = None,
               shuffle: bool = True, batch_size: int = 8, num_workers: int = 4) -> tuple[DataLoader, int]:
    # if Ur on Windows, and this runs slow switch 'num_workers' to 0 in the DataLoaders
    # apparently this is a known headache for Windows machines - bruh
    
    dataset = ImageDataset(dataset_dir, regex)
    loader = DataLoader(
        dataset,
        pin_memory=True,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return loader, len(dataset)

def get_dicom_loader(dataset_dir: str | Path, regex: re.Pattern | None = None,
                     train: bool = False, shuffle: bool = True,
                     batch_size: int = 8, num_workers: int = 4) -> tuple[DataLoader, int]:

    if not train:
        dataset = DicomDataset(dataset_dir, regex)
    else:
        dataset = DicomTrainDataset(dataset_dir, regex)
    
    loader = DataLoader(
        dataset,
        pin_memory=True,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return loader, len(dataset)
