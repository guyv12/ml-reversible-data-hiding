import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import re


global dev, mask
dev, mask = None, None


def set_device() -> None:
    global dev
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_mask(new_mask: torch.Tensor | None = None) -> None:
    """
    Sets a global mask to be used to access reference pixels.
    If none is provided it will default to ::2::2 512x512 checkerboard.
    """

    global mask, dev

    if dev is None:
        raise ValueError("Variable dev is None. Call set_device() before using this DataLoader.")

    if new_mask is None:
        mask = torch.zeros((512, 512), dtype=torch.bool).to(dev) # !GS: assumes grayscale .pgm
        mask[::2, ::2] = True

    else:
        mask = new_mask.to(dev)


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
        
        return torch.from_numpy(img).float() # !GS: assumes grayscale .pgm
    
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


def get_raw(batch: torch.Tensor) -> torch.Tensor:
    """
    Returns a 3D torch.Tensor of image pixels, moved to target device.

    Shape:
        (N_images, cols, rows)

    Each row contains pixel data (2D tensor of float32).
    """

    if dev is None:
        raise ValueError("Variable dev is None. Call set_device() before using this DataLoader.")

    return batch.to(dev, non_blocking=True)


def get_ref(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple of a 2D torch.Tensor of reference pixels for batch images,
    and a 3D torch.Tensor of image pixels, both moved to target device.

    Shape:
        [0]: (N_images, N_ref_pixels)
        [1]: (N_images, cols, rows)

    [0]: Each row contains the reference pixels (flattened) from one grayscale image (float32).
    [1]: Each row contains pixel data (2D tensor of float32).
    """
    
    global mask
    if mask is None:
        raise ValueError("Variable mask is None. Call set_mask() before using this DataLoader.")

    batch = batch.to(dev, non_blocking=True)
    B, H, W = batch.shape

    return batch.view(B, H * W)[:, mask.flatten()], batch
