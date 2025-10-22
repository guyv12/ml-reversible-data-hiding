import run
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from pathlib import Path
from typing import Union, List
import re


class ImageDataset(Dataset):
    
    def __init__(self, img_dir: Union[str, Path], regex: re.Pattern | None = None) -> None:
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


def set_mask(new_mask: torch.Tensor | None = None) -> None:
    """
    Sets a global mask to be used to access reference pixels.
    If none is provided it will default to ::2::2 512x512 checkerboard.
    """
    global mask

    if new_mask is None:
        mask = torch.zeros((512, 512), dtype=torch.bool).to(run.dev) # !GS: assumes grayscale .pgm
        mask[::2, ::2] = True

    else:
        mask = new_mask.to(run.dev)


def worker_init(worker_id: int) -> None:
    run.set_device()
    set_mask()


def ref_collate(batch: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError("mask is None. Call set_mask() before using this DataLoader.")

    batch_tensor = torch.stack(batch, dim=0).to(run.dev)
    B, H, W = batch_tensor.shape

    return batch_tensor.view(B, H * W)[:, mask.flatten()], batch_tensor


def raw_collate(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Returns a 3D torch.Tensor of image pixels moved to target device.

    Shape:
        (N_images, cols, rows)

    Each row contains pixel data (2D tensor of float32).
    """
    return torch.stack(batch, dim=0).to(run.dev)

def get_loader_ref(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init,
        collate_fn=ref_collate
        )

def get_loader_raw(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init,
        collate_fn=raw_collate
        )


def test_raw(loader: DataLoader, l_size: int, image_shape: tuple[int, int] = (512, 512)) -> None:
    """
    Loads ALL raw images from the DataLoader into a single 3D tensor (N x H x W).
    """
    N_images = l_size
    H, W = image_shape
    
    # 1. CRITICAL CORRECTION: Define the 3D tensor shape correctly
    all_imgs = torch.empty((N_images, H, W), device=run.dev, dtype=torch.float)
    
    for idx, batch in enumerate(loader):
        start = idx * batch.shape[0]
        end = start + batch.shape[0]
        
        all_imgs[start:end] = batch
    
    print(f"Successfully loaded all {N_images} raw images into a tensor of shape {all_imgs.shape}.")


def test_ref(loader: DataLoader, l_size: int, image_shape: tuple[int, int] = (512, 512)) -> None:
    """
    Loads ALL raw images from the DataLoader into a single 3D tensor (N x H x W).
    """
    N_images = l_size
    H, W = image_shape
    
    n_ref_pixels = int(mask.sum().item())
    all_ref_pixels = torch.empty((N_images, n_ref_pixels), device=run.dev, dtype=torch.float)
    
    for idx, (batch_ref, batch_raw) in enumerate(loader):
        start = idx * batch_ref.shape[0]
        end = start + batch_ref.shape[0]
        
        all_ref_pixels[start:end] = batch_ref
    
    print(f"Successfully loaded all {N_images} raw images into a tensor of shape {all_ref_pixels.shape}.")
