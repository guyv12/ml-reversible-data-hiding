from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from pathlib import Path
from typing import Union, Optional
import re


class ImageDataset(Dataset):
    
    def __init__(self, img_dir: Union[str, Path], regex: Optional[re.Pattern]) -> None:
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
        # we can do torch.load(map_location="cuda") to skip CPU & openCV completly - good idea?


def get_train_ref(dev: torch.device) -> torch.Tensor:
    ''' 
    Returns a 2D torch.Tensor of reference pixels for all images.

    Shape:
        (N_images, N_ref_pixels)

    Each row contains the reference pixels (flattened) from one grayscale image (float32).
    '''

    def ref_pixels(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return images[:, mask] # !GS: assumes grayscale .pgm

    BOSSbase_train_dataset = ImageDataset("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    BOSSbase_train_loader = DataLoader(
        BOSSbase_train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )
    
    # BOWS2_train_dataset = ImageDataset("datasets/BOWS2_512/train", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    # BOWS2_train_loader = DataLoader(
    # BOWS2_train_dataset,
    # batch_size=64, 
    # shuffle=True,
    # num_workers=4
    # pin_memory=True
    # )

    # Mask Build
    mask_shape = next(iter(BOSSbase_train_loader)).shape[1:] # take a sample image to extract shape value
    mask = torch.zeros(mask_shape, dtype=torch.bool).to(dev)
    mask[::2, ::2] = True # !GS: assumes grayscale .pgm

    # Ref Pixel Tensor Build
    n_ref_pixels = int(mask.sum().item())
    all_ref_pixels = torch.empty((len(BOSSbase_train_dataset), n_ref_pixels), device=dev, dtype=torch.float)

    for idx, batch in enumerate(BOSSbase_train_loader):
        batch = batch.to(dev)

        start = idx * batch.shape[0]
        end = start + batch.shape[0]

        all_ref_pixels[start:end] = ref_pixels(batch, mask)

    return all_ref_pixels


def get_train_raw(dev: torch.device) -> torch.Tensor:
    ''' 
    Returns a 3D torch.Tensor of image pixels.

    Shape:
        (N_images, cols, rows)

    Each row contains pixel data (2D tensor of float32).
    '''

    BOSSbase_train_dataset = ImageDataset("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    BOSSbase_train_loader = DataLoader(
        BOSSbase_train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )
    
    # BOWS2_train_dataset = ImageDataset("datasets/BOWS2_512/train", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    # BOWS2_train_loader = DataLoader(
    # BOWS2_train_dataset,
    # batch_size=64, 
    # shuffle=True,
    # num_workers=4
    # pin_memory=True
    # )

    img_shape = next(iter(BOSSbase_train_loader)).shape[1:]
    all_imgs = torch.empty((len(BOSSbase_train_dataset), *img_shape), device=dev, dtype=torch.float)

    for idx, batch in enumerate(BOSSbase_train_loader):
        batch = batch.to(dev)

        start = idx * batch.shape[0]
        end = start + batch.shape[0]

        all_imgs[start:end] = batch

    return all_imgs
