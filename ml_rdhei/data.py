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
        
        return torch.from_numpy(img)
    
        # if we do a dummy run to save images with torch.save()
        # we can do torch.load(map_location="cuda") to skip CPU & openCV completly - good idea?


def ref_pixels(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(images.device)
    batch_mask = mask.unsqueeze(0).expand_as(images)

    return images[batch_mask]


def get_train_data() -> None:
    
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

    #mask = simple bin

    for batch in BOSSbase_train_loader:
        batch = batch.cuda()
        #ref_pixels(batch, mask)

    #return all_ref_pixels
