from . import dev, mask
from .data import get_loader, get_ref, get_raw
import torch
import time
import re


def test_data_ref() -> None:
    """
    time check
    """
    
    start_time = time.time()

    BOSSbase_train_loader, data_len = get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    # BOWS2_train_loader, data_len = data.get_loader("datasets/BOWS2_512/train", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    
    # Ref Pixel Tensor Build
    n_ref_pixels = int(mask.sum().item())
    all_ref_pixels = torch.empty((data_len, n_ref_pixels), device=dev, dtype=torch.float)

    for idx, batch in enumerate(BOSSbase_train_loader):
        start = idx * batch.shape[0]
        end = start + batch.shape[0]

        all_ref_pixels[start:end], _ = get_ref(batch)

    end_time = time.time()

    print(f"elapsed: {end_time - start_time}")



def test_data_raw() -> None:
    """
    time check
    """
    
    start_time = time.time()

    BOSSbase_train_loader, data_len = get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    # BOWS2_train_loader, data_len = data.get_loader("datasets/BOWS2_512/train", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    
    img_shape = next(iter(BOSSbase_train_loader)).shape[1:]
    all_imgs = torch.empty((data_len, *img_shape), device=dev, dtype=torch.float)

    for idx, batch in enumerate(BOSSbase_train_loader):
        start = idx * batch.shape[0]
        end = start + batch.shape[0]

        all_imgs[start:end] = get_raw(batch)

    end_time = time.time()

    print(f"elapsed: {end_time - start_time}")
