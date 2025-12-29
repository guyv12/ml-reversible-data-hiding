from . import loader
import torch
import time
import re


def test_dataloader() -> None:
    """
    time check for dataloader
    """
    start_time = time.time()

    BOSSbase_train_loader, data_len = loader.get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    # BOWS2_train_loader, data_len = data.get_loader("datasets/BOWS2_512/train", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    
    img_shape = next(iter(BOSSbase_train_loader)).shape[1:]
    all_imgs = torch.empty((data_len, *img_shape), dtype=torch.float)

    for idx, batch in enumerate(BOSSbase_train_loader):
        start = idx * batch.shape[0]
        end = start + batch.shape[0]

        all_imgs[start:end] = batch

    end_time = time.time()

    print(f"elapsed: {end_time - start_time}")
