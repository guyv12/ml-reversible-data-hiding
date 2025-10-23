import ml_rdhei.data.loader as loader
from ml_rdhei.data.features import extract_features
from .model import Ridge
import torch
import re


def train_kernel(kernel_size: int = 5) -> None:
    loader.set_device()
    loader.set_mask()


    BOSSBase, _ = loader.get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))

    for idx, batch in enumerate(BOSSBase):

        ref, raw = loader.get_ref(batch)
        feat = extract_features(ref)

        # for img in feat[0]:

        #     for row in img:
        #         for pixel in row:
        #             if pixel in feat continue
        #             model = Ridge(kernel_size)
                    
        #             X, y = img, img # trimmed to 5x5?
        #             model.fit(X, y)

        #             new_val = model.predict(X)
        #             feat[img[row[pixel]]] = new_val


    return


def train_mask() -> None:
    loader.set_device()

    return