import ml_rdhei.data.loader as loader
from ml_rdhei.data.features import extract_features
from sklearn.linear_model import Ridge
import torch
import re


def train_kernel(K: int = 5) -> None:

    BOSSBase, _ = loader.get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))

    H, W = 512, 512 # !GS: assumes grayscale .pgm

    mask = torch.zeros((H, W), dtype=torch.bool).to(dev)
    mask[::2, ::2] = True

    model = Ridge(alpha=1, solver="svd", fit_intercept=False)

    for idx, batch in enumerate(BOSSBase):
        X, y = extract_features(batch, mask, K)
        model.fit(feature_matrix, y)

    return


def train_mask() -> None:
    return