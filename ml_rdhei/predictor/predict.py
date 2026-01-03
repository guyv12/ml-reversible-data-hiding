from . import models
from ml_rdhei.data.features import extract_features
import torch


def get_raw_ad(img: torch.Tensor, model: models.Ridge | models.TorchRidge, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, W = img.shape

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    X, y, ref_pixels = extract_features(img, mask, K)

    model.fit(X, y)

    y_pred = model.predict(X)
    error_map = (y - y_pred).to(torch.uint8)

    kernel_weights = model.coef_

    return kernel_weights, ref_pixels, error_map