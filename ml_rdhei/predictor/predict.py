from . import models
from ml_rdhei.data.features import extract_features
import torch


def get_raw_ad_sklearn(img: torch.Tensor, model: models.Ridge, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, W = img.shape

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    X, y, ref_pixels = extract_features(img, mask, K)
    X, y = X.numpy(), y.numpy()

    model.fit(X, y)

    y_pred = torch.from_numpy(model.predict(X))
    error_map = (y - y_pred)

    kernel_weights = torch.from_numpy(model.coef_)

    return kernel_weights, ref_pixels, error_map


def get_raw_ad_torch(img: torch.Tensor, model: models.TorchRidge, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, W = img.shape

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    X, y, ref_pixels = extract_features(img, mask, K)

    model.fit(X, y)

    y_pred = model.predict(X)
    error_map = (y - y_pred)

    kernel_weights = model.weights.to(torch.uint8)

    return kernel_weights, ref_pixels, error_map