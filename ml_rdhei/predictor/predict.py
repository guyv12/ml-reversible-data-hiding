from . import models
from ml_rdhei.data.features import extract_features
import torch
from collections.abc import Iterator


def get_raw_ad_sklearn(batch: torch.Tensor, model: models.Ridge, K: int = 5) -> Iterator[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, H, W = batch.shape

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    X_batch, y_batch, ref_pixels_batch = extract_features(batch, mask, K)

    for X, y, ref_pixels in zip(X_batch, y_batch, ref_pixels_batch):
        model.fit(X.numpy(), y.numpy())

        y_pred = torch.from_numpy(model.predict(X.numpy()))
        error_map = (y - y_pred)

        kernel_weights = torch.from_numpy(model.coef_).to(torch.float64) # stored as float64 to ensure full image recovery

        yield kernel_weights, ref_pixels, error_map


def get_raw_ad_torch(batch: torch.Tensor, model: models.TorchRidge, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, H, W = batch.shape

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    X, y, ref_pixels = extract_features(batch, mask, K)

    model.fit(X, y)

    y_pred = model.predict(X)
    error_map = (y - y_pred)

    kernel_weights = model.weights.to(torch.uint8)

    return kernel_weights, ref_pixels, error_map