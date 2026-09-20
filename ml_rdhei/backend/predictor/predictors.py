import torch.nn.functional as fn
import torch
from collections.abc import Callable

from backend.models.models import *


def quantize_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Quantizes the weights to int64 by rounding.
    """
    return torch.round(weights).to(torch.int64)


def ridge_prediction(X: torch.Tensor, y: torch.Tensor,
                     pred_fn: Callable, quantization: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map.
    Works on a single image input.

    :return: kernel weights, error map
    :rtype: torch.Tensor[f64] | torch.Tensor[i64], torch.Tensor[i16]
    """
    model = get_ridge_model()
    valid_pred_range = (torch.iinfo(y.dtype).min, torch.iinfo(y.dtype).max)
    
    X_np, y_np = X.double().numpy(), y.double().numpy() # sklearn requires float & numpy
    model.fit(X_np, y_np)

    W = torch.from_numpy(model.coef_)

    if quantization:
        kernel_weights = quantize_weights(W)
    else:
        kernel_weights = W.to(torch.float64)

    y_pred = pred_fn(X, kernel_weights, valid_pred_range)

    error_map = y.to(torch.int16) - y_pred.to(torch.int16) # convert to int16, error in <-255, 255>

    return kernel_weights, error_map


def batch_ridge_prediction(X_batch: torch.Tensor, y_batch: torch.Tensor,
                           pred_fn: Callable, quantization: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map.
    Works on a batched input.
    
    :return: kernel weights, error map
    :rtype: torch.Tensor[f64] | torch.Tensor[i64], torch.Tensor[i16]
    """
    model = get_batch_ridge_model()
    valid_pred_range = (torch.iinfo(y_batch.dtype).min, torch.iinfo(y_batch.dtype).max)

    model.fit(X_batch, y_batch)

    W = model.weights
    
    if quantization:
        kernel_weights_batch = quantize_weights(W)
    else:
        kernel_weights_batch = W.to(torch.float64)

    y_pred_batch = pred_fn(X_batch, kernel_weights_batch, valid_pred_range)

    error_map_batch = y_batch.to(torch.int16) - y_pred_batch.to(torch.int16)

    return kernel_weights_batch, error_map_batch
