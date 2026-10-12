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


def cnn_prediction(X: torch.Tensor, y: torch.Tensor, 
                   model_fn: Callable, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a CNN model prediction and error map.
    Works on a single image input.

    :return: error map
    :rtype: torch.Tensor[i16]
    """
    model = model_fn() # The model output has to match the image dimensions
    valid_pred_range = (torch.iinfo(y.dtype).min, torch.iinfo(y.dtype).max)

    # Normalize for CNN input
    X_pre = X.unsqueeze(0).unsqueeze(0).float() / 255.0
   
    model = model_fn()
    model.eval()
   
    with torch.inference_mode():
        # (B, C, H, W) output
        y_pred = model(X_pre)

    y_pred = y_pred.squeeze(0).squeeze(0)[~mask] # remove batch and channel dimensions, and ref pixels
    y_pred.clamp(valid_pred_range[0], valid_pred_range[1]) # clamp to avoid big errors

    error_map = y.to(torch.int16) - y_pred.to(torch.int16)

    return error_map
