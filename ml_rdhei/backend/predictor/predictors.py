import torch.nn.functional as fn
import torch
from collections.abc import Callable

from backend.models.models import *
from backend.predictor.features import build_patch_input


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
    valid_pred_range = (torch.iinfo(y.dtype).min, torch.iinfo(y.dtype).max)

    # Normalize for CNN input
    X_pre = X.unsqueeze(0).unsqueeze(0).float() / 255.0
   
    model = model_fn() # The model output has to match the image dimensions
    model.eval()
   
    with torch.inference_mode():
        # (B, C, H, W) output
        y_pred = model(X_pre)

    y_pred = y_pred.squeeze(0).squeeze(0)[~mask] # remove batch and channel dimensions, and ref pixels
    y_pred = (y_pred * 255.0).round() # Re-Normalize
    y_pred = y_pred.clamp(valid_pred_range[0], valid_pred_range[1]) # clamp to avoid big errors

    error_map = y.to(torch.int16) - y_pred.to(torch.int16)

    return error_map


def autoencoder_predict_image(masked_image: torch.Tensor, model: torch.nn.Module,
                              mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    H, W = masked_image.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"Autoencoder predictor requires image dimensions divisible by patch_size; got {(H, W)} and {patch_size}"
        )

    device = next(model.parameters()).device
    patches = []
    coords = []

    for row in range(0, H, patch_size):
        for column in range(0, W, patch_size):
            patch = masked_image[row: row + patch_size, column: column + patch_size].float() / 255.0
            patch_mask = mask[row: row + patch_size, column: column + patch_size]
            patches.append(build_patch_input(patch, patch_mask))
            coords.append((row, column))

    input_batch = torch.stack(patches, dim=0).to(device)

    with torch.inference_mode():
        prediction_batch = model(input_batch).squeeze(1).clamp(0.0, 1.0).cpu() * 255.0

    prediction_image = torch.zeros((H, W), dtype=torch.float32)
    for predicted_patch, (row, column) in zip(prediction_batch, coords):
        prediction_image[row: row + patch_size, column: column + patch_size] = predicted_patch

    return prediction_image


def autoencoder_prediction(X: torch.Tensor, y: torch.Tensor,
                           model_fn: Callable, mask: torch.Tensor) -> torch.Tensor:
    model = model_fn()
    model.eval()

    patch_size = getattr(model, "patch_size", 64)
    y_pred = autoencoder_predict_image(X, model, mask, patch_size)
    y_pred = torch.round(y_pred)[~mask].clamp(torch.iinfo(y.dtype).min, torch.iinfo(y.dtype).max)

    error_map = y.to(torch.int16) - y_pred.to(torch.int16)

    return error_map
