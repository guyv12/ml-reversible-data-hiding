import torch
from backend.models import *


def sklearn_ridge(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map. Assumes 8bit image.
    Works on a single image input

    :return: kernel weights, error map
    :rtype: torch.Tensor[f64], torch.Tensor[i16]
    """
    model = get_sklearn_ridge_model()
    
    X_np, y_np = X.float().numpy(), y.float().numpy() # sklearn requires float & numpy
    model.fit(X_np, y_np)

    y_pred = torch.from_numpy(model.predict(X_np))
    error_map = (y.to(torch.int16) - y_pred.to(torch.int16)) # convert to int16 for accurate output 

    kernel_weights = torch.from_numpy(model.coef_).to(torch.float64) # stored as float64 to ensure full image recovery

    return kernel_weights, error_map

def torch_ridge(X_batch: torch.Tensor, y_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map. Assumes 8bit image.
    Works on a batched input.
    
    :return: kernel weights, error map
    :rtype: torch.Tensor[f64], torch.Tensor[i16]
    """
    model = get_torch_ridge_model()
    
    model.fit(X_batch, y_batch)

    y_pred_batch = model.predict(X_batch)
    error_map_batch = (y_batch.to(torch.int16) - y_pred_batch.to(torch.int16))

    kernel_weights_batch = torch.from_numpy(model.coef_).to(torch.float64)

    return kernel_weights_batch, error_map_batch

def torch_mobilenet_v2(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Creates a mobilenet_v2 model prediction.

    :return: kernel weights, error map
    :rtype: torch.Tensor[i16]
    """
    pass