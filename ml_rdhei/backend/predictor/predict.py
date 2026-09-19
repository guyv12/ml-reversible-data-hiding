from .models import predict_sklearn_ridge, predict_torch_ridge
from backend.data.features import extract_features, lr_decompose
import torch
from backend.models import *

def reference_mask(H: int, W: int) -> torch.Tensor:
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True
    return mask

def sklearn_ridge(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map. Assumes 8bit image.
    Works on a single image input
    :return: kernel weights, error map
    :rtype: torch.Tensor[f64], torch.Tensor[i16]
    """

    mask = reference_mask(H, W)

    y_pred = torch.from_numpy(model.predict(X_np))
    error_map = (y.to(torch.int16) - y_pred.to(torch.int16)) # convert to int16 for accurate output 

    for i, (X, y, ref_pixels) in enumerate(zip(X_batch, y_batch, ref_pixels_batch)):
        kernel_weights, error_map = predict_sklearn_ridge(X, y, mask)

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

    mask = reference_mask(H, W)

    X_batch, y_batch, ref_pixels_batch = extract_features(batch, mask, K)
    kernel_weights_batch, error_map_batch = predict_torch_ridge(X_batch, y_batch)

    return kernel_weights_batch, error_map_batch

def torch_mobilenet_v2(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Creates a mobilenet_v2 model prediction.
    :return: kernel weights, error map
    :rtype: torch.Tensor[i16]
    """

def dicom_raw_ad_sklearn(batch: torch.Tensor, K: int = 5) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    _, H, W = batch.shape

    mask = reference_mask(H, W)

    img1_batch, img2_batch = lr_decompose(batch)

    X_img2_batch, y_img2_batch, ref_pixels_img2_batch = extract_features(img2_batch, mask, K)

    for img1, img2_X, img2_y, img2_ref_pixels in zip(img1_batch, X_img2_batch, y_img2_batch, ref_pixels_img2_batch):
        # image1 -> fixed prediction
        img1_error_map = (15 - img1.flatten()).to(torch.int16)
        
        # image2 -> classic approach
        img2_kernel_weights, img2_error_map = predict_sklearn_ridge(img2_X, img2_y, mask)

        yield img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map

def dicom_raw_ad_torch(batch: torch.Tensor, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, H, W = batch.shape

    mask = reference_mask(H, W)

    img1_batch, img2_batch = lr_decompose(batch)

    img1_error_map_batch = (15 - img1_batch.flatten()).to(torch.int16)

    X_img2_batch, y_img2_batch, ref_pixels_img2_batch = extract_features(img2_batch, mask, K)
    kernel_weights_img2_batch, error_map_img2_batch = predict_torch_ridge(X_img2_batch, y_img2_batch)

    return img1_error_map_batch, kernel_weights_img2_batch, ref_pixels_img2_batch, error_map_img2_batch
