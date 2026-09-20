import torch
from collections.abc import Callable, Iterator

from backend.predictor.features import lr_decompose


def reference_mask(H: int, W: int) -> torch.Tensor:
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True
    return mask


def get_ad(batch: torch.Tensor, mask: torch.Tensor, feature_fn: Callable, predictor_fn: Callable
           ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Traverses ad pipeline for batched input, one image at a time
    Args:
        batch (torch.Tensor): batch of images (B, H, W)
        mask (torch.Tensor): mask to retrieve ref_pixels (H, W)
        feature_fn (Callable): fn to extract features
        predictor_fn (Callable): fn to predict from ref_p

    Yields:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kernel_weights, ref_pixels, error_map, image - per image
    """
    # 1. Extract features and ref_pixels
    X_batch, y_batch, ref_pixels_batch = feature_fn(batch, mask)

    # 2. Use predictor to predict the img from ref_pixels
    for i, (X, y, ref_pixels) in enumerate(zip(X_batch, y_batch, ref_pixels_batch)):
            kernel_weights, error_map = predictor_fn(X, y)
    
            yield kernel_weights, ref_pixels, error_map, batch[i]

def get_dicom_ad(batch: torch.Tensor, mask: torch.Tensor, feature_fn: Callable, predictor_fn: Callable
                 ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Traverses ad pipeline for batched input, one image at a time.
    For DICOM images - applies L/R decomposition first.
    Args:
        batch (torch.Tensor): batch of images (B, H, W)
        mask (torch.Tensor): mask to retrieve ref_pixels (H, W)
        feature_fn (Callable): fn to extract features
        predictor_fn (Callable): fn to predict from ref_p
    
    Yields:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        im1 error_map, im2 kernel_weights, im2 ref_pixels, im2_error_map, image - per image
    """
    # 1. Decompose the image in 2 parts, and extract features, ref_pixels    
    img1_batch, img2_batch = lr_decompose(batch)
    X_img2_batch, y_img2_batch, ref_pixels_img2_batch = feature_fn(img2_batch, mask)

    # 2. Use predictor to predict the img from ref_pixels for img2
    #    Use the fixed prediction for img1 values
    for i, (img1, img2_X, img2_y, img2_ref_pixels) in enumerate(zip(img1_batch, X_img2_batch, y_img2_batch, ref_pixels_img2_batch)):
        # image1 -> fixed prediction
        img1_error_map = (15 - img1.flatten()).to(torch.int16)
            
        # image2 -> classic approach
        img2_kernel_weights, img2_error_map = predictor_fn(img2_X, img2_y)
    
        yield img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, batch[i]


def get_ad_batch(batch: torch.Tensor, mask: torch.Tensor, feature_fn: Callable, batch_predictor_fn: Callable
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    X_batch, y_batch, ref_pixels_batch = feature_fn(batch, mask)
    kernel_weights_batch, error_map_batch = batch_predictor_fn(X_batch, y_batch)
    
    return kernel_weights_batch, ref_pixels_batch, error_map_batch

def get_ad_batch(batch: torch.Tensor, mask: torch.Tensor, feature_fn: Callable, batch_predictor_fn: Callable
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    img1_batch, img2_batch = lr_decompose(batch)
    img1_error_map_batch = (15 - img1_batch.flatten()).to(torch.int16)
    
    X_img2_batch, y_img2_batch, ref_pixels_img2_batch = feature_fn(batch, mask)
    kernel_weights_img2_batch, error_map_img2_batch = batch_ridge_prediction(X_img2_batch, y_img2_batch)
    
    return img1_error_map_batch, kernel_weights_img2_batch, ref_pixels_img2_batch, error_map_img2_batch

### Old API - to remove
import torch
from backend.predictor.pipelines import *
from backend.predictor.predictors import *
from backend.predictor.features import *
from backend.predictor.operations import *


def pgm_raw_ad_sklearn(batch: torch.Tensor, K: int = 5) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    _, H, W = batch.shape

    mask = reference_mask(H, W)

    X_batch, y_batch, ref_pixels_batch = unfold_features(batch, mask, K)

    for i, (X, y, ref_pixels) in enumerate(zip(X_batch, y_batch, ref_pixels_batch)):
        kernel_weights, error_map = ridge_prediction(X, y, mask)

        yield kernel_weights, ref_pixels, error_map, batch[i]

def pgm_raw_ad_torch(batch: torch.Tensor, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, H, W = batch.shape

    mask = reference_mask(H, W)

    X_batch, y_batch, ref_pixels_batch = unfold_features(batch, mask, K)
    kernel_weights_batch, error_map_batch = batch_ridge_prediction(X_batch, y_batch)

    return kernel_weights_batch, ref_pixels_batch, error_map_batch


def dicom_raw_ad_sklearn(batch: torch.Tensor, K: int = 5) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    _, H, W = batch.shape

    mask = reference_mask(H, W)

    img1_batch, img2_batch = lr_decompose(batch)

    X_img2_batch, y_img2_batch, ref_pixels_img2_batch = unfold_features(img2_batch, mask, K)

    for img1, img2_X, img2_y, img2_ref_pixels in zip(img1_batch, X_img2_batch, y_img2_batch, ref_pixels_img2_batch):
        # image1 -> fixed prediction
        img1_error_map = (15 - img1.flatten()).to(torch.int16)
        
        # image2 -> classic approach
        img2_kernel_weights, img2_error_map = ridge_prediction(img2_X, img2_y, mask)

        yield img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map

def dicom_raw_ad_torch(batch: torch.Tensor, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, H, W = batch.shape

    mask = reference_mask(H, W)

    img1_batch, img2_batch = lr_decompose(batch)

    img1_error_map_batch = (15 - img1_batch.flatten()).to(torch.int16)

    X_img2_batch, y_img2_batch, ref_pixels_img2_batch = unfold_features(img2_batch, mask, K)
    kernel_weights_img2_batch, error_map_img2_batch = batch_ridge_prediction(X_img2_batch, y_img2_batch)

    return img1_error_map_batch, kernel_weights_img2_batch, ref_pixels_img2_batch, error_map_img2_batch