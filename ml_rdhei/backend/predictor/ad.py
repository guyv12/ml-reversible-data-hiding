from .models import sklearn_ridge, torch_ridge, torch_mobilenet_v2
from backend.data.features import extract_features, lr_decompose
import torch
from collections.abc import Iterator


class ADRidge:
    def get_ad(batch: torch.Tensor, K: int = 5) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        _, H, W = batch.shape

        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[::2, ::2] = True

        X_batch, y_batch, ref_pixels_batch = extract_features(batch, mask, K)

        for i, (X, y, ref_pixels) in enumerate(zip(X_batch, y_batch, ref_pixels_batch)):
            kernel_weights, error_map = sklearn_ridge(X, y)

            yield kernel_weights, ref_pixels, error_map, batch[i]

    def get_ad_batch(batch: torch.Tensor, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, H, W = batch.shape

        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[::2, ::2] = True

        X_batch, y_batch, ref_pixels_batch = extract_features(batch, mask, K)
        kernel_weights_batch, error_map_batch = torch_ridge(X_batch, y_batch)

        return kernel_weights_batch, ref_pixels_batch, error_map_batch


class ADRidgeDicom:
    def get_ad(batch: torch.Tensor, K: int = 5) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        _, H, W = batch.shape

        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[::2, ::2] = True

        img1_batch, img2_batch = lr_decompose(batch)
        X_img2_batch, y_img2_batch, ref_pixels_img2_batch = extract_features(img2_batch, mask, K)

        for img1, img2_X, img2_y, img2_ref_pixels in zip(img1_batch, X_img2_batch, y_img2_batch, ref_pixels_img2_batch):
            # image1 -> fixed prediction
            img1_error_map = (15 - img1.flatten()).to(torch.int8)
            
            # image2 -> classic approach
            img2_kernel_weights, img2_error_map = sklearn_ridge(img2_X, img2_y)

            yield img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map

    def get_ad_batch(batch: torch.Tensor, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, H, W = batch.shape

        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[::2, ::2] = True

        img1_batch, img2_batch = lr_decompose(batch)

        img1_error_map_batch = (15 - img1_batch.flatten()).to(torch.int8)

        X_img2_batch, y_img2_batch, ref_pixels_img2_batch = extract_features(img2_batch, mask, K)
        kernel_weights_img2_batch, error_map_img2_batch = torch_ridge(X_img2_batch, y_img2_batch)

        return img1_error_map_batch, kernel_weights_img2_batch, ref_pixels_img2_batch, error_map_img2_batch


class ADCNN:
    def get_ad(batch: torch.Tensor, K: int = 5) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        _, H, W = batch.shape
    
        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[::2, ::2] = True
    
        X_batch, y_batch, ref_pixels_batch = extract_features(batch, mask, K)
    
        for image, X, y, ref_pixels in enumerate(zip(X_batch, y_batch, ref_pixels_batch)):
            error_map = torch_mobilenet_v2(X, y)
    
            yield ref_pixels, error_map, image