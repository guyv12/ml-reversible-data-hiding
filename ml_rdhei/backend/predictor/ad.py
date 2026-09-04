from .predict import sklearn_ridge, torch_ridge, torch_mobilenet_v2
from backend.data.features import unfold_features, lr_decompose
import torch
from collections.abc import Iterator


class RidgePredictor:
    def __init__(self, K: int = 5, feat_func = unfold_features):
        self.K = K
        self.feat_func = feat_func

    def __get_mask(self, H: int, W: int) -> torch.Tensor:
        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[::2, ::2] = True
        return mask

    def get_ad(self, batch: torch.Tensor) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        _, H, W = batch.shape

        mask = self.__get_mask(H, W)

        X_batch, y_batch, ref_pixels_batch = self.feat_func(batch, mask, self.K)

        for i, (X, y, ref_pixels) in enumerate(zip(X_batch, y_batch, ref_pixels_batch)):
            kernel_weights, error_map = sklearn_ridge(X, y)

            yield kernel_weights, ref_pixels, error_map, batch[i]

    def get_ad_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, H, W = batch.shape

        mask = self.__get_mask(H, W)

        X_batch, y_batch, ref_pixels_batch = self.feat_func(batch, mask, self.K)
        kernel_weights_batch, error_map_batch = torch_ridge(X_batch, y_batch)

        return kernel_weights_batch, ref_pixels_batch, error_map_batch


class RidgePredictorDicom:
    def __init__(self, K: int = 5, feat_func = unfold_features):
        self.K = K
        self.feat_func = feat_func
    
    def __get_mask(self, H: int, W: int) -> torch.Tensor:
        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[::2, ::2] = True
        return mask
    
    def get_ad(self, batch: torch.Tensor) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        _, H, W = batch.shape

        mask = self.__get_mask(H, W)

        img1_batch, img2_batch = lr_decompose(batch)
        X_img2_batch, y_img2_batch, ref_pixels_img2_batch = self.feat_func(img2_batch, mask, self.K)

        for img1, img2_X, img2_y, img2_ref_pixels in zip(img1_batch, X_img2_batch, y_img2_batch, ref_pixels_img2_batch):
            # image1 -> fixed prediction
            img1_error_map = (15 - img1.flatten()).to(torch.int8)
            
            # image2 -> classic approach
            img2_kernel_weights, img2_error_map = sklearn_ridge(img2_X, img2_y)

            yield img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map

    def get_ad_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, H, W = batch.shape

        mask = self.__get_mask(H, W)

        img1_batch, img2_batch = lr_decompose(batch)
        img1_error_map_batch = (15 - img1_batch.flatten()).to(torch.int8)

        X_img2_batch, y_img2_batch, ref_pixels_img2_batch = self.feat_func(img2_batch, mask, self.K)
        kernel_weights_img2_batch, error_map_img2_batch = torch_ridge(X_img2_batch, y_img2_batch)

        return img1_error_map_batch, kernel_weights_img2_batch, ref_pixels_img2_batch, error_map_img2_batch


# class ADCNN:
#     def get_ad(self,batch: torch.Tensor, K: int = 5) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#         _, H, W = batch.shape
    
#         mask = torch.zeros((H, W), dtype=torch.bool)
#         mask[::2, ::2] = True
    
#         X_batch, y_batch, ref_pixels_batch = unfold_features(batch, mask, K)
    
#         for image, X, y, ref_pixels in enumerate(zip(X_batch, y_batch, ref_pixels_batch)):
#             error_map = torch_mobilenet_v2(X, y)
    
#             yield ref_pixels, error_map, image