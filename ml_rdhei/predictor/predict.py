from .models import sklearn_ridge 
from ml_rdhei.data.features import extract_features, lr_decompose
import torch
from collections.abc import Iterator


def get_raw_ad_sklearn(batch: torch.Tensor, K: int = 5) -> Iterator[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, H, W = batch.shape

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    X_batch, y_batch, ref_pixels_batch = extract_features(batch, mask, K)

    for X, y, ref_pixels in zip(X_batch, y_batch, ref_pixels_batch):
        kernel_weights, error_map = sklearn_ridge(X, y)

        yield kernel_weights, ref_pixels, error_map

def get_raw_ad_torch(batch: torch.Tensor, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # _, H, W = batch.shape

    # mask = torch.zeros((H, W), dtype=torch.bool)
    # mask[::2, ::2] = True

    # X, y, ref_pixels = extract_features(batch, mask, K)

    # # model.fit(X, y)

    # # y_pred = model.predict(X)
    # # error_map = (y - y_pred)

    # # kernel_weights = model.weights.to(torch.uint8)

    # return kernel_weights, ref_pixels, error_map
    pass


def get_raw_ad_sklearn_dicom(batch: torch.Tensor, model: models.Ridge, K: int = 5) -> Iterator[torch.Tensor, torch.Tensor, torch.Tensor]:
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