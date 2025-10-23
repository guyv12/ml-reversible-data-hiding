import data.loader as loader
import torch


def extract_features(batch_ref_pixels: torch.Tensor) -> torch.Tensor:

    B = batch_ref_pixels.shape[0]  # Batch size
    H, W = loader.mask.shape # Image dimensions

    feature_tensor = torch.zeros((B, H, W), device=loader.dev, dtype=torch.float)
    # pack H and W together, and index with flat (H, and W together) mask
    feature_tensor.view(B, H * W)[:, loader.mask.flatten()] = batch_ref_pixels
    
    return feature_tensor
