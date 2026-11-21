from sklearn.linear_model import Ridge
import segmentation_models_pytorch as smp
import torch
from pathlib import Path

from backend.models.linear import *
from backend.models.cnn import *


# ----- Utils -----

def save_model(model: torch.nn.Module, path: Path | str = None) -> None:
    torch.save(model, path)

def load_model(path: Path | str = None, map_location: torch.device = 'cpu') -> torch.nn.Module:
    model = torch.load(path, weights_only=False, map_location=map_location)
    model.eval()
    return model

# ----- Getters -----

def get_ridge_model():
    return Ridge(alpha=1, solver="svd", fit_intercept=False)

def get_batch_ridge_model():
    raise NotImplementedError("Torch model is not implemented")

def get_mobilenet_v2_unet_model(path: Path | str = None, in_channels: int = 1, classes: int = 1):
    if path is not None:
        return load_model(path)

    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )

def get_resnet_50_unet_model(path: Path | str = None, in_channels: int = 1, classes: int = 1):
    if path is not None:
        return load_model(path)
    
    return smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )

def get_res_block_cnn_model():
    return SeqDilatedCNNP(channels=32, kernel_size=3)

def get_parallel_dilated_res_block_cnn_model():
    return ParDilatedCCNP(channels=32, kernel_size=3)

def get_custom_model(path: Path | str, target_dev: torch.device = 'cpu'):
    return load_model(path, map_location=target_dev).to(device=target_dev)
