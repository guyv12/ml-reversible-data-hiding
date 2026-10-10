from sklearn.linear_model import Ridge
import segmentation_models_pytorch as smp
import torch
from pathlib import Path

from backend.models.linear import *
from backend.models.cnn import *


# ----- Utils -----

def save_model(model: torch.nn.Module, path: Path | str = None) -> None:
    torch.save(model, path)

def load_model(path: Path | str = None) -> torch.nn.Module:
    model = torch.load(path, weights_only=False)
    model.eval()
    return model

# ----- Getters -----

def get_ridge_model():
    return Ridge(alpha=1, solver="svd", fit_intercept=False)

def get_batch_ridge_model():
    raise NotImplementedError("Torch model is not implemented yet...")

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

def get_mobilenet_v2_unetpp_model(path: Path | str = None, in_channels: int = 1, classes: int = 1):
    if path is not None:
        return load_model(path)
    
    return smp.UnetPlusPlus(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )
