from sklearn.linear_model import Ridge
import torch
import segmentation_models_pytorch as smp


def get_sklearn_ridge_model():
    return Ridge(alpha=1, solver="svd", fit_intercept=False)

def get_torch_ridge_model():
    raise NotImplementedError("Torch model is not implemented yet...")

def get_torch_unet_model(in_channels: int = 1, classes: int = 1):
    return smp.Unet(
            encoder_name="mobilenet_v2", 
            encoder_weights="imagenet", 
            in_channels=in_channels,
            classes=classes
        )

class TorchRidge:
    def __init__(self, lambda_: float = 1e-1) -> None:
        self.L = lambda_
        self.weights = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        Features = X.shape[1]
        I = torch.eye(Features, dtype=X.dtype)
        self.weights = torch.linalg.solve(X.T @ X + self.L * I, X.T @ y)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise TypeError("Model weights need to be set first")

        return self.weights @ X
