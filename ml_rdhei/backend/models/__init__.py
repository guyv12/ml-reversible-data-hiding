from sklearn.linear_model import Ridge
import torch
import torchvision

def get_sklearn_ridge_model():
    return Ridge(alpha=1, solver="svd", fit_intercept=False)

def get_torch_ridge_model():
    raise NotImplementedError("Torch model is not implemented yet...")

def get_MobileNet_v2_model():
    return torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT, progress=True)


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
