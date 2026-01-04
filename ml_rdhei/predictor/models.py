from sklearn.linear_model import Ridge
import torch


class TorchRidge:

    def __init__(self, lambda_: float = 1e-1) -> None:
        self.L = lambda_
        self.weights = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        Features = X.shape[1]
        I = torch.eye(Features, dtype=X.dtype)
        self.weights = torch.linalg.solve(X.T @ X + self.L * I, X.T @ y)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.weights @ X


def get_sklearn_model():
    return Ridge(alpha=1, solver="svd", fit_intercept=False)


def get_torch_model(): # don't use it's not implemented
    return TorchRidge()