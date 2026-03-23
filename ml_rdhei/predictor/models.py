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


def sklearn_ridge(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model = get_sklearn_model()
    
    X_np, y_np = X.float().numpy(), y.float().numpy() # sklearn requires float & numpy
    model.fit(X_np, y_np)

    y_pred = torch.from_numpy(model.predict(X_np))
    error_map = (y.to(torch.int16) - y_pred.to(torch.int16))

    kernel_weights = torch.from_numpy(model.coef_).to(torch.float64) # stored as float64 to ensure full image recovery

    return kernel_weights, error_map
