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
        if self.weights is None:
            raise TypeError("Model weights need to be set first")

        return self.weights @ X


def __get_sklearn_model():
    return Ridge(alpha=1, solver="svd", fit_intercept=False)


def __get_torch_model():
    raise NotImplementedError("Torch model is not implemented yet...")


def sklearn_ridge(X: torch.Tensor, y: torch.Tensor, quantization: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map.
    Works on a single image input.

    :return: kernel weights, error map
    :rtype: torch.Tensor[f64] | torch.Tensor[i64], torch.Tensor[i16]
    """
    model = __get_sklearn_model()
    
    X_np, y_np = X.double().numpy(), y.double().numpy() # sklearn requires float & numpy
    model.fit(X_np, y_np)

    W = torch.from_numpy(model.coef_)

    if not quantization:
        kernel_weights = W.to(torch.float64)
    else:
        kernel_weights = torch.round(W).to(torch.int64) # cut to int

    y_pred = torch.round(X.to(kernel_weights.dtype) @ kernel_weights)
    y_pred = y_pred.clamp(0, 255) # cut the vals to avoid big errors

    error_map = y.to(torch.int16) - y_pred.to(torch.int16) # convert to int16, error in <-255, 255>

    return kernel_weights, error_map

def torch_ridge(X_batch: torch.Tensor, y_batch: torch.Tensor, quantization: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map.
    Works on a batched input.
    
    :return: kernel weights, error map
    :rtype: torch.Tensor[f64] | torch.Tensor[i64], torch.Tensor[i16]
    """
    model = __get_torch_model()
    
    model.fit(X_batch, y_batch)

    W = model.weights
    
    if not quantization:
        kernel_weights_batch = W.to(torch.float64)
    else:
        kernel_weights_batch = torch.round(W).to(torch.int64)

    y_pred_batch = torch.round(X_batch.to(kernel_weights_batch.dtype) @ kernel_weights_batch)
    y_pred_batch = y_pred_batch.clamp(0, 255) # cut the vals to avoid big errors

    error_map_batch = y_batch.to(torch.int16) - y_pred_batch.to(torch.int16)

    return kernel_weights_batch, error_map_batch
