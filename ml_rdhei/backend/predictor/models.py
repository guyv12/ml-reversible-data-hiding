from sklearn.linear_model import Ridge
import torch
import torchvision


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


def __get_sklearn_ridge_model():
    return Ridge(alpha=1, solver="svd", fit_intercept=False)

def __get_torch_ridge_model():
    raise NotImplementedError("Torch model is not implemented yet...")

def __get_MobileNet_v2_model():
    return torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT, progress=True)


def sklearn_ridge(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map. Assumes 8bit image.
    Works on a single image input

    :return: kernel weights, error map
    :rtype: torch.Tensor[f64], torch.Tensor[i16]
    """
    model = __get_sklearn_ridge_model()
    
    X_np, y_np = X.float().numpy(), y.float().numpy() # sklearn requires float & numpy
    model.fit(X_np, y_np)

    y_pred = torch.from_numpy(model.predict(X_np))
    error_map = (y.to(torch.int16) - y_pred.to(torch.int16)) # convert to int16 for accurate output 

    kernel_weights = torch.from_numpy(model.coef_).to(torch.float64) # stored as float64 to ensure full image recovery

    return kernel_weights, error_map

def torch_ridge(X_batch: torch.Tensor, y_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map. Assumes 8bit image.
    Works on a batched input.
    
    :return: kernel weights, error map
    :rtype: torch.Tensor[f64], torch.Tensor[i16]
    """
    model = __get_torch_ridge_model()
    
    model.fit(X_batch, y_batch)

    y_pred_batch = model.predict(X_batch)
    error_map_batch = (y_batch.to(torch.int16) - y_pred_batch.to(torch.int16))

    kernel_weights_batch = torch.from_numpy(model.coef_).to(torch.float64)

    return kernel_weights_batch, error_map_batch

def torch_mobilenet_v2(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Creates a mobilenet_v2 model prediction.

    :return: kernel weights, error map
    :rtype: torch.Tensor[i16]
    """
    model = __get_MobileNet_v2_model()

    ## model.fit(X, y) # TODO: finish
    
    ## y_pred = torch.round(model.predict(X))
    ## error_map = y_pred.to(torch.int16) - y.to(torch.int16) 

    # return error_map