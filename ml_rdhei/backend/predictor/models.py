from sklearn.linear_model import Ridge
import torch.nn.functional as fn
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

def __predict_ridge(X: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor, 
                    valid_pred_range: tuple[int, int]) -> torch.Tensor:
    """
    Computes ridge based predictions:
    - Interior target pixels: X @ weights
    - Border target pixels: local mean of available reference pixels
    """
    # 1. Basic prediction
    y_pred = torch.round(X.to(weights.dtype) @ weights)

    # 2. Border prediction
    H, W = mask.shape
    K = int(X.shape[-1] ** 0.5) # The X shape should be (B?, H*W, K*K)
    pad = K // 2

    interior = torch.zeros((H, W), dtype=torch.bool)
    interior[pad : H - pad, pad : W - pad] = True
    border = ~interior.flatten()[~mask.flatten()]
    if border.any():
        X_border = X[..., border, :] # all leading dimensions, target pixels, K*K

        # We need to know which pixels are ref
        # No other way but to rebuild the padded mask and unfold it...
        padded_mask = fn.pad(mask, (pad, pad, pad, pad))
        ref_valid_mask = padded_mask.unfold(0, K, 1).unfold(1, K, 1)
        ref_valid_mask = ref_valid_mask.reshape(H * W, K * K)[~mask.flatten()][border]

        # Calculate mean using the positional validity mask instead of pixel values > 0
        ref_sum = X_border.masked_fill(~ref_valid_mask, 0).sum(dim=-1)
        ref_count = ref_valid_mask.sum(dim=-1).clamp(min=1)
        border_preds = torch.round(ref_sum / ref_count)

        # Overwrite the border predictions, '...' for both batched and single image inputs
        y_pred[..., border] = border_preds.to(y_pred.dtype)

    return y_pred.clamp(valid_pred_range[0], valid_pred_range[1]) # clamp to avoid big errors


def predict_sklearn_ridge(X: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, quantization: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map.
    Works on a single image input.

    :return: kernel weights, error map
    :rtype: torch.Tensor[f64] | torch.Tensor[i64], torch.Tensor[i16]
    """
    model = __get_sklearn_model()
    valid_pred_range = (torch.iinfo(y.dtype).min, torch.iinfo(y.dtype).max)
    
    X_np, y_np = X.double().numpy(), y.double().numpy() # sklearn requires float & numpy
    model.fit(X_np, y_np)

    W = torch.from_numpy(model.coef_)

    if not quantization:
        kernel_weights = W.to(torch.float64)
    else:
        kernel_weights = torch.round(W).to(torch.int64) # cut to int

    y_pred = __predict_ridge(X, kernel_weights, mask, valid_pred_range)

    error_map = y.to(torch.int16) - y_pred.to(torch.int16) # convert to int16, error in <-255, 255>

    return kernel_weights, error_map

def predict_torch_ridge(X_batch: torch.Tensor, y_batch: torch.Tensor, mask: torch.Tensor, quantization: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a ridge model prediction and error map.
    Works on a batched input.
    
    :return: kernel weights, error map
    :rtype: torch.Tensor[f64] | torch.Tensor[i64], torch.Tensor[i16]
    """
    model = __get_torch_model()
    valid_pred_range = (torch.iinfo(y_batch.dtype).min, torch.iinfo(y_batch.dtype).max)

    model.fit(X_batch, y_batch)

    W = model.weights
    
    if not quantization:
        kernel_weights_batch = W.to(torch.float64)
    else:
        kernel_weights_batch = torch.round(W).to(torch.int64)

    y_pred_batch = __predict_ridge(X_batch, kernel_weights_batch, mask, valid_pred_range)

    error_map_batch = y_batch.to(torch.int16) - y_pred_batch.to(torch.int16)

    return kernel_weights_batch, error_map_batch
