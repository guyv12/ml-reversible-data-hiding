import torch
import torch.nn.functional as fn


def dot_product(X: torch.Tensor, weights: torch.Tensor, 
                    valid_pred_range: tuple[int, int]) -> torch.Tensor:

    y_pred = torch.round(X.to(weights.dtype) @ weights)
    return y_pred.clamp(valid_pred_range[0], valid_pred_range[1])


def dot_product_with_border(X: torch.Tensor, weights: torch.Tensor,
                    valid_pred_range: tuple[int, int], mask: torch.Tensor) -> torch.Tensor:
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
