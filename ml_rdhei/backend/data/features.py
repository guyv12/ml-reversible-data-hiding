import torch.nn.functional as fn
import torch
from backend.models import get_torch_unet_model


def unfold_features(batch: torch.Tensor, mask: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts features via sliding window operation on a padded image.
    Works on a batched input. Requires no channel dimension.

    :return: X, y, ref_p
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor - image's dtype
    """
    if batch.dim() != 3:
        raise TypeError("Feature extraction requires a batch of single-channel images (B, H, W)")

    B, H, W = batch.shape
    ref_p = batch.view(B, H * W)[:, mask.flatten()]

    # apply mask to all images in the batch
    masked_batch = torch.zeros(batch.shape, dtype=batch.dtype)
    masked_batch.view(B, H * W)[:, mask.flatten()] = ref_p

    pad = K // 2 # Last 2 dimensions padded with K // 2 0s
    padded_batch = fn.pad(masked_batch, (pad, pad, pad, pad))

    # patches shape = (B, H, W, K, K)
    patches = padded_batch.unfold(1, K, 1).unfold(2, K, step=1)

    X = patches.reshape(B, H * W, K * K)[:, ~mask.flatten(), :]
    y = batch.view(B, H * W)[:, ~mask.flatten()]
    return X, y, ref_p


def cnn_features(batch: torch.Tensor, mask: torch.Tensor, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts features through CNN pass.
    Works on a batched input. Requires no channel dimension.
    
    :return: X, y, ref_p
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor - image's dtype
    """
    if batch.dim() != 3:
        raise TypeError("Feature extraction requires a batch of single-channel images (B, H, W)")
        
    B, H, W = batch.shape
    ref_p = batch.view(B, H * W)[:, mask.flatten()]

    masked_batch = torch.zeros(batch.shape, dtype=batch.dtype)
    masked_batch.view(B, H * W)[:, mask.flatten()] = ref_p

    # Add channel dimension and normalize to <0, 1>
    X_pre = masked_batch.unsqueeze(1).float() / 255.0

    model = get_torch_unet_model(classes=25)
    model.eval()

    with torch.inference_mode():
        # (B,H,C,W), change have it in sklearn [samples[features]] format because C are the features
        feature_map = model(X_pre).permute(0, 2, 3, 1)
        _, _, _, C = feature_map.shape

    X = feature_map.reshape(B, H * W, C)[:, ~mask.flatten(), :]
    y = batch.view(B, H * W)[:, ~mask.flatten()]
    return X, y, ref_p


def lr_decompose(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch.dtype in (torch.int16, torch.int32, torch.uint16, torch.uint32):
        raise TypeError("Left-Right Decomposition requires (u)int16/int32")

    image1_batch = (batch >> 8).to(torch.uint8) # left
    image2_batch = (batch & 0x00FF).to(torch.uint8) # right
    
    return image1_batch, image2_batch

def oe_decompose(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("Odd-Even Decomposition is not implemented yet...")

    if not batch.dtype in (torch.int16, torch.int32, torch.uint16, torch.uint32):
        raise TypeError("Odd-Even Decomposition requires (u)int16/int32")
    
    pass
