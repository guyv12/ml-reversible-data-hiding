import torch.nn.functional as fn
import torch


def extract_features(batch: torch.Tensor, mask: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch.dim() != 3:
        raise TypeError("Feature extraction requires single channel images")

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
