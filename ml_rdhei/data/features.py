import torch.nn.functional as fn
import torch


def extract_features(batch: torch.Tensor, mask: torch.Tensor, K) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch.dim() != 3:
        raise TypeError("Feature extraction requires single channel images")
    
    B, H, W = batch.shape
    ref_p = batch.view(B, H * W)[:, mask.flatten()]

    # apply mask to all images in the batch
    masked_batch = torch.zeros((B, H, W), dtype=batch.dtype)
    masked_batch.view(B, H * W)[:, mask.flatten()] = ref_p

    # X_all shape = (B, K*K, H*W)
    X_all = fn.unfold(
        masked_batch.unsqueeze(1), # !GS: assumes grayscale .pgm - insert channel dimension
        kernel_size=K,
        stride=1,
        padding=(K // 2, K // 2)
    )

    X = X_all[:, :, ~mask.flatten()].permute(0, 2, 1) # get rid of refs and transform to (B, L, K*K)
    y = batch.view(B, H * W)[:, ~mask.flatten()]
    return X, y, ref_p


def lr_decompose(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.is_floating_point:
        raise TypeError("Left-Right Decomposition requires (u)int16/int32")

    image1_batch = (batch >> 8).to(torch.uint8) # left
    image2_batch = (batch & 0x00FF).to(torch.uint8) # right
    
    return image1_batch, image2_batch

def oe_decompose(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.is_floating_point:
        raise TypeError("Odd-Even Decomposition requires (u)int16/int32")
    
    pass
