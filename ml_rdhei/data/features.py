import torch.nn.functional as fn
import torch


def extract_features(batch: torch.Tensor, mask: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, W = batch.shape
    ref_p = batch.view(B, H * W)[:, mask.flatten()]

    # apply mask to all images in the batch
    masked_batch = torch.zeros((B, H, W), dtype=torch.float)
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
