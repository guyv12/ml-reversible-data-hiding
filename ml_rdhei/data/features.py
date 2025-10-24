import data.loader as loader
import torch.nn.functional as fn
import torch


def extract_features(batch: torch.Tensor, mask: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, W = batch.shape
    ref_p = batch.view(B, H * W)[:, mask.flatten()]
    
    masked_batch = torch.zeros((B, H, W), device=ref_p.device, dtype=torch.float)
    masked_batch.view(B, H * W)[:, mask.flatten()] = ref_p
    
    X_all = fn.unfold(
        masked_batch.unsqueeze(1), # !GS: assumes grayscale .pgm - insert channel dimension
        kernel_size=K,
        stride=(1, 1),
        padding=(K // 2, K // 2)
    )

    X = X_all[:, :, ~mask.flatten()].permute(0, 2, 1).reshape(-1, K * K)
    y = batch.view(B, H * W)[:, ~mask.flatten()]

    return X, y
