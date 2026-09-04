import torch.nn.functional as fn
import torch
from backend.models import get_MobileNet_v2_model


def unfold_features(batch: torch.Tensor, mask: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def extract_features_cnn(batch: torch.Tensor, mask: torch.Tensor, K: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("Not implemented")
    if batch.dim() != 3:
        raise TypeError("Feature extraction requires a batch of single-channel images (B, H, W)")
        
    B, H, W = batch.shape
    
    # 1. Extract reference pixels from the checkerboard mask
    flat_mask = mask.flatten()
    ref_p = batch.view(B, H * W)[:, flat_mask]
    
    # 2. Apply mask to create the masked input batch
    masked_batch = torch.zeros((B, H, W), dtype=batch.dtype, device=batch.device)
    masked_batch.view(B, H * W)[:, flat_mask] = ref_p

    # 3. MobileNetV2 expects 3-channel RGB inputs scaled/normalized for pretrained weights
    # Expand (B, H, W) -> (B, 1, H, W) -> (B, 3, H, W)
    x_input = masked_batch.unsqueeze(1).repeat(1, 3, 1, 1).float() / 255.0
    
    # Standard ImageNet normalization expected by MobileNetV2 weights
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
    x_input = (x_input - mean) / std

    # 4. Pass through CNN backbone to get dense feature maps: shape (B, C, H', W')
    model = get_MobileNet_v2_model().to(batch.device)
    model.eval()
    
    with torch.no_grad():
        feature_map = model(x_input)
        
    _, C, H_feat, W_feat = feature_map.shape

    # 5. Downscale or interpolate the mask to match the CNN output spatial dimensions 
    # (MobileNet downsamples resolution by a factor of 32 across its layers)
    resized_mask = torch.nn.functional.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0), 
        size=(H_feat, W_feat), 
        mode='nearest'
    ).squeeze().bool().flatten()

    # 6. Extract target and feature vectors dynamically
    # Reshape feature map from (B, C, H_feat, W_feat) -> (B, H_feat * W_feat, C)
    features_flat = feature_map.permute(0, 2, 3, 1).reshape(B, H_feat * W_feat, C)
    image_flat = batch.unsqueeze(1).float() # or corresponding target downsampled layout
    
    # For a direct dense spatial match, slice features at target indices
    # (Note: If dimensions match your grid, extract target pixels y similarly)
    target_mask_flat = ~resized_mask
    X = features_flat[:, target_mask_flat, :]  # Shape: (B, Num_Targets, C)
    
    # Downsample target image to match feature map spatial grid if shapes differ, 
    # or use standard un-downsampled alignment. Here we align with feature spatial dims:
    y_downsampled = torch.nn.functional.interpolate(
        batch.unsqueeze(1).float(), 
        size=(H_feat, W_feat), 
        mode='nearest'
    ).squeeze(1).view(B, H_feat * W_feat)
    
    y = y_downsampled[:, target_mask_flat]    # Shape: (B, Num_Targets)

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
