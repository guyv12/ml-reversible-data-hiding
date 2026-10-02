import torch
from collections.abc import Callable
from backend.predictor.predict import reference_mask

def recovery(weights: torch.Tensor, ref_pixels: torch.Tensor, error_map: torch.Tensor,
             img_size: tuple[int, int], k: int = 5) -> torch.Tensor:

    h, w = img_size
    half = k // 2
    n_ref = int(reference_mask(h, w).sum().item())
    
    if len(ref_pixels) != n_ref or len(error_map) != h * w - n_ref:
        raise ValueError(
            f"Extracted data does not match a {h}x{w} image: "
            f"{len(ref_pixels)} reference pixels, {len(error_map)} error values"
        )

    # reference pixels
    reconstructed_img = torch.zeros((h, w), dtype=torch.uint8)
    reconstructed_img[::2, ::2] = (ref_pixels.to(dtype=torch.uint8).reshape(reconstructed_img[::2, ::2].shape))
    only_ref_pixels = reconstructed_img.clone()

    # error map
    target_mask = torch.ones((h, w), dtype=torch.bool)
    target_mask[::2, ::2] = False

    # error map is stored in the same order torch.nonzero returns
    rows, cols = torch.nonzero(target_mask, as_tuple=True)

    boundary = (
        (rows < half) |
        (rows >= h - half) |
        (cols < half) |
        (cols >= w - half)
    )

    error_map = error_map.to(dtype=torch.int64)
    interior_errors = error_map[~boundary]
    boundary_errors = error_map[boundary]

    # INTERIOR CASE
    windows = only_ref_pixels.unfold(0,k,1).unfold(1,k,1)

    interior_mask = target_mask[half:h-half,half:w-half]
    feature_matrix = windows.reshape(-1, k * k).to(torch.float64)
    feature_matrix = feature_matrix[interior_mask.ravel()]

    predictions = torch.round(feature_matrix @ weights)
    predictions = predictions.clamp(0, 255)

    values = predictions.to(torch.int64) + interior_errors
    reconstructed_img[half:h-half,half:w-half][interior_mask] = (
        torch.clamp(values,0,255).to(torch.uint8)
    )

    #BORDER CASE
    boundary_rows = rows[boundary]
    boundary_cols = cols[boundary]

    for r, c, error in zip(boundary_rows, boundary_cols, boundary_errors):
        r_start = max(0, r - half)
        r_end = min(h, r + half + 1)
        c_start = max(0, c - half)
        c_end = min(w, c + half + 1)

        window = only_ref_pixels[r_start:r_end, c_start:c_end]

        # Select only true reference pixels (even if their value is 0)
        window_valid = (~target_mask)[r_start:r_end, c_start:c_end]
        valid_pixels = window[window_valid]

        prediction = torch.round(valid_pixels.float().mean())

        value = int(prediction) + error
        reconstructed_img[r, c] = torch.clamp(value,0,255)

    return reconstructed_img


def dicom_recovery(
    img1_err_map: torch.Tensor, img2_weights: torch.Tensor,
    img2_ref_pixels: torch.Tensor, img2_err_map: torch.Tensor,
    img_size: tuple[int, int], K: int = 5
) -> torch.Tensor:

    H, W = img_size
    half = K // 2

    # Image 1
    img1 = 15 - img1_err_map.to(dtype=torch.int16).reshape(H, W)

    # Image2
    # reference pixels
    img2 = torch.zeros((H, W), dtype=torch.int16)
    img2[::2, ::2] = (img2_ref_pixels.to(dtype=torch.uint8).reshape(img2[::2, ::2].shape))
    only_ref_pixels = img2.clone()

    # error map
    target_mask = torch.ones((H, W), dtype=torch.bool)
    target_mask[::2, ::2] = False

    # error map is stored in the same order torch.nonzero returns
    rows, cols = torch.nonzero(target_mask, as_tuple=True)

    boundary = (
        (rows < half) |
        (rows >= H - half) |
        (cols < half) |
        (cols >= W - half)
    )

    error_map = img2_err_map.to(dtype=torch.int64)
    interior_errors = error_map[~boundary]
    boundary_errors = error_map[boundary]

    # weights
    weights = img2_weights.to(dtype=torch.float64)

    # INTERIOR CASE
    #windows = torch.lib.stride_tricks.sliding_window_view(only_ref_pixels,(k, k))
    windows = only_ref_pixels.unfold(0, K, 1).unfold(1, K, 1)

    interior_mask = target_mask[half : H - half, half : W - half]
    feature_matrix = windows.reshape(-1, K * K).to(torch.float64)
    feature_matrix = feature_matrix[interior_mask.ravel()]

    predictions = torch.round(feature_matrix @ weights)
    predictions = predictions.clamp(0, 255)

    values = predictions.to(torch.int64) + interior_errors
    img2[half : H - half, half : W - half][interior_mask] = (
        torch.clamp(values, 0, 255).to(torch.int16)
    )

    #BORDER CASE
    boundary_rows = rows[boundary]
    boundary_cols = cols[boundary]

    for r, c, error in zip(boundary_rows, boundary_cols, boundary_errors):
        r_start = max(0, r - half)
        r_end = min(H, r + half + 1)
        c_start = max(0, c - half)
        c_end = min(W, c + half + 1)

        window = only_ref_pixels[r_start:r_end, c_start:c_end]

        # Select only true reference pixels (even if their value is 0)
        window_valid = (~target_mask)[r_start:r_end, c_start:c_end]
        valid_pixels = window[window_valid]

        prediction = torch.round(valid_pixels.float().mean())

        value = int(prediction) + error
        img2[r, c] = torch.clamp(value, 0, 255)

    reconstructed_image = (img1 << 8) | img2
    return reconstructed_image


def cnn_feat_recovery(
    weights: torch.Tensor, ref_pixels: torch.Tensor, error_map: torch.Tensor,
    model_fn: Callable, img_size: tuple[int, int], k: int = 5
) -> torch.Tensor:

    h, w = img_size

    # reference pixels
    reconstructed_img = torch.zeros((h, w), dtype=torch.uint8)
    reconstructed_img[::2, ::2] = ref_pixels.to(dtype=torch.uint8).reshape(reconstructed_img[::2, ::2].shape)
    only_ref_pixels = reconstructed_img.clone()

    # error map
    target_mask = torch.ones((h, w), dtype=torch.bool)
    target_mask[::2, ::2] = False
    error_map = error_map.to(dtype=torch.int64)

    # weights
    weights = weights.to(dtype=torch.float64)

    # rest
    model = model_fn()
    model.eval()

    with torch.inference_mode():
        feature_map = model(
            only_ref_pixels.unsqueeze(0).unsqueeze(0).float() / 255.0
        ).permute(0, 2, 3, 1)

        _, _, _, C = feature_map.shape

    X = feature_map.reshape(h * w, C)
    X = X[target_mask.flatten()]

    predictions = torch.round(X.to(weights.dtype) @ weights)
    predictions = predictions.clamp(0, 255)

    reconstructed_img[target_mask] = torch.clamp(
        predictions.to(torch.int64) + error_map,
        0,
        255,
    ).to(torch.uint8)

    return reconstructed_img
