import torch

from backend.predictor.predict import reference_mask


def recovery(weights: list[float], ref_pixels: list[int], error_map: list[int],
             img_size: tuple[int, int], k: int = 5) -> np.ndarray:

    h, w = img_size
    half = k // 2
    n_ref = int(reference_mask(h, w).sum().item())
    
    if len(ref_pixels) != n_ref or len(error_map) != h * w - n_ref:
        raise ValueError(
            f"Extracted data does not match a {h}x{w} image: "
            f"{len(ref_pixels)} reference pixels, {len(error_map)} error values"
        )

    # reference pixels
    reconstructed_img = np.zeros((h, w), dtype=np.uint8)
    reconstructed_img[::2, ::2] = (np.asarray(ref_pixels, dtype=np.uint8).reshape(reconstructed_img[::2, ::2].shape))
    only_ref_pixels = reconstructed_img.copy()

    # error map
    target_mask = np.ones((h, w), dtype=bool)
    target_mask[::2, ::2] = False
    error_img = np.zeros((h, w), dtype=np.int64)
    error_img[target_mask] = np.asarray(error_map, dtype=np.int64)

    # weights
    weights = np.asarray(weights, dtype=np.float64)

    # INTERIOR CASE
    windows = np.lib.stride_tricks.sliding_window_view(only_ref_pixels,(k, k))

    interior_mask = target_mask[half:h-half,half:w-half]
    feature_matrix = windows.reshape(-1, k * k)
    feature_matrix = feature_matrix[interior_mask.ravel()]

    predictions = np.round(feature_matrix @ weights)
    predictions = predictions.clip(0, 255)

    errors = error_img[half:h-half,half:w-half][interior_mask]

    values = predictions.astype(np.int64) + errors
    reconstructed_img[half:h-half,half:w-half][interior_mask] = (
        np.clip(values,0,255).astype(np.uint8)
    )

    #BORDER CASE
    rows, cols = np.nonzero(target_mask)

    boundary = (
        (rows < half) |
        (rows >= h - half) |
        (cols < half) |
        (cols >= w - half)
    )

    boundary_rows = rows[boundary]
    boundary_cols = cols[boundary]

    for r, c in zip(boundary_rows, boundary_cols):
        r_start = max(0, r - half)
        r_end = min(h, r + half + 1)
        c_start = max(0, c - half)
        c_end = min(w, c + half + 1)

        window = only_ref_pixels[r_start:r_end, c_start:c_end]

        # Select only true reference pixels (even if their value is 0)
        window_valid = (~target_mask)[r_start:r_end, c_start:c_end]
        valid_pixels = window[window_valid]

        prediction = np.round(valid_pixels.mean())

        value = int(prediction) + error_img[r, c]
        reconstructed_img[r, c] = np.clip(value,0,255)

    return reconstructed_img


def dicom_recovery(img1_err_map: list[int], img2_weights: list[float], img2_ref_pixels: list[int], img2_err_map: list[int],
                   img_size: tuple[int, int], K: int = 5):
    H, W = img_size
    half = K // 2

    # Image 1
    img1 = 15 - torch.tensor(img1_err_map, dtype=torch.int16).reshape(H, W)

    # Image2
    # reference pixels
    img2 = torch.zeros((H, W), dtype=torch.int16)
    img2[::2, ::2] = (torch.tensor(img2_ref_pixels, dtype=torch.uint8).reshape(img2[::2, ::2].shape))
    only_ref_pixels = img2.clone()

    # error map
    target_mask = torch.ones((H, W), dtype=torch.bool)
    target_mask[::2, ::2] = False
    error_img = torch.zeros((H, W), dtype=torch.int64)
    error_img[target_mask] = torch.tensor(img2_err_map, dtype=torch.int64)

    # weights
    weights = torch.tensor(img2_weights, dtype=torch.float64)

    # INTERIOR CASE
    #windows = torch.lib.stride_tricks.sliding_window_view(only_ref_pixels,(k, k))
    windows = only_ref_pixels.unfold(0, K, 1).unfold(1, K, 1)

    interior_mask = target_mask[half : H - half, half : W - half]
    feature_matrix = windows.reshape(-1, K * K).to(torch.float64)
    feature_matrix = feature_matrix[interior_mask.ravel()]

    predictions = torch.round(feature_matrix @ weights)
    predictions = predictions.clamp(0, 255)

    errors = error_img[half : H -half, half : W - half][interior_mask]

    values = predictions.to(torch.int64) + errors
    img2[half : H - half, half : W - half][interior_mask] = (
        torch.clamp(values, 0, 255).to(torch.int16)
    )

    #BORDER CASE
    rows, cols = torch.nonzero(target_mask, as_tuple=True)

    boundary = (
        (rows < half) |
        (rows >= H - half) |
        (cols < half) |
        (cols >= W - half)
    )

    boundary_rows = rows[boundary]
    boundary_cols = cols[boundary]

    for r, c in zip(boundary_rows, boundary_cols):
        r_start = max(0, r - half)
        r_end = min(H, r + half + 1)
        c_start = max(0, c - half)
        c_end = min(W, c + half + 1)

        window = only_ref_pixels[r_start:r_end, c_start:c_end]

        # Select only true reference pixels (even if their value is 0)
        window_valid = (~target_mask)[r_start:r_end, c_start:c_end]
        valid_pixels = window[window_valid]

        prediction = torch.round(valid_pixels.float().mean())

        value = int(prediction) + error_img[r, c]
        img2[r, c] = torch.clamp(value, 0, 255)

    reconstructed_image = (img1 << 8) | img2
    return reconstructed_image
    