import numpy as np
from backend.predictor.predict import reference_mask

def recovery(weights: list[float], ref_pixels: list[int], error_map: list[int],
             img_size: tuple[int, int], k: int = 5):

    H, W = img_size
    n_ref = int(reference_mask(H, W).sum().item())
    
    if len(ref_pixels) != n_ref or len(error_map) != H * W - n_ref:
        raise ValueError(
            f"Extracted data does not match a {H}x{W} image: "
            f"{len(ref_pixels)} reference pixels, {len(error_map)} error values"
        )
        
    reconstructed_img = np.zeros((H, W), dtype=np.uint8)

    ref_idx = 0
    for r in range(H):
        for c in range(W):
            if r % 2 == 0 and c % 2 == 0:
                reconstructed_img[r,c] = ref_pixels[ref_idx]
                ref_idx += 1

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
