import numpy as np


def recovery(weights: list[float], ref_pixels: list[int], error_map: list[int], k: int = 5):

    h, w = 512, 512
    half = k // 2

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

    predictions = feature_matrix @ weights

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

        positive = window[window > 0]
        prediction = positive.mean()

        value = int(prediction) + error_img[r, c]
        reconstructed_img[r, c] = np.clip(value,0,255)

    return reconstructed_img