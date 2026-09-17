import torch

def recovery(weights: list[float], ref_pixels: list[int], error_map: list[int], k: int = 5):

    h, w = 512, 512
    half = k // 2

    # reference pixels
    reconstructed_img = torch.zeros((h, w), dtype=torch.uint8)
    reconstructed_img[::2, ::2] = (torch.tensor(ref_pixels, dtype=torch.uint8).reshape(reconstructed_img[::2, ::2].shape))
    only_ref_pixels = reconstructed_img.clone()

    # error map
    target_mask = torch.ones((h, w), dtype=torch.bool)
    target_mask[::2, ::2] = False
    error_img = torch.zeros((h, w), dtype=torch.int64)
    error_img[target_mask] = torch.tensor(error_map, dtype=torch.int64)

    # weights
    weights = torch.tensor(weights, dtype=torch.float64)

    # INTERIOR CASE
    #windows = torch.lib.stride_tricks.sliding_window_view(only_ref_pixels,(k, k))
    windows = only_ref_pixels.unfold(0,k,1).unfold(1,k,1)

    interior_mask = target_mask[half:h-half,half:w-half]
    feature_matrix = windows.reshape(-1, k * k).to(torch.float64)
    feature_matrix = feature_matrix[interior_mask.ravel()]

    predictions = torch.round(feature_matrix @ weights)
    predictions = predictions.clamp(0, 255)

    errors = error_img[half:h-half,half:w-half][interior_mask]

    values = predictions.to(torch.int64) + errors
    reconstructed_img[half:h-half,half:w-half][interior_mask] = (
        torch.clamp(values,0,255).to(torch.uint8)
    )

    #BORDER CASE
    rows, cols = torch.nonzero(target_mask, as_tuple=True)

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

        prediction = torch.round(valid_pixels.float().mean())

        value = int(prediction) + error_img[r, c]
        reconstructed_img[r, c] = torch.clamp(value,0,255)

    return reconstructed_img
