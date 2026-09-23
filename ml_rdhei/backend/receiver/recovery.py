import numpy as np
from backend.predictor.predict import reference_mask

def recovery(weights: np.ndarray, ref_pixels: np.ndarray, error_map: np.ndarray,
             img_size: tuple[int, int], k: int = 5):

    h, w = img_size
    half = k // 2
    n_ref = int(reference_mask(h, w).sum().item())
    
    if len(ref_pixels) != n_ref or len(error_map) != h * w - n_ref:
        raise ValueError(
            f"Extracted data does not match a {h}x{w} image: "
            f"{len(ref_pixels)} reference pixels, {len(error_map)} error values"
        )

    ref_pixels = np.asarray(ref_pixels)
    if ref_pixels.size and (ref_pixels.min() < 0 or ref_pixels.max() > 255):
        raise ValueError("Extracted reference pixels are outside the 0-255 range")

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
    # Only reference pixels (even row, even column) are non-zero in a window,
    # so a target's dot product reduces to the weights that land on reference
    # pixels (4 or 6 of 25 for k=5). Process each target parity with strided
    # slices, accumulating in the same order as the flattened window.
    ref_float = only_ref_pixels.astype(np.float64)
    weights_2d = weights.reshape(k, k)

    for row_parity, col_parity in ((0, 1), (1, 0), (1, 1)):
        r0 = half + (row_parity - half) % 2
        c0 = half + (col_parity - half) % 2
        n_rows = len(range(r0, h - half, 2))
        n_cols = len(range(c0, w - half, 2))
        if n_rows == 0 or n_cols == 0:
            continue

        predictions = np.zeros((n_rows, n_cols))
        for i in range(k):
            if (row_parity + i - half) % 2:
                continue
            rs = r0 + i - half
            for j in range(k):
                if (col_parity + j - half) % 2:
                    continue
                cs = c0 + j - half
                predictions += weights_2d[i, j] * ref_float[rs:rs + 2 * n_rows - 1:2, cs:cs + 2 * n_cols - 1:2]

        predictions = np.round(predictions).clip(0, 255)

        targets = (slice(r0, r0 + 2 * n_rows - 1, 2), slice(c0, c0 + 2 * n_cols - 1, 2))
        values = predictions.astype(np.int64) + error_img[targets]
        reconstructed_img[targets] = np.clip(values, 0, 255).astype(np.uint8)

    #BORDER CASE
    # Targets within `half` of the edge are predicted as the mean of the true
    # reference pixels (even if their value is 0) in their window clipped to
    # the image. Process the four edge strips on small zero-padded crops.
    is_ref = ~target_mask
    top_end = min(half, h)
    bottom_start = max(half, h - half)
    left_end = min(half, w)
    right_start = max(half, w - half)
    strips = (
        (0, top_end, 0, w),
        (bottom_start, h, 0, w),
        (half, bottom_start, 0, left_end),
        (half, bottom_start, right_start, w),
    )

    for r0, r1, c0, c1 in strips:
        if r1 <= r0 or c1 <= c0:
            continue

        ref_sums = _window_sums(only_ref_pixels, r0, r1, c0, c1, half)
        ref_counts = _window_sums(is_ref, r0, r1, c0, c1, half)

        strip = (slice(r0, r1), slice(c0, c1))
        targets = target_mask[strip]
        predictions = np.round(ref_sums[targets] / ref_counts[targets]).astype(np.int64)

        values = predictions + error_img[strip][targets]
        reconstructed_img[strip][targets] = np.clip(values, 0, 255).astype(np.uint8)

    return reconstructed_img


def _window_sums(img: np.ndarray, r0: int, r1: int, c0: int, c1: int, half: int) -> np.ndarray:
    """Sum of each (2*half+1)^2 window centred on [r0:r1, c0:c1], zeros outside the image."""
    h, w = img.shape
    top, bottom, left, right = r0 - half, r1 + half, c0 - half, c1 + half
    crop = img[max(0, top):min(h, bottom), max(0, left):min(w, right)].astype(np.int64)
    crop = np.pad(crop, ((max(0, -top), max(0, bottom - h)), (max(0, -left), max(0, right - w))))
    k = 2 * half + 1
    return np.lib.stride_tricks.sliding_window_view(crop, (k, k)).sum(axis=(2, 3))
