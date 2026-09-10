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

    only_ref_pixels = reconstructed_img.copy()
    error_idx = 0
    for r in range(H):
        for c in range(W):
            if r % 2 != 0 or c % 2 != 0:
                feature_vector = get_feature_vector(r, c, only_ref_pixels, k)

                prediction = np.dot(feature_vector, weights)

                original_val = int(round(prediction)) + error_map[error_idx]
                #if original_val < 0:
                #    print(f"{original_val}: [{r},{c}]")
                error_idx += 1
                reconstructed_img[r,c] = max(0, min(255, original_val))

    return reconstructed_img


def get_feature_vector(r: int, c: int, reconstructed_img, k: int = 5):
    H, W = reconstructed_img.shape
    feature_vector = []
    half = k // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            row, col = r + i, c + j
            if 0 <= row < H and 0 <= col < W:
                feature_vector.append(int(reconstructed_img[row, col]))
            else:
                feature_vector.append(0)

    return feature_vector