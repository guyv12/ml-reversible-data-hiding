import numpy as np

def recovery(weights: list[float], ref_pixels: list[int], error_map: list[int], k: int = 5):
    reconstructed_img = np.zeros((512,512), dtype=np.uint8)

    ref_idx = 0
    for r in range(512):
        for c in range(512):
            if r % 2 == 0 and c % 2 == 0:
                reconstructed_img[r,c] = ref_pixels[ref_idx]
                ref_idx += 1

    error_idx = 0
    for r in range(512):
        for c in range(512):
            if r % 2 != 0 or c % 2 != 0:
                feature_vector = get_feature_vector(r, c, reconstructed_img)

                if len(feature_vector) == k^2-1:
                    prediction = np.dot(feature_vector, weights)
                else:
                    counter = 0
                    for f in feature_vector:
                        if f > 0:
                            counter +=1
                    prediction = np.sum(feature_vector) / counter

                original_val = int(round(prediction)) + error_map[error_idx]
                error_idx += 1
                reconstructed_img[r,c] = max(0, min(255, original_val))

    return reconstructed_img


def get_feature_vector(r: int, c: int, reconstructed_img, k: int = 5):
    feature_vector = []
    for i in range(-(k//2), k//2+1):
        for j in range(-(k//2), k//2+1):
            if 0 <= r - i < 512 and 0 <= c - j < 512 :
                feature_vector.append(reconstructed_img[r-i, c-j])

    return feature_vector