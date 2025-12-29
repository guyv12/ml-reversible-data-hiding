import re
import ml_rdhei.data.loader as loader
import time
import numpy as np
from sklearn.linear_model import Ridge
import ml_rdhei.performence_metric.results as res


def train_kernel(K: int = 5) -> None:
    start_time = time.time()
    BOSSbase_train_loader, _ = loader.get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    H, W = 512, 512

    BOSSBase = next(iter(BOSSbase_train_loader))[0].numpy()
    print(BOSSBase)
    mask = np.zeros((H, W), dtype=bool)
    mask[::2, ::2] = True

    non_reference = ~mask
    coords = np.argwhere(non_reference)
    feature_matrix = np.zeros((len(coords), K ** 2), dtype=np.uint8)
    y = BOSSBase[non_reference].reshape(len(coords), )
    R = K // 2

    for idx, (r, c) in enumerate(coords):
        r_min = max(r - R, 0)
        r_max = min(r + R + 1, H)
        c_min = max(c - R, 0)
        c_max = min(c + R + 1, W)
        r0 = abs(r - R) if (r - R) < 0 else 0
        c0 = abs(c - R) if (c - R) < 0 else 0

        patch = BOSSBase[r_min:r_max, c_min:c_max]

        feature_vector = np.zeros((K, K), dtype=np.uint8)
        feature_vector[r0: r0 + patch.shape[0], c0: c0 + patch.shape[1]] = patch
        # print(r, c)
        # print(feature_vector)

        patch_mask = non_reference[r_min:r_max, c_min:c_max]
        feature_vector[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]][patch_mask] = 0

        # print(feature_vector)
        feature_matrix[idx] = feature_vector.reshape(1, K ** 2)

    model = Ridge(alpha=1, solver="svd", fit_intercept=False)
    model.fit(feature_matrix, y)
    kelner = model.coef_

    output_matrix = np.copy(BOSSBase)

    for idx, (r, c) in enumerate(coords):
        if (r < R or c < R):
            output_matrix[r, c] = np.sum(feature_matrix[idx])

        output_matrix[r, c] = round(np.sum(feature_matrix[idx] * kelner))

    np.set_printoptions(threshold=np.inf, suppress=True)
    print(output_matrix)
    print()
    print(kelner.reshape(K, K))
    print()

    end_time = time.time()
    print(f"elapsed: {end_time - start_time}")
    psnr = res.psnr(BOSSBase, output_matrix)
    ssim = res.ssim(BOSSBase, output_matrix)
    print("psnr: ", psnr)
    print("ssim: ", ssim)
