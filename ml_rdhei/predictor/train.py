import ml_rdhei.data.loader as loader
from ml_rdhei.data.features import extract_features
import ml_rdhei.performence_metric.results as res
from sklearn.linear_model import Ridge
import torch
import re


def train_kernel(K: int = 5) -> None:

    BOSSBase, _ = loader.get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))

    H, W = 512, 512 # !GS: assumes grayscale .pgm

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    model = Ridge(alpha=1, solver="svd", fit_intercept=False)

    for idx, batch in enumerate(BOSSBase):
        X, y = extract_features(batch, mask, K)

        for image_X, image_y in zip(X, y):
            model.fit(image_X, image_y)

        for image_X, image_y in zip(X, y):
            y_pred = model.predict(image_X)
            psnr = res.psnr(image_y, y_pred)
            ssim = res.ssim(image_y, y_pred)

            print(psnr, ssim)