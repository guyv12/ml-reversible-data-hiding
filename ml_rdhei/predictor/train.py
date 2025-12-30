import ml_rdhei.data.loader as loader
from ml_rdhei.data.features import extract_features
import ml_rdhei.performence_metric.results as res
from sklearn.linear_model import Ridge
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import re


# with scikit learn there will be loops - we can implement ridge in pyTorch to handle batched input but idk if worth it
def train_kernel(K: int = 5) -> None:

    BOWS2_loader, _ = loader.get_loader("datasets/BOWS2_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))

    H, W = 512, 512 # !GS: assumes grayscale .pgm

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    model = Ridge(alpha=1, solver="svd", fit_intercept=False)

    for idx, batch in enumerate(BOWS2_loader):
        X, y, ref_pixels = extract_features(batch, mask, K)

        for image_X, image_y in zip(X, y):
            model.fit(image_X, image_y)

        for image_X, image_y in zip(X, y):
            y_pred = model.predict(image_X).astype("float32")

            psnr = peak_signal_noise_ratio(image_y.numpy(), y_pred, data_range=255.0)
            ssim = structural_similarity(image_y.numpy(), y_pred, data_range=255.0)
            print(psnr, ssim)