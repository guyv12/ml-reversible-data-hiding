import ml_rdhei.data.loader as loader
from ml_rdhei.data.features import extract_features
from sklearn.linear_model import Ridge
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from pathlib import Path
import re


def __build_path(filename: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    res_dir = base_dir / "test-results"
    res_dir.mkdir(exist_ok=True)
    return res_dir / filename


def test_sklearn_kernel(K: int = 5, results_file: str | Path = None) -> None:
    if results_file is None:
        results_dir = __build_path('sklearn_results.txt')

    BOSSbase_loader, _ = loader.get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))

    H, W = 512, 512 # !GS: assumes grayscale .pgm

    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[::2, ::2] = True

    model = Ridge(alpha=1, solver="svd", fit_intercept=False)

    with open(results_dir, "a") as f:
        for idx, batch in enumerate(BOSSbase_loader):
            X, y, ref_pixels = extract_features(batch, mask, K)

            for image_X, image_y in zip(X, y):
                model.fit(image_X, image_y)
                y_pred = model.predict(image_X).astype("float32")

                psnr = peak_signal_noise_ratio(image_y.numpy(), y_pred, data_range=255.0)
                ssim = structural_similarity(image_y.numpy(), y_pred, data_range=255.0)

                # some assert?

                print(psnr, ssim)
                f.write(f"{psnr},{ssim}\n")


def test_torch_kernel(K: int = 5, results_file: str | Path = None) -> None:
    pass