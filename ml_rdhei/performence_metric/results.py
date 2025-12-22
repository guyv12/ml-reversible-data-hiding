import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error as Mse

psnr_results = []
ssim_results = []


def psnr(original, processed):
    mse = Mse(original, processed)
    psnr = 10 * np.log10(255 / mse)
    psnr_results.append(psnr)
    return psnr

def ssim(original, proccesed, K1=0.01, K2=0.03, L=255, window_size=11, sigma=1.5):
    img1 = original.astype(np.float64)
    img2 = proccesed.astype(np.float64)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = gaussian_filter(img1, sigma=sigma)
    mu2 = gaussian_filter(img2, sigma=sigma)

    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma) - mu1 ** 2
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma) - mu2 ** 2
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_results.append(ssim_map.mean())
    return ssim_map.mean()

# zamiast samemu pisać to można zaimportować to