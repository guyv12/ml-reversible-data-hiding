import torch
import numpy as np

import backend.predictor.predict as ppredict
from backend.predictor.results import compute_metrics, Prediction
import backend.compressor.compress as ccompress

def predict(image: np.ndarray, bpp: int = 8) -> Prediction:
    H, W = image.shape[:2]
    tensor = torch.from_numpy(image[np.newaxis]).float()
    raw_ad = ppredict.pgm_raw_ad_sklearn(tensor)
    kernel_weights, ref_pixels, error_map, original = next(raw_ad)
    mask = ppredict.reference_mask(H, W)

    ad = ccompress.compress_pgm_ad((H, W), kernel_weights, ref_pixels, error_map)
    metrics = compute_metrics(original, error_map, mask, len(ad), bpp)

    return Prediction(ad, metrics)