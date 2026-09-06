import torch
import numpy as np
from bitarray import bitarray

import backend.predictor.predict as ppredict
from backend.predictor.results import compute_metrics, Prediction
import backend.compressor.compress as ccompress
import backend.compressor.encryption as encryption
from backend.compressor.hiding import hider

def _transform_bits_to_image(bits: bitarray, img_size: tuple[int, int], bpp: int = 8) -> np.ndarray:
    H, W = img_size[0], img_size[1]
    total_bits = H * W * bpp

    if len(bits) > total_bits:
        raise ValueError(f"Bitstream has {len(bits)} bits, image can hold {total_bits}")

    padded = bitarray(bits)
    padded.extend([0] * (total_bits - len(bits)))

    dtype = {8: np.uint8, 16: np.dtype(">u2")}[bpp]
    return np.frombuffer(padded.tobytes(), dtype=dtype).reshape(H, W)

def predict(image: np.ndarray, bpp: int = 8) -> Prediction:
    H, W = image.shape[:2]
    tensor = torch.from_numpy(image[np.newaxis]).float()
    raw_ad = ppredict.pgm_raw_ad_sklearn(tensor)
    kernel_weights, ref_pixels, error_map, original = next(raw_ad)
    mask = ppredict.reference_mask(H, W)

    ad = ccompress.compress_pgm_ad((H, W), kernel_weights, ref_pixels, error_map)
    metrics = compute_metrics(original, error_map, mask, len(ad), bpp)
    pixels = (metrics.payload_capacity + len(ad)) // bpp
    return Prediction(ad, metrics, bpp, (H, W))

def hide(prediction: Prediction, key: str, message: str) -> np.ndarray:
    ad_enrypted = encryption.encrypt_ad(
        prediction.ad,
        prediction.pixels,
        prediction.bpp,
        key
    )

    bits = hider(
        ad_enrypted,
        prediction.metrics.payload_capacity,
        message,
        key
    )
 
    return _transform_bits_to_image(bits, prediction.shape, prediction.bpp)