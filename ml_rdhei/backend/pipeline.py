import torch
import numpy as np
from cv2 import imread, IMREAD_UNCHANGED
from pydicom import dcmread
from bitarray import bitarray
from pathlib import Path

import backend.predictor.predict as ppredict
from backend.predictor.results import (
    compute_metrics, compute_dicom_metrics, Prediction
)
import backend.compressor.compress as ccompress
import backend.compressor.encryption as encryption
from backend.compressor.hiding import hider
from backend.receiver.receive import receive

def _transform_bits_to_image(bits: bitarray, img_size: tuple[int, int], bpp: int = 8) -> np.ndarray:
    H, W = img_size[0], img_size[1]
    total_bits = H * W * bpp

    if len(bits) > total_bits:
        raise ValueError(f"Bitstream has {len(bits)} bits, image can hold {total_bits}")

    padded = bitarray(bits)
    padded.extend([0] * (total_bits - len(bits)))

    dtype = {8: np.uint8, 16: np.uint16}[bpp]
    return np.frombuffer(padded.tobytes(), dtype=dtype).reshape(H, W)

def transform_image_to_ndarray(image_path: Path) -> np.ndarray:
		if image_path.suffix.lower() == ".dcm":
			try:
				dicom = dcmread(image_path)
				image = dicom.pixel_array
			except Exception as e:
				raise ValueError(f"Failed to decode DICOM file '{image_path}': {e}")
		else:
			image = imread(str(image_path), IMREAD_UNCHANGED)

		if image is None:
			raise FileNotFoundError(
				f"""Failed to load image: File not found or unreadable at '{image_path}'"""
			)

		return image


def predict(image: np.ndarray, fmt: str) -> Prediction:
    H, W = image.shape[:2]
    mask = ppredict.reference_mask(H, W)

    if fmt.lower() == ".pgm":
        bpp = 8
        tensor = torch.from_numpy(image[np.newaxis]).int()
        raw_ad = ppredict.pgm_raw_ad_sklearn(tensor)
        kernel_weights, ref_pixels, error_map, _ = next(raw_ad)

        ad = ccompress.compress_pgm_ad((H, W), kernel_weights, ref_pixels, error_map)
        metrics = compute_metrics(tensor, error_map, mask, len(ad), bpp)
        return Prediction(ad, metrics, bpp, (H, W))

    elif fmt.lower() == ".dcm":
        bpp = 16
        tensor = torch.from_numpy(image[np.newaxis]).int()
        raw_ad = ppredict.dicom_raw_ad_sklearn(tensor)
        msb_error_map, lsb_kernel_weights, lsb_ref_pixels, lsb_error_map = next(raw_ad)
        
        ad = ccompress.compress_dicom_ad((H, W), msb_error_map, lsb_kernel_weights, lsb_ref_pixels, lsb_error_map)
        metrics = compute_dicom_metrics(tensor, lsb_error_map, mask, len(ad), bpp)
        return Prediction(ad, metrics, bpp, (H, W))

    else:
        raise ValueError(f"Unsupported image format: '{fmt}'")
    

def hide(
    prediction: Prediction,
    ad_encryption_key: str,
    message_encryption_key: str,
    message: str
) -> np.ndarray:
    
    ad_enrypted = encryption.encrypt_ad(
        prediction.ad,
        prediction.pixels,
        prediction.bpp,
        ad_encryption_key
    )

    bits = hider(
        ad_enrypted,
        prediction.metrics.payload_capacity,
        message,
        message_encryption_key
    )
 
    return _transform_bits_to_image(bits, prediction.shape, prediction.bpp)

def extract(source_image: np.ndarray, ad_decryption_key: str, message_decryption_key: str) -> np.ndarray:
    H, W = source_image.shape[:2]

    ba = bitarray(endian='big')
    ba.frombytes(source_image.tobytes())
    
    return receive(ba, ad_decryption_key, message_decryption_key, (H, W))