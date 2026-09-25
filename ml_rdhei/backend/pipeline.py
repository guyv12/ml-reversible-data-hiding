import torch
import numpy as np
from cv2 import imread, IMREAD_UNCHANGED
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from bitarray import bitarray
from pathlib import Path

import backend.predictor.predict as ppredict
from backend.predictor.results import (
    compute_metrics, compute_dicom_metrics, Prediction
)
import backend.compressor.compress as ccompress
import backend.compressor.encryption as encryption
from backend.compressor.hiding import hider
from backend.receiver.receive import receive, receive_dicom

def _transform_bits_to_image(bits: bitarray, img_size: tuple[int, int], bpp: int = 8) -> np.ndarray:
    H, W = img_size[0], img_size[1]
    total_bits = H * W * bpp

    if len(bits) > total_bits:
        raise ValueError(f"Bitstream has {len(bits)} bits, image can hold {total_bits}")

    padded = bitarray(bits)
    padded.extend([0] * (total_bits - len(bits)))

    dtype = {8: np.uint8, 16: np.uint16}[bpp]
    return np.frombuffer(padded.tobytes(), dtype=dtype).reshape(H, W)

def _load_dicom(image_path: Path) -> np.ndarray:
	try:
		dicom = dcmread(image_path)
	except InvalidDicomError as e:
		raise ValueError(f"'{image_path.name}' is not a valid DICOM file") from e

	if "PixelData" not in dicom:
		raise ValueError(f"'{image_path.name}' does not contain any image data")
	if dicom.get("NumberOfFrames", 1) != 1:
		raise ValueError(f"'{image_path.name}' has multiple frames\nOnly single-frame images are supported")
	if dicom.get("SamplesPerPixel") != 1 or dicom.get("PhotometricInterpretation") not in ("MONOCHROME1", "MONOCHROME2"):
		raise ValueError(f"'{image_path.name}' is not a grayscale image")
	if dicom.get("BitsAllocated") != 16:
		raise ValueError(f"'{image_path.name}' is not a 16-bit image")

	try:
		image = dicom.pixel_array
	except Exception as e:
		raise ValueError(f"Failed to decode '{image_path.name}': {e}") from e

	if image.min() < 0:
		raise ValueError(f"'{image_path.name}' contains negative pixel values, which are not supported")

	return image.astype(np.uint16)

def transform_image_to_ndarray(image_path: Path) -> np.ndarray:
		if image_path.suffix.lower() == ".dcm":
			return _load_dicom(image_path)

		if image_path.suffix.lower() == ".pgm":
			image = imread(str(image_path), IMREAD_UNCHANGED)

			if image is None:
				raise FileNotFoundError(
					f"""Failed to load image: File not found or unreadable at '{image_path}'"""
				)
			return image

		raise ValueError(f"Unsupported image format: '{image_path.suffix}'")

def predict(image: np.ndarray, fmt: str) -> Prediction:
    H, W = image.shape[:2]
    mask = ppredict.reference_mask(H, W)

    if fmt.lower() == ".pgm":
        bpp = 8
        tensor = torch.from_numpy(np.ascontiguousarray(image[np.newaxis]))
        raw_ad = ppredict.ad_unfold_ridge_border(tensor)
        
        kernel_weights, ref_pixels, error_map, _ = next(raw_ad)

        ad = ccompress.compress_pgm_ad((H, W), kernel_weights, ref_pixels, error_map)
        metrics = compute_metrics(tensor, error_map, mask, len(ad), bpp)
        return Prediction(ad, metrics, bpp, (H, W))

    elif fmt.lower() == ".dcm":
        if image.max() > 0x0FFF:
            raise ValueError("Pixel values exceed 12 bits, which is not supported for hiding")
        
        bpp = 16
        tensor = torch.from_numpy(image[np.newaxis]).int()
        raw_ad = ppredict.dicom_ad_unfold_ridge_border(tensor)
        msb_error_map, lsb_kernel_weights, lsb_ref_pixels, lsb_error_map, _ = next(raw_ad)
        
        ad = ccompress.compress_dicom_ad(
            (H, W),
            msb_error_map,
            lsb_kernel_weights,
            lsb_ref_pixels,
            lsb_error_map
        )

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

def extract(
    source_image: np.ndarray,
    ad_decryption_key: str,
    message_decryption_key: str,
    fmt: str
) -> np.ndarray:

    H, W = source_image.shape[:2]

    ba = bitarray(endian='big')
    ba.frombytes(source_image.tobytes())

    if fmt.lower() == ".pgm":
        return receive(ba, ad_decryption_key, message_decryption_key, (H, W))
    elif fmt.lower() == ".dcm":
        return receive_dicom(ba, ad_decryption_key, message_decryption_key, (H, W))

    else:
        raise ValueError(f"Unsupported image format: '{fmt}'")
    
    