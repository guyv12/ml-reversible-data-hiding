from bitarray import bitarray
from backend.exceptions import *
import torch
from collections.abc import Callable
from backend.receiver.extraction import *
from backend.receiver.recovery import recovery, dicom_recovery, cnn_feat_recovery

def receive(stego_image: bitarray, key_ad: str, key_msg: str, img_size: tuple[int, int] = (512, 512), K: int = 5) -> tuple[torch.Tensor, str]:
    try:
        weights, ref_pixels, error_map, message_bits = ad_extraction(stego_image, key_ad, img_size, 8, K)
    except CorruptedDataError as e:
        raise InvalidImageKeyError("Invalid image or image decryption key") from e
    
    original_image = recovery(weights, ref_pixels, error_map, img_size, K)

    try:
        message = msg_extraction(message_bits, key_msg)
    except CorruptedDataError as e:
        raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image, message

def receive_dicom(stego_image: bitarray, key_ad: str, key_msg: str, img_size: tuple[int, int] = (512, 512), K: int = 5) -> tuple[torch.Tensor, str]:
    try:
        img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, message_bits = ad_dicom_extraction(stego_image, key_ad, img_size, 16, K)
    except CorruptedDataError as e:
        raise InvalidImageKeyError("Invalid image or image decryption key") from e

    original_image = dicom_recovery(img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, img_size, K)

    try:
        message = msg_extraction(message_bits, key_msg)
    except CorruptedDataError as e:
        raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image, message

def receive_cnn_features(stego_image: bitarray, key_ad: str, key_msg: str, model_fn: Callable,
                         img_size: tuple[int, int] = (512, 512), K: int = 5) -> tuple[torch.Tensor, str]:
    try:
        weights, ref_pixels, error_map, message = ad_extraction(stego_image, key_ad, img_size, 8, K)
    except CorruptedDataError as e:
        raise InvalidImageKeyError("Invalid image or image decryption key") from e
    
    original_image = cnn_feat_recovery(weights, ref_pixels, error_map, model_fn, img_size, K)
    
    try:
        message = msg_extraction(message, key_msg)
    except CorruptedDataError as e:
        raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image, message
