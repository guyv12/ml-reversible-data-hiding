from bitarray import bitarray
import torch
from collections.abc import Callable

from backend.exceptions import *
from backend.receiver.extraction import *
from backend.receiver.recovery import *


def receive(stego_image: bitarray, key_ad: str, key_msg: str, 
            img_size: tuple[int, int] = (512, 512), bpp: int = 8, K: int = 5) -> tuple[torch.Tensor, str]:
    original_image, message = None, None
    try:
        ad, message_bits = split_payload(stego_image, img_size, bpp)
    except CorruptedDataError as e:
        raise CorruptedStegoImageError("Corrupted image") from e

    if key_ad:
        try:        
                weights, ref_pixels, error_map = ad_extraction(ad, key_ad, img_size, K)
        except CorruptedDataError as e:
            raise InvalidImageKeyError("Invalid image or image decryption key") from e

        original_image = recovery(weights, ref_pixels, error_map, img_size, K)

    if key_msg:
        try:
            message = msg_decryption(message_bits, key_msg)
        except CorruptedDataError as e:
            raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image, message


def receive_dicom(stego_image: bitarray, key_ad: str, key_msg: str,
                  img_size: tuple[int, int] = (512, 512), bpp = 16, K: int = 5) -> tuple[torch.Tensor, str]:
    original_image, message = None, None    
    try:
        ad, message_bits = split_payload(stego_image, img_size, bpp)
        img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map = ad_dicom_extraction(ad, key_ad, img_size, K)
    except CorruptedDataError as e:
        raise InvalidImageKeyError("Invalid image or image decryption key") from e

    original_image = dicom_recovery(img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, img_size, K)

    try:
        message = msg_decryption(message_bits, key_msg)
    except CorruptedDataError as e:
        raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image, message


def receive_cnn_features(stego_image: bitarray, key_ad: str, key_msg: str, model_fn: Callable,
                         img_size: tuple[int, int] = (512, 512), bpp: int = 8, K: int = 5) -> tuple[torch.Tensor, str]:
    original_image, message = None, None
    try:
        ad, message_bits = split_payload(stego_image, img_size, bpp)
        weights, ref_pixels, error_map = ad_extraction(ad, key_ad, img_size, K)
    except CorruptedDataError as e:
        raise InvalidImageKeyError("Invalid image or image decryption key") from e
    
    original_image = cnn_feat_recovery(weights, ref_pixels, error_map, model_fn, img_size)
    
    try:
        message = msg_decryption(message_bits, key_msg)
    except CorruptedDataError as e:
        raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image, message


def receive_cnn(stego_image: bitarray, key_ad: str, key_msg: str, model_fn: Callable,
                img_size: tuple[int, int] = (512, 512), bpp: int = 8) -> tuple[torch.Tensor, str]:
    original_image, message = None, None
    try:
        ad, message_bits = split_payload(stego_image, img_size, bpp)
        ref_pixels, error_map = ad_cnn_extraction(ad, key_ad, img_size)
    except CorruptedDataError as e:
        raise InvalidImageKeyError("Invalid image or image decryption key") from e
    
    original_image = cnn_recovery(ref_pixels, error_map, model_fn, img_size)
    
    try:
        message = msg_decryption(message_bits, key_msg)
    except CorruptedDataError as e:
        raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image, message
