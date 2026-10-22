from bitarray import bitarray
import torch
from collections.abc import Callable

from backend.exceptions import *
from backend.receiver.extraction import *
from backend.receiver.recovery import *


def receive(stego_image: bitarray, key_ad: str, key_msg: str, 
            img_size: tuple[int, int] = (512, 512), bpp: int = 8, K: int = 5) -> tuple[torch.Tensor | None, str | None, list[ValueError]]:
    errors = []
    original_image, message = None, None

    try:
        ad, message_bits = split_payload(stego_image, img_size, bpp)
    except CorruptedDataError:
        raise CorruptedStegoImageError("Corrupted image")

    if key_ad:
        try:        
                weights, ref_pixels, error_map = ad_extraction(ad, key_ad, img_size, K)
        except CorruptedDataError:
            errors.append(InvalidImageKeyError("Invalid image or image decryption key"))
        else:
            original_image = recovery(weights, ref_pixels, error_map, img_size, K)

    if key_msg:
        try:
            message = msg_decryption(message_bits, key_msg)
        except CorruptedDataError:
            errors.append(InvalidMessageKeyError("Invalid message decryption key"))

    return original_image, message, errors


def receive_dicom(stego_image: bitarray, key_ad: str, key_msg: str,
                  img_size: tuple[int, int] = (512, 512), bpp = 16, K: int = 5) -> tuple[torch.Tensor | None, str | None, list[ValueError]]:
    errors = []
    original_image, message = None, None    

    try:
        ad, message_bits = split_payload(stego_image, img_size, bpp)
    except CorruptedDataError:
        raise CorruptedStegoImageError("Corrupted image")

    if key_ad:
        try:
            img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map = ad_dicom_extraction(ad, key_ad, img_size, K)
        except CorruptedDataError:
            errors.append(InvalidImageKeyError("Invalid image or image decryption key"))
        else:
            original_image = dicom_recovery(img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, img_size, K)

    if key_msg:
        try:
            message = msg_decryption(message_bits, key_msg)
        except CorruptedDataError:
            errors.append(InvalidMessageKeyError("Invalid message decryption key"))

    return original_image, message, errors


def receive_cnn_features(stego_image: bitarray, key_ad: str, key_msg: str, model_fn: Callable,
                         img_size: tuple[int, int] = (512, 512), bpp: int = 8, K: int = 5) -> tuple[torch.Tensor | None, str | None, list[ValueError]]:
    errors = []
    original_image, message = None, 
    
    try:
        ad, message_bits = split_payload(stego_image, img_size, bpp)
    except CorruptedDataError:
        raise CorruptedStegoImageError("Corrupted image")

    if key_ad:
        try:
            weights, ref_pixels, error_map = ad_extraction(ad, key_ad, img_size, K)
        except CorruptedDataError:
            errors.append(InvalidImageKeyError("Invalid image or image decryption key"))
        else:
            original_image = cnn_feat_recovery(weights, ref_pixels, error_map, model_fn, img_size)

    if key_msg:
        try:
            message = msg_decryption(message_bits, key_msg)
        except CorruptedDataError:
            errors.append(InvalidMessageKeyError("Invalid message decryption key"))

    return original_image, message, errors


def receive_cnn(stego_image: bitarray, key_ad: str, key_msg: str, model_fn: Callable,
                img_size: tuple[int, int] = (512, 512), bpp: int = 8) -> tuple[torch.Tensor | None, str | None, list[ValueError]]:
    errors = []
    original_image, message = None, None
    
    try:
        ad, message_bits = split_payload(stego_image, img_size, bpp)
    except CorruptedDataError:
        raise CorruptedStegoImageError("Corrupted image")

    if key_ad:
        try:
            ref_pixels, error_map = ad_cnn_extraction(ad, key_ad, img_size)
        except CorruptedDataError:
            errors.append(InvalidImageKeyError("Invalid image or image decryption key"))
        else:
            original_image = cnn_recovery(ref_pixels, error_map, model_fn, img_size)
    
    if key_msg:
        try:
            message = msg_decryption(message_bits, key_msg)
        except CorruptedDataError:
            errors.append(InvalidMessageKeyError("Invalid message decryption key"))

    return original_image, message, errors
