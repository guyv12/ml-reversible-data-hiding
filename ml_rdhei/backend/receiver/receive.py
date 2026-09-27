from bitarray import bitarray
from collections.abc import Callable

from backend.receiver.extraction import *
from backend.receiver.recovery import recovery, dicom_recovery, cnn_feat_recovery


def receive(stego_image: bitarray, key_ad: str, key_msg: str, img_size: tuple[int, int] = (512, 512), K: int = 5) -> tuple[bytes, str]:

    weights, ref_pixels, error_map, message = ad_extraction(stego_image, key_ad, img_size, 8, K)
    message = msg_extraction(message, key_msg)
    original_image = recovery(weights, ref_pixels, error_map, img_size, K)

    return original_image, message


def receive_dicom(stego_image: bitarray, key_ad: str, key_msg: str, img_size: tuple[int, int] = (512, 512), K: int = 5) -> tuple[bytes, str]:

    img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, message = ad_dicom_extraction(stego_image, key_ad, img_size, 16, K)
    message = msg_extraction(message, key_msg)
    original_image = dicom_recovery(img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, img_size, K)

    return original_image.numpy(), message


def receive_cnn_features(stego_image: bitarray, key_ad: str, key_msg: str, model_fn: Callable,
                         img_size: tuple[int, int] = (512, 512), K: int = 5) -> tuple[bytes, str]:

    weights, ref_pixels, error_map, message = ad_extraction(stego_image, key_ad, img_size, 8, K)
    message = msg_extraction(message, key_msg)
    original_image = cnn_feat_recovery(weights, ref_pixels, error_map, model_fn, img_size, K)

    return original_image.numpy(), message
