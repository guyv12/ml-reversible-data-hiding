from bitarray import bitarray
from numpy import uint16

from backend.exceptions import *
from backend.receiver.extraction import *
from backend.receiver.recovery import recovery, dicom_recovery


def receive(stego_image: bitarray, key_ad: str, key_msg: str, img_size: tuple[int, int] = (512, 512)) -> tuple[ndarray, str]:
    try:
        weights, ref_pixels, error_map, message_bits = ad_extraction(stego_image, key_ad, img_size)
    except CorruptedDataError as e:
        raise InvalidImageKeyError("Invalid image or image decryption key") from e
    
    original_image = recovery(weights, ref_pixels, error_map, img_size)

    try:
        message = msg_extraction(message_bits, key_msg)
    except CorruptedDataError as e:
        raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image, message

def receive_dicom(stego_image: bitarray, key_ad: str, key_msg: str, img_size: tuple[int, int] = (512, 512)) -> tuple[ndarray, str]:
    try:
        img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, message_bits = ad_dicom_extraction(stego_image, key_ad, img_size)
    except CorruptedDataError as e:
        raise InvalidImageKeyError("Invalid image or image decryption key") from e

    original_image = dicom_recovery(img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, img_size)

    try:
        message = msg_extraction(message_bits, key_msg)
    except CorruptedDataError as e:
        raise InvalidMessageKeyError("Invalid message decryption key") from e

    return original_image.numpy().astype(uint16), message
