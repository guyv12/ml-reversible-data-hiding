from bitarray import bitarray

from backend.receiver.extraction import *
from backend.receiver.recovery import recovery, dicom_recovery


def receive(stego_image: bitarray, key_ad: str, key_msg: str, img_size: tuple[int, int] = (512, 512)) -> tuple[bytes, str]:

    weights, ref_pixels, error_map, message = ad_extraction(stego_image, key_ad, img_size)

    message = msg_extraction(message, key_msg)

    original_image = recovery(weights, ref_pixels, error_map, img_size)

    return original_image, message

def receive_dicom(stego_image: bitarray, key_ad: str, key_msg: str, img_size: tuple[int, int] = (512, 512)) -> tuple[ndarray, str]:

    img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, message = ad_dicom_extraction(stego_image, key_ad, img_size)
    
    message = msg_extraction(message, key_msg)
    
    original_image = dicom_recovery(img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, img_size)
    
    return original_image, message
