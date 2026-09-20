from backend.receiver.extraction import *
from backend.receiver.recovery import recovery, dicom_recovery

def receive(image, key_ad, key_msg, img_size: tuple[int, int] = (512, 512)):

    weights, ref_pixels, error_map, message = ad_extraction(image, key_ad, img_size)

    message = msg_extraction(message, key_msg)
    print(message)

    original_image = recovery(weights, ref_pixels, error_map, img_size)

    return original_image

def receive_dicom(image, key_ad, key_msg, img_size: tuple[int, int] = (512, 512)):

    img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, message = ad_dicom_extraction(image, key_ad, img_size)
    
    message = msg_extraction(message, key_msg)
    print(message)
    
    original_image = dicom_recovery(img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, img_size)
    
    return original_image
    