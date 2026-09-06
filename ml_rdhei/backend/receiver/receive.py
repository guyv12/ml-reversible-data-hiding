from backend.receiver.extraction import ad_extraction, msg_extraction
from backend.predictor.predict import reference_mask
from backend.receiver.recovery import recovery

def receive(image, key_ad, key_msg, img_size: tuple[int, int] = (512, 512)):

    weights, ref_pixels, error_map, message = ad_extraction(image, key_ad, img_size)

    message = msg_extraction(message, key_msg)
    print(message)

    original_image = recovery(weights, ref_pixels, error_map, img_size)

    return original_image