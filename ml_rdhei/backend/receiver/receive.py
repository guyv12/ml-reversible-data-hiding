from backend.receiver.extraction import ad_extraction, msg_extraction
from backend.receiver.recovery import recovery


def receive(image, key_ad, key_msg, n_ref):

    weights, ref_pixels, error_map, message = ad_extraction(image, key_ad, n_ref)

    message = msg_extraction(message, key_msg)
    print(message)

    original_image = recovery(weights, ref_pixels, error_map)

    return original_image