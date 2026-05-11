from ml_rdhei.receiver.extraction import ad_extraction


def receive(image, key_ad, key_msg, n_ref):

    weights, codebook_pixels, ref_pixels, codebook_error, error_map = ad_extraction(image, key_ad, n_ref)
