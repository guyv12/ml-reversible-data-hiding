import math
import struct
import numpy as np

from bitarray import bitarray
from backend.exceptions import CorruptedDataError
from backend.compressor.encryption import encrypt_data
from backend.predictor.predict import reference_mask


def ad_extraction(bitstream: bitarray, key: str, image_size: tuple[int, int], bpp: int = 8, k: int = 5):
    H, W = image_size
    n = H * W
    n_ref = int(reference_mask(H, W).sum().item())
    
    # AD length
    length = math.ceil(math.log2(n * bpp))
    ad_length = bitstream[:length]
    ad_and_message = bitstream[length:]
    ad_length_int = int(ad_length.to01(), 2)
    ad = ad_and_message[:ad_length_int]
    message = ad_and_message[ad_length_int:]

    ad = encrypt_data(ad, key)  # decrypting

    # Kernel weights
    weights_float, ad = weights_extraction(ad, k)
    
    if not np.isfinite(weights_float).all():
        raise CorruptedDataError("Extracted weights are not finite")
    
    # Compressed reference pixels
    b_sym = 9
    header_length_pixels = math.ceil(math.log2(n_ref * b_sym))
    codebook_pixels, compressed_pixels, ad = huffman_extraction(ad, b_sym, header_length_pixels)
    # Compressed error map
    header_length_error = math.ceil(math.log2((n - n_ref) * b_sym))
    codebook_error, compressed_error, ad = huffman_extraction(ad, b_sym, header_length_error)

    if len(ad) > 0:
        raise CorruptedDataError(f"AD has {len(ad)} leftover bits after last section")

    # Decode Huffman
    ref_pixels = huffman_decode(codebook_pixels, compressed_pixels)
    error_map = huffman_decode(codebook_error, compressed_error)

    if len(ref_pixels) != n_ref or len(error_map) != n - n_ref:
        raise CorruptedDataError("Invalid number of extracted reference pixels or errors")

    # remove offset
    deltas = [ref_pixels[0]]
    for p in ref_pixels[1:]:
        deltas.append(p-255)
    error_map = [e - 255 for e in error_map]

    # remove delta encoding
    pixels = delta_decoding(deltas)

    return weights_float, pixels, error_map, message

def ad_dicom_extraction(bitstream: bitarray, key: str, image_size: tuple[int, int], bpp: int = 16, k: int = 5):
    H, W = image_size
    N = H * W
    n_ref = int(reference_mask(H, W).sum().item())
    
    # AD length
    length = math.ceil(math.log2(N * bpp))
    ad_length = bitstream[:length]
    ad_and_message = bitstream[length:]
    ad_length_int = int(ad_length.to01(), 2)
    ad = ad_and_message[:ad_length_int]
    message = ad_and_message[ad_length_int:]

    ad = encrypt_data(ad, key)  # decrypting

    # 1. Image1 error map
    b_sym = 4
    header_length_error = math.ceil(math.log2(N * b_sym))
    img1_codebook_error, img1_compressed_error, ad = huffman_extraction(
        ad, b_sym, header_length_error,
    )

    # 2. Image2 kernel weights
    img2_kernel_weights, ad = weights_extraction(ad, k)

    if not np.isfinite(img2_kernel_weights).all():
        raise CorruptedDataError("Extracted weights are not finite")

    # 3. Image2 compressed reference pixels
    b_sym = 9
    header_length_pixels = math.ceil(math.log2(n_ref * b_sym))
    img2_codebook_pixels, img2_compressed_pixels, ad = huffman_extraction(ad, b_sym, header_length_pixels)

    # 4. Image2 compressed error map
    header_length_error = math.ceil(math.log2((N - n_ref) * b_sym))
    img2_codebook_error, img2_compressed_error, ad = huffman_extraction(ad, b_sym, header_length_error)

    if len(ad) > 0:
        raise CorruptedDataError(f"AD has {len(ad)} leftover bits after last section")

    # Decode Huffman
    img1_error_map = huffman_decode(img1_codebook_error, img1_compressed_error)
    img2_ref_pixels = huffman_decode(img2_codebook_pixels, img2_codebook_pixels)
    img2_error_map = huffman_decode(img2_codebook_error, img2_compressed_error)

    if len(img2_ref_pixels) != n_ref or len(img2_error_map) != N - n_ref or len(img1_error_map) != N:
        raise CorruptedDataError("Invalid number of extracted reference pixels or errors")

    # remove delta encoding
    deltas = [img2_ref_pixels[0]]
    for p in img2_ref_pixels[1:]:
        deltas.append(p-255)
    img2_ref_pixels = delta_decoding(deltas)
    
    # remove offset
    img2_error_map = [e - 255 for e in img2_error_map]

    return img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, message


def huffman_extraction(ad: bitarray, b_sym: int, header_length: int):
    if len(ad) < header_length:
        raise CorruptedDataError(
            "AD too short for codebook header: "
            f"need {header_length} bits, {len(ad)} left"
        )

    header = ad[:header_length]
    header_int = int(header.to01(), 2)
    ad = ad[header_length:]

    if len(ad) < header_int:
        raise CorruptedDataError(
            "AD too short for codebook: "
            f"need {header_int} bits, {len(ad)} left"
        )

    codebook = ad[:header_int]
    ad = ad[header_int:]

    extracted_codebook: dict = {}

    while len(codebook) > 0:
        if len(codebook) < b_sym + 5: # do zmiany
            raise CorruptedDataError(
                "Codebook too short for value and code length: " + 
                f"need {b_sym + 5} bits, {len(codebook)} left" # do zmiany
            )

        value = codebook[:b_sym]
        value_int = int(value.to01(), 2)
        codebook = codebook[b_sym:]

        code_length = codebook[:5]  # do zmiany
        code_length_int = int(code_length.to01(), 2)
        codebook = codebook[5:]

        if len(codebook) < code_length_int:
            raise CorruptedDataError(
                "Codebook too short for code: "
                f"need {code_length_int} bits, {len(codebook)} left"
            )

        code = (codebook[:code_length_int]).to01()
        codebook = codebook[code_length_int:]

        extracted_codebook.update({code: value_int})
    
    if len(extracted_codebook) > 2 ** b_sym:
        raise CorruptedDataError(f"Codebook has {len(extracted_codebook)} entries, max {2 ** b_sym}")

    if len(ad) < header_length:
        raise CorruptedDataError(
            "AD too short for data header: "
            f"need {header_length} bits, {len(ad)} left"
        )

    header = ad[:header_length]
    header_int = int(header.to01(), 2)
    ad = ad[header_length:]

    if len(ad) < header_int:
        raise CorruptedDataError(
            "AD too short for data extraction: "
            f"need {header_length} bits, {len(ad)} left"
        )

    compressed_data = (ad[:header_int]).to01()
    ad = ad[header_int:]

    return extracted_codebook, compressed_data, ad

def weights_extraction(ad: bitarray, k: int):
    weights_bits = (k ** 2) * 64
    if len(ad) < weights_bits:
        raise CorruptedDataError(
            "AD too short for weigths extraction: "
            f"need {weights_bits} bits, {len(ad)} left"
        )

    weights_float = []
    for i in range(k ** 2):
        weight = ad[:64] # Assume storing W as 64bit
        weight_bytes = weight.tobytes()
        weight_float = struct.unpack('>d', weight_bytes)[0]
        weights_float.append(weight_float)
        ad = ad[64:]

    return weights_float, ad

def huffman_decode(codebook: dict[str, int], compressed_data: str):
    decoded = []
    buffer = ""

    for bit in compressed_data:
        buffer += bit

        if buffer in codebook:
            symbol = codebook[buffer]
            decoded.append(symbol)
            buffer = ""

    return decoded

def delta_decoding(deltas: list[int]):
    pixels = []
    current_pixel = deltas[0]
    pixels.append(current_pixel)

    for i in range(1, len(deltas)):
        current_pixel += deltas[i]
        pixels.append(current_pixel)

    return pixels

def msg_extraction(image, key):
    image = image[:len(image) // 8 * 8]
    message = encrypt_data(image, key)
    message = message.tobytes()
    padding_start = message.find(b'\x00')
    try:
        if padding_start != -1:
            padding = message[padding_start:]
            is_not_zeros = np.frombuffer(padding, dtype=np.uint8).any()
            if is_not_zeros:
                raise CorruptedDataError("Corrupted padding")

            decoded_msg = message[:padding_start].decode('utf-8')
        else:   
            decoded_msg = message.decode('utf-8')
    except UnicodeDecodeError as e:
        raise CorruptedDataError("Decrypted message is not valid UTF-8") from e
    
    return decoded_msg
