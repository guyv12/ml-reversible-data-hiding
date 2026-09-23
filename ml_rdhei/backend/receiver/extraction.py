import math

import numpy as np
from bitarray import bitarray, decodetree
from backend.compressor.encryption import encrypt_data
from backend.predictor.predict import reference_mask


def ad_extraction(bitstream: bitarray, key: str, image_size: tuple[int, int], bpp: int = 8, k: int = 5):
    #ba = bitarray()
    #ba.frombytes(bitstream)

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

    # Compressed reference pixels
    b_sym = 9
    header_length_pixels = math.ceil(math.log2(n_ref * b_sym))
    codebook_pixels, compressed_pixels, ad = huffman_extraction(ad, b_sym, header_length_pixels)

    # Compressed error map
    header_length_error = math.ceil(math.log2((n - n_ref) * b_sym))
    codebook_error, compressed_error, ad = huffman_extraction(ad, b_sym, header_length_error)

    # Decode Huffman
    ref_pixels = huffman_decode(codebook_pixels, compressed_pixels)
    error_map = huffman_decode(codebook_error, compressed_error)

    # remove offset
    deltas = ref_pixels.astype(np.int32)
    deltas[1:] -= 255
    error_map = error_map.astype(np.int32) - 255

    # remove delta encoding
    pixels = delta_decoding(deltas)

    return weights_float, pixels, error_map, message


def _read_uint(bits: bitarray, pos: int, width: int) -> tuple[int, int]:
    return int(bits[pos:pos + width].to01(), 2), pos + width


def huffman_extraction(ad: bitarray, b_sym: int, header_length: int, b_code: int = 5):
    # Layout (see compressor): [len(codebook)][codebook][len(data)][data],
    # both length fields are header_length bits wide.
    pos = 0

    codebook_length, pos = _read_uint(ad, pos, header_length)
    codebook_end = pos + codebook_length

    extracted_codebook: dict[str, int] = {}
    while pos < codebook_end:
        value, pos = _read_uint(ad, pos, b_sym)
        code_length, pos = _read_uint(ad, pos, b_code)
        code = ad[pos:pos + code_length].to01()
        pos += code_length
        extracted_codebook[code] = value

    data_length, pos = _read_uint(ad, pos, header_length)
    compressed_data = ad[pos:pos + data_length]
    pos += data_length

    return extracted_codebook, compressed_data, ad[pos:]

def weights_extraction(ad: bitarray, k: int):
    n_bits = 64 * k ** 2
    weights_float = np.frombuffer(ad[:n_bits].tobytes(), dtype='>f8').astype(np.float64)

    return weights_float, ad[n_bits:]

def huffman_decode(codebook: dict[str, int], compressed_data: bitarray) -> np.ndarray:
    if len(compressed_data) == 0:
        return np.empty(0, dtype=np.int16)

    tree = decodetree({value: bitarray(code) for code, value in codebook.items()})
    return np.fromiter(compressed_data.decode(tree), dtype=np.int16)

def delta_decoding(deltas: np.ndarray) -> np.ndarray:
    return np.cumsum(deltas, dtype=np.int32)

def msg_extraction(image, key):
    # The sender embeds whole bytes only; trailing bits are zero padding that
    # would decrypt to a garbage byte and defeat the rstrip below.
    image = image[:len(image) // 8 * 8]
    message = encrypt_data(image, key)
    message = message.tobytes()
    # Strip the zero padding before decoding; in UTF-8 a 0x00 byte is only ever '\x00'
    decoded_msg = message.rstrip(b'\x00').decode('utf-8')

    return decoded_msg
