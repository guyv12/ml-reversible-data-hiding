import math
import struct

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
    deltas = [ref_pixels[0]]
    for p in ref_pixels[1:]:
        deltas.append(p-255)
    error_map = [e - 255 for e in error_map]

    # remove delta encoding
    pixels = delta_decoding(deltas)

    return weights_float, pixels, error_map, message

def _read_uint(bits: bitarray, pos: int, width: int) -> tuple[int, int]:
    return int(bits[pos:pos + width].to01(), 2), pos + width

def huffman_extraction(ad: bitarray, b_sym: int, header_length: int, b_code: int = 5):
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
    weights_float = []
    for i in range(k ** 2):
        weight = ad[:64]
        weight_bytes = weight.tobytes()
        weight_float = struct.unpack('>d', weight_bytes)[0]
        weights_float.append(weight_float)
        ad = ad[64:]

    return weights_float, ad

def huffman_decode(codebook: dict[str, int], compressed_data: bitarray):
    if len(compressed_data) == 0:
        return []

    tree = decodetree({value: bitarray(code) for code, value in codebook.items()})
    return list(compressed_data.decode(tree))

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
    decoded_msg = message.decode('utf-8').rstrip('\x00')

    return decoded_msg