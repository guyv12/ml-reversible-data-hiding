import math
import struct

import torch
from bitarray import bitarray
from ml_rdhei.compressor.encryption import encrypt_data


def ad_extraction(bitstream: bytes, key: str, n_ref: int, n: int = 512 * 512, bpp: int = 8, k: int = 5):
    ba = bitarray()
    ba.frombytes(bitstream)

    # AD length
    length = math.ceil(math.log2(n * bpp))
    ad_length = ba[:length]
    ad_and_message = ba[length:]
    ad_length_int = int(ad_length.to01(), 2)
    ad = ad_and_message[:ad_length_int]

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

    return weights_float, pixels, error_map


def huffman_extraction(ad: bitarray, b_sym: int, header_length: int):
    header = ad[:header_length]
    header_int = int(header.to01(), 2)
    ad = ad[header_length:]
    codebook = ad[:header_int]
    ad = ad[header_int:]

    extracted_codebook: dict = {}

    while len(codebook) > 0:
        value = codebook[:b_sym]
        value_int = int(value.to01(), 2)
        codebook = codebook[b_sym:]

        code_length = codebook[:5]  # do zmiany
        code_length_int = int(code_length.to01(), 2)
        codebook = codebook[5:]

        code = (codebook[:code_length_int]).to01()
        codebook = codebook[code_length_int:]

        extracted_codebook.update({code: value_int})

    header = ad[:header_length]
    header_int = int(header.to01(), 2)
    ad = ad[header_length:]
    compressed_data = (ad[:header_int]).to01()
    ad = ad[header_int:]

    return extracted_codebook, compressed_data, ad

def weights_extraction(ad: bitarray, k: int):
    weights_float = []
    for i in range(k ** 2):
        weight = ad[:64]
        weight_bytes = weight.tobytes()
        weight_float = struct.unpack('>d', weight_bytes)[0]
        weights_float.append(weight_float)
        ad = ad[64:]

    return weights_float, ad

def huffman_decode(codebook: [(str, int)], compressed_data: str):
    decoded = []
    buffer = ""

    for bit in compressed_data:
        buffer += bit

        if buffer in codebook:
            symbol = codebook[buffer]
            decoded.append(symbol)
            buffer = ""

    return decoded

def delta_decoding(deltas: [int]):
    pixels = []
    current_pixel = deltas[0]
    pixels.append(current_pixel)

    for i in range(1, len(deltas)):
        current_pixel += deltas[i]
        pixels.append(current_pixel)

    return pixels
