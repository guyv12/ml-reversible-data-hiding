import math

import torch
from .huffman import build_huffman_tree, get_huffman_codes, delta_encode

# here compression logic
def __tensor_to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().cpu().numpy().tobytes()

def huffman_codebook_to_bytes(huffman_codes, b_sym):
    codebook = ""

    L_max = max(len(code) for code in huffman_codes.values())
    b_code = math.ceil(math.log2(L_max))

    for symbol, code in huffman_codes.items():
        # symbol value
        codebook += format(int(symbol), f'0{b_sym}b')

        # code length
        codebook += format(len(code), f'0{b_code}b')

        # huffman code itself
        codebook += code

    return codebook


def bits_to_bytes(bits):
    padding = (8 - len(bits) % 8) % 8
    bits += "0" * padding

    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))

def compress_ad_classic(kernel_weights: torch.Tensor, ref_pixels: torch.Tensor, error_map: torch.Tensor, batch: torch.Tensor) -> bytes:

    # 2 delta-huffman compress (reference pixels)
    encoded_ref_pixels = delta_encode(ref_pixels)
    pixels_list = encoded_ref_pixels.tolist()

    pixels_tree = build_huffman_tree(pixels_list)
    pixels_codes = get_huffman_codes(pixels_tree)


    # 3 huffman compress (error map)
    error_map_ints = torch.round(error_map).to(torch.int16)
    error_map_ints += 255 # offset
    error_list = error_map_ints.flatten().tolist()

    error_tree = build_huffman_tree(error_list)
    error_codes = get_huffman_codes(error_tree)


    # Auxiliary Data formulation
    #ad = bytearray()
    ad = ""

    # AD length
    _, H, W = batch.shape
    N = H * W
    header_width = math.ceil(math.log2(N * 8))

    #Kernel weights
    weights_bytes = __tensor_to_bytes(kernel_weights)
    ad += "".join(f"{b:08b}" for b in weights_bytes)

    # Compressed reference pixels
    n_ref = len(ref_pixels)
    b_sym = 9  # range [0, 510] so 9 bits are required
    width = math.ceil(math.log2(n_ref * b_sym))
    codebook = huffman_codebook_to_bytes(pixels_codes, b_sym)
    compressed_data = "".join([pixels_codes[val] for val in pixels_list])

    ad += format(len(codebook), f'0{width}b')
    ad += codebook
    ad += format(len(compressed_data), f'0{width}b')
    ad += compressed_data

    # Compressed error map
    n_non_ref = N - n_ref
    # b_sym stays the same
    width = math.ceil(math.log2(n_non_ref * b_sym))
    codebook = huffman_codebook_to_bytes(error_codes, b_sym)
    compressed_data = "".join([error_codes[val] for val in error_list])

    ad += format(len(codebook), f'0{width}b')
    ad += codebook
    ad += format(len(compressed_data), f'0{width}b')
    ad += compressed_data

    # add len(ad) at the beggining
    ad = format(len(ad), f'0{header_width}b') + ad

    # change bits string to bytes
    ad_bytes = bits_to_bytes(ad)
    return ad_bytes