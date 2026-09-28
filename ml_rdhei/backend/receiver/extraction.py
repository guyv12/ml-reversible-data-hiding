import math
import struct
import torch
from bitarray import bitarray
from backend.compressor.encryption import encrypt_data
from backend.predictor.predict import reference_mask


def ad_extraction(bitstream: bitarray, key: str, image_size: tuple[int, int], bpp: int = 8, k: int = 5) -> (torch.Tensor, torch.Tensor, torch.Tensor, bitarray):
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
    ref_pixels = huffman_decode(codebook_pixels, compressed_pixels, n_ref)
    error_map = huffman_decode(codebook_error, compressed_error, n - n_ref)

    # remove offset
    deltas = torch.cat([ref_pixels[:1], ref_pixels[1:] - 255])
    error_map = error_map - 255

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
    codebook_error, compressed_error, ad = huffman_extraction(
        ad, b_sym, header_length_error,
    )

    img1_error_map = huffman_decode(
        codebook_error, compressed_error, N - n_ref
    )

    # 2. Image2 kernel weights
    img2_kernel_weights, ad = weights_extraction(ad, k)

    # 3. Image2 compressed reference pixels
    b_sym = 9
    header_length_pixels = math.ceil(math.log2(n_ref * b_sym))
    codebook_pixels, compressed_pixels, ad = huffman_extraction(ad, b_sym, header_length_pixels)

    # 4. Image2 compressed error map
    header_length_error = math.ceil(math.log2((N - n_ref) * b_sym))
    codebook_error, compressed_error, ad = huffman_extraction(ad, b_sym, header_length_error)

    # Decode Huffman
    img2_ref_pixels = huffman_decode(codebook_pixels, compressed_pixels, n_ref)
    img2_error_map = huffman_decode(codebook_error, compressed_error, N - n_ref)

    # remove delta encoding
    deltas = torch.cat([img2_ref_pixels[:1], img2_ref_pixels[1:] - 255])
    error_map = img2_error_map - 255
    
    # remove offset
    img2_error_map = [e - 255 for e in img2_error_map]

    return img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, message


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

def weights_extraction(ad: bitarray, k: int) -> (torch.Tensor, bitarray):
    num_weights = k ** 2
    weights = torch.empty(num_weights, dtype=torch.float64)

    for i in range(num_weights):
        weight_bytes = ad[:64].tobytes() # Assume storing W as 64bit
        weights[i] = struct.unpack('>d', weight_bytes)[0]

    return weights, ad

def huffman_decode(codebook: dict[str, int], compressed_data: str, n: int) -> torch.Tensor:
    decoded = torch.empty(n, dtype=torch.int64)
    buffer = ""
    i = 0

    for bit in compressed_data:
        buffer += bit

        if buffer in codebook:
            symbol = codebook[buffer]
            decoded[i] =symbol
            buffer = ""
            i += 1

    return decoded

def delta_decoding(deltas: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(deltas, dim=0)

def msg_extraction(image, key):
    image = image[:len(image) // 8 * 8]
    message = encrypt_data(image, key)
    message = message.tobytes()
    decoded_msg = message.decode('utf-8').rstrip('\x00')

    return decoded_msg
