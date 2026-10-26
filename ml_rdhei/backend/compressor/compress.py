import math
import torch
from bitarray import bitarray
from .huffman import build_huffman_tree, get_huffman_codes, delta_encode, huffman_codebook_to_bits
from experiments.compression import *

def __tensor_to_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    
    if t.is_floating_point():
        return t.numpy().astype('>f8').tobytes()
    else:
        return t.numpy().astype('>i8').tobytes()

def __bits_to_bytes(bits):
    padding = (8 - len(bits) % 8) % 8
    bits += "0" * padding

    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))

def __compress_kernel_weights(kernel_weights: torch.Tensor) -> str:
    weights_bytes = __tensor_to_bytes(kernel_weights)
    return "".join(f"{b:08b}" for b in weights_bytes)

def __compress_ref_pixels(ref_pixels: torch.Tensor) -> str:
    # delta-huffman compression
    encoded_ref_pixels = delta_encode(ref_pixels)
    pixels_list = encoded_ref_pixels.tolist()

    pixels_tree = build_huffman_tree(pixels_list)
    pixels_codes = get_huffman_codes(pixels_tree)


    n_ref = len(ref_pixels)
    b_sym = 9  # range [0, 510] so 9 bits are required
    width = math.ceil(math.log2(n_ref * b_sym))
    codebook = huffman_codebook_to_bits(pixels_codes, b_sym)
    compressed_data = "".join([pixels_codes[val] for val in pixels_list])

    print(len(codebook), len(compressed_data))

    return str(
        format(len(codebook), f'0{width}b')
        + codebook
        + format(len(compressed_data), f'0{width}b')
        + compressed_data
    )

DEFAULT_AUTOENCODER_PATH = "compression_autoencoder.pth"

def __latent_to_bits(latent: torch.Tensor) -> str:
    latent_bytes = latent.detach().contiguous().cpu().numpy().astype('>f2').tobytes()
    return "".join(f"{b:08b}" for b in latent_bytes)

def __compress_ref_pixels_autoencoder(ref_pixels: torch.Tensor,
                                      model_path: str = DEFAULT_AUTOENCODER_PATH) -> str:
    model = load_model(model_path)

    ref_img = ref_pixels.float() / 255.0

    with torch.no_grad():
        latent = model.encode(ref_img)

        latent_q = latent.to(torch.float16)
        prediction = model.decode(latent_q.float()).reshape(-1)

    prediction = prediction.round().clamp(0, 255).to(torch.int16)
    error_map = ref_pixels.to(torch.int16) - prediction  # [-255, 255]

    latent_bits = __latent_to_bits(latent_q.reshape(-1))
    latent_width = math.ceil(math.log2(ref_pixels.numel() * 8))  # latent can't exceed raw ref pixel size

    if len(latent_bits) >= 2 ** latent_width:
        raise ValueError("Latent vector is larger than raw reference pixels")

    return str(
        format(len(latent_bits), f'0{latent_width}b')
        + latent_bits
        + __compress_error_map(error_map, ref_pixels.numel(), 0, 9)  # range [0, 510] so 9 bits are required
    )

def __compress_error_map(error_map: torch.Tensor, N: int, n_ref: int, b_sym: int, add_offset: bool = True) -> str:
    # huffman compression (error map)
    if add_offset:
        error_map = error_map + 255 # offset

    error_map_list = error_map.flatten().tolist()

    error_map_tree = build_huffman_tree(error_map_list)
    error_map_codes = get_huffman_codes(error_map_tree)

    # Compressed error map
    n_non_ref = N - n_ref
    width = math.ceil(math.log2(n_non_ref * b_sym))
    codebook = huffman_codebook_to_bits(error_map_codes, b_sym)
    compressed_data = "".join([error_map_codes[val] for val in error_map_list])

    return str(
        format(len(codebook), f'0{width}b')
        + codebook
        + format(len(compressed_data), f'0{width}b')
        + compressed_data
    )

def compress_pgm_ad(img_size: tuple[int, int], 
                    kernel_weights: torch.Tensor, ref_pixels: torch.Tensor, error_map: torch.Tensor,
                    autoencoder: bool = False, autoencoder_path: str = DEFAULT_AUTOENCODER_PATH
                    ) -> bitarray:
    H, W = img_size
    N = H * W
    bpp = 8
    header_width = math.ceil(math.log2(N * bpp))

    ad = __compress_kernel_weights(kernel_weights)

    if autoencoder:
        ad += __compress_ref_pixels_autoencoder(ref_pixels, autoencoder_path)
    else:
        ad += __compress_ref_pixels(ref_pixels)

    ad += __compress_error_map(error_map, N, len(ref_pixels), 9) # range [0, 510] so 9 bits are required

    # add len(ad) at the beggining
    ad = format(len(ad), f'0{header_width}b') + ad

    # change bits string to bytes
    ad_bits = bitarray(ad)

    return ad_bits

def compress_dicom_ad(img_size: tuple[int, int], img1_error_map: torch.Tensor,
                      img2_kernel_weights: torch.Tensor, img2_ref_pixels: torch.Tensor, img2_error_map: torch.Tensor
                      ) -> bitarray:
    H, W = img_size
    N = H * W
    bpp = 16
    header_width = math.ceil(math.log2(N * bpp))

    ad = __compress_error_map(img1_error_map, N, 0, 4, add_offset=False) # range [0, 15] so 4 bits are required

    ad += __compress_kernel_weights(img2_kernel_weights)
    ad += __compress_ref_pixels(img2_ref_pixels)
    ad += __compress_error_map(img2_error_map, N, len(img2_ref_pixels), 9)

    # add len(ad) at the beggining
    ad = format(len(ad), f'0{header_width}b') + ad

    # change bits string to bytes
    ad_bits = bitarray(ad)

    return ad_bits

def compress_cnn_ad(img_size: tuple[int, int], ref_pixels: torch.Tensor, error_map: torch.Tensor) -> bitarray:
    H, W = img_size
    N = H * W
    bpp = 8
    header_width = math.ceil(math.log2(N * bpp))

    ad = __compress_ref_pixels(ref_pixels)
    ad += __compress_error_map(error_map, N, len(ref_pixels), 9)

    # add len(ad) at the beggining
    ad = format(len(ad), f'0{header_width}b') + ad

    # change bits string to bytes
    ad_bits = bitarray(ad)

    return ad_bits

def compress_cnn_dicom_ad(img_size: tuple[int, int], img1_error_map: torch.Tensor,
                          img2_ref_pixels: torch.Tensor, img2_error_map: torch.Tensor
                          ) -> bitarray:
    H, W = img_size
    N = H * W
    bpp = 16
    header_width = math.ceil(math.log2(N * bpp))

    ad = __compress_error_map(img1_error_map, N, 0, 4, add_offset=False) # range [0, 15] so 4 bits are required

    ad += __compress_ref_pixels(img2_ref_pixels)
    ad += __compress_error_map(img2_error_map, N, len(img2_ref_pixels), 9)

    # add len(ad) at the beggining
    ad = format(len(ad), f'0{header_width}b') + ad

    # change bits string to bytes
    ad_bits = bitarray(ad)

    return ad_bits
