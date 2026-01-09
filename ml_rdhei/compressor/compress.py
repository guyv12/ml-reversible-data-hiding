import torch
from .huffman import build_huffman_tree, get_huffman_codes, delta_encode

# here compression logic
def __tensor_to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().cpu().numpy().tobytes()


def compress_ad_classic(kernel_weights: torch.Tensor, ref_pixels: torch.Tensor, error_map: torch.Tensor) -> bytes:
    ad = bytearray()

    # flatten & bytes
    ad.extend(__tensor_to_bytes(kernel_weights))

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

    # 4 add len(ad) at the beggining

    return bytes(ad)