import torch

# here compression logic
def __tensor_to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().cpu().numpy().tobytes()


def compress_ad_classic(kernel_weights: torch.Tensor, ref_pixels: torch.Tensor, error_map: torch.Tensor) -> bytes:
    ad = bytearray()

    # flatten & bytes
    ad.extend(__tensor_to_bytes(kernel_weights))

    # 2 huffman compress

    # 3 huffman compress

    # 4 add len(ad) at the beggining

    return bytes(ad)

def delta_huffman_encode(ref_pixels: torch.Tensor):
    ref_pixels = ref_pixels.to(torch.int16)

    deltas = torch.diff(ref_pixels)
    deltas += 255 # offset
    encoded_deltas = torch.cat((ref_pixels[0].unsqueeze(0), deltas))

    return encoded_deltas