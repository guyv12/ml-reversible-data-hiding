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