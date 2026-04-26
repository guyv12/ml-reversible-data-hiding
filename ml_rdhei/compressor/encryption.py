import numpy as np
import math

def generate_bitstream(key, length):
    seed = sum(ord(c) for c in str(key))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, length, dtype=np.uint8)

def encrypt_data(data_bytes: bytes, key: str) -> bytes:
    random_bitstream = generate_bitstream(key, len(data_bytes))
    data = np.frombuffer(data_bytes, dtype=np.uint8)
    result = data ^ random_bitstream
    return result.tobytes()

def encrypt_ad(ad: bytes, n: int, bpp: int, key: str) -> bytes:
    header = int(math.ceil(math.log2(n * bpp)) / 8)
    ad_header = ad[:header]
    ad_rest = encrypt_data(ad[header:], key)
    return ad_header + ad_rest
