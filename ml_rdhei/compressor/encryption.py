import hashlib
import numpy as np
import math

from bitarray import bitarray


def generate_bitstream(key, length):
    seed_hash = hashlib.sha256(key.encode()).digest()
    seed = int.from_bytes(seed_hash, byteorder='big')
    rng = np.random.default_rng(seed)

    number_of_bytes = (length + 7) // 8
    random_bytes = rng.bytes(number_of_bytes)

    bits = bitarray()
    bits.frombytes(random_bytes)
    return bits[:length]

def encrypt_data(data_bits: bitarray, key: str) -> bitarray:
    random_bitstream = generate_bitstream(key, len(data_bits))
    result = data_bits ^ random_bitstream
    return result

def encrypt_ad(ad: bytes, n: int, bpp: int, key: str) -> bytes:
    header = int(math.ceil(math.log2(n * bpp)))
    ad_bits = bitarray()
    ad_bits.frombytes(ad)
    ad_header = ad_bits[:header]
    ad_rest = encrypt_data(ad_bits[header:], key)
    return (ad_header + ad_rest).tobytes()
