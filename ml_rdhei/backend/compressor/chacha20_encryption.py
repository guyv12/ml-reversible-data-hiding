import math
from bitarray import bitarray
from Crypto.Cipher import ChaCha20
from Crypto.Hash import SHA256
from Crypto.Random import get_random_bytes

NONCE_BITS = 64

def generate_bitstream(key: str, length: int, nonce: bytes) -> bitarray:
    key_hash = SHA256.new(key.encode()).digest()
    cipher = ChaCha20.new(key=key_hash, nonce=nonce)

    bits = bitarray()
    bits.frombytes(cipher.encrypt(bytes((length + 7) // 8)))
    return bits[:length]

def encrypt_data(data_bits: bitarray, key: str) -> bitarray:
    nonce = get_random_bytes(NONCE_BITS // 8)
    nonce_bits = bitarray()
    nonce_bits.frombytes(nonce)
    return nonce_bits + (data_bits ^ generate_bitstream(key, len(data_bits), nonce))

def decrypt_data(data_bits: bitarray, key: str) -> bitarray:
    nonce = data_bits[:NONCE_BITS].tobytes()
    ciphertext = data_bits[NONCE_BITS:]
    return ciphertext ^ generate_bitstream(key, len(ciphertext), nonce)

def encrypt_ad(ad: bitarray, n: int, bpp: int, key: str) -> bitarray:
    header = int(math.ceil(math.log2(n * bpp)))
    ad_header = ad[:header]
    ad_rest = encrypt_data(ad[header:], key)
    return ad_header + ad_rest
