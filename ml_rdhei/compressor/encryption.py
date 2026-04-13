import numpy as np

def generate_bitstream(key, length):
    seed = sum(ord(c) for c in str(key))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, length, dtype=np.uint8)

def encrypt_data(data_bytes, key) -> bytes:
    random_bitstream = generate_bitstream(key, len(data_bytes))
    data = np.frombuffer(data_bytes, dtype=np.uint8)
    result = data ^ random_bitstream
    return result
