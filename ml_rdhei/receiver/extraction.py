import math
import struct

from bitarray import bitarray

from ml_rdhei.compressor.encryption import encrypt_data

def ad_extraction(bitstream: bytes, key: str, n_ref : int, n: int = 512*512, bpp: int = 8, k: int = 5):
    ba = bitarray()
    ba.frombytes(bitstream)

    # AD length
    length = math.ceil(math.log2(n * bpp))
    ad_length = ba[:length]
    ad_and_message = ba[length:]
    ad_length_int = int(ad_length.to01(), 2)
    ad = ad_and_message[:ad_length_int]

    ad = encrypt_data(ad, key) # decrypting

    # Kernel weights
    weights_float = []
    for i in range(k**2):
        weight = ad[:64]
        weight_bytes = weight.tobytes()
        weight_float = struct.unpack('>d', weight_bytes)[0]
        weights_float.append(weight_float)
        #print(weights_float[i])
        ad = ad[64:]


    # Compressed reference pixels
    b_sym = 9
    header_length = math.ceil(math.log2(n_ref * b_sym))
    header = ad[:header_length]
    header_int = int(header.to01(), 2)
    ad = ad[header_length:]
