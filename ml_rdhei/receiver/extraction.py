import math
import struct

from bitarray import bitarray
from torch.distributed.tensor import empty

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
    codebook = ad[:header_int]
    ad = ad[header_int:]

    extracted_codebook : [(int, str)] = []

    while len(codebook) > 0:
        value = codebook[:b_sym]
        value_int = int(value.to01(), 2)
        codebook = codebook[b_sym:]

        code_length = codebook[:5] # do zmiany
        code_length_int = int(code_length.to01(), 2)
        codebook = codebook[5:]

        code = (codebook[:code_length_int]).to01()
        codebook = codebook[code_length_int:]

        extracted_codebook.append((value_int, code))

    header = ad[:header_length]
    header_int = int(header.to01(), 2)
    ad = ad[header_length:]
    compressed_pixels = ad[:header_int]
