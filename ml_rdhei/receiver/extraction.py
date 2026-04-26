import math
from bitarray import bitarray

from ml_rdhei.compressor.encryption import encrypt_data

def ad_extraction(ad: bytes, n: int, bpp: int, key: str):
    header = math.ceil(math.log2(n * bpp))
    ba = bitarray()
    ba.frombytes(ad)

    ad_length = ba[:header]
    ba.remove(ad_length)
    ad_length_int = int(ad_length.to01(), 2)

    ad_rest = ba[:ad_length_int]
    ba.remove(ad_rest)

    #decrypted_ad = encrypt_data(ad_rest, key)
