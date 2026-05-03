from bitarray import bitarray

from .encryption import encrypt_data


def hider(ad: bytes, length: int, message: str, key: str) -> bytes:
    msg_bytes = bytes(message.encode('utf-8'))
    padding_len = length - len(msg_bytes)

    if padding_len < 0:
        raise ValueError(f"Message is bigger than the image!")


    msg = msg_bytes.ljust(length, b'\x00')
    msg_bits = bitarray()
    msg_bits.frombytes(msg)
    encrypted_msg = encrypt_data(msg_bits, key)

    return ad + encrypted_msg.tobytes()
