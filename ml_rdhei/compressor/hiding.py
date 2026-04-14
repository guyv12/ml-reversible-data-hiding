import numpy as np

from .encryption import encrypt_data


def hider(ad: bytes, length: int, message: str, key: str) -> bytes:
    msg_bytes = bytes(message.encode('utf-8'))
    padding_len = length - len(msg_bytes)

    if padding_len < 0:
        raise ValueError(f"Message is bigger than the image!")

    message = np.zeros(length, dtype=np.uint8)
    message[:len(msg_bytes)] = np.frombuffer(msg_bytes, dtype=np.uint8)

    encrypted_msg = encrypt_data(message.tobytes(), key)

    return ad + encrypted_msg
