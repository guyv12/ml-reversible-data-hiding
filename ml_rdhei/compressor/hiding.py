import numpy as np

from encryption import encrypt_data, generate_bitstream

def hider(ad, length, message, key) -> bytes:
    msg_bytes = message.encode('utf-8')
    padding_len = length - len(msg_bytes)

    if padding_len < 0:
        raise ValueError(f"Message is bigger than the image!")


    message = np.zeros(length, dtype=np.uint8)
    message[:len(msg_bytes)] = msg_bytes
    #padding = generate_bitstream(key, padding_len)
    #message = msg_bytes + padding

    encrypted_msg = encrypt_data(message, key)

    return ad + encrypted_msg
