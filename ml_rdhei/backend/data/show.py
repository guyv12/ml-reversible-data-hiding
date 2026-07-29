import matplotlib.pyplot as plt
import numpy as np


def show_image(bytes, width=512, height=512):
    stego_array = np.frombuffer(bytes, dtype=np.uint8)

    image_2d = stego_array.reshape((height, width))

    plt.figure(figsize=(8, 8))
    plt.imshow(image_2d, cmap='gray')
    plt.title("Zaszyfrowany obraz z ukrytymi danymi")
    plt.axis('off')
    plt.show()

def check_images(original: bytes, reconstructed: bytes):
    if original == reconstructed:
        print("ZGODNOSC 100%")
    else:
        print("BŁĄD")
        byte_errors = 0
        for b1, b2 in zip(original, reconstructed):
            xor_byte = b1 ^ b2
            if xor_byte == 1: byte_errors += 1
        print(byte_errors)