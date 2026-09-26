import matplotlib.pyplot as plt
import numpy as np
import torch


def show_image(image: torch.Tensor, width=512, height=512):
    image.reshape(height, width)

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title("Oryginalny obraz")
    plt.axis('off')
    plt.show()


def show_bytes(image, width=512, height=512):
    stego_array = np.frombuffer(image, dtype=np.uint8)
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