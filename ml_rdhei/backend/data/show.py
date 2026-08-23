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

def check_images(original: np.ndarray, reconstructed: np.ndarray):
    if np.equal(original, reconstructed).all():
        print("ZGODNOSC 100%")
    else:
        print("BŁĄD")
        b1 = original.view(np.uint8).ravel()
        b2 = reconstructed.view(np.uint8).ravel()

        xor_bytes = b1 ^ b2
        byte_errors = np.count_nonzero(xor_bytes)

        print(byte_errors)