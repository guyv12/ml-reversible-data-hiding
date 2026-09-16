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

        if len(original) != len(reconstructed):
            print(f"different length: {len(original)} vs {len(reconstructed)}")

        diffs = [abs(b1 - b2) for b1, b2 in zip(original, reconstructed) if b1 != b2]
        total = min(len(original), len(reconstructed))

        print(f"different bytes: {len(diffs)} / {total} ({100 * len(diffs) / total:.2f}%)")
        print(f"max abs diff: {max(diffs, default=0)}")
        print(f"off by one: {sum(1 for d in diffs if d == 1)}")