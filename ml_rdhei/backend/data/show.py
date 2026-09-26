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
