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