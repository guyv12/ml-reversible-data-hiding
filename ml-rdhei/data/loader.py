import cv2


def get_image(filename: str) -> None:
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image '{filename}'")
        return

    cv2.imshow("image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
