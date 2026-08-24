import numpy as np

def test_image_reconstruction(original: bytes, reconstructed: bytes):
    assert len(original) == len(reconstructed), (
        f"Róźna długość obrazów: "
        f"original={len(original)}, reconstructed={len(reconstructed)}"
    )

    original_np = np.frombuffer(original, dtype=np.uint8)
    reconstructed_np = np.frombuffer(reconstructed, dtype=np.uint8)

    diff = original_np.astype(np.int16) - reconstructed_np.astype(np.int16)
    different = diff != 0
    error_count = np.count_nonzero(different)

    assert error_count == 0, (
        f"Rekonstrukcja niepoprawna: "
        f"{error_count}/{len(original_np)} pikseli różni się "
        f"({100 * error_count / len(original_np):.4f}%)"
    )
