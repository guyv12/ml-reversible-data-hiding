import numpy as np

def test_image_reconstruction(original: bytes, reconstructed: bytes):
    assert len(original) == len(reconstructed), (
        f"Images have different length: "
        f"original={len(original)}, reconstructed={len(reconstructed)}"
    )

    original_np = np.frombuffer(original, dtype=np.uint8)
    reconstructed_np = np.frombuffer(reconstructed, dtype=np.uint8)

    diff = original_np.astype(np.int16) - reconstructed_np.astype(np.int16)
    different = diff != 0
    error_count = np.count_nonzero(different)

    assert error_count == 0, (
        f"Reconstruction FAILED: "
        f"{error_count}/{len(original_np)} of pixels differ "
        f"({100 * error_count / len(original_np):.4f}%)"
    )

def test_error_statistics(original: bytes, reconstructed: bytes):
    original_np = np.frombuffer(original, dtype=np.uint8)
    reconstructed_np = np.frombuffer(reconstructed, dtype=np.uint8)

    diff = (original_np.astype(np.int16) - reconstructed_np.astype(np.int16))
    nonzero = diff[diff != 0]
    values, counts = np.unique(diff[diff != 0], return_counts=True)

    if len(nonzero) == 0:
        return

    print(f"\nNumber of pixel errors: {len(nonzero)}")
    print(f"Min error: {nonzero.min()}")
    print(f"Max error: {nonzero.max()}")
    print(f"Average error: {nonzero.mean():.4f}")
    print("Error distribution:")
    for value, count in zip(values, counts):
        print(f"{value:+3d}: {count}")

