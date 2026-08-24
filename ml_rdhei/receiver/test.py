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


def test_reconstruction_region(original: bytes, reconstructed: bytes):
    original_np = np.frombuffer(original, dtype=np.uint8).reshape(512,512)
    reconstructed_np = np.frombuffer(reconstructed, dtype=np.uint8).reshape(512,512)

    diff = (original_np.astype(np.int16) - reconstructed_np.astype(np.int16))

    region = diff[:16, :16]

    print("\nRegion 16x16:")
    for row in region:
        print(" ".join(f"{value:+3d}" for value in row))

def test_reference_pixels(original, reconstructed):
    original_np = np.frombuffer(original, dtype=np.uint8).reshape(512, 512)
    reconstructed_np = np.frombuffer(reconstructed, dtype=np.uint8).reshape(512, 512)

    reference_mask = np.zeros((512, 512), dtype=bool)
    reference_mask[::2, ::2] = True

    errors = original_np[reference_mask] != reconstructed_np[reference_mask]
    error_count = np.count_nonzero(errors)

    assert error_count == 0, (
        f"Reference pixels errors: {error_count}"
    )