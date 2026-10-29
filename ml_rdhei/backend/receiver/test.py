import torch


def test_image_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor):
    assert original.shape == reconstructed.shape, (
        f"Images have different length: "
        f"original={original.shape}, reconstructed={reconstructed.shape}"
    )

    diff = original.to(torch.int16) - reconstructed.to(torch.int16)
    different = diff != 0
    error_count = torch.count_nonzero(different).item()

    assert error_count == 0, (
        f"Reconstruction FAILED: "
        f"{error_count}/{original.numel()} of pixels differ "
        f"({100 * error_count / original.numel():.4f}%)"
    )

def test_error_statistics(original: torch.Tensor, reconstructed: torch.Tensor):
    diff = (original.to(torch.int16) - reconstructed.to(torch.int16))
    nonzero = diff[diff != 0]
    values, counts = torch.unique(diff[diff != 0], return_counts=True)

    if len(nonzero) == 0:
        return

    print(f"\nNumber of pixel errors: {len(nonzero)}")
    print(f"Min error: {nonzero.min()}")
    print(f"Max error: {nonzero.max()}")
    print(f"Average error: {nonzero.mean():.4f}")
    print("Error distribution:")
    for value, count in zip(values, counts):
        print(f"{value:+3d}: {count}")


def test_reconstruction_region(original: torch.Tensor, reconstructed: torch.Tensor):
    diff = (original.to(torch.int16) - reconstructed.to(torch.int16))

    region = diff[:16, :16]

    print("\nRegion 16x16:")
    for row in region:
        print(" ".join(f"{value:+3d}" for value in row))

def test_reference_pixels(original: torch.Tensor, reconstructed: torch.Tensor):
    reference_mask = torch.zeros((512, 512), dtype=torch.bool)
    reference_mask[::2, ::2] = True

    errors = original[reference_mask] != reconstructed[reference_mask]
    error_count = torch.count_nonzero(errors)

    assert error_count == 0, (
        f"Reference pixels errors: {error_count}"
    )

def test_psnr(original: torch.Tensor, reconstructed: torch.Tensor):
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * torch.log10((255 ** 2) / mse)

    print(f"\nMSE: {mse:.4f}")
    print(f"PSNR: {psnr:.4f} dB")

    assert mse == 0, f"Reconstruction is not lossless - MSE={mse:.4f}"