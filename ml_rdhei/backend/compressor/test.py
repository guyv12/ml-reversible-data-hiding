import torch

def test_weights_compression(before: torch.Tensor, after: torch.Tensor):
    diff = before - after
    different = diff != 0
    error_count =torch.count_nonzero(different).item()

    assert error_count == 0, (
        f"Kernel weights compression FAILED: "
        f"{error_count}/{before.numel} of weights differ "
        f"({100 * error_count / after.numel():.4f}%)")

def test_error_map_compression(before: torch.Tensor, after: torch.Tensor):
    diff = before - 255 - after
    different = diff != 0
    error_count = torch.count_nonzero(different).item()

    assert error_count == 0, (
        f"Error map compression FAILED: "
        f"{error_count}/{before.numel()} of fields differ "
        f"({100 * error_count / after.numel():.4f}%)")
