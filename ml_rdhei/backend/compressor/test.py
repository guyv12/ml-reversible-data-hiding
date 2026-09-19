import torch
import numpy as np

def test_weights_compression(before: torch.Tensor, after: list[float]):
    before.flatten().tolist()
    before_np = np.array(before,dtype=np.float64)
    after_np = np.array(after, dtype=np.float64)

    diff = before_np - after_np
    different = diff != 0
    error_count =np.count_nonzero(different)

    assert error_count == 0, (
        f"Kernel weights compression FAILED: "
        f"{error_count}/{len(before_np)} of weights differ "
        f"({100 * error_count / len(after_np):.4f}%)")

def test_error_map_compression(before: torch.Tensor, after):
    before.flatten().tolist()
    before_np = np.array(before, dtype=np.int16)
    after_np = np.array(after, dtype=np.int16)

    diff = before_np - 255 - after_np
    different = diff != 0
    error_count = np.count_nonzero(different)

    assert error_count == 0, (
        f"Error map compression FAILED: "
        f"{error_count}/{len(before_np)} of fields differ "
        f"({100 * error_count / len(after_np):.4f}%)")