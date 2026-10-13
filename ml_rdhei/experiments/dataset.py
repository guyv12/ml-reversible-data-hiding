from torch.utils.data import Dataset
import torch

class PatchDataset(Dataset):
    def __init__(self, image: torch.Tensor, patch_size: int = 64, stride: int | None = None):
        super().__init__()
        self.image = image.to(torch.float32) / 255.0
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride

        H, W = self.image.shape

        self.coords = []

        for row in range(0, H - patch_size + 1, self.stride):
            for column in range(0, W - patch_size + 1, self.stride):
                self.coords.append((row, column))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        row, column = self.coords[index]

        patch = self.image[
            row: row + self.patch_size,
            column: column + self.patch_size
        ].clone()

        ref_mask = torch.zeros_like(patch, dtype=torch.bool)
        ref_mask[::2, ::2] = True
        target = patch.clone()

        input_patch = patch.clone()
        input_patch[~ref_mask] = 0.0

        input_tensor = torch.stack([input_patch, ref_mask.to(torch.float32)])

        return input_tensor, target, ref_mask
