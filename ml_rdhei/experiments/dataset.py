from torch.utils.data import Dataset
import torch

class PatchDataset(Dataset):
    def __init__(self, image, patch_size=64):
        super().__init__()
        self.image = image.to(torch.float32) / 255.0
        self.patch_size = patch_size

        H, W = self.image.shape

        self.coords = []

        for y in range(0, W - patch_size + 1, patch_size):
            for x in range(0, H - patch_size + 1, patch_size):
                self.coords.append((y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        column, row = self.coords[index]

        patch = self.image[
            column: column + self.patch_size,
            row: row + self.patch_size
        ].clone()

        ref_mask = torch.zeros_like(patch, dtype=torch.bool)
        ref_mask[::2, ::2] = True
        target = patch.clone()

        input_patch = patch.clone()
        input_patch[~ref_mask] = 0.0

        input_tensor = torch.stack([input_patch, ref_mask.to(torch.float32)])

        return input_tensor, target, ref_mask
