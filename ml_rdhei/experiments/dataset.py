from torch.utils.data import Dataset
import torch

class PixelMaskDataset(Dataset):
    def __init__(self, image, patch_size=5):
        super().__init__()
        self.image = image
        self.patch_size = patch_size

        self.padding = patch_size // 2
        self.height = image.shape[0]
        self.width = image.shape[1]

        self.valid_height = self.height - 2 * self.padding
        self.valid_width = self.width - 2 * self.padding

        self.reference_mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        self.reference_mask[::2, ::2] = True

        self.coords = [
            (row, col)
            for row in range(self.padding, self.height - self.padding)
            for col in range(self.padding, self.width - self.padding)
            if not self.reference_mask[row, col]
        ]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        row, column = self.coords[index]

        patch = self.image[
            row - self.padding : row + self.padding + 1,
            column - self.padding : column + self.padding + 1
        ].clone()

        target = patch[self.padding, self.padding].clone()

        ref_window = torch.zeros_like(patch, dtype=torch.bool)
        for dy in range(-self.padding, self.padding + 1):
            for dx in range(-self.padding, self.padding + 1):
                wy = row + dy
                wx = column + dx
                if (wy % 2 == 0) and (wx % 2 == 0):
                    ref_window[dy + self.padding, dx + self.padding] = True

        input_patch = patch.clone()
        input_patch[~ref_window] = 0.0

        mask = torch.zeros_like(patch)
        mask[ref_window] = 1.0

        input_tensor = torch.stack([input_patch, mask])

        return input_tensor, target
