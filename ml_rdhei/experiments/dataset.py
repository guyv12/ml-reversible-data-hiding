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


    def __len__(self):
        return self.valid_height * self.valid_width

    def __getitem__(self, index):
        row = index // self.valid_width + self.padding
        column = index % self.valid_width + self.padding

        patch = self.image[
            row - self.padding : row + self.padding + 1,
            column - self.padding : column + self.padding + 1
        ].clone()

        target = patch[self.padding, self.padding].clone()

        patch[self.padding, self.padding] = 0.0
        mask = torch.zeros_like(patch)
        mask[self.padding, self.padding] = 1.0

        input_tensor = torch.stack([patch, mask])

        return input_tensor, target
