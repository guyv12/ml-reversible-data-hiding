import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(32*5*5, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 32*5*5),
            nn.ReLU(),

            nn.Unflatten(1, (32*5*5)),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return reconstructed