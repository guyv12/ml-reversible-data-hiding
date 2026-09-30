import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
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

            nn.Unflatten(1, (32, 5, 5)),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return reconstructed

def test_shapes():
    model = Autoencoder()
    x = torch.randn(8, 2, 5, 5)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    latent = model.encoder(x)
    print(f"Latent shape: {latent.shape}")

def test_training():
    model = Autoencoder()

    x = torch.rand(1, 2, 5, 5)
    target = x[:, 0:1, :, :]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range (1000):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

    with torch.no_grad():
        output = model(x)

    print(f"Target:\n{target[0, 0]}")
    print(f"\nReconstructed:\n{output[0, 0]}")

test_shapes()
test_training()
