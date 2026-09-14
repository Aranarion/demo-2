"""
Variational Autoencoder for the Preprocessed OASIS brain MRI dataset.

--------------------------------------------------------------------------
DATA
--------------------------------------------------------------------------
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class OASISSliceDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No PNGs found under {root_dir}")
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # scales to [0, 1], shape (1, H, W)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, image_size=128, latent_dim=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.feat_size = image_size // 16
        flat_dim = 256 * self.feat_size * self.feat_size
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, image_size=128, latent_dim=2):
        super().__init__()
        self.feat_size = image_size // 16
        flat_dim = 256 * self.feat_size * self.feat_size
        self.fc = nn.Linear(latent_dim, flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, self.feat_size, self.feat_size)
        return self.deconv(h)

class VAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(image_size, latent_dim)
        self.decoder = Decoder(image_size, latent_dim)

    def reparameterise(self, mu, logvar):
        # z = mu + sigma * eps, eps ~ N(0, I); logvar predicted (not sigma
        # directly) for numerical stability -- exp() keeps sigma^2 > 0
        # everywhere without needing a positivity-constrained layer.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        x_logits = self.decoder(z)  # raw logits, no sigmoid applied here
        return x_logits, mu, logvar

def vae_loss(x_logits, x, mu, logvar, kl_weight=1.0):
    recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + kl_weight * kl, recon, kl

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(model, loader, optimiser, device, epochs, kl_weight=1.0, anneal_epochs=None,
          use_amp=True):
    if anneal_epochs is None:
        anneal_epochs = max(1, epochs // 2)
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        current_kl_weight = kl_weight * min(1.0, epoch / anneal_epochs)
        total, total_recon, total_kl = 0.0, 0.0, 0.0
        for x in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                 enabled=use_amp and device.type == "cuda"):
                x_hat, mu, logvar = model(x)
                loss, recon, kl = vae_loss(x_hat, x, mu, logvar, current_kl_weight)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
        n = len(loader)
        history.append((total / n, total_recon / n, total_kl / n))
        print(f"Epoch {epoch:3d}/{epochs} | beta {current_kl_weight:.3f} "
              f"| ELBO loss {total/n:.2f} | recon {total_recon/n:.2f} | KL {total_kl/n:.2f}")
    return history


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def plot_manifold(model, device, out_path, n=20, latent_range=3.0, image_size=128):
    model.eval()
    grid_x = np.linspace(-latent_range, latent_range, n)
    grid_y = np.linspace(-latent_range, latent_range, n)
    canvas = np.zeros((image_size * n, image_size * n))
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
                img = torch.sigmoid(model.decoder(z)).cpu().numpy()[0, 0]
                canvas[i * image_size:(i + 1) * image_size,
                       j * image_size:(j + 1) * image_size] = img
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.title("VAE Manifold")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved manifold grid to {out_path}")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                         default="/home/groups/comp3710/OASIS/keras_png_slices_train",
                         help="e.g. /home/groups/comp3710/OASIS/keras_png_slices_train")
    parser.add_argument("--test_root", type=str,
                         default="/home/groups/comp3710/OASIS/keras_png_slices_test",
                         help="Optional separate test/validate directory for latent scatter plot")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=1.0,
                         help="Beta in beta-VAE; >1 trades reconstruction sharpness for a more"
                              " disentangled/regular latent space (more Gaussian-like manifold)")
    parser.add_argument("--anneal_epochs", type=int, default=None,
                         help="Epochs to linearly ramp KL weight 0 -> kl_weight (fixes posterior"
                              " collapse). Defaults to epochs // 2.")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--out_dir", type=str, default="./vae_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = OASISSliceDataset(args.data_root, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, drop_last=True)

    model = VAE(args.image_size, args.latent_dim).to(device).to(memory_format=torch.channels_last)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, optimiser, device, args.epochs, args.kl_weight,
          anneal_epochs=args.anneal_epochs, use_amp=not args.no_amp)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "vae_oasis.pt"))

    if args.latent_dim == 2:
        plot_manifold(model, device, os.path.join(args.out_dir, "manifold.png"),
                      image_size=args.image_size)

    scatter_root = args.test_root or args.data_root
    test_ds = OASISSliceDataset(scatter_root, args.image_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)


if __name__ == "__main__":
    main()
