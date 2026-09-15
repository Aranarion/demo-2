"""
Generative Adversarial Network for the Preprocessed OASIS brain MRI dataset.
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
from torchvision.utils import make_grid


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class OASISSliceDataset(Dataset):
    """Real MRI slices only -- no labels/masks needed for an unconditional
    GAN. Images scaled to [-1, 1] to match the generator's tanh output."""

    def __init__(self, root_dir, image_size=64):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No PNGs found under {root_dir}")
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                       # [0, 1]
            transforms.Normalize([0.5], [0.5]),           # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]))


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Generator(nn.Module):
    """latent z (B, latent_dim) -> 1 x image_size x image_size.
    Upsampling uses nearest-neighbour Upsample + Conv2d rather than
    ConvTranspose2d. Transposed convolution upsamples via overlapping
    kernel windows that contribute unevenly to neighbouring output
    pixels; the network must precisely cancel this unevenness to produce
    flat regions (e.g. the black MRI background), and imperfect
    cancellation shows up as a fixed checkerboard/dashed pattern at the
    same spatial position across every generated sample -- exactly the
    fixed-position dotted artifact this replaces. Nearest-neighbour
    upsampling has no overlapping windows, so there's no such artifact
    to cancel in the first place.
    """

    def __init__(self, latent_dim=128, image_size=64, base_c=64):
        super().__init__()
        self.init_size = image_size // 16  # four upsampling stages -> /16
        self.base_c = base_c
        self.fc = nn.Linear(latent_dim, base_c * 8 * self.init_size * self.init_size)

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(True),
            )

        self.net = nn.Sequential(
            nn.BatchNorm2d(base_c * 8), nn.ReLU(True),
            up_block(base_c * 8, base_c * 4),
            up_block(base_c * 4, base_c * 2),
            up_block(base_c * 2, base_c),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_c, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.base_c * 8, self.init_size, self.init_size)
        return self.net(h)


class Critic(nn.Module):
    """1 x image_size x image_size -> scalar realness score (unbounded,
    no sigmoid -- this is a WGAN critic, not a classifier). InstanceNorm
    instead of BatchNorm (see theory note on why BatchNorm breaks the
    gradient penalty's per-sample independence assumption).
    """

    def __init__(self, image_size=64, base_c=64):
        super().__init__()
        final_size = image_size // 16
        self.net = nn.Sequential(
            nn.Conv2d(1, base_c, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_c, base_c * 2, 4, 2, 1), nn.InstanceNorm2d(base_c * 2, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_c * 2, base_c * 4, 4, 2, 1), nn.InstanceNorm2d(base_c * 4, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_c * 4, base_c * 8, 4, 2, 1), nn.InstanceNorm2d(base_c * 8, affine=True), nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(base_c * 8 * final_size * final_size, 1)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h).squeeze(1)


def gradient_penalty(critic, real, fake, device):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(x_hat)
    grads = torch.autograd.grad(
        outputs=scores, inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]
    grads = grads.view(B, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# --------------------------------------------------------------------------
# Diagnostics: cheap diversity check for mode collapse
# --------------------------------------------------------------------------
@torch.no_grad()
def diversity_score(samples):
    """Mean pairwise L2 distance between a batch of generated images,
    flattened. Near-zero => samples are near-identical => mode collapse.
    Not a rigorous metric (FID would be), but a fast, dependency-free
    per-checkpoint signal of whether diversity is holding up."""
    flat = samples.flatten(1)
    B = flat.size(0)
    dists = torch.cdist(flat, flat, p=2)
    mask = ~torch.eye(B, dtype=torch.bool, device=flat.device)
    return dists[mask].mean().item()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(generator, critic, loader, device, epochs, latent_dim, out_dir,
          n_critic=5, lambda_gp=10.0, lr=1e-4, sample_every=1):
    # betas=(0, 0.9) is the standard WGAN-GP choice (not the default
    # (0.9, 0.999)) -- high momentum destabilises the critic's estimate
    # of the Wasserstein distance since it lags behind the rapidly
    # changing gradient penalty landscape.
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    fixed_z = torch.randn(64, latent_dim, device=device)  # fixed noise: same seeds every
    g_losses, c_losses = [], []                           # epoch => visually comparable progress

    for epoch in range(1, epochs + 1):
        epoch_g_loss, epoch_c_loss, n_g_updates = 0.0, 0.0, 0
        for i, real in enumerate(loader):
            real = real.to(device, non_blocking=True)
            B = real.size(0)

            # ---- Critic update(s) ----
            for _ in range(n_critic):
                z = torch.randn(B, latent_dim, device=device)
                fake = generator(z).detach()
                opt_c.zero_grad(set_to_none=True)
                gp = gradient_penalty(critic, real, fake, device)
                c_loss = critic(fake).mean() - critic(real).mean() + lambda_gp * gp
                c_loss.backward()
                opt_c.step()
            epoch_c_loss += c_loss.item()

            # ---- Generator update ----
            z = torch.randn(B, latent_dim, device=device)
            fake = generator(z)
            opt_g.zero_grad(set_to_none=True)
            g_loss = -critic(fake).mean()
            g_loss.backward()
            opt_g.step()
            epoch_g_loss += g_loss.item()
            n_g_updates += 1

        g_losses.append(epoch_g_loss / n_g_updates)
        c_losses.append(epoch_c_loss / n_g_updates)
        print(f"Epoch {epoch:3d}/{epochs} | critic loss {c_losses[-1]:.4f} "
              f"| generator loss {g_losses[-1]:.4f}")

        if epoch % sample_every == 0 or epoch == epochs:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_z)
            div = diversity_score(samples)
            print(f"  -> diversity score (mean pairwise L2, higher = more diverse): {div:.3f}")
            save_sample_grid(samples, os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png"))
            generator.train()

    plot_losses(g_losses, c_losses, os.path.join(out_dir, "loss_curve.png"))
    return g_losses, c_losses


def save_sample_grid(samples, out_path, nrow=8):
    samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1] for display
    grid = make_grid(samples.cpu(), nrow=nrow)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_losses(g_losses, c_losses, out_path):
    """Critic loss approximates the negative Wasserstein distance and,
    unlike a vanilla GAN's discriminator accuracy, its trend is
    meaningfully informative: a critic loss that keeps decreasing (more
    negative) generally indicates the generator is still improving,
    rather than the loss being uninterpretable noise."""
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator loss")
    plt.plot(c_losses, label="Critic loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WGAN-GP training curves")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                         default="/home/groups/comp3710/OASIS/keras_png_slices_train")
    parser.add_argument("--image_size", type=int, default=64,
                         help="Start at 64 for faster/more stable training; try 128 once "
                              "hyperparameters are confirmed to work.")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_critic", type=int, default=3)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--sample_every", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="./gan_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = OASISSliceDataset(args.data_root, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                         drop_last=True, pin_memory=True, persistent_workers=True)

    generator = Generator(args.latent_dim, args.image_size).to(device)
    critic = Critic(args.image_size).to(device)

    train(generator, critic, loader, device, args.epochs, args.latent_dim, args.out_dir,
          n_critic=args.n_critic, lambda_gp=args.lambda_gp, lr=args.lr,
          sample_every=args.sample_every)

    torch.save(generator.state_dict(), os.path.join(args.out_dir, "generator.pt"))
    torch.save(critic.state_dict(), os.path.join(args.out_dir, "critic.pt"))

    generator.eval()
    with torch.no_grad():
        final_z = torch.randn(64, args.latent_dim, device=device)
        final_samples = generator(final_z)
    save_sample_grid(final_samples, os.path.join(args.out_dir, "final_samples.png"))
    print(f"Final diversity score: {diversity_score(final_samples):.3f}")


if __name__ == "__main__":
    main()
