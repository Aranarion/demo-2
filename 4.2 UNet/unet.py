"""
U-Net segmentation of the Preprocessed OASIS brain MRI dataset.

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
class OASISSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=128, class_values=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(self.image_paths) == len(self.mask_paths), (
            f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)} "
            f"-- check {image_dir} and {mask_dir} correspond.")
        assert len(self.image_paths) > 0, f"No PNGs found under {image_dir}"

        self.image_size = image_size
        self.img_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        if class_values is None:
            class_values = self._discover_class_values()
        self.class_values = class_values          # sorted list of raw pixel values
        self.num_classes = len(class_values)

    def _discover_class_values(self, sample_n=50):
        """Scan a sample of masks to find the full set of raw label
        intensities present, so the same mapping is used consistently
        across train/val/test (pass class_values explicitly for val/test
        so they share the train set's mapping)."""
        values = set()
        for p in self.mask_paths[:sample_n]:
            arr = np.array(Image.open(p).convert("L"))
            values.update(np.unique(arr).tolist())
        return sorted(values)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.img_transform(Image.open(self.image_paths[idx]))

        mask = Image.open(self.mask_paths[idx]).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST)  # NEAREST: never blend labels
        mask = np.array(mask)
        # Map raw intensities -> contiguous class indices via lookup.
        class_idx = np.searchsorted(self.class_values, mask)
        class_idx = torch.from_numpy(class_idx).long()

        return img, class_idx


# --------------------------------------------------------------------------
# Model: standard U-Net
# --------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two 3x3 conv-BN-ReLU layers, the basic feature-extraction block
    used at every U-Net resolution level (both encoder and decoder)."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder step: max-pool halves resolution, then DoubleConv doubles
    channel depth to build more abstract features at the coarser scale."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Decoder step: transposed conv doubles resolution, the result is
    concatenated with the matching-resolution encoder feature map (the
    skip connection -- see theory note above), then DoubleConv fuses the
    combined (high-level + spatially-precise) information."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c, out_c)  # in_c because concat doubles channels back up

    def forward(self, x, skip):
        x = self.up(x)
        # Pad in case of odd input sizes causing a 1px mismatch after pooling/upsampling.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_c=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)  # bottleneck
        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)
        self.outc = nn.Conv2d(base_c, num_classes, kernel_size=1)  # per-class logits

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)  # logits, shape (B, num_classes, H, W)


# --------------------------------------------------------------------------
# Loss and metric
# --------------------------------------------------------------------------
def dice_loss(logits, target_idx, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target_idx, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_onehot, dims)
    union = torch.sum(probs, dims) + torch.sum(target_onehot, dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1 - dice_per_class.mean()


def combined_loss(logits, target_idx, num_classes):
    ce = F.cross_entropy(logits, target_idx)
    dl = dice_loss(logits, target_idx, num_classes)
    return ce + dl, ce, dl


@torch.no_grad()
def dsc_per_class(logits, target_idx, num_classes, eps=1e-6):
    """Hard-prediction DSC per class -- this is the number the task's
    ">0.9 for all labels" requirement is checked against, computed on
    argmax predictions (not soft probabilities)."""
    pred_idx = torch.argmax(logits, dim=1)
    pred_onehot = F.one_hot(pred_idx, num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target_idx, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(pred_onehot * target_onehot, dims)
    union = torch.sum(pred_onehot, dims) + torch.sum(target_onehot, dims)
    return ((2 * intersection + eps) / (union + eps)).cpu().numpy()


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_dsc = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                 enabled=device.type == "cuda"):
                logits = model(x)
            all_dsc.append(dsc_per_class(logits, y, num_classes))
    return np.mean(all_dsc, axis=0)  # per-class mean DSC across the whole loader


def train(model, train_loader, val_loader, optimiser, device, epochs, num_classes,
          ckpt_path, use_amp=True):
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    best_mean_dsc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, total_ce, total_dl = 0.0, 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                 enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss, ce, dl = combined_loss(logits, y, num_classes)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total += loss.item(); total_ce += ce.item(); total_dl += dl.item()
        n = len(train_loader)

        val_dsc = evaluate(model, val_loader, device, num_classes)
        mean_dsc = val_dsc.mean()
        dsc_str = ", ".join(f"c{c}={d:.3f}" for c, d in enumerate(val_dsc))
        print(f"Epoch {epoch:3d}/{epochs} | loss {total/n:.4f} (CE {total_ce/n:.4f} "
              f"Dice {total_dl/n:.4f}) | val DSC: {dsc_str} | mean {mean_dsc:.4f}")

        if mean_dsc > best_mean_dsc:
            best_mean_dsc = mean_dsc
            torch.save({"model_state": model.state_dict(),
                        "num_classes": num_classes,
                        "class_values": train_loader.dataset.class_values}, ckpt_path)
            print(f"  -> saved new best checkpoint (mean DSC {mean_dsc:.4f}) to {ckpt_path}")
    return best_mean_dsc


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def visualise_predictions(model, dataset, device, out_path, n=6, num_classes=4):
    """Saves one figure with n rows of (input MRI | ground truth mask |
    predicted mask), which is what justifies the DSC numbers visually --
    exactly what the task asks for."""
    model.eval()
    idxs = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    cmap = plt.get_cmap("tab10", num_classes)

    with torch.no_grad():
        for row, idx in enumerate(idxs):
            x, y = dataset[idx]
            logits = model(x.unsqueeze(0).to(device))
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

            axes[row, 0].imshow(x[0].numpy(), cmap="gray")
            axes[row, 0].set_title("Input MRI")
            axes[row, 1].imshow(y.numpy(), cmap=cmap, vmin=0, vmax=num_classes - 1)
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1)
            axes[row, 2].set_title("Prediction")
            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction visualisation to {out_path}")


# --------------------------------------------------------------------------
# Live-demo inference on a single slice
# --------------------------------------------------------------------------
def run_inference(ckpt_path, image_path, mask_path, image_size, device, out_path):
    """Standalone single-image inference for the live demo: loads the
    saved checkpoint, runs one MRI slice through the model, prints its
    per-class DSC against the ground-truth mask, and saves a comparison
    figure -- run this during the demo to show the model working."""
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt["num_classes"]
    class_values = ckpt["class_values"]

    model = UNet(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    x = img_transform(Image.open(image_path)).unsqueeze(0).to(device)

    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    mask = np.array(mask)
    y = torch.from_numpy(np.searchsorted(class_values, mask)).long().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
    dsc = dsc_per_class(logits, y, num_classes)
    print("Per-class DSC on this slice:")
    for c, d in enumerate(dsc):
        print(f"  class {c}: {d:.4f}")
    print(f"  mean: {dsc.mean():.4f}")

    pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
    cmap = plt.get_cmap("tab10", num_classes)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(x[0, 0].cpu().numpy(), cmap="gray"); axes[0].set_title("Input MRI")
    axes[1].imshow(y[0].cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved inference figure to {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    root = "/home/groups/comp3710/OASIS"
    parser.add_argument("--train_img", default=f"{root}/keras_png_slices_train")
    parser.add_argument("--train_seg", default=f"{root}/keras_png_slices_seg_train")
    parser.add_argument("--val_img", default=f"{root}/keras_png_slices_validate")
    parser.add_argument("--val_seg", default=f"{root}/keras_png_slices_seg_validate")
    parser.add_argument("--test_img", default=f"{root}/keras_png_slices_test")
    parser.add_argument("--test_seg", default=f"{root}/keras_png_slices_seg_test")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="./unet_outputs")
    parser.add_argument("--no_amp", action="store_true")
    # Live-demo inference mode: skip training, run one image through a saved checkpoint.
    parser.add_argument("--infer_image", type=str, default=None,
                         help="Path to a single MRI slice for live-demo inference")
    parser.add_argument("--infer_mask", type=str, default=None,
                         help="Path to that slice's ground-truth mask (for DSC reporting)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = os.path.join(args.out_dir, "unet_oasis.pt")

    if args.infer_image is not None:
        run_inference(ckpt_path, args.infer_image, args.infer_mask, args.image_size, device,
                      os.path.join(args.out_dir, "inference_demo.png"))
        return

    train_ds = OASISSegDataset(args.train_img, args.train_seg, args.image_size)
    val_ds = OASISSegDataset(args.val_img, args.val_seg, args.image_size,
                              class_values=train_ds.class_values)
    test_ds = OASISSegDataset(args.test_img, args.test_seg, args.image_size,
                               class_values=train_ds.class_values)
    num_classes = train_ds.num_classes
    print(f"Discovered {num_classes} classes with raw values {train_ds.class_values}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = UNet(num_classes=num_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, val_loader, optimiser, device, args.epochs, num_classes,
          ckpt_path, use_amp=not args.no_amp)

    # Reload best checkpoint (by val mean DSC) before final test-set report.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_dsc = evaluate(model, test_loader, device, num_classes)
    print("Final TEST set per-class DSC:")
    for c, d in enumerate(test_dsc):
        print(f"  class {c}: {d:.4f}")
    print(f"  mean: {test_dsc.mean():.4f}")

    visualise_predictions(model, test_ds, device,
                           os.path.join(args.out_dir, "segmentation_results.png"),
                           num_classes=num_classes)


if __name__ == "__main__":
    main()
