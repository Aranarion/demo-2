# Demo-2 

## Repository structure

```
demo-2/
├── 4.1 VAE/        # Convolutional VAE — latent manifold visualisation
│   ├── VAE.py
│   ├── run_vae.sh
│   ├── manifold.png
│   └── slurm-589688.out
├── 4.2 UNet/        # U-Net — 4-class semantic segmentation
│   ├── unet.py
│   ├── run_unet.sh
│   ├── unet_outputs/segmentation_results.png
│   └── slurm-589775.out
└── 4.3 GAN/          # WGAN-GP — unconditional MRI slice generation
    ├── GAN.py
    ├── run_GAN.sh
    ├── final_samples.png
    ├── loss_curve.png
    └── slurm-590780.out
```

All three tasks share the same dataset family (OASIS keras PNG slices) and a common project structure: a PyTorch `Dataset`, model definition(s), a `train()` loop with mixed-precision support, a `main()` with `argparse` CLI flags, and a `run_*.sh` SLURM script for GPU cluster submission.

---

## 4.1 — Variational Autoencoder (`4.1 VAE/`)

A convolutional VAE trained on 2D OASIS brain MRI slices, with the learned latent space visualised as a grid of decoded images.

**Architecture:** 4-layer strided-convolution encoder (32→64→128→256 channels) producing `mu`/`logvar` for a 2D latent space, mirrored by a transposed-convolution decoder. Trained with the standard VAE ELBO — binary cross-entropy reconstruction plus closed-form Gaussian KL divergence — using linear KL annealing (beta ramps 0 → `kl_weight` over the first half of training) to avoid posterior collapse, and mixed-precision (AMP) training.

**Outputs:**
- `manifold.png` — a 20×20 grid of images decoded from an evenly-spaced grid over `z ∈ [-3, 3]²`, giving a direct visual picture of the learned latent space.
- `vae_oasis.pt` — trained model weights.

**Result:** training converged to an ELBO of ~4186 (recon ~4180, KL ~6.8) after 20 epochs. The manifold shows clear anatomical variation along one latent axis but little along the other — one latent dimension dominates the encoding, i.e. partial (not full) posterior collapse, since total KL remains nonzero.

**Run:**
```bash
sbatch run_vae.sh
# or directly:
python3 VAE.py --epochs 20 --latent_dim 2 --kl_weight 1.0
```

| Argument | Default |
|---|---|
| `--data_root` | `/home/groups/comp3710/OASIS/keras_png_slices_train` |
| `--test_root` | `/home/groups/comp3710/OASIS/keras_png_slices_test` |
| `--latent_dim` | `2` |
| `--epochs` | `30` |
| `--kl_weight` | `1.0` (beta in beta-VAE) |
| `--anneal_epochs` | `epochs // 2` |

---

## 4.2 — U-Net (`4.2 UNet/`)

A standard U-Net trained for 4-class semantic segmentation of OASIS brain MRI slices (background + 3 tissue classes).

**Architecture:** classic encoder–decoder with skip connections — four downsampling stages (max-pool + double-conv, 64→128→256→512→1024 channels) and four upsampling stages (transposed conv + concatenated skip feature map + double-conv), with a final 1×1 conv producing per-class logits. Trained with a combined cross-entropy + soft Dice loss, and evaluated with hard-prediction Dice Similarity Coefficient (DSC) per class.

**Outputs:**
- `unet_outputs/segmentation_results.png` — input MRI / ground-truth mask / predicted mask comparison grid for a random sample of test slices.
- `unet.py` also supports a **live single-image inference mode** (`--infer_image` / `--infer_mask`) that loads a saved checkpoint, runs one slice, prints per-class DSC, and saves a comparison figure — useful for a quick demo without retraining.

**Result (30 epochs):** final **test-set DSC: class 0 = 0.999, class 1 = 0.945, class 2 = 0.954, class 3 = 0.975, mean = 0.968** — all four classes exceed the 0.9 DSC target.

**Run:**
```bash
sbatch run_unet.sh
# or directly:
python3 unet.py --epochs 30

# live-demo inference on a single slice + mask, using a saved checkpoint:
python3 unet.py --infer_image path/to/slice.png --infer_mask path/to/mask.png
```

Key arguments: `--train_img/--train_seg`, `--val_img/--val_seg`, `--test_img/--test_seg` (default to the `/home/groups/comp3710/OASIS/` cluster paths), `--image_size 128`, `--batch_size 16`, `--epochs 30`, `--lr 1e-3`.

---

## 4.3 — GAN (`4.3 GAN/`)

A Wasserstein GAN with gradient penalty (WGAN-GP), trained unconditionally to generate synthetic OASIS brain MRI slices.

**Architecture:**
- **Generator:** latent vector (default dim 128) → dense projection → four nearest-neighbour-upsample + conv blocks (base_c·8 → base_c·4 → base_c·2 → base_c) → final conv + tanh. Nearest-neighbour upsampling is used instead of transposed convolution specifically to avoid the fixed-position checkerboard/dashed artifacts that transposed-conv overlap produces.
- **Critic:** 4-layer strided-convolution network with InstanceNorm (not BatchNorm, to preserve the gradient penalty's per-sample independence assumption) and no final activation, since it estimates a Wasserstein distance rather than a real/fake probability.
- **Training:** standard WGAN-GP loop — `n_critic` critic updates per generator update, gradient penalty (`lambda_gp=10`) enforcing the 1-Lipschitz constraint, Adam with `betas=(0, 0.9)`. A lightweight **diversity score** (mean pairwise L2 distance between generated samples) is logged every few epochs as a cheap, dependency-free proxy for detecting mode collapse.

**Outputs:**
- `final_samples.png` — an 8×8 grid of generated MRI slices from the final generator.
- `loss_curve.png` — generator/critic loss curves over training.
- `generator.pt` / `critic.pt` — trained weights.

**Result (90 epochs):** final critic loss ≈ −3.27, generator loss ≈ −17.96, final diversity score ≈ 13.28 (broadly stable/increasing across training) — no evidence of mode collapse.

**Run:**
```bash
sbatch run_GAN.sh
# or directly:
python3 GAN.py --epochs 90
```

| Argument | Default |
|---|---|
| `--data_root` | `/home/groups/comp3710/OASIS/keras_png_slices_train` |
| `--image_size` | `64` (start small/stable; try 128 once hyperparameters are confirmed) |
| `--latent_dim` | `128` |
| `--batch_size` | `128` |
| `--epochs` | `90` |
| `--n_critic` | `3` |
| `--lambda_gp` | `10.0` |

---

## Environment

All three scripts expect a `conda` environment named `torch` with PyTorch (+CUDA), `torchvision`, `numpy`, `matplotlib`, and `PIL` installed, and are set up to submit as SLURM jobs (`--gres=gpu:1`) on a cluster with the OASIS dataset preprocessed under `/home/groups/comp3710/OASIS/`. Each script also runs standalone on CPU (falls back automatically via `torch.cuda.is_available()`), just without mixed precision.

```bash
conda activate torch
cd "4.1 VAE"   && python3 VAE.py --epochs 5      # quick smoke test
cd "4.2 UNet"  && python3 unet.py --epochs 5
cd "4.3 GAN"   && python3 GAN.py --epochs 5
```
