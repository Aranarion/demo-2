# vae-oasis
COMP3710 Recognition Task 1: a convolutional VAE trained on 2D brain MRI slices, with the learned latent manifold visualised as a grid of decoded images.

## Files
**VAE.py** — full implementation: OASIS slice dataset loader, convolutional encoder/decoder, VAE loss (BCE-with-logits reconstruction closed-form Gaussian KL), training loop with KL annealing and mixed precision, and the manifold visualisation code.
**run_vae.sh** — SLURM batch script
**manifold.png** — the required manifold visualisation: a 20x20 grid of images, each decoded from an evenly-spaced point in the 2D latent space (z1, z2 ∈ [-3, 3]), tiled into a single image.
**vae_oasis.pt** — saved trained model weights.
**slurm-589688.out** — SLURM job log from the training run.

## Usage
sbatch run_vae.sh

## Argument	Default
--data_root	/home/groups/comp3710/OASIS/keras_png_slices_train
--test_root	/home/groups/comp3710/OASIS/keras_png_slices_test
--latent_dim	2
--epochs	30
--kl_weight	1.0
Notes

The manifold shows clear variation along z1 but little along z2 — one latent dimension is carrying most of the encoded information. Total KL divergence is nonzero, so this is partial rather than full posterior collapse.
