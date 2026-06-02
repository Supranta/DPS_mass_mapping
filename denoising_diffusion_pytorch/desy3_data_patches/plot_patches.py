"""
Visualize a DES-Y3 shear patch created by create_desy3_patches.py.

Usage
-----
  python plot_patches.py <patch_index>
  python plot_patches.py <patch_index> --out figures/

Each figure shows all 4 tomographic bins × {g1, g2, sigma_noise, survey_mask}.
"""

import argparse
import os
import sys

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = ('/home2/supranta/PosteriorSampling/denoising_diffusion_pytorch'
            '/denoising_diffusion_pytorch/data/des_y3_metacal_big')

TOMO_LABELS = ['bin 0', 'bin 1', 'bin 2', 'bin 3']
MAP_LABELS  = ['g1', 'g2', 'sigma_noise', 'survey_mask']
CMAPS       = ['RdBu_r', 'RdBu_r', 'viridis', 'gray']


def load_patch(patch_idx, data_dir):
    path = os.path.join(data_dir, f'patch_{patch_idx}.h5')
    if not os.path.exists(path):
        sys.exit(f'File not found: {path}')
    with h5.File(path, 'r') as f:
        noisy_shear  = f['noisy_shear'][:]   # (4, 2, npix, npix)
        sigma_noise  = f['sigma_noise'][:]   # (4,    npix, npix)
        survey_mask  = f['survey_mask'][:]   # (4,    npix, npix)
        patch_center = f['patch_center'][:]  # (2,)
    return noisy_shear, sigma_noise, survey_mask, patch_center


def plot_patch(patch_idx, data_dir, out_dir=None):
    noisy_shear, sigma_noise, survey_mask, centre = load_patch(patch_idx, data_dir)

    n_bins = noisy_shear.shape[0]
    n_maps = len(MAP_LABELS)

    fig, axes = plt.subplots(n_bins, n_maps, figsize=(4 * n_maps, 3.5 * n_bins))
    fig.suptitle(
        f'Patch {patch_idx}  —  centre RA={centre[0]:.2f}°  Dec={centre[1]:.2f}°',
        fontsize=13, y=1.01,
    )

    for b in range(n_bins):
        maps = [
            noisy_shear[b, 0],   # g1  (already sign-flipped)
            noisy_shear[b, 1],   # g2
            sigma_noise[b],
            survey_mask[b],
        ]
        for m, (data, key, cmap) in enumerate(zip(maps, MAP_LABELS, CMAPS)):
            ax = axes[b, m]

            if key in ('g1', 'g2'):
                vmax = np.nanpercentile(np.abs(data[data != 0]), 99) if data.any() else 1.0
                vmin = -vmax
            else:
                vmin, vmax = None, None

            im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(f'{TOMO_LABELS[b]} — {key}', fontsize=9)
            ax.set_xlabel('x pixel')
            ax.set_ylabel('y pixel')

    plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'patch_{patch_idx}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to {out_path}')
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Plot a DES-Y3 shear patch.')
    parser.add_argument('patch_index', type=int, help='Patch index to visualize')
    parser.add_argument('--data-dir', default=DATA_DIR,
                        help='Directory containing patch HDF5 files')
    parser.add_argument('--out', default=None, metavar='DIR',
                        help='Save figure to this directory instead of displaying')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    plot_patch(args.patch_index, args.data_dir, out_dir=args.out)
