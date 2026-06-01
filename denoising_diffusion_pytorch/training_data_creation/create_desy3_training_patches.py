import pandas as pd
import numpy as np
import healpy as hp
from tqdm import trange
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Create DES-Y3 training patches from a simulation.')
parser.add_argument('simnum', type=int, help='Simulation number (e.g. 1, 42, 200)')
args = parser.parse_args()

# --- Configuration ---
simnum   = args.simnum
sim_base = '/global/cfs/cdirs/m5099/GowerSt2/Fiducial'
simdir   = '%s/%d_big' % (sim_base, simnum)
npix     = 256
reso     = 2.0   # arcmin/pixel -> 512 arcmin patch
savedir  = '/pscratch/sd/s/ssarmabo/DPS_data/flat_patches/%d' % simnum

# --- Build kappa maps from particle snapshots ---
kappa_weights = np.load('/global/u1/s/ssarmabo/DPS_mass_mapping/denoising_diffusion_pytorch/des_y3_kappa_weights.npy').astype(np.float32)
nside = 4096
kappa_maps = np.zeros((4, 12 * nside**2), dtype=np.float32)
for i in trange(15, 100):
    particle_filename = simdir + '/particles_%d_4096.parquet' % (i + 1)
    df = pd.read_parquet(particle_filename)
    N_parts = np.array(df['0']).astype(np.float32)
    delta = N_parts / N_parts.mean() - 1.
    kappa_maps = kappa_maps + kappa_weights[:, i, np.newaxis] * delta[np.newaxis]


def generate_nonoverlapping_coordinates(npix, reso, overlap_buffer=0.0):
    """
    Generate (lon, lat) patch centers tiling the full sky with no overlap.

    Args:
        npix (int): Number of pixels per side.
        reso (float): Pixel scale in arcmin/pixel.
        overlap_buffer (float): Extra gap in degrees between patches (default 0).

    Returns:
        list of (lon, lat) tuples in degrees.
    """
    patch_size_deg = (npix * reso) / 60.0
    spacing = patch_size_deg + overlap_buffer

    coordinates = []
    lat_min = -90 + patch_size_deg / 2
    lat_max =  90 - patch_size_deg / 2
    n_lat = int((lat_max - lat_min) / spacing) + 1

    print("n_lat: %d" % n_lat)
    for i in range(n_lat):
        lat = lat_min + i * spacing
        cos_lat = np.cos(np.radians(lat))
        if cos_lat > 0.01:
            lon_spacing = spacing / cos_lat
            n_lon = int(360 / lon_spacing)
            for j in range(n_lon):
                lon = j * lon_spacing
                if lon >= 360:
                    lon -= 360
                coordinates.append((lon, lat))

    return coordinates


def extract_patch(hp_map, lon, lat, npix, reso):
    """
    Extract a gnomonic (flat) patch from a HEALPix map.

    Args:
        hp_map (np.ndarray): Full-sky HEALPix map (RING ordering).
        lon (float): Center longitude in degrees.
        lat (float): Center latitude in degrees.
        npix (int): Number of pixels per side.
        reso (float): Pixel scale in arcmin/pixel.

    Returns:
        np.ndarray: 2D array of shape (npix, npix).
    """
    patch = hp.visufunc.gnomview(
        hp_map, rot=(lon, lat, 0.), xsize=npix,
        reso=reso,
        return_projected_map=True,
        nest=False,
        no_plot=True,
    )
    return np.asarray(patch)


def extract_all_nonoverlapping_patches(kappa_maps, npix, reso):
    """
    Extract all valid non-overlapping patches from a set of kappa maps.

    Args:
        kappa_maps (np.ndarray): Shape (n_zbins, npix_hp).
        npix (int): Patch size in pixels.
        reso (float): Pixel scale in arcmin/pixel.

    Returns:
        patches (np.ndarray): Shape (N_valid, n_zbins, npix, npix).
        valid_coords (list of (lon, lat)): Centers of valid patches.
    """
    coordinates = generate_nonoverlapping_coordinates(npix, reso)

    patches      = []
    valid_coords = []

    for i in trange(len(coordinates)):
        lon, lat = coordinates[i]
        try:
            patch = np.array([
                extract_patch(kappa_maps[z], lon, lat, npix, reso)
                for z in range(kappa_maps.shape[0])
            ])
            if not np.any(np.isnan(patch)) and not np.any(patch == hp.UNSEEN):
                patches.append(patch)
                valid_coords.append((lon, lat))
        except Exception:
            continue

    return np.array(patches), valid_coords


# --- Extract patches ---
patches, valid_coords = extract_all_nonoverlapping_patches(kappa_maps, npix, reso)

# --- Save patches and metadata ---
os.makedirs(savedir, exist_ok=True)

metadata = []
for i in trange(len(patches)):
    np.save(os.path.join(savedir, '%d.npy' % i), patches[i].astype(np.float32))
    metadata.append({
        'patch_id':   i,
        'sim_id':     simnum,
        'center_lon': valid_coords[i][0],
        'center_lat': valid_coords[i][1],
        'pixel_scale_arcmin': reso,
        'npix':       npix,
        'patch_size_arcmin': npix * reso,
    })

with open(os.path.join(savedir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("Saved %d patches to %s" % (len(patches), savedir))
