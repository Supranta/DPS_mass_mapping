"""
Create pixelized shear map patches from DES-Y3 metacal data in DPS-ready format.

Patch geometry (all dimensions in arcmin unless noted):
  Full patch:   512 arcmin  = 256 px × 2 arcmin/px
  Buffer:        30 arcmin  = 15 px  (each side, masked in survey_mask)
  Usable area:  452 arcmin  = 226 px
  Step size:    332 arcmin  between patch centres
  Overlap:      180 arcmin  between adjacent patches (~2.4× sky redundancy)

Output HDF5 per patch:
  noisy_shear   (4, 2, 256, 256)  — [-g1, -g2] per tomo bin; NaN → 0
  sigma_noise   (4,    256, 256)  — sqrt(sigma2) per tomo bin; NaN → 100
  survey_mask   (4,    256, 256)  — float 0/1; buffer pixels forced to 0
  patch_center  (2,)              — [ra_centre, dec_centre] in degrees
"""

import os
import h5py as h5
import numpy as np
from tqdm import trange

DATA_DIR  = '/home2/supranta/PosteriorSampling/data/DES_Y3_metacal'
OUT_DIR   = ('/home2/supranta/PosteriorSampling/denoising_diffusion_pytorch'
             '/denoising_diffusion_pytorch/data/des_y3_metacal_big')
COORD_FILE = os.path.join(OUT_DIR, 'patch_coords.npy')

NPIX          = 256           # pixels per side
RESO          = 2.0           # arcmin per pixel
PATCH_ARCMIN  = NPIX * RESO   # 512 arcmin full patch
STEP_ARCMIN   = 332.0         # patch-centre spacing (arcmin)
BUFFER_PIX    = 15            # 30 arcmin / 2 arcmin per pixel
MIN_GALAXIES  = 100           # minimum total galaxies across all 4 bins


def load_tomo_bins(data_dir):
    """Load all four tomographic bins from metacal HDF5 files."""
    bins = []
    for i in range(4):
        path = os.path.join(data_dir, f'metacal_shear_bin_{i}.h5')
        with h5.File(path, 'r') as f:
            bins.append({
                'ra':  f['ra'][:],
                'dec': f['dec'][:],
                'g1':  f['g1'][:],
                'g2':  f['g2'][:],
                'w':   f['w'][:],
            })
    return bins


def centres_to_bounds(centres, half_deg):
    """
    Derive [ra_min, ra_max, dec_min, dec_max] for each patch centre.
    RA half-width is cosine-corrected for the centre's declination.
    """
    bounds = []
    for ra_c, dec_c in centres:
        cos_dec = np.cos(np.radians(dec_c))
        half_ra = half_deg / cos_dec if cos_dec > 1e-10 else 180.0
        bounds.append([ra_c - half_ra, ra_c + half_ra,
                       dec_c - half_deg, dec_c + half_deg])
    return np.array(bounds)


def generate_patch_grid(dec_all, step_arcmin, patch_arcmin):
    """
    Tile the Dec range of the survey with patch centres spaced step_arcmin
    apart.  RA always spans the full 0–360° circle; footprint filtering is
    handled downstream by the galaxy-count threshold.  RA spacing is
    cosine-corrected so patches have equal angular size at every declination.

    Returns
    -------
    centres : list of (ra, dec) tuples in degrees
    bounds  : np.ndarray of shape (N, 4) — [ra_min, ra_max, dec_min, dec_max]
    """
    step_deg = step_arcmin / 60.0
    half_deg = (patch_arcmin / 60.0) / 2.0

    dec_min = dec_all.min() - half_deg
    dec_max = dec_all.max() + half_deg

    centres = []

    dec = dec_min + half_deg
    while dec <= dec_max:
        cos_dec = np.cos(np.radians(dec))
        if cos_dec > 0.01:
            ra_step = step_deg / cos_dec
            ra = 0.0
            while ra < 360.0:
                centres.append((ra, dec))
                ra += ra_step
        dec += step_deg

    return centres, centres_to_bounds(centres, half_deg)


def objects_in_patch(ra, dec, bounds):
    """
    Boolean mask of objects whose (ra, dec) fall within patch bounds.
    Handles RA wraparound at 0°/360°.
    """
    ra_min, ra_max, dec_min, dec_max = bounds

    dec_ok = (dec >= dec_min) & (dec <= dec_max)

    if ra_min <= ra_max:
        ra_ok = (ra >= ra_min) & (ra <= ra_max)
    else:
        ra_ok = (ra >= ra_min) | (ra <= ra_max)

    return ra_ok & dec_ok


def radec_to_gnomonic(ra, dec, ra_c, dec_c):
    """
    Project (ra, dec) onto a flat plane centred at (ra_c, dec_c) using the
    gnomonic projection.  Returns (x, y) in degrees.
    """
    ra_rad  = np.radians(ra)
    dec_rad = np.radians(dec)
    ra_c_r  = np.radians(ra_c)
    dec_c_r = np.radians(dec_c)

    dra = ra_rad - ra_c_r
    dra = np.where(dra >  np.pi, dra - 2*np.pi, dra)
    dra = np.where(dra < -np.pi, dra + 2*np.pi, dra)

    cos_c = (np.sin(dec_c_r) * np.sin(dec_rad)
             + np.cos(dec_c_r) * np.cos(dec_rad) * np.cos(dra))

    x = np.cos(dec_rad) * np.sin(dra) / cos_c
    y = (np.cos(dec_c_r) * np.sin(dec_rad)
         - np.sin(dec_c_r) * np.cos(dec_rad) * np.cos(dra)) / cos_c

    return np.degrees(x), np.degrees(y)


def pixelize_shear_patch(ra, dec, g1, g2, w, bounds, centre, npix, reso):
    """
    Bin shear measurements into a 2D npix×npix map using a weighted mean.

    Returns a dict with g1_map, g2_map, sigma2_map, survey_mask
    (all shape npix×npix).  Pixels with no galaxies are NaN / False.
    """
    patch_mask = objects_in_patch(ra, dec, bounds)

    ra_p = ra[patch_mask];  dec_p = dec[patch_mask]
    g1_p = g1[patch_mask];  g2_p  = g2[patch_mask];  w_p = w[patch_mask]

    half_deg = (npix * reso / 60.0) / 2.0
    edges    = np.linspace(-half_deg, half_deg, npix + 1)

    ra_c, dec_c = centre
    x_p, y_p   = radec_to_gnomonic(ra_p, dec_p, ra_c, dec_c)

    i_pix = np.clip(np.digitize(y_p, edges) - 1, 0, npix - 1)
    j_pix = np.clip(np.digitize(x_p, edges) - 1, 0, npix - 1)

    # Drop galaxies with invalid shear or non-positive weight
    good = ~(np.isnan(g1_p) | np.isnan(g2_p)) & (w_p > 0)
    i_pix, j_pix = i_pix[good], j_pix[good]
    g1_p, g2_p, w_p = g1_p[good], g2_p[good], w_p[good]

    weight_map = np.zeros((npix, npix))
    g1_acc     = np.zeros((npix, npix))
    g2_acc     = np.zeros((npix, npix))
    sigma2_acc = np.zeros((npix, npix))

    np.add.at(weight_map, (i_pix, j_pix), w_p)
    np.add.at(g1_acc,     (i_pix, j_pix), g1_p * w_p)
    np.add.at(g2_acc,     (i_pix, j_pix), g2_p * w_p)
    np.add.at(sigma2_acc, (i_pix, j_pix), 0.5 * w_p**2 * (g1_p**2 + g2_p**2))

    valid      = weight_map > 0
    w2         = np.where(valid, weight_map, 1.0)   # avoid divide-by-zero in invalid pixels
    g1_map     = np.where(valid, g1_acc     / w2,       np.nan)
    g2_map     = np.where(valid, g2_acc     / w2,       np.nan)
    sigma2_map = np.where(valid, sigma2_acc / w2**2,    np.nan)

    return {
        'g1_map':      g1_map,
        'g2_map':      g2_map,
        'sigma2_map':  sigma2_map,
        'survey_mask': valid,
    }


def apply_buffer_mask(survey_mask, buffer_pix):
    """Zero out the border buffer region in every tomographic bin's mask (in-place)."""
    survey_mask[:, :buffer_pix, :]  = 0
    survey_mask[:, -buffer_pix:, :] = 0
    survey_mask[:, :, :buffer_pix]  = 0
    survey_mask[:, :, -buffer_pix:] = 0


def assemble_dps_arrays(bin_results, buffer_pix):
    """
    Convert per-bin pixelization results into DPS-format tensors.

    noisy_shear : (4, 2, npix, npix)  — sign-flipped; NaN → 0
    sigma_noise : (4,    npix, npix)  — sqrt(sigma2); NaN → 100
    survey_mask : (4,    npix, npix)  — float; buffer pixels forced to 0
    """
    n_bins = len(bin_results)
    npix   = bin_results[0]['g1_map'].shape[0]

    noisy_shear  = np.zeros((n_bins, 2, npix, npix))
    sigma_noise  = np.zeros((n_bins, npix, npix))
    survey_mask  = np.zeros((n_bins, npix, npix))

    for i, res in enumerate(bin_results):
        g1 = res['g1_map'].copy()
        g2 = res['g2_map'].copy()
        s2 = res['sigma2_map'].copy()

        # Sign convention
        g1[np.isnan(g1)] = 0.0
        g2[np.isnan(g2)] = 0.0

        noisy_shear[i, 0] = -g1
        noisy_shear[i, 1] = -g2

        # Large noise for unobserved pixels → down-weighted in DPS likelihood
        s2[np.isnan(s2)] = 10000.0
        sigma_noise[i]   = np.sqrt(s2)

        survey_mask[i] = res['survey_mask'].astype(float)

    apply_buffer_mask(survey_mask, buffer_pix)

    return noisy_shear, sigma_noise, survey_mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading tomographic bins...')
    tomo_bins = load_tomo_bins(DATA_DIR)

    dec_all = np.concatenate([b['dec'] for b in tomo_bins])

    # ------------------------------------------------------------------
    # Patch grid — computed once and cached
    # ------------------------------------------------------------------
    half_deg = (PATCH_ARCMIN / 60.0) / 2.0

    if os.path.exists(COORD_FILE):
        print(f'Loading cached patch coords from {COORD_FILE}')
        patch_coords = np.load(COORD_FILE)
        centres = [(c[0], c[1]) for c in patch_coords]
        bounds  = centres_to_bounds(centres, half_deg)
    else:
        print('Generating patch grid...')
        centres, bounds = generate_patch_grid(dec_all, STEP_ARCMIN, PATCH_ARCMIN)

        print('Filtering patches by galaxy count...')
        kept_centres = []
        kept_bounds  = []

        for idx in trange(len(centres)):
            total_count = sum(
                objects_in_patch(b['ra'], b['dec'], bounds[idx]).sum()
                for b in tomo_bins
            )
            if total_count >= MIN_GALAXIES:
                kept_centres.append(centres[idx])
                kept_bounds.append(bounds[idx])

        centres = kept_centres
        bounds  = np.array(kept_bounds)

        patch_coords = np.array(centres)   # shape (N, 2)
        np.save(COORD_FILE, patch_coords)
        print(f'Saved {len(centres)} patch centres to {COORD_FILE}')

    print(f'Processing {len(centres)} patches...')

    for idx in trange(len(centres)):
        out_path = os.path.join(OUT_DIR, f'patch_{idx}.h5')
        centre   = centres[idx]
        bound    = bounds[idx]

        bin_results = []
        for tbin in tomo_bins:
            result = pixelize_shear_patch(
                tbin['ra'], tbin['dec'],
                tbin['g1'], tbin['g2'], tbin['w'],
                bound, centre, NPIX, RESO,
            )
            bin_results.append(result)

        noisy_shear, sigma_noise, survey_mask = assemble_dps_arrays(
            bin_results, BUFFER_PIX
        )

        with h5.File(out_path, 'w') as f:
            f['noisy_shear']  = noisy_shear
            f['sigma_noise']  = sigma_noise
            f['survey_mask']  = survey_mask
            f['patch_center'] = np.array(centre)

    print('Done.')


if __name__ == '__main__':
    main()
