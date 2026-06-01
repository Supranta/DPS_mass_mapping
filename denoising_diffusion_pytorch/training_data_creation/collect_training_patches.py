import numpy as np
import os
import json
import shutil
from tqdm import tqdm

patch_base = '/pscratch/sd/s/ssarmabo/DPS_data/flat_patches'
out_dir    = '/pscratch/sd/s/ssarmabo/DPS_data/all_patches'
os.makedirs(out_dir, exist_ok=True)

# --- Validate simulations ---
valid_sims = []
skipped    = []

for simnum in range(1, 201):
    simdir        = os.path.join(patch_base, str(simnum))
    metadata_file = os.path.join(simdir, 'metadata.json')

    if not os.path.exists(metadata_file):
        skipped.append((simnum, 'no metadata.json'))
        continue

    with open(metadata_file) as f:
        sim_metadata = json.load(f)

    n_patches    = len(sim_metadata)
    missing_files = [
        os.path.join(simdir, '%d.npy' % i)
        for i in range(n_patches)
        if not os.path.exists(os.path.join(simdir, '%d.npy' % i))
    ]

    if missing_files:
        skipped.append((simnum, '%d/%d patch files missing' % (len(missing_files), n_patches)))
        continue

    valid_sims.append((simnum, simdir, sim_metadata))

print("Valid simulations : %d" % len(valid_sims))
print("Skipped           : %d" % len(skipped))
for simnum, reason in skipped:
    print("  sim %d: %s" % (simnum, reason))

# --- Copy patches and compute per-bin kappa min/max ---
# kappa_maps have shape (4, 256, 256); track min/max per tomographic bin
kappa_min = None
kappa_max = None

global_idx    = 1
patch_registry = []   # records source sim + local index for each global patch

for simnum, simdir, sim_metadata in tqdm(valid_sims, desc='Collecting sims'):
    n_patches = len(sim_metadata)

    for local_i in range(n_patches):
        src  = os.path.join(simdir, '%d.npy' % local_i)
        dst  = os.path.join(out_dir, '%d.npy' % global_idx)

        patch = np.load(src)   # shape (4, 256, 256)

        # Update running per-bin min/max
        patch_min = patch.min(axis=(1, 2))   # shape (4,)
        patch_max = patch.max(axis=(1, 2))   # shape (4,)

        if kappa_min is None:
            kappa_min = patch_min.copy()
            kappa_max = patch_max.copy()
        else:
            kappa_min = np.minimum(kappa_min, patch_min)
            kappa_max = np.maximum(kappa_max, patch_max)

        shutil.copy2(src, dst)

        patch_registry.append({
            'global_id': global_idx,
            'sim_id':    simnum,
            'local_id':  local_i,
            'center_lon': sim_metadata[local_i]['center_lon'],
            'center_lat': sim_metadata[local_i]['center_lat'],
        })

        global_idx += 1

# --- Save metadata ---
summary = {
    'n_patches':          global_idx - 1,
    'n_sims':             len(valid_sims),
    'skipped_sims':       [s[0] for s in skipped],
    'kappa_min':          kappa_min.tolist(),   # list of 4 values, one per tomo bin
    'kappa_max':          kappa_max.tolist(),
    'pixel_scale_arcmin': 2.0,
    'npix':               256,
    'patch_size_arcmin':  512.0,
}

with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(out_dir, 'patch_registry.json'), 'w') as f:
    json.dump(patch_registry, f, indent=2)

print("\nDone.")
print("Total patches : %d" % summary['n_patches'])
print("kappa_min     : %s" % kappa_min)
print("kappa_max     : %s" % kappa_max)
print("Output        : %s" % out_dir)
