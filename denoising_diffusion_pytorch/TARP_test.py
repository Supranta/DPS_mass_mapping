# IMPORTANT - IN CURRENT ITERATION OF THIS SCRIPT, THE CONFIG FILE MUST SET
# N_TARP_SIMULATION, N_TARP_REFERENCE, AND N_TARP_POSTERIORS (TARP FIELD) TO BE DIVISIBLE BY THE BATCH SIZE

import subprocess
import yaml
import os
import sys
import shutil
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
import random

# config file
configfile = sys.argv[1]

base_config_path = configfile
with open(base_config_path, 'r') as f:
    config = yaml.safe_load(f)

sample_output = config['diffusion']['savedir']
sample_output_dir = os.path.join('./samples', sample_output)
noisy_mock_dir = config['tarp']['mock_folder']
n_mocks = config['tarp']['n_noisy_mocks']
os.makedirs(noisy_mock_dir, exist_ok=True)
# number of simulation (prior) maps to generate
n_simulation_samples = config['tarp']['n_tarp_simulation']
# number of posterior samples to generate
n_posterior_samples = config['tarp']['n_tarp_posteriors']
# number of reference maps to generate
n_reference_samples = config['tarp']['n_tarp_reference']
# folder to save TARP results
tarp_results = './tarp_results'
os.makedirs(tarp_results, exist_ok=True)

# create noisy data which will be sampled to create simulation maps
subprocess.run(["python", "create_noisy_data.py", base_config_path])

# run sample.py on noisy data to generate simulation maps
# set n_prior_iterations in config based on n_simulation_samples and batch size
if (n_simulation_samples % config['diffusion']['batch_size']) == 0:
    config['diffusion']['n_prior_iterations'] = int(n_simulation_samples / config['diffusion']['batch_size'])
else:
    print("In the config file, n_tarp_simulation should be divisible by batch_size")
    exit(1)
config['diffusion']['n_dps_iterations'] = 0
# create temp config
with open('temp_config.yaml', 'w') as f:
    yaml.dump(config, f)
subprocess.run(["python", "sample.py", 'temp_config.yaml'])
# delete temp config file
os.remove('temp_config.yaml')
# delete data file
os.remove('./data/desy3/data_0.h5')

# move simulation maps to subfolder
sample_subfolder = os.path.join(sample_output_dir, 'simulation_samples')
os.makedirs(sample_subfolder, exist_ok=True)
for i in range(n_simulation_samples):
    default_output_dir = os.path.join(sample_output_dir, f'prior_sample_{i}.h5')
    new_output_dir = os.path.join(sample_subfolder, f'simulation_sample_{i}.h5')
    if os.path.exists(default_output_dir):
        shutil.move(default_output_dir, new_output_dir)

# run create_noisy_data.py to create noisy data for each simulation map
for i in range(n_mocks):
    # update create_noisy_data.py for creating noisy mocks
    tmp_dir = os.path.dirname(os.path.abspath('create_noisy_data.py'))
    with open('create_noisy_data.py', 'r') as f:
        lines = f.readlines()
    for x, line in enumerate(lines):
        if 'filename  = data_folder + "/%d.npy"%(n_ind)' in line:
            lines[x] = 'filename = data_folder\n'
        elif 'kappa_map = np.load(filename)' in line:
            lines[x] = (
                "with h5.File(filename, 'r') as f:\n"
                "    kappa_map = f['kappa'][:]\n"
            )
    temp_path = os.path.join(tmp_dir, 'create_noisy_data_temp.py')
    with open(temp_path, 'w') as f:
        f.writelines(lines)

    # input simulation map
    input_h5 = os.path.join(sample_subfolder, f"simulation_sample_{i}.h5")
    output_h5 = os.path.join(noisy_mock_dir, f"data_{i}.h5")

    if not os.path.exists(input_h5):
        print(f"File not found: {input_h5}")
        continue

    # update config
    config['train']['data_folder'] = input_h5
    config['data']['datafile'] = output_h5

    # create temp config
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)

    # call script to create noisy data
    subprocess.run(["python", temp_path, 'temp_config.yaml'])
    os.remove('temp_config.yaml')

# remove temp python file
os.remove(temp_path)

# run sample.py on noisy mocks to create posteriors
for i in range(n_mocks):
    datafile_path = os.path.join(noisy_mock_dir, f"data_{i}.h5")

    # create output subfolder
    iter_output_dir = os.path.join(sample_output_dir, f'posteriors_iter_{i}')
    os.makedirs(iter_output_dir, exist_ok=True)

    # update config
    config['data']['datafile'] = datafile_path

    config['diffusion']['n_prior_iterations'] = 0

    if (n_posterior_samples % config['diffusion']['batch_size'] == 0):
        config['diffusion']['n_dps_iterations'] = int(n_posterior_samples / config['diffusion']['batch_size'])
    else:
        print("In the config file, n_tarp_posteriors should be divisible by batch_size")
        exit(1)

    # create temp config
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)

    # run sample.py with new config
    subprocess.run(['python', 'sample.py', 'temp_config.yaml', str(i)], check=True)

    # move posterior samples to subfolder
    for j in range(n_posterior_samples):
        default_output_dir = os.path.join(sample_output_dir, f'posterior_sample_{j}.h5')
        new_output_dir = os.path.join(iter_output_dir, f'posterior_sample_{j}.h5')
        if os.path.exists(default_output_dir):
            shutil.move(default_output_dir, new_output_dir)
            print(f'Saved: {new_output_dir}')

    # delete temp config
    os.remove('temp_config.yaml')

# generate reference maps from a randomly chosen simulation map
simulation_idx = random.randint(0, n_simulation_samples - 1)
print("Reference samples generated from simulation index: {}".format(simulation_idx))
datafile_path = os.path.join(noisy_mock_dir, f"data_{simulation_idx}.h5")
reference_output_dir = os.path.join(sample_output_dir, 'reference_samples')
os.makedirs(reference_output_dir, exist_ok=True)
# update config
config['data']['datafile'] = datafile_path

if (n_reference_samples % config['diffusion']['batch_size']) == 0:
    config['diffusion']['n_prior_iterations'] = int(n_reference_samples / config['diffusion']['batch_size'])
else:
    print("In the config file, n_tarp_reference should be divisible by batch_size")
    exit(1)
config['diffusion']['n_dps_iterations'] = 0

# create temp config
with open('temp_config.yaml', 'w') as f:
    yaml.dump(config, f)
# run sample.py with new config
subprocess.run(['python', 'sample.py', 'temp_config.yaml'])

# move reference samples to subfolder
for i in range(n_reference_samples):
    default_output_dir = os.path.join(sample_output_dir, f'prior_sample_{i}.h5')
    new_output_dir = os.path.join(reference_output_dir, f'reference_sample_{i}.h5')
    if os.path.exists(default_output_dir):
        shutil.move(default_output_dir, new_output_dir)
        print(f'Saved: {new_output_dir}')
# delete temp config
os.remove('temp_config.yaml')

# calculate distances
distance_results = []
distance_per_bin_results = []
for ref_idx in range(n_reference_samples):
    for i in range(n_simulation_samples):
        # load ith simulation map
        with h5py.File(os.path.join(sample_subfolder, f"simulation_sample_{i}.h5"), 'r') as f:
            kappa_simulation = f['kappa'][:]

        # load reference maps
        with h5py.File(os.path.join(reference_output_dir, f"reference_sample_{ref_idx}.h5"), 'r') as f:
            kappa_reference = f['kappa'][:]

        # normalization
        kappa_min = np.minimum(kappa_simulation.min(), kappa_reference.min())
        kappa_max = np.maximum(kappa_simulation.max(), kappa_reference.max())
        # normalization per bin
        kappa_per_bin_min = np.minimum(kappa_reference.min(axis = (1, 2)), kappa_simulation.min(axis = (1, 2)))
        kappa_per_bin_max = np.maximum(kappa_reference.max(axis = (1, 2)), kappa_simulation.max(axis = (1, 2)))

        # define distances
        y_simulation = (kappa_simulation - kappa_min) / (kappa_max - kappa_min)
        y_reference = (kappa_reference - kappa_min) / (kappa_max - kappa_min)
        # distances per bin
        y_simulation_per_bin = (kappa_simulation - kappa_per_bin_min[:, None, None]) / (kappa_per_bin_max[:, None, None] - kappa_per_bin_min[:, None, None])
        y_reference_per_bin = (kappa_reference - kappa_per_bin_min[:, None, None]) / (kappa_per_bin_max[:, None, None] - kappa_per_bin_min[:, None, None])

        # distance: simulation to reference
        dist_simulation = np.sum((y_reference - y_simulation) ** 2)
        # distance per bin: simulation to reference
        dist_per_bin_simulation = np.sum((y_reference_per_bin - y_simulation_per_bin) ** 2, axis = (1, 2))

        # distance: posterior to reference
        dist_posterior = []
        dist_per_bin_posterior = []
        for j in range(n_posterior_samples):
            with h5py.File(os.path.join(sample_output_dir, f"posteriors_iter_{i}/posterior_sample_{j}.h5"), 'r') as f:
                kappa_posterior = f['kappa'][:]
            y_posterior = (kappa_posterior - kappa_min) / (kappa_max - kappa_min)
            y_posterior_per_bin = (kappa_posterior - kappa_per_bin_min[:, None, None]) / (kappa_per_bin_max[:, None, None] - kappa_per_bin_min[:, None, None])
            dist_posterior.append(np.sum((y_reference - y_posterior) ** 2))
            dist_per_bin_posterior.append(np.sum((y_reference_per_bin - y_posterior_per_bin) ** 2, axis = (1, 2)))

        # append distances to lists
        distance_results.append({
            'simulation_index': i,
            'reference_index': ref_idx,
            'simulation_distance': dist_simulation,
            'posterior_distances': dist_posterior
        })

        distance_per_bin_results.append({
            'simulation_index': i,
            'reference_index': ref_idx,
            'simulation_distance': dist_per_bin_simulation,
            'posterior_distances': dist_per_bin_posterior
        })

# get f_closer values
def get_closer_fraction(dist_samples, dist_true):
    frac_closer = (dist_samples < dist_true).sum() / len(dist_samples)
    return frac_closer

def get_frac_closer_list(dist, N_maps):
    frac_closer_list = []
    for i in range(N_maps):
        frac_closer = get_closer_fraction(dist[i]['posterior_distances'], dist[i]['simulation_distance'])
        frac_closer_list.append(frac_closer)
    return np.array(frac_closer_list)

stats_rows = []
def get_frac_closer_list_bin(dist, bin_num, N_maps):
    frac_closer_list = []
    for i in range(N_maps):
        bin_posterior_dists = []
        bin_posterior_dists = [posterior[bin_num] for posterior in dist[i]['posterior_distances']]
        frac_closer = get_closer_fraction(np.array(bin_posterior_dists), dist[i]['simulation_distance'][bin_num])
        frac_closer_list.append(frac_closer)

        stats_rows.append({
            "Map": i,
            "Bin": bin_num,
            "Simulation": round(dist[i]['simulation_distance'][bin_num], 4),
            "Min Posterior": round(np.min(bin_posterior_dists), 4),
            "Max Posterior": round(np.max(bin_posterior_dists), 4),
            "f_closer": round(frac_closer, 4)
        })
    return np.array(frac_closer_list)

# add f_closer values to respective lists
frac_closer_list = []
frac_closer_list_bin0 = []
frac_closer_list_bin1 = []
frac_closer_list_bin2 = []
frac_closer_list_bin3 = []

frac_closer_list.append(get_frac_closer_list(distance_results, n_reference_samples * n_simulation_samples))
frac_closer_list_bin0.append(get_frac_closer_list_bin(distance_per_bin_results, 0, n_reference_samples * n_simulation_samples))
frac_closer_list_bin1.append(get_frac_closer_list_bin(distance_per_bin_results, 1, n_reference_samples * n_simulation_samples))
frac_closer_list_bin2.append(get_frac_closer_list_bin(distance_per_bin_results, 2, n_reference_samples * n_simulation_samples))
frac_closer_list_bin3.append(get_frac_closer_list_bin(distance_per_bin_results, 3, n_reference_samples * n_simulation_samples))

frac_closer_arr = np.array(frac_closer_list).flatten()
frac_closer_arr_bin0 = np.array(frac_closer_list_bin0).flatten()
frac_closer_arr_bin1 = np.array(frac_closer_list_bin1).flatten()
frac_closer_arr_bin2 = np.array(frac_closer_list_bin2).flatten()
frac_closer_arr_bin3 = np.array(frac_closer_list_bin3).flatten()

# save TARP results to csv
df = pd.DataFrame(stats_rows)
df.to_csv(f'{tarp_results}/bin_stats.csv', index=False)

alpha = np.arange(0, 1.01, 0.05) # alpha used to calculate credibility level

coverage_probability_list = []
coverage_probability_list_bin0 = []
coverage_probability_list_bin1 = []
coverage_probability_list_bin2 = []
coverage_probability_list_bin3 = []

# calculate coverage probabilities
for i in range(len(alpha)):
    select_alpha_bin = (frac_closer_arr < alpha[i])
    coverage_probability = select_alpha_bin.sum() / len(select_alpha_bin)
    coverage_probability_list.append(coverage_probability)

    select_alpha_bin0 = (frac_closer_arr_bin0 < alpha[i])
    coverage_probability_bin0 = select_alpha_bin0.sum() / len(select_alpha_bin0)
    coverage_probability_list_bin0.append(coverage_probability_bin0)

    select_alpha_bin1 = (frac_closer_arr_bin1 < alpha[i])
    coverage_probability_bin1 = select_alpha_bin1.sum() / len(select_alpha_bin1)
    coverage_probability_list_bin1.append(coverage_probability_bin1)

    select_alpha_bin2 = (frac_closer_arr_bin2 < alpha[i])
    coverage_probability_bin2 = select_alpha_bin2.sum() / len(select_alpha_bin2)
    coverage_probability_list_bin2.append(coverage_probability_bin2)

    select_alpha_bin3 = (frac_closer_arr_bin3 < alpha[i])
    coverage_probability_bin3 = select_alpha_bin3.sum() / len(select_alpha_bin3)
    coverage_probability_list_bin3.append(coverage_probability_bin3)

coverage_probability_arr = np.array(coverage_probability_list)
coverage_probability_arr_bin0 = np.array(coverage_probability_list_bin0)
coverage_probability_arr_bin1 = np.array(coverage_probability_list_bin1)
coverage_probability_arr_bin2 = np.array(coverage_probability_list_bin2)
coverage_probability_arr_bin3 = np.array(coverage_probability_list_bin3)

# create and save coverage probability plots
plt.figure()
plt.xlabel('Credibility level')
plt.ylabel('Coverage probability')
plt.title('Coverage probability vs credibility level (all bins)')
plt.plot(alpha, alpha, 'r')
plt.plot(alpha, coverage_probability_arr)
plt.savefig(f'{tarp_results}/coverage_probability.png', dpi = 300, bbox_inches = 'tight')
plt.close()

plt.figure()
plt.xlabel('Credibility level')
plt.ylabel('Coverage probability')
plt.title('Coverage probability vs credibility level (bin 0)')
plt.plot(alpha, alpha, 'r')
plt.plot(alpha, coverage_probability_arr_bin0)
plt.savefig(f'{tarp_results}/coverage_probability_bin0.png', dpi = 300, bbox_inches = 'tight')
plt.close()

plt.figure()
plt.xlabel('Credibility level')
plt.ylabel('Coverage probability')
plt.title('Coverage probability vs credibility level (bin 1)')
plt.plot(alpha, alpha, 'r')
plt.plot(alpha, coverage_probability_arr_bin1)
plt.savefig(f'{tarp_results}/coverage_probability_bin1.png', dpi = 300, bbox_inches = 'tight')
plt.close()

plt.figure()
plt.xlabel('Credibility level')
plt.ylabel('Coverage probability')
plt.title('Coverage probability vs credibility level (bin 2)')
plt.plot(alpha, alpha, 'r')
plt.plot(alpha, coverage_probability_arr_bin2)
plt.savefig(f'{tarp_results}/coverage_probability_bin2.png', dpi = 300, bbox_inches = 'tight')
plt.close()

plt.figure()
plt.xlabel('Credibility level')
plt.ylabel('Coverage probability')
plt.title('Coverage probability vs credibility level (bin 3)')
plt.plot(alpha, alpha, 'r')
plt.plot(alpha, coverage_probability_arr_bin3)
plt.savefig(f'{tarp_results}/coverage_probability_bin3.png', dpi = 300, bbox_inches = 'tight')
plt.close()