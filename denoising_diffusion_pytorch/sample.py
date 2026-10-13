import torch
import os as os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import numpy as np
import h5py as h5
from tqdm import trange
import sys
from utils import *
import yaml

configfile = sys.argv[1]

with open(configfile, 'r') as f:
	config = yaml.safe_load(f)

data_folder    = config['train']['data_folder']
results_folder = config['train']['results_folder']

n_tomo    = config['map']['n_tomo']
n_grid    = config['map']['n_grid']
theta_max = config['map']['theta_max']

datafile  = config['data']['datafile']

modelname   = str(config['diffusion']['trained_model_name'])
savedir     = config['diffusion']['savedir']
batch_size  = config['diffusion']['batch_size']                   # Number of samples for diffusion sampling
n_iters     = config['diffusion']['n_prior_iterations']
n_iters_dps = config['diffusion']['n_dps_iterations']
n_iters_tds         = config['diffusion'].get('n_tds_iterations', 0)
n_tds_particles     = config['diffusion'].get('n_tds_particles', 8)
tds_grad_batch_size = config['diffusion'].get('tds_grad_batch_size', None)
tds_ess_threshold   = config['diffusion'].get('tds_ess_threshold', 0.5)

KAPPA_MIN = string_to_numpy_array(config['train']['transform']['kappa_min'])[:,np.newaxis,np.newaxis]
KAPPA_MAX = string_to_numpy_array(config['train']['transform']['kappa_max'])[:,np.newaxis,np.newaxis]
def unnorm_kappa(field, kappa_min = KAPPA_MIN, kappa_max = KAPPA_MAX):
    return (field * (kappa_max - kappa_min)) + kappa_min

# Define the diffusion model architecture
model = Unet(
    dim = 64,
    flash_attn = False,
    channels = n_tomo
).cuda()

with h5.File(datafile, 'r') as f:
    noisy_shear_map = torch.tensor(f['noisy_shear'][:], device='cuda')
    sigma_noise     = torch.tensor(f['sigma_noise'][:], device='cuda')
    survey_mask     = torch.tensor(f['survey_mask'][:], device='cuda')

guidance_method = config['diffusion'].get('guidance_method', 'dps_sigmoid')

# Load measured tau_hat_sq for the calibrated pi-GDM schedule.
# Only needed for pigdm_calibrated; all other methods ignore it.
tau_hat_sq = None
if guidance_method == 'pigdm_calibrated':
    tau_hat_sq_file = config['diffusion'].get(
        'tau_hat_sq_file',
        os.path.join(os.path.dirname(__file__), 'tds_experiment', 'tau_hat_sq.npy')
    )
    tau_hat_sq = torch.tensor(np.load(tau_hat_sq_file), dtype=torch.float32)
    print(f"Loaded tau_hat_sq from {tau_hat_sq_file}: {tau_hat_sq.numpy()}")

diffusion = GaussianDiffusion(
    model,
    image_size = n_grid,
    theta_max = theta_max,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 1000,
    noisy_image = noisy_shear_map, #Map to condition to
    sigma_noise = sigma_noise, #Noise specification
    survey_mask = survey_mask,
    kappa_min = KAPPA_MIN,
    kappa_max = KAPPA_MAX,
    ddim_sampling_eta = 1.,
    tau_hat_sq = tau_hat_sq,
).cuda()

trainer = Trainer(
    diffusion,
    data_folder,
    calculate_fid = False,            # whether to calculate fid during training
    results_folder = results_folder
)

# In this step we are loading the trained model and creating samples with it.
trainer.load(modelname)

name = savedir
samples_root = os.path.join("./samples", name)
os.makedirs(samples_root, exist_ok=True)

# Sample unconditioned maps from the diffusion model
# Creates a total of n_iters * batch_size maps
for n in trange(n_iters):
	sampled_images = diffusion.sample(batch_size = batch_size, return_all_timesteps = False)
	for i in trange(batch_size):
		sample_i = sampled_images[i].detach().cpu().squeeze().numpy()
		kappa_map = unnorm_kappa(sample_i)
		ind = n * batch_size + i
		with h5.File(samples_root + '/prior_sample_%d.h5'%(ind), 'w') as f:
			f['kappa'] = kappa_map

def convert_string_to_array(string):
    str_list = string.split(',')
    arr_list = []
    for x in str_list:
        arr_list.append(float(x))
    return np.array(arr_list)

if guidance_method in ('dps_sigmoid', 'dps_sigmoid_annealed'):
    delta_t = convert_string_to_array(config['diffusion']['lkl_scaling']['delta_t'])
    sigma_t = convert_string_to_array(config['diffusion']['lkl_scaling']['sigma_t'])
    diffusion.set_dps_lkl_scale(delta_t, sigma_t)

# Sample maps from the diffusion model posterior
for n in trange(n_iters_dps):
    sampled_images_posterior = diffusion.sample_posterior(
        batch_size=batch_size, guidance_method=guidance_method, return_all_timesteps=False
    )
    for i in range(batch_size):
        sample_i = sampled_images_posterior[i].detach().cpu().squeeze().numpy()
        kappa_map = unnorm_kappa(sample_i)
        ind = n * batch_size + i
        with h5.File(samples_root + '/posterior_sample_%d.h5'%(ind), 'w') as f:
            f['kappa'] = kappa_map

# Sample maps from the diffusion model posterior using TDS.
# One unweighted posterior sample is drawn per iteration by resampling a
# single particle proportional to the final importance weights.
for n in trange(n_iters_tds):
    tds_samples, tds_log_weights = diffusion.sample_tds(n_particles=n_tds_particles,
                                                         grad_batch_size=tds_grad_batch_size,
                                                         ess_threshold=tds_ess_threshold,
                                                         guidance_method=guidance_method)
    log_w = tds_log_weights.cpu().numpy()
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    idx = np.random.choice(n_tds_particles, p=w)
    sample_i = tds_samples[idx].detach().cpu().squeeze().numpy()
    kappa_map = unnorm_kappa(sample_i)
    with h5.File(samples_root + '/tds_sample_%d.h5' % n, 'w') as f:
        f['kappa'] = kappa_map
    np.save(samples_root + '/tds_log_weights_%d.npy' % n,
            log_w)
