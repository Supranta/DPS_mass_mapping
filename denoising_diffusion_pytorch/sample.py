import torch
import os as os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import numpy as np
import h5py as h5
from tqdm import trange
import sys
import yaml

configfile = sys.argv[1]

with open(configfile, 'r') as f:
	config = yaml.safe_load(f)

data_folder    = config['train']['data_folder']
results_folder = config['train']['results_folder']

n_tomo    = config['map']['n_tomo']
n_grid    = config['map']['n_grid']

datafile  = config['data']['datafile']

modelname  = str(config['diffusion']['trained_model_name'])
savedir    = config['diffusion']['savedir']
batch_size = config['diffusion']['prior_batch_size']                   # Number of samples for Unconditioned sampling, only one sample produced for conditioned at a time
n_iters    = config['diffusion']['n_prior_iterations']
n_dps      = config['diffusion']['n_dps_samples']

KAPPA_MIN = np.array([-0.03479804,-0.05888689,-0.08089042])[:,np.newaxis,np.newaxis]
KAPPA_MAX = np.array([0.4712809,  0.58141315, 0.6327746])[:,np.newaxis,np.newaxis]

def unnorm_kappa(field, kappa_min = KAPPA_MIN, kappa_max = KAPPA_MAX):
    shift = 1.1 * kappa_min
    y_min = np.log(kappa_min - shift)
    y_max = np.log(kappa_max - shift)
    y = (field * (y_max - y_min)) + y_min
    kappa = np.exp(y) + shift
    return kappa

# Define the diffusion model architecture
model = Unet(
    dim = 64,
    flash_attn = False, 
    channels = n_tomo
).cuda()

with h5.File(datafile, 'r') as f:
	noisy_shear_map = torch.tensor(f['noisy_shear'][:], device='cuda') 
	sigma_noise     = f['sigma_noise'][()]

diffusion = GaussianDiffusion(
    model,
    image_size = n_grid,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 1000, 
    noisy_image = noisy_shear_map, #Map to condition to
    sigma_noise = sigma_noise, #Noise specification
    kappa_min = KAPPA_MIN,
    kappa_max = KAPPA_MAX,
    ddim_sampling_eta = 1.
).cuda()

trainer = Trainer(
    diffusion,
    data_folder,
 #   gradient_accumulate_every = 2,    # gradient accumulation steps
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

# Sample maps from the diffusion model posterior using DPS
N_samples = 1 # Will create a total of 10 posterior maps 
for sample_id in trange(n_dps):
    sampled_images_posterior = diffusion.sample_posterior(batch_size = 1, return_all_timesteps=False)
    with h5.File(samples_root + '/posterior_sample_%d.h5'%(sample_id), 'w') as f:
        kappa_dps = sampled_images_posterior.detach().cpu().numpy()
        kappa_dps = unnorm_kappa(kappa_dps)
        f['kappa'] = kappa_dps[0]

