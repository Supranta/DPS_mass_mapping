import torch
import os as os
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from matplotlib.animation import FuncAnimation
import torchvision.transforms as transforms
import imageio
from FieldAnalysis import PowerSpectrumCalculator#, FieldCorrelations
from MapTools import TorchMapTools
import math
import numpy as np
import h5py as h5
from tqdm import trange


#Functions used to perform sampling
def compare_tensors_with_tolerance(tensor1, tensor2, tolerance=.0039):
    """
    Compares two tensors element-wise and prints the differences,
    setting the difference to 0 if it is within the specified tolerance.

    """
    # Ensure both tensors are the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape to compare")
    
    # Calculate the absolute difference
    difference = torch.abs(tensor1 - tensor2)
    
    # Set differences within tolerance to 0
    difference_with_tolerance = torch.where(difference <= tolerance, torch.tensor(0.0), difference)
    
    # Print the differences
    print("Differences (0 if within tolerance):")
    print(difference_with_tolerance)

def torch_kappa_to_shear_old(kappa, N_grid = 128, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    kappa_fourier = torch_map_tool.map2fourier(kappa)
    y_1, y_2   = torch_map_tool.do_fwd_KS(kappa_fourier)
    shear_map = torch.stack((y_1, y_2))
    return shear_map

def torch_shear_to_kappa(shear, N_grid = 128, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    kappa = torch_map_tool.do_KS_inversion(shear)
    return kappa

def torch_kappa_to_shear(kappa, N_grid = 128, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    y_1, y_2   = torch_map_tool.do_fwd_KS1(kappa)
    shear_map = torch.stack((y_1, y_2))
    return shear_map

def add_noise_to_map(input_map, std_map):
    """
    Adds Gaussian noise to a shear map while ensuring gradients are tracked.

    Parameters:
    shear_map (torch.Tensor): The input shear map with shape (256, 256).
    std_map (torch.Tensor, optional): A 256x256 tensor specifying the standard deviation at each pixel or a single scalar
    """
    assert isinstance(input_map, torch.Tensor), "shear_map must be a torch.Tensor"
    
    if not input_map.requires_grad:
        input_map.requires_grad = True

    noise = torch.randn_like(input_map) * std_map
    noisy_map = input_map + noise

    # Ensure noisy_shear_map tracks gradients
    if not noisy_map.requires_grad:
        noisy_map.requires_grad = True

    return noisy_map

def neff2noise(neff, pix_area):
    """
    :neff: Effective number density of galaxies per arcmin^2
    :pix_area: pixel area in arcmin^2
    """
    N = neff * pix_area    # avg. number of galaxies per pixel
    sigma_e = 0.26      # avg. shape noise per galaxy
    total_noise = sigma_e / math.sqrt(N)
    return total_noise

KAPPA_MIN = np.array([-0.03479804,-0.05888689,-0.08089042])[:,np.newaxis,np.newaxis]
KAPPA_MAX = np.array([0.4712809,  0.58141315, 0.6327746])[:,np.newaxis,np.newaxis]

def unnorm_kappa(field, kappa_min = KAPPA_MIN, kappa_max = KAPPA_MAX, exp_transform = False):
    if(exp_transform):        
        shift = 1.1 * kappa_min
        y_min = np.log(kappa_min - shift)
        y_max = np.log(kappa_max - shift)
        y = (field * (y_max - y_min)) + y_min
        kappa = np.exp(y) + shift
    else:
        kappa = (field * (kappa_max - kappa_min)) + kappa_min
    return kappa

def norm_kappa(kappa, kappa_min = -0.08201675, kappa_max = 0.7101586):
    x = (kappa - kappa_min) / (kappa_max - kappa_min)
    return x

#Hyperparameters
Delta_theta       = 3.5 / 128 * 60.      # Pixel side in arcmin
pix_area          = Delta_theta**2       # Pixel area in arcmin^2
ddim_sampling_eta = 1
batch_size        = 16                   # Number of samples for Unconditioned sampling, only one sample produced for conditioned at a time
neff              = 7.5                  # Effective number density of source galaxies in arcmin^{-2}
sigma_noise       = neff2noise(neff, pix_area) # Estimates the Gaussian noise in each pixel given the effective number density and the pixel area

return_all_timesteps = False

# Create noisy data measurement
n_ind     = 439
filename  = f"/home2/supranta/PosteriorSampling/data/Columbia_lensing/MassiveNuS/kappa_128_3bins/%d.npy"%(n_ind)
kappa_map = np.load(filename)    
kappa_map = torch.tensor(kappa_map)
x0_mean   = norm_kappa(kappa_map.mean())

kappa_map = kappa_map.to('cuda:0')
kappa_map.requires_grad = False

# Create a noisy shear map and also the associated Kaiser-Squires inverted map
noisy_shear_map_tomo = []
KS_inv_map_tomo      = []
for i in range(3):
	shear_map = torch_kappa_to_shear(kappa_map[i])
	noisy_shear_map = add_noise_to_map(shear_map, sigma_noise)
	noisy_shear_map_tomo.append(noisy_shear_map.detach())
	KS_inverse = torch_shear_to_kappa(noisy_shear_map)
	KS_inv_map_tomo.append(KS_inverse.detach())

noisy_kappa_map_tomo = []
for i in range(3):
	noisy_kappa_map = add_noise_to_map(kappa_map[i], sigma_noise)
	noisy_kappa_map_tomo.append(noisy_kappa_map.detach())

#This chunk of code creates the corresponding target shear maps(overwriting pervious saves of shear maps)
savedir      = 'diffusion_samples' # Directory where to save the outputs
name         = savedir
samples_root = os.path.join("./samples", name)
os.makedirs(samples_root, exist_ok=True)

# Save the data file
with h5.File(samples_root + '/data.h5', 'w') as f:
    f['kappa']       = kappa_map.cpu().numpy()
    f['noisy_shear'] = np.array([noisy_shear_map_tomo[i].cpu().numpy() for i in range(3)])
    f['KS_map']      = np.array([KS_inv_map_tomo[i].cpu().numpy() for i in range(3)])
    f['neff']        = neff       

schedule = 'exp_transform'
exp_transform = False
schedule_fn_kwargs = dict()
beta_schedule = schedule

# Folder where the diffusion model is saved
diffusion_folder = './results'
beta_schedule = 'sigmoid'
schedule_fn_kwargs = {'start': -5}
exp_transform = True
dim_mults = (1, 2, 4, 8)

# Define the diffusion model architecture
model = Unet(
    dim = 64,
    dim_mults = dim_mults,
    flash_attn = False, 
    channels = 3
).cuda()

noisy_shear_map = noisy_shear_map.unsqueeze(0)
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 1000, 
    noisy_image = noisy_shear_map_tomo, #Map to condition to
    sigma_noise = sigma_noise, #Noise specification
    beta_schedule = beta_schedule,
    schedule_fn_kwargs = schedule_fn_kwargs, 
    kappa_min = KAPPA_MIN,
    kappa_max = KAPPA_MAX,
    exp_transform = exp_transform,
    ddim_sampling_eta = ddim_sampling_eta
).cuda()

trainer = Trainer(
    diffusion,
    '/home2/supranta/PosteriorSampling/data/Columbia_lensing/MassiveNuS/kappa_128_3bins/',
    train_batch_size = 16,
    train_lr = 8e-5,
    save_and_sample_every = 20000,
    num_samples = 100, 
    train_num_steps = 200000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,            # whether to calculate fid during training
    exp_transform = exp_transform,
    results_folder = diffusion_folder
)

# The diffusion model saves the model periodically. Here 32 signifies the 32nd model that was saved. 
# In this step we are loading the trained model and creating samples with it.  
trainer.load('32')

name = savedir
samples_root = os.path.join("./samples", name)
os.makedirs(samples_root, exist_ok=True)

# Sample unconditioned maps from the diffusion model
# Creates a total of n_iters * batch_size maps
n_iters = 4
for n in trange(n_iters):
	sampled_images = diffusion.sample(batch_size = batch_size, return_all_timesteps = False)
	for i in trange(batch_size):
		sample_i = sampled_images[i].detach().cpu().squeeze().numpy()
		kappa_map = unnorm_kappa(sample_i, exp_transform = exp_transform)
		ind = n * batch_size + i
		with h5.File(samples_root + '/prior_sample_%d.h5'%(ind), 'w') as f:
			f['kappa'] = kappa_map 

# Sample maps from the diffusion model posterior using DPS
name = savedir
samples_root = os.path.join("./samples", name)
os.makedirs(samples_root, exist_ok=True)

N_samples = 10 # Will create a total of 10 posterior maps 
for sample_id in trange(N_samples):
    sampled_images_posterior = diffusion.sample_posterior(batch_size = 1, return_all_timesteps=False)
    with h5.File(samples_root + '/posterior_sample_%d.h5'%(sample_id), 'w') as f:
        kappa_dps = sampled_images_posterior.detach().cpu().numpy()
        kappa_dps = unnorm_kappa(kappa_dps, exp_transform = exp_transform)
        f['kappa'] = kappa_dps[0]

