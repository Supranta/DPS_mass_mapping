import torch
import os as os
from MapTools import TorchMapTools
import math
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
theta_max = config['map']['theta_max']

datafile = config['data']['datafile']
neff     = config['data']['neff']
sigma_e  = config['data']['sigma_e']

torch_map_tool  = TorchMapTools(n_grid, theta_max)

def torch_shear_to_kappa(shear): 
    kappa = torch_map_tool.do_KS_inversion(shear)
    return kappa

def torch_kappa_to_shear(kappa): 
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

def neff2noise(sigma_e, neff, pix_area):
    """
    :neff: Effective number density of galaxies per arcmin^2
    :pix_area: pixel area in arcmin^2
    """
    N = neff * pix_area    # avg. number of galaxies per pixel
    total_noise = sigma_e / math.sqrt(N)
    return total_noise

KAPPA_MIN = np.array([-0.03479804,-0.05888689,-0.08089042])[:,np.newaxis,np.newaxis]
KAPPA_MAX = np.array([0.4712809,  0.58141315, 0.6327746])[:,np.newaxis,np.newaxis]

def unnorm_kappa(field, kappa_min = KAPPA_MIN, kappa_max = KAPPA_MAX):
    shift = 1.1 * kappa_min
    y_min = np.log(kappa_min - shift)
    y_max = np.log(kappa_max - shift)
    y = (field * (y_max - y_min)) + y_min
    kappa = np.exp(y) + shift
    return kappa

def norm_kappa(kappa, kappa_min = -0.08201675, kappa_max = 0.7101586):
    x = (kappa - kappa_min) / (kappa_max - kappa_min)
    return x

#Hyperparameters
Delta_theta       = theta_max / n_grid * 60.            # Pixel side in arcmin
pix_area          = Delta_theta**2                      # Pixel area in arcmin^2
sigma_noise       = neff2noise(sigma_e, neff, pix_area) # Estimates the Gaussian noise in each pixel given the effective number density and the pixel area

# Create noisy data measurement
n_ind     = 439
filename  = data_folder + "/%d.npy"%(n_ind)
kappa_map = np.load(filename)    
kappa_map = torch.tensor(kappa_map)

kappa_map = kappa_map.to('cuda:0')
kappa_map.requires_grad = False

# Create a noisy shear map and also the associated Kaiser-Squires inverted map
noisy_shear_map_tomo = []
KS_inv_map_tomo      = []
for i in range(n_tomo):
	shear_map = torch_kappa_to_shear(kappa_map[i])
	noisy_shear_map = add_noise_to_map(shear_map, sigma_noise)
	noisy_shear_map_tomo.append(noisy_shear_map.detach())
	KS_inverse = torch_shear_to_kappa(noisy_shear_map)
	KS_inv_map_tomo.append(KS_inverse.detach())

noisy_kappa_map_tomo = []
for i in range(n_tomo):
	noisy_kappa_map = add_noise_to_map(kappa_map[i], sigma_noise)
	noisy_kappa_map_tomo.append(noisy_kappa_map.detach())

# Save the data file
with h5.File(datafile, 'w') as f:
    f['kappa']       = kappa_map.cpu().numpy()
    f['noisy_shear'] = np.array([noisy_shear_map_tomo[i].cpu().numpy() for i in range(3)])
    f['KS_map']      = np.array([KS_inv_map_tomo[i].cpu().numpy() for i in range(3)])
    f['neff']        = neff       
    f['sigma_noise'] = sigma_noise

