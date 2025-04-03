import torch
import numpy as np
import sys
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import yaml

configfile = sys.argv[1]

with open(configfile, 'r') as f:
	config = yaml.safe_load(f)

data_folder    = config['train']['data_folder']
results_folder = config['train']['results_folder']

n_tomo    = config['map']['n_tomo']
n_grid    = config['map']['n_grid']

KAPPA_MIN = np.array([-0.03479804,-0.05888689,-0.08089042])[:,np.newaxis,np.newaxis]
KAPPA_MAX = np.array([0.4712809,  0.58141315, 0.6327746])[:,np.newaxis,np.newaxis]


#Adjust model architecture 
model = Unet(
    dim = 64,
    flash_attn = False, 
    channels = n_tomo
).cuda()


diffusion = GaussianDiffusion(
    model,
    image_size = n_grid,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 999,
    kappa_min = KAPPA_MIN,
    kappa_max = KAPPA_MAX,
).cuda()

#Adjust training specifications
trainer = Trainer(
    diffusion,
    data_folder, 
    train_batch_size = 16,
    train_lr = 8e-7,
    save_and_sample_every = 10000,
    train_num_steps = 500000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,            # whether to calculate fid during training
    #exp_transform = exp_transform,
    results_folder = results_folder
)

trainer.train()






