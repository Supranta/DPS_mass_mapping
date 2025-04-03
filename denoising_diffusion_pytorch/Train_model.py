import torch
import numpy as np
import sys
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

N_channels = 3         # Number of tomographic bins
data_folder = '/home2/supranta/PosteriorSampling/data/Columbia_lensing/MassiveNuS/kappa_128_3bins/'
KAPPA_MIN = np.array([-0.03479804,-0.05888689,-0.08089042])[:,np.newaxis,np.newaxis]
KAPPA_MAX = np.array([0.4712809,  0.58141315, 0.6327746])[:,np.newaxis,np.newaxis]

results_folder = './results'
beta_schedule = 'sigmoid'
schedule_fn_kwargs = {'start': -5}
exp_transform = True
dim_mults = (1, 2, 4, 8)

#Adjust model architecture 
model = Unet(
    dim = 64,
    dim_mults = dim_mults,
    flash_attn = False, 
    channels = N_channels
).cuda()


diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 999,
    beta_schedule = beta_schedule,
    schedule_fn_kwargs = schedule_fn_kwargs,
    kappa_min = KAPPA_MIN,
    kappa_max = KAPPA_MAX,
    exp_transform = exp_transform
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
    exp_transform = exp_transform,
    results_folder = results_folder
)

trainer.train()






