import torch
import numpy as np
import sys
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import yaml
from utils import *

configfile = sys.argv[1]

with open(configfile, 'r') as f:
	config = yaml.safe_load(f)

data_folder    = config['train']['data_folder']
results_folder = config['train']['results_folder']
ntrain         = config['train']['ntrain']

n_tomo    = config['map']['n_tomo']
n_grid    = config['map']['n_grid']

try:
	modelname = str(config['diffusion']['trained_model_name'])
except:
	modelname = None

KAPPA_MIN = string_to_numpy_array(config['train']['transform']['kappa_min'])[:,np.newaxis,np.newaxis]
KAPPA_MAX = string_to_numpy_array(config['train']['transform']['kappa_max'])[:,np.newaxis,np.newaxis]
exp_transform = config['train']['transform']['exp_transform']

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
    exp_transform = exp_transform
).cuda()

n_gpus       = 3
base_lr      = 8e-5
n_accumulate = 1
lr           = base_lr * n_gpus * n_accumulate
#Adjust training specifications
trainer = Trainer(
    diffusion,
    data_folder, 
    train_batch_size = 24,
    train_lr = lr,
    save_and_sample_every = 2000,
    train_num_steps = ntrain,                 # total training steps
    gradient_accumulate_every = n_accumulate, # gradient accumulation steps
    ema_decay = 0.995,                        # exponential moving average decay
    amp = True,                               # turn on mixed precision
    calculate_fid = False,                    # whether to calculate fid during training
    results_folder = results_folder
)

if modelname is not None:
	trainer.load(modelname)

trainer.train()






