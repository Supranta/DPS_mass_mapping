import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from tqdm import trange
from ema_pytorch import EMA

from accelerate import Accelerator
from MapTools import TorchMapTools
import time
import psutil
import os

from attend import Attend

from version import __version__

def check_memory_status():
    # GPU Memory
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Total: {total:.2f} GB")
    print(f"GPU - Free: {total - reserved:.2f} GB")

    # CPU Memory (System-wide)
    cpu_memory = psutil.virtual_memory()
    cpu_used = cpu_memory.used / 1e9
    cpu_total = cpu_memory.total / 1e9
    cpu_available = cpu_memory.available / 1e9
    print(f"CPU - Used: {cpu_used:.2f} GB, Available: {cpu_available:.2f} GB, Total: {cpu_total:.2f} GB")
    print(f"CPU - Usage: {cpu_memory.percent:.1f}%")
    
    # Your process specifically
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info().rss / 1e9  # Resident Set Size
    print(f"Your process - Memory: {process_memory:.2f} GB")
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)
        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -5, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    print("START kwarg: %2.5f"%(start))
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        theta_max,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        noisy_image = None,
        kappa_min = None,
        kappa_max = None,
        sigma_noise = None,
        survey_mask = None,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        tau_hat_sq = None,           # measured per-channel variance of x_start; used by pigdm_calibrated
        # GRF-DPS parameters (identity forward operator A=I)
        power_spectrum = None,       # (C, H, W//2+1) real tensor — Pκ(ℓ) in [-1,1] units
        y_fft = None,                # (C, H, W//2+1) complex tensor — rfft2(y, norm='ortho')
        y_observation_norm = None,   # (C, H, W) real tensor — y in [-1,1] units (for dps_identity)
        sigma_n_sq = None,           # float — per-pixel noise variance in [-1,1] units
        grf_guidance_scale = 1.0,    # scale factor on the GRF likelihood step
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.noisy_image = noisy_image
        self.kappa_min     = torch.tensor(kappa_min).to('cuda')
        self.kappa_max     = torch.tensor(kappa_max).to('cuda')
        self.sigma_noise   = sigma_noise
        self.survey_mask   = survey_mask
        # Per-channel variance of x_start (network's [-1,1] representation).
        # Used by pigdm_calibrated to correct the r_t^2 schedule.
        # Defaults to ones (= tau^2=1 assumption, same as pigdm_flat).
        _tau = (tau_hat_sq if tau_hat_sq is not None
                else torch.ones(self.channels))
        self.register_buffer('tau_hat_sq',
                             torch.as_tensor(_tau, dtype=torch.float32)
                             .reshape(self.channels, 1, 1))

        # GRF-DPS buffers — only registered when the method is used.
        # These are optional; other guidance methods leave them as None.
        self.sigma_n_sq        = float(sigma_n_sq) if sigma_n_sq is not None else None
        self.grf_guidance_scale = float(grf_guidance_scale)
        if power_spectrum is not None:
            self.register_buffer('power_spectrum',
                                 power_spectrum.float())
        else:
            self.power_spectrum = None
        if y_fft is not None:
            self.register_buffer('y_fft',
                                 y_fft.to(torch.complex64))
        else:
            self.y_fft = None
        if y_observation_norm is not None:
            self.register_buffer('y_observation_norm',
                                 y_observation_norm.float())
        else:
            self.y_observation_norm = None

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size
        self.n_grid     = image_size[0]
        self.theta_max  = theta_max
        self.map_tool   = TorchMapTools(self.n_grid, self.theta_max)
        self.n_pix = self.channels * self.image_size[0] * self.image_size[1]
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = True
        #self.is_ddim_sampling = self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = False)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    
    
    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]
        print(eta)
        x_start = None
        t_list = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if time in [980, 900, 800, 740, 700, 600, 500, 400]:
                t_list.append(time)
                imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        print(self.is_ddim_sampling)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        print('IS DDIM Sampling', self.is_ddim_sampling)
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    def unnorm_kappa(self, field):
        return (field * (self.kappa_max - self.kappa_min)) + self.kappa_min

    def torch_kappa_to_shear(self, kappa_batch, N_grid=256, theta_max=12., J=1j, EPS=1e-20):
        y_1_batch, y_2_batch = self.map_tool.do_fwd_KS_batch(kappa_batch)
        shear_batch = torch.stack((y_1_batch, y_2_batch), dim=1)
        return shear_batch

    def _internal_normalize(self, x):
        return 0.5 * (x + 1.)

    def compute_complex_log_likelihood(self, noisy_pred_batch, noisy_shear_map, sigma_noise, survey_mask):
        y_1_batch = noisy_pred_batch[:, 0]
        y_2_batch = noisy_pred_batch[:, 1]
    
        y_1_obs = noisy_shear_map[0]
        y_2_obs = noisy_shear_map[1]
    
        real_mse = -0.5 * survey_mask * (y_1_batch - y_1_obs)**2 / sigma_noise
        imag_mse = -0.5 * survey_mask * (y_2_batch - y_2_obs)**2 / sigma_noise
    
        mse_batch = real_mse + imag_mse
    
        total_loglkl = -mse_batch.sum()
    
        return total_loglkl

    def compute_log_lkl(self, shears, noisy_image, sigma_noise, survey_mask):
        loglkl = 0.
        for i in range(self.channels):
            loglkl += self.compute_complex_log_likelihood(shears[i], noisy_image[i], sigma_noise[i], survey_mask[i])
        return loglkl

    def compute_grad_log_lkl(self, x_t, x_start, noisy_image, sigma_noise_var, survey_mask):
        x_start = self._internal_normalize(x_start)
        kappa_map = self.unnorm_kappa(x_start)
    
        batch_size = x_start.shape[0]
        total_loglkl = 0.
   
        for i in range(self.channels):
            kappa_channel_i = kappa_map[:, i]
        
            shears_batch = self.torch_kappa_to_shear(kappa_channel_i)
       
            batch_loglkl = self.compute_complex_log_likelihood(
                shears_batch, 
                noisy_image[i], 
                sigma_noise_var[i], 
                survey_mask[i]
            )
        
            total_loglkl += batch_loglkl
    
        total_loglkl.backward(retain_graph=True)
        grad_log_lkl = x_t.grad

        return grad_log_lkl

    def pigdm_sigma_eff2(self, time, sigma_noise):
        """Effective noise variance for the ΠGDM flat-noise approximation.

        Approximates the extra covariance term r_t^2 * H H^T as a flat scalar
        per tomographic channel, valid in the no-mask limit where |F(l)| = 1
        for all modes l (a property of the Kaiser-Squires operator).

        The effective variance is:
            sigma_eff2_i = sigma_noise_i^2 + r_t^2 * c_i^2 / 2
        where c_i = 0.5 * (kappa_max_i - kappa_min_i) is the Jacobian of the
        affine unnorm_kappa transform, and r_t^2 = 1 - alphas_cumprod[t].

        Args:
            time: integer diffusion timestep
            sigma_noise: per-channel noise std, broadcastable to [channels, 1, 1]
        Returns:
            sigma_eff2: effective noise variance, shape [channels, 1, 1]
        """
        r_t_sq = 1. - self.alphas_cumprod[time]           # scalar
        c = 0.5 * (self.kappa_max - self.kappa_min)        # [channels, 1, 1]
        extra_noise = r_t_sq * c ** 2 / 2.
        return sigma_noise ** 2 + extra_noise

    def pigdm_r_t_sq(self, time):
        """Posterior variance of x_0 | x_t under the calibrated prior N(0, tau_hat_sq).

        General conjugate-Gaussian formula:
            r_t^2 = (v_t * tau_hat_sq) / (alpha_t * tau_hat_sq + v_t)

        where v_t = 1 - alphas_cumprod[t] and alpha_t = alphas_cumprod[t].

        When tau_hat_sq == 1 this reduces exactly to 1 - alphas_cumprod[t],
        recovering the pigdm_flat schedule as a special case.

        Returns shape [channels, 1, 1].
        """
        v_t     = 1. - self.alphas_cumprod[time]   # scalar
        alpha_t = self.alphas_cumprod[time]         # scalar
        return (v_t * self.tau_hat_sq) / (alpha_t * self.tau_hat_sq + v_t)

    def pigdm_sigma_eff2_calibrated(self, time, sigma_noise):
        """Effective noise variance using the calibrated r_t^2 schedule.

        Identical to pigdm_sigma_eff2 except r_t^2 is computed from the
        measured training-data variance tau_hat_sq rather than assuming
        tau_hat_sq = 1. Use guidance_method='pigdm_calibrated' to invoke this.

        Args:
            time: integer diffusion timestep
            sigma_noise: per-channel noise std, broadcastable to [channels, 1, 1]
        Returns:
            sigma_eff2: effective noise variance, shape [channels, 1, 1]
        """
        r_t_sq      = self.pigdm_r_t_sq(time)              # [channels, 1, 1]
        c           = 0.5 * (self.kappa_max - self.kappa_min)
        extra_noise = r_t_sq * c ** 2 / 2.
        return sigma_noise ** 2 + extra_noise

    def ddim_sample_posterior(self, shape, noisy_image, sigma_noise, survey_mask,
                               guidance_method="dps_sigmoid", return_all_timesteps=False):
        print("DDIM Sample Posterior called")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_t = torch.randn(shape, device = device)
        imgs = [x_t]
        t_list = []
        x_start = torch.zeros(shape, device = device)

        # GRF-DPS computes an analytic likelihood score; no backprop through the model needed.
        needs_backprop = guidance_method != 'grf_dps'

        ind = 0
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            if needs_backprop:
                x_t.requires_grad = True
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            if needs_backprop:
                pred_noise, x_start, *_ = self.model_predictions(x_t, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)
            else:
                with torch.no_grad():
                    pred_noise, x_start, *_ = self.model_predictions(x_t, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            alpha_t = alpha/alpha_next   # single-step alpha (ᾱt / ᾱ_{t-1})
            noise = torch.randn_like(x_t)

            if guidance_method == "grf_dps":
                # ----------------------------------------------------------
                # Analytic GRF likelihood score ∇_xt log pG(y|xt), Eq. 27.
                # αt_bar = sqrt(ᾱt) and σt² = 1 - ᾱt use the cumulative schedule.
                # See grf_score.py for the derivation; inlined here to avoid
                # import-path complexity.
                # ----------------------------------------------------------
                alpha_t_bar = self.sqrt_alphas_cumprod[time]        # αt in PDF notation
                sigma_t_sq  = 1.0 - self.alphas_cumprod[time]       # σt²
                x_t_fft     = torch.fft.rfft2(x_t, norm='ortho')    # (B, C, H, W//2+1)
                sigma_post  = 1.0 / (
                    1.0 / self.sigma_n_sq
                    + alpha_t_bar**2 / sigma_t_sq
                    + 1.0 / self.power_spectrum
                )
                mu_fft = sigma_post * (
                    self.y_fft / self.sigma_n_sq
                    + alpha_t_bar * x_t_fft / sigma_t_sq
                )
                score_fft = (
                    (alpha_t_bar / sigma_t_sq) * mu_fft
                    - x_t_fft / sigma_t_sq
                    + x_t_fft / (alpha_t_bar**2 * self.power_spectrum + sigma_t_sq)
                )
                grf_score = torch.fft.irfft2(
                    score_fft, s=(x_t.shape[-2], x_t.shape[-1]), norm='ortho'
                )
            elif guidance_method == "dps_identity":
                # ----------------------------------------------------------
                # DPS with identity forward model (A = I): y = x0 + n.
                # Backprop the MSE residual through the Tweedie denoiser.
                # Sign convention: grad_log_lkl = ∇_xt [sum (y-x_hat_0)²/σn²]
                # is the gradient of the *negative* log-likelihood, so
                # x_t -= (1-alpha_t)*grad_log_lkl is a gradient-ascent step.
                # ----------------------------------------------------------
                residual_sq_sum = (
                    (self.y_observation_norm.unsqueeze(0) - x_start) ** 2
                    / self.sigma_n_sq
                ).sum()
                residual_sq_sum.backward(retain_graph=True)
                grad_log_lkl = x_t.grad
            elif guidance_method == "pigdm_flat":
                sigma_eff2 = self.pigdm_sigma_eff2(time, sigma_noise)
                grad_log_lkl = self.compute_grad_log_lkl(x_t, x_start, noisy_image, sigma_eff2, survey_mask)
            elif guidance_method == "pigdm_calibrated":
                sigma_eff2 = self.pigdm_sigma_eff2_calibrated(time, sigma_noise)
                grad_log_lkl = self.compute_grad_log_lkl(x_t, x_start, noisy_image, sigma_eff2, survey_mask)
            else:  # "dps_sigmoid" or "dps_sigmoid_annealed"
                lkl_scale = self.dps_lkl_scale(time)
                grad_log_lkl = self.compute_grad_log_lkl(x_t, x_start, noisy_image, sigma_noise**2, survey_mask)

            if(time > 0):
                with torch.no_grad():
                    x_t = (x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise)
                    if guidance_method == "grf_dps":
                        # grf_score = ∇_xt log pG(y|xt): add to ascend the likelihood.
                        x_t = x_t + self.grf_guidance_scale * (1 - alpha_t) * grf_score
                    elif guidance_method == "dps_identity":
                        x_t = x_t - (1 - alpha_t) * grad_log_lkl.detach()
                    elif guidance_method in ("pigdm_flat", "pigdm_calibrated"):
                        x_t = x_t - (1 - alpha_t) * grad_log_lkl.detach()
                    else:  # "dps_sigmoid"
                        x_t = x_t - lkl_scale * (1 - alpha_t) * grad_log_lkl.detach()
                    t_list.append(time)
                    if return_all_timesteps:
                        imgs.append(x_t)

            else:
                with torch.no_grad():
                    x_t = (x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise)
            ind += 1

        ret = x_t if not return_all_timesteps else torch.stack(imgs, dim = 1)
        self.t_list = t_list
        ret = self.unnormalize(ret)
        return ret
    
    def set_dps_lkl_scale(self, delta_t, sigma_t):
        self.delta_t = torch.tensor(delta_t, dtype=torch.float32).to(self.device)
        self.sigma_t = torch.tensor(sigma_t, dtype=torch.float32).to(self.device)
        print("delta_t set to: "+str(self.delta_t))
        print("sigma_t set to: "+str(self.sigma_t))


    def dps_lkl_scale(self, time):
        lkl_scale = 1. / (1. + torch.exp(- (-time + self.delta_t) / self.sigma_t))
        return lkl_scale.view(1, self.channels, 1, 1)

    def sample_posterior(self, batch_size=16, guidance_method="dps_sigmoid", return_all_timesteps=False):
        print("Sample Posterior called")
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample_posterior
        return sample_fn((batch_size, channels, h, w), self.noisy_image, self.sigma_noise,
                         self.survey_mask, guidance_method=guidance_method,
                         return_all_timesteps=return_all_timesteps)

    # ------------------------------------------------------------------
    # TDS (Twisted Diffusion Sampler) helpers and main loop
    # ------------------------------------------------------------------

    def _log_lkl_per_particle_channel(self, shear_pred, obs_shear, sigma_noise_sq, mask):
        """Gaussian log-likelihood per particle for one tomographic bin.

        Sums over (H, W) but keeps the particle (batch) dimension.

        Args:
            shear_pred:    predicted shear [K, 2, H, W]
            obs_shear:     observed shear  [2, H, W]
            sigma_noise_sq: per-pixel noise variance [H, W]
            mask:          survey mask [H, W]

        Returns:
            log_lkl: [K]
        """
        chi2 = (
            -0.5 * mask * (shear_pred[:, 0] - obs_shear[0]) ** 2 / sigma_noise_sq
            - 0.5 * mask * (shear_pred[:, 1] - obs_shear[1]) ** 2 / sigma_noise_sq
        )  # [K, H, W]
        return chi2.sum(dim=(-2, -1))  # [K]

    def _compute_log_lkl_per_particle(self, x_start, sigma_eff2=None):
        """Twisting function: log p(d | x_hat_0) evaluated per particle.

        Takes the Tweedie estimate in network space ([-1, 1]), converts to
        physical kappa via the stored normalisation, applies the KS forward
        operator, and evaluates the Gaussian shear likelihood.

        Args:
            x_start:    Tweedie estimate [K, C, H, W] in [-1, 1]
            sigma_eff2: effective noise variance per channel [C, 1, 1], or None.
                        If None, uses self.sigma_noise[i]**2 (DPS / fixed noise).
                        Pass pigdm_sigma_eff2(time, self.sigma_noise) for the
                        ΠGDM time-dependent twisting function.

        Returns:
            log_lkl: [K] per-particle log-likelihood (no gradient)
        """
        with torch.no_grad():
            x_norm  = self._internal_normalize(x_start)   # [-1,1] -> [0,1]
            kappa   = self.unnorm_kappa(x_norm)            # [0,1]  -> physical
            log_lkl = torch.zeros(x_start.shape[0], device=x_start.device)
            for i in range(self.channels):
                noise_var = sigma_eff2[i] if sigma_eff2 is not None else self.sigma_noise[i] ** 2
                shears    = self.torch_kappa_to_shear(kappa[:, i])   # [K, 2, H, W]
                log_lkl  += self._log_lkl_per_particle_channel(
                    shears,
                    self.noisy_image[i],
                    noise_var,
                    self.survey_mask[i],
                )
        return log_lkl

    def _compute_dps_scaled_log_lkl_per_particle(self, x_start, time):
        """Sigmoid-annealed twisting function for TDS + DPS.

        Computes lkl_scale(t) * log p(y | x_hat_0; sigma_noise²), where
        lkl_scale(t) is the same per-channel sigmoid schedule used in the
        DPS proposal gradient.  This ensures the SMC twisting function is
        consistent with the proposal, unlike the plain 'dps_sigmoid' mode
        which only scales the proposal but uses an unscaled twisting function.

        Requires set_dps_lkl_scale() to have been called beforehand.

        Args:
            x_start: Tweedie estimate [K, C, H, W] in [-1, 1]
            time:    integer diffusion timestep (used to evaluate lkl_scale)

        Returns:
            log_lkl: [K] scaled per-particle log-likelihood (no gradient)
        """
        lkl_scale = self.dps_lkl_scale(time)   # [1, C, 1, 1]
        with torch.no_grad():
            x_norm  = self._internal_normalize(x_start)
            kappa   = self.unnorm_kappa(x_norm)
            log_lkl = torch.zeros(x_start.shape[0], device=x_start.device)
            for i in range(self.channels):
                shears      = self.torch_kappa_to_shear(kappa[:, i])
                channel_lkl = self._log_lkl_per_particle_channel(
                    shears,
                    self.noisy_image[i],
                    self.sigma_noise[i] ** 2,
                    self.survey_mask[i],
                )
                log_lkl += lkl_scale[0, i, 0, 0] * channel_lkl
        return log_lkl

    @staticmethod
    def _compute_ess(log_weights):
        """Effective sample size from unnormalised log-weights.

        ESS = (sum w)^2 / sum(w^2), computed stably in log-space.
        Returns a scalar in [1, K].
        """
        log_w = log_weights - log_weights.max()
        w     = log_w.exp()
        return (w.sum() ** 2 / (w ** 2).sum()).item()

    @staticmethod
    def _systematic_resample(log_weights):
        """Systematic resampling from unnormalised log-weights.

        Returns an index tensor of shape [K] suitable for indexing into the
        particle batch dimension.
        """
        K     = log_weights.shape[0]
        log_w = log_weights - torch.logsumexp(log_weights, dim=0)
        w     = log_w.exp()
        cumw  = torch.cumsum(w, dim=0)

        # One random offset shared across all K strata
        u         = torch.rand(1, device=log_weights.device) / K
        positions = u + torch.arange(K, device=log_weights.device, dtype=torch.float32) / K
        indices   = torch.searchsorted(cumw.contiguous(), positions.contiguous())
        return indices.clamp(0, K - 1)

    def ddim_sample_tds(self, shape, noisy_image, sigma_noise, survey_mask,
                        ess_threshold=0.95, grad_batch_size=None,
                        guidance_method="dps_sigmoid", return_all_timesteps=False):
        """Twisted Diffusion Sampler (TDS) via SMC particle filtering.

        Wraps the existing DDIM-DPS proposal inside an SMC loop with
        exact importance weights.

        Three guidance methods are supported via `guidance_method`:

        "dps_sigmoid" (default):
            Proposal gradient scaled by dps_lkl_scale(t); twisting function
            uses fixed sigma_noise**2 (time-independent).  The proposal and
            twisting function are inconsistent in their time-dependence.

        "dps_sigmoid_annealed":
            Proposal gradient scaled by dps_lkl_scale(t) (same as above).
            Twisting function is also sigmoid-annealed:
                psi_t(x_hat_0) = lkl_scale(t) * log p(y | x_hat_0; sigma_noise²)
            This makes the SMC weights consistent with the time-annealed
            proposal and is the principled TDS variant for DPS-style guidance.
            Requires set_dps_lkl_scale() to have been called.

        "pigdm_flat":
            The twisting function uses the ΠGDM effective noise variance
                sigma_eff2_i(t) = sigma_noise_i**2 + (1-alphabar_t) * c_i**2 / 2
            where c_i = 0.5*(kappa_max_i - kappa_min_i) is the affine-transform
            Jacobian.  This is a principled, schedule-free approximation that
            accounts for the uncertainty in x_hat_0 given x_t.  The
            dps_lkl_scale factor is dropped from the proposal shift (it is
            already absorbed into sigma_eff2).

        "pigdm_calibrated":
            Like "pigdm_flat" but uses the measured tau_hat_sq (per-channel
            variance of x_start) instead of assuming tau²=1.

        Args:
            shape:           (K, C, H, W) — K is the number of particles
            noisy_image:     observed shear [n_tomo, 2, H, W]
            sigma_noise:     per-pixel noise std [n_tomo, H, W]
            survey_mask:     binary mask [n_tomo, H, W]
            ess_threshold:   resample when ESS < ess_threshold * K (default 0.5)
            guidance_method: "dps_sigmoid", "dps_sigmoid_annealed", "pigdm_flat",
                             or "pigdm_calibrated"
            return_all_timesteps: if True, also return intermediate states

        Returns:
            samples:    unnormalised kappa maps [K, C, H, W]
            log_weights: final unnormalised log-weights [K]
        """
        K, device = shape[0], self.device
        total_timesteps    = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta                = self.ddim_sampling_eta

        # grad_batch_size controls how many particles are processed per backward
        # pass.  Defaults to K (no sub-batching).  Set it smaller to trade
        # runtime for lower peak GPU memory when K is large.
        if grad_batch_size is None:
            grad_batch_size = K

        times      = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times      = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # ---- Initialise K particles from the prior ----
        x_t = torch.randn(shape, device=device)

        # Initial twisting function at t=T, sub-batched to control memory
        if guidance_method == "pigdm_flat":
            sigma_eff2_T = self.pigdm_sigma_eff2(times[0], sigma_noise)
        elif guidance_method == "pigdm_calibrated":
            sigma_eff2_T = self.pigdm_sigma_eff2_calibrated(times[0], sigma_noise)
        else:
            sigma_eff2_T = None
        log_w_parts = []
        with torch.no_grad():
            for b0 in range(0, K, grad_batch_size):
                b1     = min(b0 + grad_batch_size, K)
                tc_sub = torch.full((b1 - b0,), times[0], device=device, dtype=torch.long)
                _, x_start_sub = self.model_predictions(
                    x_t[b0:b1], tc_sub, clip_x_start=True, rederive_pred_noise=True
                )
                if guidance_method == "dps_sigmoid_annealed":
                    log_w_parts.append(
                        self._compute_dps_scaled_log_lkl_per_particle(x_start_sub, times[0])
                    )
                else:
                    log_w_parts.append(
                        self._compute_log_lkl_per_particle(x_start_sub, sigma_eff2=sigma_eff2_T)
                    )
        log_w = torch.cat(log_w_parts)

        ess_history = []
        imgs        = [x_t] if return_all_timesteps else []

        for time, time_next in tqdm(time_pairs, desc='TDS sampling loop'):
            # ---- Conditionally resample ----
            ess = self._compute_ess(log_w)
            ess_history.append(ess)

            if ess < ess_threshold * K:
                indices = self._systematic_resample(log_w)
                x_t     = x_t[indices].clone()
                log_w   = torch.zeros(K, device=device)

            # Effective noise variance at the current and next timestep.
            # For "dps_sigmoid" both are None (fixed sigma_noise^2 used inside
            # _compute_log_lkl_per_particle).  For "pigdm_flat" the twisting
            # function is time-dependent: sigma_eff2(t) is used for log_twist_old
            # and the proposal gradient, sigma_eff2(t_next) for log_twist_new.
            if guidance_method == "pigdm_flat":
                sigma_eff2_t      = self.pigdm_sigma_eff2(time,      sigma_noise)
                sigma_eff2_t_next = self.pigdm_sigma_eff2(time_next, sigma_noise) if time_next >= 0 else None
            elif guidance_method == "pigdm_calibrated":
                sigma_eff2_t      = self.pigdm_sigma_eff2_calibrated(time,      sigma_noise)
                sigma_eff2_t_next = self.pigdm_sigma_eff2_calibrated(time_next, sigma_noise) if time_next >= 0 else None
            else:
                sigma_eff2_t      = None
                sigma_eff2_t_next = None

            # ---- Sub-batched model forward + proposal gradient ----
            # Each sub-batch of grad_batch_size particles gets its own
            # forward+backward pass, keeping peak activation memory fixed
            # regardless of K.
            x_start_parts       = []
            pred_noise_parts    = []
            grad_parts          = []
            log_twist_old_parts = []

            for b0 in range(0, K, grad_batch_size):
                b1      = min(b0 + grad_batch_size, K)
                x_t_sub = x_t[b0:b1].detach().requires_grad_(True)
                tc_sub  = torch.full((b1 - b0,), time, device=device, dtype=torch.long)

                pred_noise_sub, x_start_sub = self.model_predictions(
                    x_t_sub, tc_sub, clip_x_start=True, rederive_pred_noise=True
                )

                if guidance_method == "dps_sigmoid_annealed":
                    log_twist_old_parts.append(
                        self._compute_dps_scaled_log_lkl_per_particle(x_start_sub.detach(), time)
                    )
                else:
                    log_twist_old_parts.append(
                        self._compute_log_lkl_per_particle(x_start_sub.detach(), sigma_eff2=sigma_eff2_t)
                    )

                if time > 0:
                    grad_noise_var = sigma_eff2_t if guidance_method in ("pigdm_flat", "pigdm_calibrated") else sigma_noise ** 2
                    grad_sub = self.compute_grad_log_lkl(
                        x_t_sub, x_start_sub, noisy_image, grad_noise_var, survey_mask
                    )
                    grad_parts.append(grad_sub.detach())
                else:
                    grad_parts.append(torch.zeros_like(x_t_sub))

                x_start_parts.append(x_start_sub.detach())
                pred_noise_parts.append(pred_noise_sub.detach())

            x_start       = torch.cat(x_start_parts)
            pred_noise    = torch.cat(pred_noise_parts)
            grad_log_lkl  = torch.cat(grad_parts)
            log_twist_old = torch.cat(log_twist_old_parts)

            # ---- Terminal step: x_0 = x_start, no stochastic proposal ----
            # At t=0, r_t^2 = 0, so sigma_eff2 = sigma_noise^2 for both methods.
            if time_next < 0:
                x_t = x_start
                log_twist_new = self._compute_log_lkl_per_particle(x_t, sigma_eff2=None)
                log_w = log_w + log_twist_new - log_twist_old
                break

            # ---- DDIM step parameters ----
            alpha      = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            alpha_t    = alpha / alpha_next

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c     = (1 - alpha_next - sigma ** 2).sqrt()

            # x_start and pred_noise are already detached; mu_uncond has no grad
            mu_uncond = x_start * alpha_next.sqrt() + c * pred_noise
            noise     = torch.randn(shape, device=device)

            if time > 0 and sigma > 1e-8:
                if guidance_method in ("pigdm_flat", "pigdm_calibrated"):
                    # No sigmoid rescaling — sigma_eff2 already encodes the
                    # time-dependent attenuation of the likelihood gradient.
                    lkl_shift = (1 - alpha_t) * grad_log_lkl
                else:  # "dps_sigmoid" or "dps_sigmoid_annealed"
                    lkl_scale = self.dps_lkl_scale(time)
                    lkl_shift = lkl_scale * (1 - alpha_t) * grad_log_lkl

                mu_proposal = mu_uncond - lkl_shift
                x_next      = mu_proposal + sigma * noise

                # log p_theta(x_next|x_t) - log q(x_next|x_t,d)
                # x_next - mu_uncond   = sigma*noise - lkl_shift
                # x_next - mu_proposal = sigma*noise
                # ratio = [(noise·lkl_shift)/sigma - ||lkl_shift||²/(2σ²)]  per particle
                dot_noise_shift   = (noise * lkl_shift).sum(dim=(1, 2, 3))   # [K]
                dot_sq_shift      = (lkl_shift ** 2).sum(dim=(1, 2, 3))      # [K]
                log_density_ratio = dot_noise_shift / sigma - dot_sq_shift / (2 * sigma ** 2)
            else:
                x_next            = mu_uncond + sigma * noise
                log_density_ratio = torch.zeros(K, device=device)

            # ---- New twisting function at proposed x_next, sub-batched ----
            # Uses sigma_eff2 at time_next (different from the current step).
            log_twist_new_parts = []
            with torch.no_grad():
                for b0 in range(0, K, grad_batch_size):
                    b1     = min(b0 + grad_batch_size, K)
                    tc_sub = torch.full((b1 - b0,), time_next, device=device, dtype=torch.long)
                    _, x_start_next_sub = self.model_predictions(
                        x_next[b0:b1], tc_sub, clip_x_start=True, rederive_pred_noise=True
                    )
                    if guidance_method == "dps_sigmoid_annealed":
                        log_twist_new_parts.append(
                            self._compute_dps_scaled_log_lkl_per_particle(x_start_next_sub, time_next)
                        )
                    else:
                        log_twist_new_parts.append(
                            self._compute_log_lkl_per_particle(x_start_next_sub, sigma_eff2=sigma_eff2_t_next)
                        )
            log_twist_new = torch.cat(log_twist_new_parts)

            # ---- Update log-weights ----
            log_w = log_w + log_density_ratio + log_twist_new - log_twist_old

            x_t = x_next.detach()
            if return_all_timesteps:
                imgs.append(x_t)

        self.tds_ess_history = ess_history

        ret = self.unnormalize(x_t)
        if return_all_timesteps:
            ret = torch.stack([self.unnormalize(img) for img in imgs], dim=1)

        return ret, log_w.detach()

    def sample_tds(self, n_particles=8, grad_batch_size=None,
                   ess_threshold=0.5, guidance_method="dps_sigmoid",
                   return_all_timesteps=False):
        """Draw posterior samples using the Twisted Diffusion Sampler.

        Args:
            n_particles:     number of SMC particles (K=1 reduces to DPS)
            grad_batch_size: particles per backward pass; controls peak GPU
                             memory.  None means process all K at once.
            ess_threshold:   resample when ESS < ess_threshold * K
            guidance_method: "dps_sigmoid" or "pigdm_flat" — controls both the
                             proposal gradient and the twisting function used in
                             the importance weights (see ddim_sample_tds docstring)
            return_all_timesteps: return intermediate states too

        Returns:
            samples:     [K, C, H, W] unnormalised kappa maps
            log_weights: [K] final log-importance weights
        """
        (h, w), channels = self.image_size, self.channels
        shape = (n_particles, channels, h, w)
        return self.ddim_sample_tds(
            shape, self.noisy_image, self.sigma_noise, self.survey_mask,
            ess_threshold=ess_threshold,
            grad_batch_size=grad_batch_size,
            guidance_method=guidance_method,
            return_all_timesteps=return_all_timesteps,
        )

    def _grf_marginal_lkl_per_particle(self, x_t, time):
        """Analytic GRF-marginalized log-likelihood log p_G(y | x_t).

        Marginalizes x_0 out under the GRF prior using the exact Gaussian
        convolution formula in Fourier space.  No neural network is invoked.

        The normalization constant (which depends on t but not on x_t) is
        dropped: it contributes identically to all K particles at a given t
        and cancels after weight normalization.

        Args:
            x_t:  [K, C, H, W] particles in [-1, 1] normalized space.
            time: int, current diffusion timestep.

        Returns:
            log_lkl: [K] per-particle log-likelihoods (unnormalized).
        """
        alpha_t    = self.sqrt_alphas_cumprod[time]          # scalar
        sigma_t_sq = 1.0 - self.alphas_cumprod[time]         # scalar

        # GRF posterior variance of x_0 | x_t, per Fourier mode.
        # Written to avoid division by zero when sigma_t_sq → 0.
        # Shape: [C, H, W//2+1]
        Gamma_t   = sigma_t_sq * self.power_spectrum / (
            sigma_t_sq + alpha_t ** 2 * self.power_spectrum
        )

        # Prediction of y in Fourier space:
        # ρ_ℓ = (α_t P_κ_ℓ / (σ_t² + α_t² P_κ_ℓ)) * x_t_ℓ
        # Numerically stable form avoids dividing by sigma_t_sq.
        W_pred   = alpha_t * self.power_spectrum / (
            sigma_t_sq + alpha_t ** 2 * self.power_spectrum
        )  # [C, H, W//2+1]
        x_t_fft  = torch.fft.rfft2(x_t, norm='ortho')        # [K, C, H, W//2+1]
        rho_fft  = W_pred.unsqueeze(0) * x_t_fft             # [K, C, H, W//2+1]

        # Total variance per mode and residual.
        total_var = Gamma_t + self.sigma_n_sq                 # [C, H, W//2+1]
        resid_fft = self.y_fft.unsqueeze(0) - rho_fft        # [K, C, H, W//2+1]

        # Unnormalized log-likelihood: −Σ_ℓ |r_ℓ|² / (2 V_ℓ)
        log_lkl = -(
            resid_fft.abs() ** 2 / (2.0 * total_var.unsqueeze(0))
        ).sum(dim=(1, 2, 3))
        return log_lkl

    def ddim_sample_tds_grf(self, shape, ess_threshold=0.5, guidance_scale=1.0,
                            return_all_timesteps=False):
        """TDS using the analytic GRF-marginalized likelihood as twisting function.

        Replaces the standard TDS Tweedie twisting function ψ_t(x̂₀) with the
        exact GRF-marginalized likelihood p_G(y | x_t), which is analytic in
        Fourier space.  This removes the need for backpropagation entirely —
        both the proposal gradient and the twisting function evaluations are
        computed analytically, at the cost of assuming a GRF prior on x_0.

        The proposal uses the same analytic GRF score as in the grf_dps method.

        The importance weight increment per step is:
            Δ log w = log p_uncond(x_next | x_t) − log q(x_next | x_t)
                    + log p_G(y | x_next) − log p_G(y | x_t)

        where q is the GRF-guided DDIM proposal and the density ratio is
        computed analytically from the Gaussian DDIM transition kernels.

        Args:
            shape:               (K, C, H, W) — K is the number of particles.
            ess_threshold:       resample when ESS < ess_threshold * K.
            guidance_scale:      multiplier on the GRF analytic score (default 1.0).
            return_all_timesteps: if True, also return intermediate states.

        Returns:
            samples:     [K, C, H, W] unnormalized kappa maps.
            log_weights: [K] final unnormalized log-importance weights.
        """
        assert self.power_spectrum is not None, "power_spectrum must be set for TDS-GRF"
        assert self.y_fft         is not None, "y_fft must be set for TDS-GRF"
        assert self.sigma_n_sq    is not None, "sigma_n_sq must be set for TDS-GRF"

        K, device = shape[0], self.device
        total_timesteps    = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta                = self.ddim_sampling_eta

        times      = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times      = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Initialise K particles from the prior.
        x_t = torch.randn(shape, device=device)

        # Initial twisting function at t = T (analytic, no NN).
        with torch.no_grad():
            log_w = self._grf_marginal_lkl_per_particle(x_t, times[0])

        ess_history = []
        imgs        = [x_t] if return_all_timesteps else []

        for time, time_next in tqdm(time_pairs, desc='TDS-GRF sampling loop'):
            # ---- Conditionally resample ----
            ess = self._compute_ess(log_w)
            ess_history.append(ess)

            if ess < ess_threshold * K:
                indices = self._systematic_resample(log_w)
                x_t     = x_t[indices].clone()
                log_w   = torch.zeros(K, device=device)

            # ---- NN forward pass (unconditional score, no grad) ----
            time_cond = torch.full((K,), time, device=device, dtype=torch.long)
            with torch.no_grad():
                pred_noise, x_start = self.model_predictions(
                    x_t, time_cond, clip_x_start=True, rederive_pred_noise=True
                )

            # ---- Old twisting function ψ_t(x_t) ----
            with torch.no_grad():
                log_twist_old = self._grf_marginal_lkl_per_particle(x_t, time)

            # ---- Terminal step: x_0 = x_start, final weight update ----
            if time_next < 0:
                x_t = x_start
                # At t=0: Γ_t → 0, total_var → σ_n², W_pred → 1 (α_t→1, σ_t²→0).
                with torch.no_grad():
                    log_twist_new = self._grf_marginal_lkl_per_particle(x_t, time)
                log_w = log_w + log_twist_new - log_twist_old
                break

            # ---- DDIM parameters ----
            alpha      = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            alpha_t    = alpha / alpha_next   # single-step α

            sigma = eta * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # Unconditional DDIM mean.
            mu_uncond = x_start * alpha_next.sqrt() + c * pred_noise  # [K, C, H, W]

            # ---- Analytic GRF score ∇_{x_t} log p_G(x_t | y) ----
            # Same formula as the grf_dps branch in ddim_sample_posterior.
            with torch.no_grad():
                alpha_t_bar = self.sqrt_alphas_cumprod[time]
                sigma_t_sq  = 1.0 - self.alphas_cumprod[time]
                x_t_fft     = torch.fft.rfft2(x_t, norm='ortho')
                sigma_post  = 1.0 / (
                    1.0 / self.sigma_n_sq
                    + alpha_t_bar ** 2 / sigma_t_sq
                    + 1.0 / self.power_spectrum
                )
                mu_fft = sigma_post * (
                    self.y_fft / self.sigma_n_sq
                    + alpha_t_bar * x_t_fft / sigma_t_sq
                )
                score_fft = (
                    (alpha_t_bar / sigma_t_sq) * mu_fft
                    - x_t_fft / sigma_t_sq
                    + x_t_fft / (alpha_t_bar ** 2 * self.power_spectrum + sigma_t_sq)
                )
                grf_score = torch.fft.irfft2(
                    score_fft, s=(shape[-2], shape[-1]), norm='ortho'
                )  # [K, C, H, W]

            # ---- Guided proposal step ----
            noise     = torch.randn(shape, device=device)

            if time > 0 and sigma > 1e-8:
                # Proposal: ascend the GRF log-likelihood.
                grf_shift   = guidance_scale * (1 - alpha_t) * grf_score  # [K, C, H, W]
                mu_proposal = mu_uncond + grf_shift
                x_next      = mu_proposal + sigma * noise

                # log p_uncond(x_next | x_t) − log q(x_next | x_t)
                # = −noise·grf_shift/σ − ||grf_shift||²/(2σ²)
                dot_noise_shift   = (noise * grf_shift).sum(dim=(1, 2, 3))   # [K]
                dot_sq_shift      = (grf_shift ** 2).sum(dim=(1, 2, 3))      # [K]
                log_density_ratio = (
                    -dot_noise_shift / sigma - dot_sq_shift / (2 * sigma ** 2)
                )
            else:
                x_next            = mu_uncond + sigma * noise
                log_density_ratio = torch.zeros(K, device=device)

            # ---- New twisting function ψ_{t−1}(x_next) ----
            with torch.no_grad():
                log_twist_new = self._grf_marginal_lkl_per_particle(x_next, time_next)

            # ---- Weight update ----
            log_w = log_w + log_density_ratio + log_twist_new - log_twist_old

            x_t = x_next.detach()
            if return_all_timesteps:
                imgs.append(x_t)

        self.tds_ess_history = ess_history

        ret = self.unnormalize(x_t)
        if return_all_timesteps:
            ret = torch.stack([self.unnormalize(img) for img in imgs], dim=1)
        return ret, log_w.detach()

    def sample_tds_grf(self, n_particles=8, ess_threshold=0.5, guidance_scale=1.0,
                       return_all_timesteps=False):
        """Draw posterior samples using TDS with the analytic GRF twisting function.

        Args:
            n_particles:          number of SMC particles K.
            ess_threshold:        resample when ESS < ess_threshold * K.
            guidance_scale:       multiplier on the GRF analytic score.
            return_all_timesteps: return intermediate states too.

        Returns:
            samples:     [K, C, H, W] unnormalized kappa maps.
            log_weights: [K] final log-importance weights.
        """
        (h, w), channels = self.image_size, self.channels
        shape = (n_particles, channels, h, w)
        return self.ddim_sample_tds_grf(
            shape,
            ess_threshold=ess_threshold,
            guidance_scale=guidance_scale,
            return_all_timesteps=return_all_timesteps,
        )

    def _compute_identity_lkl_per_particle(self, x_start):
        """Twisting function for TDS with identity forward model.

        Evaluates log p(y | x̂₀) = −||y − x̂₀||² / (2 σ_n²) per particle,
        where y is the pixel-space observation (self.y_observation_norm) and
        x̂₀ is the Tweedie denoised estimate.  No gradient is tracked.

        Args:
            x_start: Tweedie estimate [K, C, H, W] in [-1, 1] normalized space.

        Returns:
            log_lkl: [K] per-particle log-likelihoods (no gradient).
        """
        with torch.no_grad():
            resid   = self.y_observation_norm.unsqueeze(0) - x_start   # [K, C, H, W]
            log_lkl = -(resid ** 2).sum(dim=(1, 2, 3)) / (2.0 * self.sigma_n_sq)
        return log_lkl

    def ddim_sample_tds_dps_identity(self, shape, ess_threshold=0.5,
                                     return_all_timesteps=False):
        """TDS with identity forward model and DPS (Tweedie) twisting function.

        This is the SMC-corrected version of the DPS-identity sampler.  The
        twisting function is the pixel-space Gaussian likelihood evaluated at
        the Tweedie denoised estimate,

            ψ_t(x_t) = log p(y | x̂₀(x_t, t)) = −||y − x̂₀||² / (2 σ_n²),

        which is identical to the DPS-identity proposal objective.  The SMC
        importance weights correct for the approximation error in using this as
        a proxy for the true marginal likelihood p_θ(y | x_t).

        Both the proposal gradient (backprop of ||y − x̂₀||² through the NN)
        and the new-state twisting function (separate no-grad forward pass at
        x_next) require a NN forward pass per step.

        Args:
            shape:               (K, C, H, W) — K is the number of particles.
            ess_threshold:       resample when ESS < ess_threshold * K.
            return_all_timesteps: if True, also return intermediate states.

        Returns:
            samples:     [K, C, H, W] unnormalized kappa maps.
            log_weights: [K] final unnormalized log-importance weights.
        """
        assert self.y_observation_norm is not None, \
            "y_observation_norm must be set for TDS-DPS-identity"
        assert self.sigma_n_sq is not None, \
            "sigma_n_sq must be set for TDS-DPS-identity"

        K, device = shape[0], self.device
        total_timesteps    = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta                = self.ddim_sampling_eta

        times      = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times      = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Initialise K particles from the prior.
        x_t = torch.randn(shape, device=device)

        # Initial twisting function at t = T: needs NN forward pass for x̂₀.
        with torch.no_grad():
            tc_init = torch.full((K,), times[0], device=device, dtype=torch.long)
            _, x_start_init = self.model_predictions(
                x_t, tc_init, clip_x_start=True, rederive_pred_noise=True
            )
        log_w = self._compute_identity_lkl_per_particle(x_start_init)

        ess_history = []
        imgs        = [x_t] if return_all_timesteps else []

        for time, time_next in tqdm(time_pairs, desc='TDS-DPS-identity sampling loop'):
            # ---- Conditionally resample ----
            ess = self._compute_ess(log_w)
            ess_history.append(ess)

            if ess < ess_threshold * K:
                indices = self._systematic_resample(log_w)
                x_t     = x_t[indices].clone()
                log_w   = torch.zeros(K, device=device)

            # ---- NN forward pass with backprop for DPS proposal gradient ----
            x_t_grad  = x_t.detach().requires_grad_(True)
            time_cond = torch.full((K,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x_t_grad, time_cond, clip_x_start=True, rederive_pred_noise=True
            )

            # Old twisting function ψ_t(x_t) — value only, detached.
            log_twist_old = self._compute_identity_lkl_per_particle(x_start.detach())

            # ---- Terminal step: x_0 = x_start, final weight update ----
            if time_next < 0:
                x_t = x_start.detach()
                log_twist_new = self._compute_identity_lkl_per_particle(x_t)
                log_w = log_w + log_twist_new - log_twist_old
                break

            # ---- DPS proposal gradient via backprop ----
            # grad_log_lkl = ∂||y − x̂₀||²/∂x_t  (gradient of negative log-likelihood)
            residual_sq = (
                (self.y_observation_norm.unsqueeze(0) - x_start) ** 2
                / self.sigma_n_sq
            ).sum()
            residual_sq.backward()
            grad_log_lkl = x_t_grad.grad.detach()   # [K, C, H, W]

            # ---- DDIM step parameters ----
            alpha      = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            alpha_t    = alpha / alpha_next

            sigma = eta * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            mu_uncond = (
                x_start.detach() * alpha_next.sqrt() + c * pred_noise.detach()
            )   # [K, C, H, W]
            noise = torch.randn(shape, device=device)

            if time > 0 and sigma > 1e-8:
                # Proposal: descend the residual (= ascend the log-likelihood).
                # Same sign convention as ddim_sample_tds.
                lkl_shift   = (1 - alpha_t) * grad_log_lkl
                mu_proposal = mu_uncond - lkl_shift
                x_next      = mu_proposal + sigma * noise

                # log p_uncond(x_next | x_t) − log q(x_next | x_t)
                dot_noise_shift   = (noise * lkl_shift).sum(dim=(1, 2, 3))   # [K]
                dot_sq_shift      = (lkl_shift ** 2).sum(dim=(1, 2, 3))      # [K]
                log_density_ratio = (
                    dot_noise_shift / sigma - dot_sq_shift / (2 * sigma ** 2)
                )
            else:
                x_next            = mu_uncond + sigma * noise
                log_density_ratio = torch.zeros(K, device=device)

            # ---- New twisting function ψ_{t−1}(x_next) — separate no-grad pass ----
            with torch.no_grad():
                tc_next = torch.full((K,), time_next, device=device, dtype=torch.long)
                _, x_start_next = self.model_predictions(
                    x_next, tc_next, clip_x_start=True, rederive_pred_noise=True
                )
            log_twist_new = self._compute_identity_lkl_per_particle(x_start_next)

            # ---- Weight update ----
            log_w = log_w + log_density_ratio + log_twist_new - log_twist_old

            x_t = x_next.detach()
            if return_all_timesteps:
                imgs.append(x_t)

        self.tds_ess_history = ess_history

        ret = self.unnormalize(x_t)
        if return_all_timesteps:
            ret = torch.stack([self.unnormalize(img) for img in imgs], dim=1)
        return ret, log_w.detach()

    def sample_tds_dps_identity(self, n_particles=8, ess_threshold=0.5,
                                return_all_timesteps=False):
        """Draw posterior samples using TDS with the DPS identity twisting function.

        Args:
            n_particles:          number of SMC particles K.
            ess_threshold:        resample when ESS < ess_threshold * K.
            return_all_timesteps: return intermediate states too.

        Returns:
            samples:     [K, C, H, W] unnormalized kappa maps.
            log_weights: [K] final log-importance weights.
        """
        (h, w), channels = self.image_size, self.channels
        shape = (n_particles, channels, h, w)
        return self.ddim_sample_tds_dps_identity(
            shape,
            ess_threshold=ess_threshold,
            return_all_timesteps=return_all_timesteps,
        )

    def compute_maximum_likelihood_map(self, noisy_image, sigma_noise, survey_mask,
                                   learning_rate=1e-3, num_iterations=1000, 
                                   verbose=True):
        """
        Compute maximum likelihood estimate of kappa map by direct optimization.
    
        Args:
            noisy_image: Observed noisy shear data [channels, 2, H, W]
            sigma_noise: Noise standard deviation for each channel [channels]
            survey_mask: Binary mask indicating valid pixels [channels, H, W]
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            verbose: Whether to print progress
        
        Returns:
            kappa_ml: Maximum likelihood kappa map [1, channels, H, W]
        """
        device = self.device
        (h, w), channels = self.image_size, self.channels
    
        # Initialize kappa in normalized space (between -1 and 1)
        x_init = torch.zeros(1, channels, h, w, device=device, requires_grad=True)
    
        optimizer = torch.optim.Adam([x_init], lr=learning_rate)
    
        if verbose:
            iterator = tqdm(range(num_iterations), desc='ML Optimization')
        else:
            iterator = range(num_iterations)
    
        for iteration in iterator:
            optimizer.zero_grad()
        
            # Convert normalized x to kappa
            x_normalized = self._internal_normalize(x_init)
            kappa_map = self.unnorm_kappa(x_normalized)
        
            # Compute negative log likelihood (loss to minimize)
            total_loglkl = 0.
            for i in range(channels):
                kappa_channel_i = kappa_map[:, i]
            
                # Compute shear from kappa
                shears_batch = self.torch_kappa_to_shear(kappa_channel_i)
            
                # Compute log likelihood for this channel
                batch_loglkl = self.compute_complex_log_likelihood(
                    shears_batch, 
                    noisy_image[i], 
                    sigma_noise[i]**2, 
                    survey_mask[i]
                )
            
                total_loglkl += batch_loglkl
        
            # Minimize negative log likelihood
            loss = total_loglkl
            loss.backward()
            optimizer.step()
        
            if verbose and iteration % 100 == 0:
                iterator.set_postfix({'neg_log_likelihood': loss.item()})
    
        # Return final maximum likelihood estimate
        with torch.no_grad():
            x_normalized = self._internal_normalize(x_init)
            kappa_ml = self.unnorm_kappa(x_normalized)
    
        return kappa_ml

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes
class WLDataset(Dataset):
    def __init__(self, folder, KAPPA_MIN, KAPPA_MAX, N_patches=100000):
        super().__init__()
        self.N_patches = N_patches
        self.folder = folder
        self.kappa_min = KAPPA_MIN
        self.kappa_max = KAPPA_MAX

    def __len__(self):
        return self.N_patches

    def __getitem__(self, index):
        kappa = np.load(self.folder + '/%d.npy'%(index+1))
        train_map = (kappa - self.kappa_min) / (self.kappa_max - self.kappa_min)
        return train_map.astype(np.float32)

# trainer class

class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        n_patches = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling
        
        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = WLDataset(folder,
                            diffusion_model.kappa_min.cpu().numpy(),
                            diffusion_model.kappa_max.cpu().numpy(),
                            N_patches=n_patches)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 1)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        # strict=False allows loading checkpoints saved before tau_hat_sq was
        # added as a buffer; the buffer is already correctly initialized in __init__
        model.load_state_dict(data['model'], strict=False)

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"], strict=False)

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        print("Entering training loop.... ")
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    torch.cuda.empty_cache()
                    data = next(self.dl).to(device)
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()


                    self.accelerator.backward(loss)
                pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        print('Milestone saved?')
                pbar.update(1)

        accelerator.print('training complete')
