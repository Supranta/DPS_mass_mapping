#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from PIL import Image
from torch import optim
from scipy.misc import face
import random
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5


class MapTools:
    def __init__(self, N_grid, theta_max):
        self.set_map_properties(N_grid, theta_max)
        self.set_fft_properties(N_grid, self.theta_max)
        self.imag_indices = self.get_imag_indices()

    def do_fwd_KS(self, kappa_l, J = 1j, EPS = 1e-20):
        kappa_l = self.symmetrize_fourier(kappa_l)
        kappa_l_complex = kappa_l[0] + J * kappa_l[1] 

        F_gamma_1 = (self.ell_x**2 - self.ell_y**2) * kappa_l_complex / (self.ell**2)
        F_gamma_2 = 2. * self.ell_x * self.ell_y    * kappa_l_complex / (self.ell**2)
        
        gamma_1 =  np.fft.irfftn(F_gamma_1) / self.PIX_AREA
        gamma_2 =  np.fft.irfftn(F_gamma_2) / self.PIX_AREA
        
        return gamma_1, gamma_2    
    
    def do_KS_inversion(self, eps, J = 1j, EPS = 1e-20):    

        A_ell = ((self.ell_x_full**2 - self.ell_y_full**2) - J * (2 * self.ell_x_full * self.ell_y_full)) \
                                            /(self.ell_full**2)  
        
        eps_1, eps_2 = eps
        eps_ell = self.PIX_AREA * np.fft.fftn(eps_1 + J * eps_2)
        kappa_ell = A_ell * eps_ell
        kappa_map_KS = np.fft.ifftn(kappa_ell).real /  self.PIX_AREA
        return kappa_map_KS
    
    def map2fourier(self, x_map):
        Fx_complex =  self.PIX_AREA * np.fft.rfftn(x_map)
        return np.array([Fx_complex.real, Fx_complex.imag])
    
    
    def fourier2map(self, Fx, J = 1j, EPS = 1e-20):
        Fx         = self.symmetrize_fourier(Fx)
        Fx_complex = (Fx[0] + J * Fx[1])
        return np.fft.irfftn(Fx_complex) /  self.PIX_AREA
    
    def symmetrize_fourier(self, Fx):
        return np.array([(self.fourier_symm_mask * Fx[0]) + 
                          (~self.fourier_symm_mask * Fx[0,self.fourier_symm_flip_ind])
                         ,(self.fourier_symm_mask * Fx[1]) - 
                          (~self.fourier_symm_mask * Fx[1,self.fourier_symm_flip_ind])])

    def set_map_properties(self, N_grid, theta_max):
        self.N_grid      = N_grid
        self.theta_max   = theta_max
        self.Omega_s     = self.theta_max**2
        self.PIX_AREA    = self.Omega_s / self.N_grid**2
        self.Delta_theta = theta_max / N_grid
        
    def set_fft_properties(self, N_grid, theta_max):
        lx = 2*np.pi*np.fft.fftfreq(N_grid, d=theta_max / N_grid)
        ly = 2*np.pi*np.fft.fftfreq(N_grid, d=theta_max / N_grid)

        N_Y = (N_grid//2 +1)
        self.N_Y = N_Y
        
        # mesh of the 2D frequencies
        self.ell_x = np.tile(lx[:, None], (1, N_Y))       
        self.ell_y = np.tile(ly[None, :N_Y], (N_grid, 1))
        self.ell = np.sqrt(self.ell_x**2 + self.ell_y**2)
        self.ell[0,0] = 1.
        
        self.ell_x_full = np.tile(lx[:, None], (1, N_grid))       
        self.ell_y_full = np.tile(ly[None, :], (N_grid, 1))
        self.ell_full   = np.sqrt(self.ell_x_full**2 + self.ell_y_full**2)
        self.ell_full[0,0] = 1.
        
        fourier_symm_mask = np.ones((N_grid, self.N_Y))
        fourier_symm_mask[(self.N_Y):,0]  = 0
        fourier_symm_mask[(self.N_Y):,-1] = 0
        fourier_symm_mask[0,0]            = 0
        self.fourier_symm_mask = fourier_symm_mask.astype(bool)        
        
        fourier_symm_mask_imag = fourier_symm_mask.copy()
        fourier_symm_mask_imag[0,-1]        = 0
        fourier_symm_mask_imag[self.N_Y-1,0]  = 0
        fourier_symm_mask_imag[self.N_Y-1,-1] = 0
        self.fourier_symm_mask_imag = fourier_symm_mask_imag.astype(bool)
        
        fourier_symm_flip_ind      = np.arange(N_grid)
        fourier_symm_flip_ind[1:]  = fourier_symm_flip_ind[1:][::-1]
        self.fourier_symm_flip_ind = fourier_symm_flip_ind

# ================== 1D array to Fourier plane ===============================             
    def set_fourier_plane_face(self, F_x, x):
        N = self.N_grid
        F_x[:,:,1:-1] = x[:N**2 - 2*N].reshape(2,N,N//2-1)
        return F_x

    def set_fourier_plane_edge(self, F_x, x):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    
        
        F_x[:,1:N_Y-1,0]  = x[N**2 - 2*N:N**2 - 2*N+2*N_edge].reshape((2,-1))
        F_x[:,1:N_Y-1,-1] = x[N**2 - 2*N+2*N_edge:-3].reshape((2,-1))
        return F_x

    def set_fourier_plane_corner(self, F_x, x):    
        N = self.N_grid
        N_Y = N//2+1
               
        F_x[0,N_Y-1,-1] = x[-3]
        F_x[0,0,-1]     = x[-2]
        F_x[0,N_Y-1,0]  = x[-1]
        return F_x
    
    def array2fourier_plane(self, x):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    

        F_x_plane = np.zeros((2,N,N_Y))
        F_x_plane = self.set_fourier_plane_face(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_edge(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_corner(F_x_plane, x)

        F_x_plane = self.symmetrize_fourier(F_x_plane)        
        return F_x_plane
    
    def fourier_plane2array(self, Fx):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    

        x = np.zeros(shape=N*N-1)

        x[:N**2 - 2*N]                    = Fx[:,:,1:-1].reshape(-1)
        x[N**2 - 2*N:N**2 - 2*N+2*N_edge] = Fx[:,1:N_Y-1,0].reshape(-1)
        x[N**2 - 2*N+2*N_edge:-3]         = Fx[:,1:N_Y-1,-1].reshape(-1)
        
        x[-3] = Fx[0,N_Y-1,-1]
        x[-2] = Fx[0,0,-1]
        x[-1] = Fx[0,N_Y-1,0]
        
        return x
    
    def get_imag_indices(self):
        x0 = np.zeros(self.N_grid**2-1)
        Fx = np.array(self.array2fourier_plane(x0))
        Fx[1] = 1

        x0 = np.array(self.fourier_plane2array(Fx)).astype(int)

        indices = np.arange(x0.shape[0])
        imag_indices_1d = indices[(x0 == 1)]

        return imag_indices_1d


# In[ ]:

class TorchMapTools:
    def __init__(self, N_grid, theta_max):
        self.set_map_properties(N_grid, theta_max)
        self.set_fft_properties(N_grid, self.theta_max)
        self.imag_indices = self.get_imag_indices()

    def do_fwd_KS(self, kappa_l):
        kappa_l = self.symmetrize_fourier(kappa_l)
        kappa_l_complex = kappa_l[0] + 1j * kappa_l[1] 
        F_gamma_1 = (self.ell_x**2 - self.ell_y**2) * kappa_l_complex / (self.ell**2)
        F_gamma_2 = 2. * self.ell_x * self.ell_y    * kappa_l_complex / (self.ell**2)
        
        gamma_1 =  torch.fft.irfftn(F_gamma_1, s=(self.N_grid, self.N_grid)) / self.PIX_AREA
        gamma_2 =  torch.fft.irfftn(F_gamma_2, s=(self.N_grid, self.N_grid)) / self.PIX_AREA
        
        return gamma_1, gamma_2    

    def do_fwd_KS1(self, kappa, N_grid = 256, theta_max = 12., J = 1j, EPS = 1e-20): 
        A_ell = ((self.ell_x_full**2 - self.ell_y_full**2) - 1j * (2 * self.ell_x_full * self.ell_y_full)) \
                                            /(self.ell_full**2)
        A_ell[0,0] = 1.
        kappa_l = torch.fft.fftn(kappa)
        gamma = torch.fft.ifftn(kappa_l / A_ell)
        gamma1 = gamma.real
        gamma2 = gamma.imag
        return gamma1,gamma2
    
    def do_KS_inversion(self, eps):        
        A_ell = ((self.ell_x_full**2 - self.ell_y_full**2) - 1j * (2 * self.ell_x_full * self.ell_y_full)) \
                                            /(self.ell_full**2)
        
        eps_1, eps_2 = eps
        eps_ell = self.PIX_AREA * torch.fft.fftn(eps_1 + 1j * eps_2, s=(self.N_grid, self.N_grid))
        kappa_ell = A_ell * eps_ell
        kappa_map_KS = torch.fft.ifftn(kappa_ell, s=(self.N_grid, self.N_grid)).real /  self.PIX_AREA
        return kappa_map_KS
    
    def map2fourier(self, x_map):
        Fx_complex =  self.PIX_AREA * torch.fft.rfftn(x_map, s=(self.N_grid, self.N_grid))
        return torch.stack([Fx_complex.real, Fx_complex.imag])
    
    
    def fourier2map(self, Fx):
        Fx         = self.symmetrize_fourier(Fx)
        Fx_complex = Fx[0] + 1j * Fx[1]
        return torch.fft.irfftn(Fx_complex, s=(self.N_grid, self.N_grid)) /  self.PIX_AREA
    
    def symmetrize_fourier(self, Fx):
        return torch.stack([(self.fourier_symm_mask * Fx[0]) + 
                            (~self.fourier_symm_mask * Fx[0,self.fourier_symm_flip_ind])
                         ,(self.fourier_symm_mask * Fx[1]) - 
                            (~self.fourier_symm_mask * Fx[1,self.fourier_symm_flip_ind])])

    def set_map_properties(self, N_grid, theta_max):
        self.N_grid      = N_grid
        self.theta_max   = theta_max
        self.Omega_s     = self.theta_max**2
        self.PIX_AREA    = self.Omega_s / self.N_grid**2
        self.Delta_theta = theta_max / N_grid
        
    def set_fft_properties(self, N_grid, theta_max):
        lx = 2 * torch.pi * torch.fft.fftfreq(N_grid, d=theta_max / N_grid).to(device = 'cuda:0')
        ly = 2 * torch.pi * torch.fft.fftfreq(N_grid, d=theta_max / N_grid).to(device = 'cuda:0')

    
        N_Y = (N_grid // 2 + 1)
        self.N_Y = N_Y
        
        # mesh of the 2D frequencies
        self.ell_x = torch.tile(lx[:, None], (1, N_Y)).to(device = 'cuda:0')
        self.ell_y = torch.tile(ly[None, :N_Y], (N_grid, 1)).to(device = 'cuda:0')
        self.ell = torch.sqrt(self.ell_x**2 + self.ell_y**2).to(device = 'cuda:0')
        self.ell[0,0] = 1.
        
        self.ell_x_full = torch.tile(lx[:, None], (1, N_grid))       
        self.ell_y_full = torch.tile(ly[None, :], (N_grid, 1))
        self.ell_full   = torch.sqrt(self.ell_x_full**2 + self.ell_y_full**2)
        self.ell_full[0,0] = 1.
        
        fourier_symm_mask = torch.ones((N_grid, self.N_Y), dtype=bool, device = 'cuda:0')
        fourier_symm_mask[(self.N_Y):,0]  = 0
        fourier_symm_mask[(self.N_Y):,-1] = 0
        fourier_symm_mask[0,0]            = 0
        self.fourier_symm_mask = fourier_symm_mask
        
        fourier_symm_mask_imag = fourier_symm_mask.clone()
        fourier_symm_mask_imag[0,-1]        = 0
        fourier_symm_mask_imag[self.N_Y-1,0]  = 0
        fourier_symm_mask_imag[self.N_Y-1,-1] = 0
        self.fourier_symm_mask_imag = fourier_symm_mask_imag
        
        fourier_symm_flip_ind = torch.arange(N_grid)
        fourier_symm_flip_ind[1:] = torch.flip(fourier_symm_flip_ind[1:], dims=[0])
        self.fourier_symm_flip_ind = fourier_symm_flip_ind

    def set_fourier_plane_face(self, F_x, x):
        N = self.N_grid
        F_x[:,:,1:-1] = x[:N**2 - 2*N].view(2, N, N//2 - 1)
        return F_x

    def set_fourier_plane_edge(self, F_x, x):
        N = self.N_grid
        N_Y = N//2 + 1
        N_edge = N//2 - 1    
        
        F_x[:,1:N_Y-1,0]  = x[N**2 - 2*N:N**2 - 2*N+2*N_edge].view(2, -1)
        F_x[:,1:N_Y-1,-1] = x[N**2 - 2*N+2*N_edge:-3].view(2, -1)
        return F_x

    def set_fourier_plane_corner(self, F_x, x):    
        N = self.N_grid
        N_Y = N//2 + 1
               
        F_x[0,N_Y-1,-1] = x[-3]
        F_x[0,0,-1]     = x[-2]
        F_x[0,N_Y-1,0]  = x[-1]
        return F_x
    
    def array2fourier_plane(self, x):
        N = self.N_grid
        N_Y = N//2 + 1
        N_edge = N//2 - 1    

        F_x_plane = torch.zeros((2, N, N_Y), device = 'cuda:0')
        F_x_plane = self.set_fourier_plane_face(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_edge(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_corner(F_x_plane, x)

        F_x_plane = self.symmetrize_fourier(F_x_plane)        
        return F_x_plane
    
    def fourier_plane2array(self, Fx):
        N = self.N_grid
        N_Y = N // 2 + 1
        N_edge = N // 2 - 1    
    
        x = torch.zeros(N * N - 1, device = 'cuda:0')
    
        x[:N**2 - 2 * N] = Fx[:, :, 1:-1].reshape(-1)
        x[N**2 - 2 * N:N**2 - 2 * N + 2 * N_edge] = Fx[:, 1:N_Y - 1, 0].reshape(-1)
        x[N**2 - 2 * N + 2 * N_edge:-3] = Fx[:, 1:N_Y - 1, -1].reshape(-1)
        
        x[-3] = Fx[0, N_Y - 1, -1]
        x[-2] = Fx[0, 0, -1]
        x[-1] = Fx[0, N_Y - 1, 0]
        
        return x
    
    def get_imag_indices(self):
        x0 = torch.zeros(self.N_grid**2 - 1, device = 'cuda:0')
        Fx = self.array2fourier_plane(x0).to(device = 'cuda:0')
        Fx[1] = 1

        x0 = self.fourier_plane2array(Fx).int()

        indices = torch.arange(x0.shape[0]).to(device = 'cuda:0')
        imag_indices_1d = indices[(x0 == 1)]

        return imag_indices_1d
