#!/usr/bin/env python
#Import Statements
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from scipy.interpolate import make_interp_spline
from astropy.io import fits
from PIL import Image
from torchvision import transforms
from scipy.stats import gaussian_kde

class PowerSpectrumCalculator:
    def __init__(self, N_grid, theta_max):
        self.set_map_properties(N_grid, theta_max)
        self.set_fft_properties(N_grid, self.theta_max)
            
    def set_map_properties(self, N_grid, theta_max):
        self.N_grid    = N_grid
        self.theta_max = theta_max * np.pi / 180.      # theta_max in radians
        self.Omega_s   = self.theta_max**2             # Area of the map  
        self.PIX_AREA  = self.Omega_s / self.N_grid**2 

    def set_fft_properties(self, N_grid, theta_max):
        lx = 2 * np.pi * np.fft.fftfreq(N_grid, d=theta_max / N_grid)
        ly = 2 * np.pi * np.fft.fftfreq(N_grid, d=theta_max / N_grid)

        self.N_Y = (N_grid//2 +1)
        
        # mesh of the 2D frequencies
        self.ell_x = np.tile(lx[:, None], (1, self.N_Y))
        self.ell_y = np.tile(ly[None, :self.N_Y], (N_grid, 1))
        self.ell   = np.sqrt(self.ell_x**2 + self.ell_y**2)

        self.ell_max = self.ell.max()
        self.ell_min = np.sort(self.ell.flatten())[1]
        
        print("ELL_MIN: "+str(self.ell_min))
        print("ELL_MAX: "+str(self.ell_max))

        fourier_symm_mask = np.ones((N_grid, self.N_Y))
        fourier_symm_mask[(self.N_Y):,0]  = 0
        fourier_symm_mask[(self.N_Y):,-1] = 0
        fourier_symm_mask[0,0]            = 0
        self.fourier_symm_mask = fourier_symm_mask.astype(bool)
        
    def map2fourier(self, x_map):
        Fx_complex =  self.PIX_AREA * np.fft.rfftn(x_map)
        return np.array([Fx_complex.real, Fx_complex.imag])

    def set_kappa_bins(self, KAPPA_MIN, KAPPA_MAX, N_KAPPA):
        self.N_tomo_bins     = len(KAPPA_MIN)
        self.kappa_bins      = [np.linspace(KAPPA_MIN[i], KAPPA_MAX[i], N_KAPPA) for i in range(self.N_tomo_bins)] 
        self.kappa_bincentre = [0.5 * (self.kappa_bins[i][1:] + self.kappa_bins[i][:-1]) for i in range(self.N_tomo_bins)]

    def get_kappa_pdf(self, kappa):
        return np.array([np.histogram(kappa[i], bins=self.kappa_bins[i], density=True)[0] for i in range(self.N_tomo_bins)])

    def get_ell_bins(self, N_bins):
        return np.logspace(np.log10(self.ell_min), np.log10(self.ell_max), N_bins)
    
    def set_ell_bins(self, N_bins):
        self.ell_bins = self.get_ell_bins(N_bins)

    def binned_Cl(self, delta1, delta2=None):
        cross_Cl_bins = []
        ell_bin_centre = []
        delta_ell_1 = self.map2fourier(delta1)
        if delta2 is not None:
            delta_ell_2 = self.map2fourier(delta2)
        else:
            delta_ell_2 = delta_ell_1
        for i in range(len(self.ell_bins) - 1):
            select_ell = (self.ell > self.ell_bins[i]) & (self.ell < self.ell_bins[i+1]) & self.fourier_symm_mask
            ell_bin_centre.append(np.mean(self.ell[select_ell]))
            # The factor of 2 needed because there are both real and imaginary modes in the l selection!
            cross_Cl = 2. * np.mean(delta_ell_1[:,select_ell] * delta_ell_2[:,select_ell]) / self.Omega_s
            cross_Cl_bins.append(cross_Cl)

        return np.array(ell_bin_centre), np.array(cross_Cl_bins)

    def compute_crosscorr(self, delta1, delta2):
        n_tomo_bins = delta1.shape[0]
        rho_c_list = []
        for i in range(n_tomo_bins):
            _, C11 = self.binned_Cl(delta1[i])
            _, C22 = self.binned_Cl(delta2[i])
            _, C12 = self.binned_Cl(delta1[i], delta2[i])
            rho_c = C12 / np.sqrt(C11 * C22)
            rho_c_list.append(rho_c)
        return np.array(rho_c_list)

    def get_neighbor_maps(self, flat_map):
        n, m = flat_map.shape
        neighbor_maps = []

        # Define the shifts for neighbors (8 directions)
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Top-Left, Top-Right, Bottom-Left, Bottom-Right

        for dx, dy in shifts:
            shifted_map = np.zeros_like(flat_map)
            for i in range(n):
                for j in range(m):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < m:
                        shifted_map[i, j] = flat_map[ni, nj]
                    else:
                        shifted_map[i, j] = 0  # Or some other boundary value
            neighbor_maps.append(shifted_map)

        return np.array(neighbor_maps)

    def get_kappa_peaks(self, flat_map):
        neighbor_maps = self.get_neighbor_maps(flat_map)
        max_neighbor_map = np.max(neighbor_maps, axis=0)
        select_peaks = (flat_map > max_neighbor_map)
        return flat_map[select_peaks]

    def get_kappa_voids(self, flat_map):
        neighbor_maps = self.get_neighbor_maps(flat_map)
        min_neighbor_map = np.min(neighbor_maps, axis=0)
        select_voids = (flat_map < min_neighbor_map)
        return flat_map[select_voids]

    def get_peak_counts(self, kappa):
        kappa_peaks = [self.get_kappa_peaks(kappa[i]) for i in range(self.N_tomo_bins)]
        return np.array([np.histogram(kappa_peaks[i], bins=self.kappa_bins[i])[0] for i in range(self.N_tomo_bins)])

    def get_void_counts(self, kappa):
        kappa_voids = [self.get_kappa_voids(kappa[i]) for i in range(self.N_tomo_bins)]
        return np.array([np.histogram(kappa_voids[i], bins=self.kappa_bins[i])[0] for i in range(self.N_tomo_bins)])

