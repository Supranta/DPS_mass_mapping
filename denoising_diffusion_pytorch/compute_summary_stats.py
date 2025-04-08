import numpy as np
import h5py as h5
import sys
from FieldAnalysis import PowerSpectrumCalculator 
from tqdm import trange

import yaml

configfile = sys.argv[1]

with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

n_tomo    = config['map']['n_tomo']
n_grid    = config['map']['n_grid']
theta_max = config['map']['theta_max']

datafile  = config['data']['datafile']

savedir    = './samples/' + config['diffusion']['savedir']
n_iters    = config['diffusion']['n_prior_iterations']
n_dps      = config['diffusion']['n_dps_samples']
batch_size = config['diffusion']['prior_batch_size']

try:
	print("Computing cross-correlations....")
	compute_crosscorr = bool(int(sys.argv[2]))
except:
	compute_crosscorr = False

N_ell_bins  = 21

ps_calculator = PowerSpectrumCalculator(n_grid, theta_max)
ps_calculator.set_ell_bins(N_ell_bins)

KAPPA_MIN = np.array([-0.03479804, -0.05888689, -0.08089042])
KAPPA_MAX = np.array([0.4712809, 0.58141315, 0.6327746])

N_KAPPA = 101

ps_calculator.set_kappa_bins(KAPPA_MIN, KAPPA_MAX, N_KAPPA)

####### ============ Scattering transform ================
sys.path.append('/home2/supranta/software/scattering_transform')
import scattering
st_calc = scattering.Scattering2d(M=n_grid, N=n_grid, J=6, L=4,device='gpu')
def get_scattering_coefficients(kappa):
	results = st_calc.scattering_coef(kappa)
	S1 = results['S1_iso'].cpu().data.numpy()
	S2 = results['S2_iso'].cpu().data.numpy().mean(axis=3)
	return S1, S2
####### ==================================================
def get_tomo_Cl(kappa):
	binned_Cl = np.zeros((n_tomo,n_tomo,N_ell_bins-1))
	for i in range(n_tomo):
		for j in range(i+1):
			ell_bin_centre, Cl_ij = ps_calculator.binned_Cl(kappa[i], kappa[j])
			binned_Cl[i,j] = Cl_ij
			binned_Cl[j,i] = Cl_ij
	return ell_bin_centre, binned_Cl

def get_all_summary_stats(kappa):
	ell_bin_centre, binned_Cl = get_tomo_Cl(kappa)
	####### ==================================================
	####### ============ PDF / peaks / void ================
	kappa_pdf         = ps_calculator.get_kappa_pdf(kappa)
	kappa_peak_counts = ps_calculator.get_peak_counts(kappa)
	kappa_void_counts = ps_calculator.get_void_counts(kappa)
	####### ==================================================
	####### ============ Scattering transform ================
	S1, S2 = get_scattering_coefficients(kappa)
	summary_stats = [binned_Cl, kappa_pdf, kappa_peak_counts, kappa_void_counts, S1, S2]
	return ell_bin_centre, summary_stats

def save_summary_stats(f, ell_bin_centre, summary_stats):
	binned_Cl, kappa_pdf, kappa_peak_counts, kappa_void_counts, S1, S2 = summary_stats	

	for x in ['Cl', 'ng_stats', 'scattering_transform', 'crosscorr']:
		if x in f:
			del f[x]
	
	Cl = f.create_group('Cl')
	Cl['ell_bin_centre'] = ell_bin_centre
	Cl['Cl']             = binned_Cl	
		
	ng_stats = f.create_group('ng_stats')
	ng_stats['kappa_bincentre'] = ps_calculator.kappa_bincentre
	ng_stats['kappa_pdf']       = kappa_pdf
	ng_stats['peak_count']      = kappa_peak_counts
	ng_stats['void_count']      = kappa_void_counts	
		
	scattering_transform_f = f.create_group('scattering_transform')
	scattering_transform_f['S1'] = S1
	scattering_transform_f['S2'] = S2

def process_file(filename, compute_crosscorr=False):
	with h5.File(filename, 'r') as f:
		kappa = f['kappa'][:]
	ell_bin_centre, summary_stats = get_all_summary_stats(kappa)
	if(compute_crosscorr):
		rho_c = ps_calculator.compute_crosscorr(kappa, kappa_true)	
	with h5.File(filename, 'r+') as f:
		save_summary_stats(f, ell_bin_centre, summary_stats)	
		if(compute_crosscorr):
			crosscorr_grp = f.create_group('crosscorr')
			crosscorr_grp['ell_bin_centre'] = ell_bin_centre
			crosscorr_grp['crosscorr']      = rho_c

if(compute_crosscorr):
	print("Computing crosscorr...")
	process_file(datafile)

n_prior = n_iters * batch_size 
for i in trange(n_prior):
	filename = savedir + '/prior_sample_%d.h5'%(i)
	process_file(filename)

for i in trange(n_dps):
	filename = savedir + '/posterior_sample_%d.h5'%(i)
	process_file(filename, compute_crosscorr)
