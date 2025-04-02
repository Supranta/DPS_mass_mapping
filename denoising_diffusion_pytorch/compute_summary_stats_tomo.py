import numpy as np
import h5py as h5
import sys
from FieldAnalysis import PowerSpectrumCalculator 
from tqdm import trange

savedir = sys.argv[1]
N_grid  = 128
#N_grid  = int(sys.argv[2])
N       = int(sys.argv[2])
try:
	print("Computing cross-correlations....")
	compute_crosscorr = bool(int(sys.argv[3]))
except:
	compute_crosscorr = False

theta_max = 3.5
N_ell_bins  = 21

ps_calculator = PowerSpectrumCalculator(N_grid, theta_max)
ps_calculator.set_ell_bins(N_ell_bins)


if(N_grid==256):
	KAPPA_MIN = -0.08201675
	KAPPA_MAX = 0.7101586
	N_KAPPA = 101
if(N_grid==128):
	KAPPA_MIN = np.array([-0.03479804, -0.05888689, -0.08089042])
	KAPPA_MAX = np.array([0.4712809, 0.58141315, 0.6327746])
	#KAPPA_MIN = -0.056813
	#KAPPA_MAX = 0.512567
	#N_KAPPA = 151
	#N_KAPPA = 51
	N_KAPPA = 101


ps_calculator.set_kappa_bins(KAPPA_MIN, KAPPA_MAX, N_KAPPA)

####### ============ Scattering transform ================
sys.path.append('/home2/supranta/software/scattering_transform')
import scattering
st_calc = scattering.Scattering2d(M=N_grid, N=N_grid, J=6, L=4,device='gpu')
def get_scattering_coefficients(kappa):
	results = st_calc.scattering_coef(kappa)
	S1 = results['S1_iso'].cpu().data.numpy()
	S2 = results['S2_iso'].cpu().data.numpy().mean(axis=3)
	return S1, S2
####### ==================================================
def get_tomo_Cl(kappa):
	N_Z_BINS = kappa.shape[0]
	binned_Cl = np.zeros((N_Z_BINS,N_Z_BINS,N_ell_bins-1))
	for i in range(N_Z_BINS):
		for j in range(i+1):
			ell_bin_centre, Cl_ij = ps_calculator.binned_Cl(kappa[i], kappa[j])
			binned_Cl[i,j] = Cl_ij
			binned_Cl[j,i] = Cl_ij
	return ell_bin_centre, binned_Cl

if(compute_crosscorr):
	print("Computing crosscorr...")
	with h5.File(savedir + '/data.h5', 'r') as f:
		kappa_true = f['kappa'][:]
	ell_bin_centre, binned_Cl = get_tomo_Cl(kappa_true)
	####### ==================================================
	####### ============ PDF / peaks / void ================
	kappa_pdf         = ps_calculator.get_kappa_pdf(kappa_true)
	kappa_peak_counts = ps_calculator.get_peak_counts(kappa_true)
	kappa_void_counts = ps_calculator.get_void_counts(kappa_true)
	####### ==================================================
	####### ============ Scattering transform ================
	S1, S2 = get_scattering_coefficients(kappa_true)
	with h5.File(savedir + '/data.h5', 'r+') as f:
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

for i in trange(N):
	with h5.File(savedir + '/sample_%d.h5'%(i), 'r') as f:
		kappa = f['kappa'][:]

	####### ============ Power spectrum ================
	ell_bin_centre, binned_Cl = get_tomo_Cl(kappa)
	####### ==================================================
	####### ============ PDF / peaks / void ================
	kappa_pdf         = ps_calculator.get_kappa_pdf(kappa)
	kappa_peak_counts = ps_calculator.get_peak_counts(kappa)
	kappa_void_counts = ps_calculator.get_void_counts(kappa)
	####### ==================================================
	####### ============ Scattering transform ================
	S1, S2 = get_scattering_coefficients(kappa)
	####### ==================================================
	if(compute_crosscorr):
		rho_c = ps_calculator.compute_crosscorr(kappa, kappa_true)
	with h5.File(savedir + '/sample_%d.h5'%(i), 'r+') as f:
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

		if(compute_crosscorr):
			crosscorr_grp = f.create_group('crosscorr')
			crosscorr_grp['ell_bin_centre'] = ell_bin_centre
			crosscorr_grp['crosscorr']      = rho_c
