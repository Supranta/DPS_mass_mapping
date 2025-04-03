## Steps for running the DPS-WL code

### Download the repository and the training data
- Clone this repository into your HPC system. You can use: 
```git
	git clone git@github.com:Supranta/DPS_mass_mapping.git
```
to download the git repository.
- Download the training data into your computer. The training data for the Columbia lensing simulations used in our [paper](https://arxiv.org/pdf/2502.04158) can be found [here](https://drive.google.com/file/d/10NFlppmLl3U8V6S-B-kGlMVx6KeBwROU/view?usp=sharing). 

### Create a conda environment for the diffusion code 
- Create a conda environment for running the code. I have included an environment file [here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/diffusion_environment.yml). Install this environment using the command, 
```conda
	conda env create -f diffusion_environment.yml
``` 
You might need to change the prefix in the file.

### The configfile
- We set various changable parameters of our diffusion run in a config file. [Here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/config/columbia_lensing.yaml) is an example. For example, in the `map` part of the YAML config, we set the number of tomographic bins, number of pixels and the size of the map.   
```yaml
map:
        n_tomo: 3                   # Number of tomographic bins
        n_grid: 128                 # Number of pixel grids in each map
        theta_max: 3.5              # Size of the maps in degrees
```

### Train the diffusion model
- We can train the diffusion model using the [`train.py`](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/train.py) script as,
```bash
	CONFIGFILE=./config/columbia_lensing.yaml
	python3 train.py $CONFIGFILE
```
- The slurm script I used to train the model can be found [here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/train_diffusion.sh).
- The crucial part of the configfile is the `train` section. 
```yaml
train:
        # Training data location
        data_folder: /home2/supranta/PosteriorSampling/data/Columbia_lensing/MassiveNuS/kappa_128_3bins/
        # Where to save the diffusion models
        results_folder: ./results

data:
        datafile: ./data/columbia_lensing/data.h5    # Data containing noisy shear data 
        neff: 7.5                                    # Effective number density in arcmin^{-2}
        sigma_e: 0.26                                # Shape noise

diffusion:
        trained_model_name: 32                       # The trained model to use for diffusion sampling
        savedir: diffusion_samples                   # where to save the diffusion outputs
        prior_batch_size: 16                         # Batch size while sampling the prior with the diffusion model
```
### Create noisy data
- If the training is successfull, we first create a noisy mock data from the simulations using the script [`create_noisy_data.py`](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/create_noisy_data.py)
```python
        python3 create_noisy_data.py $CONFIGFILE
```
- The properties of the noisy data is guided by the `data` section of the config file, where we specify the datafile local, effective number density of sources and the shape noise
```yaml
data:
        datafile: ./data/columbia_lensing/data.h5    # Data containing noisy shear data 
        neff: 7.5                                    # Effective number density in arcmin^{-2}
        sigma_e: 0.26                                # Shape noise

diffusion:
        trained_model_name: 32                       # The trained model to use for diffusion sampling
        savedir: diffusion_samples                   # where to save the diffusion outputs
        prior_batch_size: 16                         # Batch size while sampling the prior with the diffusion model
```
### Sampling diffusion maps
- We can then sample maps from the diffusion model using the [`sample.py`](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/sample.py) script
```python
        python3 sample.py $CONFIGFILE
```
- The same script is used for sampling in the generative mode as well as to sample from the posterior. How many samples to generate of each and the trained model to use for the sampling is specified by the `diffusion` sections of the config file.  
```yaml
diffusion:
        trained_model_name: 32                       # The trained model to use for diffusion sampling
        savedir: diffusion_samples                   # where to save the diffusion outputs
        n_prior_iterations: 1                        # Number of sampling iterations for the prior sampling
        prior_batch_size: 16                         # Batch size while sampling the prior with the diffusion model
        n_dps_samples: 1                             # Number of DPS samples to generate
```
### Compute the summary statistics of the sampled maps
- After we have the sample the diffusion maps, we would like to validate the maps by computing various different summary statistics. This can be done by running 
```python
	python3 compute_summary_stats_tomo.py SAMPLE_DIRECTORY N_SAMPLES 1
``` 
