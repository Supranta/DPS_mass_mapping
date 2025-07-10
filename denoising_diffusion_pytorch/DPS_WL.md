## Steps for running the DPS-WL code

### Download the repository and the training data
- Clone this repository into your HPC system. You can use: 
```git
	git clone git@github.com:Supranta/DPS_mass_mapping.git
```
to download the git repository.

### Create a conda environment for the diffusion code 
- Create a conda environment for running the code. I have included an environment file [here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/diffusion_environment.yml). Install this environment using the command, 
```conda
	conda env create -f diffusion_environment.yml
``` 
You might need to change the prefix in the file.

### The configfile
- We set various changable parameters of our diffusion run in a config file. [Here]([https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/config/columbia_lensing.yaml](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/config/desy3.yaml)) is an example. For example, in the `map` part of the YAML config, we set the number of tomographic bins, number of pixels and the size of the map.   
```yaml
map:
        n_tomo: 4                   # Number of tomographic bins
        n_grid: 256                 # Number of pixel grids in each map
        theta_max: 4.26666666666    # Size of the maps in degrees
```

### Train the diffusion model
- We can train the diffusion model using the [`train.py`](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/train.py) script as,
```bash
	CONFIGFILE=./config/desy3.yaml
	python3 train.py $CONFIGFILE
```
- The slurm script I used to train the model can be found [here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/train_desy3.sh).
- The crucial part of the configfile is the `train` section. 
```yaml
train:
        # Training data location
        data_folder: /home2/supranta/PosteriorSampling/data/Marco_Flagship/Flagship_covariance_small/all_patches/
        # Where to save the diffusion models
        results_folder: ./results_desy3
        ntrain: 50000
        transform:
                exp_transform: false
                kappa_min: -0.00842643,-0.01598292,-0.02735016,-0.03807191
                kappa_max: 0.76519483,0.9558846,1.1022788,1.1788471

```
### Create noisy data
- If the training is successfull, we first create a noisy mock data from the simulations using the script [`create_noisy_data.py`](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/create_noisy_data.py)
```python
        python3 create_noisy_data.py $CONFIGFILE
```
- The properties of the noisy data is guided by the `data` section of the config file, where we specify the datafile local, effective number density of sources and the shape noise
```yaml
data:
        datafile: ./data/desy3/data.h5               # Data containing noisy shear data 
        neff: 1.5                                    # Effective number density in arcmin^{-2}
        sigma_e: 0.26                                # Shape noise
```
### Sampling diffusion maps
- We can then sample maps from the diffusion model using the [`sample.py`](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/sample.py) script
```python
        python3 sample.py $CONFIGFILE
```
- The same script is used for sampling in the generative mode as well as to sample from the posterior. How many samples to generate of each and the trained model to use for the sampling is specified by the `diffusion` sections of the config file.  
```yaml
diffusion:
        trained_model_name: 10                   # Comment this out when running this initially
        savedir: desy3_samples                   # where to save the diffusion outputs
        n_prior_iterations: 4                    # Number of sampling iterations for the prior sampling
        prior_batch_size: 8                      # Batch size while sampling the prior with the diffusion model
        n_dps_samples: 0                         # Number of DPS samples to generate
```
### Compute the summary statistics of the sampled maps
- After we have the sample the diffusion maps, we would like to validate the maps by computing various different summary statistics. This can be done by running 
```python
	python3 compute_summary_stats.py $CONFIGFILE
``` 
