## Steps for getting the code going

- Step 0: Clone this repository into your HPC system
- Step 1: Create a conda environment for running the code. I have included an environment file [here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/diffusion_environment.yml). Install this environment using the command, `conda env create -f diffusion_environment.yml`. You might want to change the prefix in the file.
- Step 2: Download the training data into your computer. The training data can be found [here](https://drive.google.com/file/d/10NFlppmLl3U8V6S-B-kGlMVx6KeBwROU/view?usp=sharing). 
- Step 3: CONFIGFILE
```yaml
map:
        n_tomo: 3                   # Number of tomographic bins
        n_grid: 128                 # Number of pixel grids in each map
        theta_max: 3.5              # Size of the maps in degrees
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
Set the config file as: 
```bash
	CONFIGFILE=./config/columbia_lensing.yaml
```
- Step 4: After you download the training data, try to train the diffusion model using the `train.py` script. The slurm script I used to train the model can be found [here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/train_diffusion.sh).
```python
	python3 train.py $CONFIGFILE
``` 
- Step 5: If the training is successful, you can sample the diffusion maps using the `sample.py` script. You should be able to execute it simply using
```python
        python3 sample.py $CONFIGFILE
```
- Step 6: After we have the diffusion samples, we would like to validate the maps by computing various different summary statistics. This can be done by running `python3 compute_summary_stats_tomo.py SAMPLE_DIRECTORY N_SAMPLES 1`. 
