## Steps for getting the code going

- Step 0: Clone this repository into your HPC system
- Step 1: Create a conda environment for running the code. I have included an environment file [here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/diffusion_environment.yml). Install this environment using the command, `conda env create -f diffusion_environment.yml`. You might want to change the prefix in the file.
- Step 2: Download the training data into your computer. The training data can be found [here](https://drive.google.com/file/d/10NFlppmLl3U8V6S-B-kGlMVx6KeBwROU/view?usp=sharing). 
- Step 3: After you download the training data, try to train the diffusion model using the `Train_model.py` script. The slurm script I used to train the model can be found [here](https://github.com/Supranta/DPS_mass_mapping/blob/main/denoising_diffusion_pytorch/train_diffusion.sh). 
- Step 4: If the training is successful, you can sample the diffusion maps using the `Sample_tomo.py` script. You should be able to execute it simply using `python3 Sample_tomo.py`.
- Step 5: After we have the diffusion samples, we would like to validate the maps by computing various different summary statistics. This can be done by running `python3 compute_summary_stats_tomo.py SAMPLE_DIRECTORY N_SAMPLES 1`. 
