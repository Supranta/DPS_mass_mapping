#!/bin/bash
#
# JOB SPECIFICATIONS:
#SBATCH --job-name=train_diffusion
#SBATCH --partition=low_gpu_titan
#SBATCH --qos=low
#SBATCH --time=95:59:59
#SBATCH --output=Diffusion_train.out
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=supranta@sas.upenn.edu

echo "NODES=1"

s=$(date)
echo "starting"
echo $s
echo "running the thing"
source ~/.bash_profile
conda activate diffusion
cd /home2/supranta/PosteriorSampling/denoising_diffusion_pytorch/denoising_diffusion_pytorch 

CONFIGFILE=./config/columbia_lensing.yaml
python3 train.py $CONFIGFILE

e=$(date)
echo "ending"
echo $e             
