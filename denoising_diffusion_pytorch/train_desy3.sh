#!/bin/bash
#
# JOB SPECIFICATIONS:
#SBATCH --job-name=train_diffusion
#SBATCH --partition=bhuv_gpu
#SBATCH --qos=bhuv
#SBATCH --time=23:59:59
#SBATCH --output=train_desy3.out
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

CONFIGFILE=./config/desy3.yaml
export CUDA_VISIBLE_DEVICES=3,4,5
accelerate launch train.py $CONFIGFILE

e=$(date)
echo "ending"
echo $e             
