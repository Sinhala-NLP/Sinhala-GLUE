#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache

python -m tasks.named_entity_recognition.run_experiment
