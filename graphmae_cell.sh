#!/bin/bash

#SBATCH --job-name=cell_only
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/pretraining/%x_%j.out
#SBATCH --error=./logs/slurm/pretraining/%x_%j.err

cd /gpfs/gibbs/project/ying_rex/hm638/SCGFM
conda init
conda activate SCGFM

python nonhie_pretraining.py --input_modality "pos"
