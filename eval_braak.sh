#!/bin/bash

#SBATCH --job-name=braak_eval
#SBATCH --time=25:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_ying_rex
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/eval/%x_%j.out
#SBATCH --error=./logs/slurm/eval/%x_%j.err

cd /home/hm638/SCGFM
conda init
conda activate SCGFM

python eval_braak.py --model sea_graphs_3M_attention_anchor_pe_ranknorm_cross_blending.pt --num_epochs 500 --wd 1e-2