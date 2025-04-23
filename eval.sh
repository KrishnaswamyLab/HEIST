#!/bin/bash

#SBATCH --job-name=eval_contra
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/eval/%x_%j.out
#SBATCH --error=./logs/slurm/eval/%x_%j.err

cd /gpfs/gibbs/project/ying_rex/hm638/SCGFM
conda init
conda activate SCGFM

python calculate_rep_spacegm.py --data_name dfci --model_name final_model_sea_pe_concat_no_cross
python calculate_rep_spacegm.py --data_name upmc --model_name final_model_sea_pe_concat_no_cross
python calculate_rep_spacegm.py --data_name charville --model_name final_model_sea_pe_concat_no_cross
python calculate_rep_sea.py --model_name final_model_sea_pe_concat_no_cross
python calculate_rep_melanoma.py --model_name final_model_sea_pe_concat_no_cross

python eval_mlp.py  --data_name dfci --label_name pTR_label
python eval_mlp.py  --data_name dfci --label_name pTR_label
python eval_mlp.py  --data_name dfci --label_name pTR_label
python eval_mlp.py  --data_name dfci --label_name pTR_label

python eval_mlp_charville.py  --data_name charville --label_name primary_outcome
python eval_mlp_charville.py  --data_name charville --label_name primary_outcome
python eval_mlp_charville.py  --data_name charville --label_name primary_outcome
python eval_mlp_charville.py  --data_name charville --label_name primary_outcome

python eval_mlp_charville.py  --data_name charville --label_name recurrence
python eval_mlp_charville.py  --data_name charville --label_name recurrence
python eval_mlp_charville.py  --data_name charville --label_name recurrence
python eval_mlp_charville.py  --data_name charville --label_name recurrence

python eval_mlp_charville.py  --data_name upmc --label_name primary_outcome
python eval_mlp_charville.py  --data_name upmc --label_name primary_outcome
python eval_mlp_charville.py  --data_name upmc --label_name primary_outcome
python eval_mlp_charville.py  --data_name upmc --label_name primary_outcome

python eval_mlp_charville.py  --data_name upmc --label_name recurred
python eval_mlp_charville.py  --data_name upmc --label_name recurred
python eval_mlp_charville.py  --data_name upmc --label_name recurred
python eval_mlp_charville.py  --data_name upmc --label_name recurred

python eval_melanoma.py
python eval_cell_clustering.py
