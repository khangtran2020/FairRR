#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J FairDP
#SBATCH -p datasci
#SBATCH --output=results/logs/adult_ns_0.8.out
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
module load python
conda activate torch
for RUN in 1 2 3 4 5
do
    python main.py --mode clean --dataset adult --lr 0.02 --batch_size 64 --sampling_rate 0.01 --model_type Logit --epochs 250 --seed $RUN
done