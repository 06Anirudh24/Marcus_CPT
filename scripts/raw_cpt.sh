#!/bin/bash
#SBATCH -p genai-goyal-highpri
#SBATCH --gres=gpu:nvidia_b200:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH -J raw_cpt
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@60


echo "Running on $(hostname)"
nvidia-smi

bash /home/aa2886/Marcus_CPT/scripts/train.sh \
    --lr 4e-05 \
    --rr 0.15 \
    --epochs 1 \
    --bs 16 \
    --wd 0.01 \
    --warmup 0.05 \
    --task_name quality

echo "Job completed on $(date)"
