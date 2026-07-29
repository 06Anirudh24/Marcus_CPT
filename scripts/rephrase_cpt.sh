#!/bin/bash
#SBATCH -J rephrase                      # Job name
#SBATCH -o rephrase%j.out               # output file (%j expands to jobID)
#SBATCH -e rephrase%j.err               # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=mep296@cornell.edu       # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 16                                # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH -t 24:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                      # Request partition
#SBATCH --gres=gpu:a6000:2                   # Type/number of GPUs needed
#SBATCH --mem-per-cpu=22500


echo "Running on $(hostname)"
nvidia-smi

bash /share/goyal/mposey/Synthetic_Continued_Pretraining/scripts/train.sh \
    --lr 5e-06 \
    --rr 0.1 \
    --epochs 4 \
    --bs 16 \
    --wd 0.01 \
    --warmup 0.05 \
    --task_name quality

echo "Job completed on $(date)"
