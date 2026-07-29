#!/bin/bash
#SBATCH -J qa_eval                           # Job name
#SBATCH -o qa_eval_%j.out                    # output file (%j expands to jobID)
#SBATCH -e qa_eval_%j.err                    # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 16                                # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=90000                          # server memory requested (per node)
#SBATCH -t 24:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                      # Request partition
#SBATCH --gres=gpu:a6000:2                   # Type/number of GPUs needed

echo "Running on $(hostname)"
nvidia-smi

python /home/aa2886/Marcus_CPT/evaluation.py

echo "Job completed on $(date)"
