#!/bin/bash
#SBATCH --job-name=flexsn-sweep
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=flexsn_sweep_%j.log

# Full timestep sweep (FlexSN recompiles per (T, N); T=64 training ~= 36 min).
# --append lets a resubmit resume a partial run.
cd "$SLURM_SUBMIT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
micromamba run -n ml-py312 python bench_sweep.py --N 32768 --T 4 16 64 256 1024
