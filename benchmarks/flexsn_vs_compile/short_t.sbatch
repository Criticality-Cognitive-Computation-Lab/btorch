#!/bin/bash
#SBATCH --job-name=flexsn-shortt
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=flexsn_shortt_%j.log

# Short-T bars (Figure 1): inductor (unrolled) vs FlexSN vs CuPy.
cd "$SLURM_SUBMIT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
micromamba run -n ml-py312 python bench_short_t.py --N 1048576 --T 2 4 8 16
