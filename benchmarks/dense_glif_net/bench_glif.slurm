#!/bin/bash
#SBATCH --job-name=glif-bench
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=glif_bench_%j.log

# Full GLIF3 kernel benchmark: sweep N and T for
#   {neuron, dense, sparse} x {inference, training} x {triton, warp, cupy}.
# Submit from the repo root:
#   sbatch benchmarks/dense_glif_net/bench_glif.slurm
# Writes benchmarks/dense_glif_net/bench_glif_full_results.json for the plots.
cd "$SLURM_SUBMIT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
micromamba run -n ml-py312 python -m benchmarks.dense_glif_net.bench_glif_full \
    --out benchmarks/dense_glif_net/bench_glif_full_results.json
