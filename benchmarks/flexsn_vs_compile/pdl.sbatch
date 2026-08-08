#!/bin/bash
#SBATCH --job-name=pdl-rsnn
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=pdl_rsnn_%j.log

# PDL vs CUDA-graph vs persistent kernel for the sparse RSNN recurrent multistep
# (GLIF3, 5% scale-free recurrent W), across Triton / CuPy / TileLang. Needs an
# sm_90+ GPU for PDL (griddepcontrol). Run from the repo root so the
# `benchmarks.*` package imports resolve.
cd "$SLURM_SUBMIT_DIR/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Sweep N from launch-bound (512) to GPU-bound (24576) so the crossover where
# PDL / CUDA-graph / persistent stop paying off is visible. A dedicated SLURM GPU
# gives clean timing (no shared-GPU contention).
micromamba run -n ml-py312 python -m benchmarks.flexsn_vs_compile.bench_pdl \
    --N ${N:-512 2048 8192 24576} --T "${T:-64}" \
    --out benchmarks/flexsn_vs_compile/bench_pdl_results.json
