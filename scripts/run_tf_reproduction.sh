#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
cuda_libraries=""
for directory in "$project_root"/.venv/lib/python3.11/site-packages/nvidia/*/lib "$project_root"/.venv-tf/lib/python3.11/site-packages/nvidia/*/lib; do
  [ -d "$directory" ] || continue
  cuda_libraries="$directory${cuda_libraries:+:$cuda_libraries}"
done
export LD_LIBRARY_PATH="$cuda_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_USE_LEGACY_KERAS=1 REQUIRE_TF_GPU=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$project_root/.venv-tf/lib/python3.11/site-packages/nvidia/cuda_nvcc"
exec .venv-tf/bin/python -u scripts/run_baseline_reproduction.py "$@"
