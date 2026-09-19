#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
mkdir -p OUTPUT/venv_logs
for index in 0 1 2; do
  seed=$((2026 + index))
  for method in diffusion_ts pad_ts; do
    gpu=$index
    if [ "$method" = pad_ts ]; then gpu=$((index + 3)); fi
    session="kpd_${method}_${seed}"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Already running: $session"
      continue
    fi
    tmux new-session -d -s "$session" -c "$project_root" \
      ".venv/bin/python -u scripts/resume_fmri_venv.py --method $method --seed $seed --gpu $gpu > OUTPUT/venv_logs/${method}_${seed}.log 2>&1"
  done
done
if ! tmux has-session -t kpd_conference_fmri 2>/dev/null; then
  tmux new-session -d -s kpd_conference_fmri -c "$project_root" \
    'for seed in 2026 2027; do .venv/bin/python -u scripts/resume_fmri_venv.py --method k_protodiff --seed "$seed" --gpu 6 >> "OUTPUT/venv_logs/k_protodiff_${seed}.log" 2>&1 || exit; done'
fi
if ! tmux has-session -t kpd_conference_fmri_gpu7 2>/dev/null; then
  tmux new-session -d -s kpd_conference_fmri_gpu7 -c "$project_root" \
    '.venv/bin/python -u scripts/resume_fmri_venv.py --method k_protodiff --seed 2028 --gpu 7 >> OUTPUT/venv_logs/k_protodiff_2028_gpu7.log 2>&1'
fi
tmux list-sessions
