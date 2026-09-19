#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
mkdir -p OUTPUT/venv_logs
for method in gtgan timevqvae timegan timevae; do
  for index in 0 1 2; do
    seed=$((2026+index))
    session="kpd_${method}_${seed}"
    if tmux has-session -t "$session" 2>/dev/null; then continue; fi
    if [ -f "OUTPUT/main_results_records/fmri__${method}__seed${seed}.json" ]; then continue; fi
    case "$method" in
      gtgan) gpu=$index; runner='.venv/bin/python -u scripts/run_remaining_fmri.py' ;;
      timevqvae) gpu=$((3+index)); runner='.venv/bin/python -u scripts/run_remaining_fmri.py' ;;
      timegan) gpu=$((3+index)); runner='bash scripts/run_tf_baseline.sh --tf-xla' ;;
      timevae) gpu=$((6+index%2)); runner='bash scripts/run_tf_baseline.sh' ;;
    esac
    tmux new-session -d -s "$session" -c "$project_root" \
      "$runner --method $method --seed $seed --gpu $gpu >> OUTPUT/venv_logs/${method}_${seed}.log 2>&1"
  done
done
tmux list-sessions
