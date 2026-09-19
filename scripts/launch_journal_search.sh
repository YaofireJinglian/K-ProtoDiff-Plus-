#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
mkdir -p OUTPUT/journal_search_logs
for gpu in 6 7; do
  session="kpd_journal_search_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then continue; fi
  if [ "$gpu" -eq 6 ]; then
    datasets='fmri traffic etth illness stocks'
  else
    datasets='energy electricity weather exchange eeg'
  fi
  tmux new-session -d -s "$session" -c "$project_root" \
    ".venv/bin/python -u scripts/journal_search.py --gpu $gpu --datasets $datasets >> OUTPUT/journal_search_logs/worker_gpu${gpu}.log 2>&1"
done
tmux list-sessions
