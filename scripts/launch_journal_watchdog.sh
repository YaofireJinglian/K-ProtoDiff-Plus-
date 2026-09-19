#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
mkdir -p OUTPUT/journal_monitor
if ! tmux has-session -t kpd_journal_watchdog 2>/dev/null; then
  tmux new-session -d -s kpd_journal_watchdog -c "$project_root" \
    '.venv/bin/python -u scripts/journal_watchdog.py >> OUTPUT/journal_monitor/watchdog.log 2>&1'
fi
tmux list-sessions
