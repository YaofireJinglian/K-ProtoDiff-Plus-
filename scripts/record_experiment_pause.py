"""Mark terminated project jobs as user-paused without losing completion records."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'OUTPUT/journal_monitor/scheduler_state.json'
status = json.loads(STATE.read_text())
paused_at = datetime.now(timezone.utc).isoformat()
for job in status.get('jobs', {}).values():
    if job.get('status') == 'running':
        job.update(status='paused_by_user', paused_at=paused_at)
for key, job in status.get('uncertainty_jobs', {}).items():
    if job.get('status') == 'running':
        record = ROOT / 'OUTPUT/main_results_records' / f'{key}.json'
        metrics = json.loads(record.read_text()).get('metrics', {}) if record.exists() else {}
        if {'C-FID', 'KL', 'DS', 'PS', 'Seg-DTW-L6', 'Seg-DTW-L8', 'Seg-DTW-L10'} <= set(metrics):
            job['status'] = 'complete'
        else:
            job.update(status='paused_by_user', paused_at=paused_at)
temporary = STATE.with_suffix('.pause.tmp')
temporary.write_text(json.dumps(status, indent=2) + '\n')
temporary.replace(STATE)

report = ROOT / 'BASELINE_UNCERTAINTY_STATUS.md'
lines = report.read_text().splitlines()
for index, line in enumerate(lines):
    if line.startswith('完成 '):
        jobs = status['uncertainty_jobs'].values()
        lines[index] = (f'完成 {sum(j["status"]=="complete" for j in jobs)}/{len(status["uncertainty_jobs"])}；'
                        f'用户暂停 {sum(j["status"]=="paused_by_user" for j in jobs)}；'
                        f'失败待检查 {sum(j["status"]=="failed_needs_review" for j in jobs)}。')
    elif line.startswith('| ') and '__seed' in line:
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        job = status['uncertainty_jobs'].get(cells[0])
        if job:
            cells[1] = job['status']
            lines[index] = '| ' + ' | '.join(cells) + ' |'
lines.insert(2, f'所有项目实验于 {paused_at} 暂停；本文件的 GPU 列为暂停前使用的设备。')
report.write_text('\n'.join(lines) + '\n')
print('Recorded pause at', paused_at)
