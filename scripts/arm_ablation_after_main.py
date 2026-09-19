"""Reserve GPU 1 and resume ablation once all runnable baselines finish."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'Config/journal_runtime.yaml'
STATE=ROOT/'OUTPUT/journal_monitor/scheduler_state.json'


def main():
    lock=(ROOT/'OUTPUT/journal_ablation/arm.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        policy=yaml.safe_load(POLICY.read_text())
        if policy.get('experiments_paused'):
            print('Experiments paused; ablation arm exiting.',flush=True)
            return
        requested={0,1,2,3}
        if not requested <= set(range(8)):
            raise RuntimeError('Invalid ablation GPU reservation')
        state=json.loads(STATE.read_text())
        jobs=state.get('uncertainty_jobs',{})
        terminal={'complete','failed_needs_review','skipped_by_user'}
        active=[key for key,job in jobs.items() if job['status'] not in terminal]
        if len(jobs)==191 and not active:
            policy['allowed_gpus']=sorted(set(policy['allowed_gpus'])-requested)
            policy['baseline_allowed_gpus']=sorted(set(policy.get('baseline_allowed_gpus',[]))-requested)
            policy['ablation_reserved_gpus']=sorted(requested)
            temporary=POLICY.with_suffix('.ablation_arm.tmp')
            temporary.write_text(yaml.safe_dump(policy,sort_keys=False))
            temporary.replace(POLICY)
            assignments={0:'stocks',1:'eeg',2:'traffic',3:'fmri'}
            for gpu,dataset in assignments.items():
                subprocess.run(['tmux','new-session','-d','-s',f'kpd_journal_ablation_gpu{gpu}','-c',str(ROOT),
                    f'.venv-cu130/bin/python -u scripts/run_ablation_queue.py --gpu {gpu} --datasets {dataset} '
                    f'>> OUTPUT/journal_ablation/queue_gpu{gpu}.log 2>&1'],check=True)
            print('Main runnable baselines drained; ablation queued on GPUs 0-3.',flush=True)
            return
        print(f'Waiting: {len(active)} runnable baselines remain.',flush=True)
        time.sleep(60)


if __name__=='__main__': main()
