"""Run the efficiency ablation exclusively after GPU 1's existing job exits."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'Config/journal_runtime.yaml'
TOKEN='sampling_efficiency_20260913'


def main():
    lock=(ROOT/'OUTPUT/efficiency_runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy=yaml.safe_load(POLICY.read_text())
    if policy.get('efficiency_reservation')!=TOKEN or 1 in policy['allowed_gpus']:
        raise RuntimeError('GPU 1 must be reserved before starting')
    print('Waiting for EEG TimeGAN seed2028 and other GPU 1 compute processes to exit.',flush=True)
    try:
        while True:
            job_alive=subprocess.run(['tmux','has-session','-t','kpd_uncertainty_eeg__timegan__seed2028'],capture_output=True).returncode==0
            uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
            if not job_alive and uuid not in processes:
                break
            time.sleep(30)
        print('GPU 1 idle: starting formal efficiency ablation.',flush=True)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        for script in ['scripts/benchmark_training_efficiency.py', 'scripts/benchmark_sampling_efficiency.py']:
            subprocess.run([str(ROOT/'.venv-cu130/bin/python'),'-u',script],cwd=ROOT,env=env,check=True)
    finally:
        policy=yaml.safe_load(POLICY.read_text())
        if policy.get('efficiency_reservation')==TOKEN:
            policy['allowed_gpus']=sorted(set(policy['allowed_gpus'])|{1})
            del policy['efficiency_reservation']
            temporary=POLICY.with_suffix('.efficiency.tmp')
            temporary.write_text(yaml.safe_dump(policy,sort_keys=False))
            temporary.replace(POLICY)
            print('Released GPU 1 back to main experiment queue.',flush=True)


if __name__=='__main__': main()
