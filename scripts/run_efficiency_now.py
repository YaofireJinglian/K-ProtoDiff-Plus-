"""Pause only the verified in-project TimeGAN; restore it even on benchmark failure."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
PID=3980792


def main():
    lock=(ROOT/'OUTPUT/efficiency_runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=Path(f'/proc/{PID}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    cwd=Path(f'/proc/{PID}/cwd').resolve()
    if cwd != ROOT or not all(s in cmd for s in ['run_baseline_reproduction.py','--dataset eeg','--method timegan','--seed 2028','--gpu 1']):
        raise RuntimeError('Refusing to pause a different process')
    child=None
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]: signal.signal(sig,interrupted)
    try:
        os.kill(PID,signal.SIGSTOP)
        print(f'Paused in-project TimeGAN PID {PID}; other project processes untouched. Shared-GPU preliminary measurements.',flush=True)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        for script, output in [('benchmark_training_efficiency.py','training_efficiency_shared'),
                               ('benchmark_sampling_efficiency.py','efficiency_ablation_shared')]:
            child=subprocess.Popen([str(ROOT/'.venv-cu130/bin/python'),'-u',f'scripts/{script}',
                                    '--shared-gpu','--output',f'OUTPUT/{output}'],cwd=ROOT,env=env,start_new_session=True)
            result=child.wait()
            if result: raise RuntimeError(f'{script} failed with exit {result}')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            child.wait()
        if Path(f'/proc/{PID}').exists(): os.kill(PID,signal.SIGCONT)
        print('Resumed original TimeGAN.',flush=True)
        path=ROOT/'Config/journal_runtime.yaml'
        policy=yaml.safe_load(path.read_text())
        if policy.get('efficiency_reservation')=='sampling_efficiency_20260913':
            policy['allowed_gpus']=sorted(set(policy['allowed_gpus'])|{1})
            del policy['efficiency_reservation']
            temporary=path.with_suffix('.efficiency.tmp')
            temporary.write_text(yaml.safe_dump(policy,sort_keys=False))
            temporary.replace(path)


if __name__=='__main__': main()
