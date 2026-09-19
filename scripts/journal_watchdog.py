"""Persistent bounded tuning scheduler and experiment monitor.

No architecture mutations, test-driven selection, publication-table writes,
or termination of other running experiments are performed here.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.journal_search import save_json, select, architecture, apply_options
from scripts.journal_extended_search import expanded_specs
from scripts.journal_refinement_search import specs as refinement_specs
from scripts.baseline_reproduction_queue import tick as uncertainty_tick

DATASETS=['fmri','energy','traffic','electricity','etth','weather','illness','exchange','stocks','eeg']
STATE=ROOT/'OUTPUT/journal_monitor'
PYTHON=ROOT/'.venv/bin/python'


def read(path):
    return json.loads(path.read_text())


def sessions():
    result=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
    return set(result.stdout.splitlines())


def artifact(spec,trial):
    return ROOT/'checkpoints'/spec.get('namespace','journal_search')/spec['dataset']/trial


def emit_spec(folder,dataset,spec):
    path=ROOT/'Config'/folder/f'{dataset}.yaml'
    path.parent.mkdir(parents=True,exist_ok=True)
    # Once emitted, a spec is immutable. Changing it requires a new namespace.
    if not path.exists():
        path.write_text(yaml.safe_dump(spec,sort_keys=False))
    return path,yaml.safe_load(path.read_text())


def next_specs():
    result=[]
    for dataset in DATASETS:
        first=yaml.safe_load((ROOT/'Config/journal_search'/f'{dataset}.yaml').read_text())
        first_root=ROOT/'checkpoints/journal_search'/dataset
        selected_file=first_root/'selection.json'
        if not selected_file.exists(): continue
        selected=read(selected_file)['selected']
        spec=copy.deepcopy(first)
        spec.update(namespace='journal_search_round2',screen_steps=6000)
        control={'name':'control','overrides':{},'warm_start_from':str(first_root/'control/latest.pt')}
        trials=[control]
        if selected['train_overrides']:
            trials.append({'name':'incumbent','overrides':selected['train_overrides'],
                           'warm_start_from':str(first_root/selected['trial']/'latest.pt')})
        refined=dict(selected['train_overrides'])
        refined.update({'solver.scheduler.params.warmup_lr':0.0003,
                        'model.params.prototype_loss_weight':0.01})
        if refined in [t['overrides'] for t in trials]:
            refined['model.params.prototype_loss_weight']=0.1
        trials.append({'name':'refined','overrides':refined})
        spec['training_trials']=trials
        incumbent=selected['sampler_params']
        neighbor=dict(incumbent)
        neighbor['adaptive_sampling_timesteps']=min(300,int(incumbent['adaptive_sampling_timesteps'])*2)
        neighbor['reflection_strength']=max(0.005,float(incumbent['reflection_strength'])/2)
        samplers=[{'name':'original50','params':first['samplers'][0]['params']}]
        for name,params in [('incumbent_sampling',incumbent),('refined_sampling',neighbor)]:
            if params not in [s['params'] for s in samplers]: samplers.append({'name':name,'params':params})
        spec['samplers']=samplers
        path,spec=emit_spec('journal_search_round2',dataset,spec)
        result.append((path,spec))
        if all((artifact(spec,t['name'])/'complete.json').exists() for t in spec['training_trials']):
            selection=artifact(spec,'selection.json')
            if not selection.exists(): select(spec)
            frozen=read(selection)['selected']
            base=yaml.safe_load((ROOT/spec['base_config']).read_text())
            assert architecture(apply_options(base,frozen['train_overrides']))==architecture(base)
            # The choice is frozen using validation BEFORE inspecting test metrics.
            for seed in [2026,2027,2028]:
                confirmation=copy.deepcopy(spec)
                namespace=f'journal_confirmation_seed{seed}'
                confirmation.update(namespace=namespace,screen_seed=seed,
                                    screen_steps=base['solver']['max_epochs'],evaluation_split='test')
                confirmation['frozen_validation_choice']=frozen
                original_sampler=first['samplers'][0]
                chosen_sampler={'name':'frozen_selected','params':frozen['sampler_params']}
                control={'name':'control','overrides':{},'samplers':[original_sampler]}
                candidate={'name':'selected','overrides':frozen['train_overrides'],'samplers':[chosen_sampler]}
                if seed==2026:
                    control['warm_start_from']=str(artifact(spec,'control')/'latest.pt')
                    candidate['warm_start_from']=str(artifact(spec,frozen['trial'])/'latest.pt')
                if not frozen['train_overrides']:
                    # Same trained weights can serve both samplers without retraining.
                    candidate['warm_start_from']=str(ROOT/'checkpoints'/namespace/dataset/'control/latest.pt')
                    candidate['depends_on']=str(ROOT/'checkpoints'/namespace/dataset/'control/complete.json')
                confirmation['training_trials']=[control,candidate]
                path,confirmation=emit_spec(namespace,dataset,confirmation)
                result.append((path,confirmation))
    return result


def gpu_state():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=[]
    for line in result.stdout.splitlines():
        index,uuid,util,memory=[v.strip() for v in line.split(',')]
        rows.append({'gpu':int(index),'uuid':uuid,'utilization':int(util),'memory_mib':int(memory)})
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],
                        capture_output=True,text=True,check=True).stdout
    occupied={line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
    return rows,{row['gpu'] for row in rows if row['uuid'] in occupied}


def runtime_health(state,gpus):
    path=ROOT/'Config/journal_runtime.yaml'
    policy=yaml.safe_load(path.read_text()) if path.exists() else {}
    allowed=set(policy.get('allowed_gpus',range(8)))
    rows=[row for row in gpus if row['gpu'] in allowed]
    health=state.setdefault('cuda_health',{})
    alerts=[]
    missing=allowed-{row['gpu'] for row in rows}
    if missing:
        alerts.append(f'允许使用但 NVML 无法枚举的 GPU：{sorted(missing)}；强制复查其余卡 CUDA 健康')
    unhealthy=set()
    for row in rows:
        key=row['uuid']
        cached=health.get(key,{})
        if missing or time.time()-cached.get('checked_at',0)>=policy.get('cuda_probe_interval_seconds',300):
            env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=key
            try:
                probe=subprocess.run([PYTHON,'-c',
                    'import ctypes,sys; code=ctypes.CDLL("libcuda.so.1").cuInit(0); print(code); sys.exit(0 if code==0 else 1)'],
                    env=env,capture_output=True,text=True,timeout=15)
                cached={'checked_at':time.time(),'healthy':probe.returncode==0,
                        'detail':(probe.stdout+probe.stderr)[-500:]}
            except subprocess.TimeoutExpired:
                cached={'checked_at':time.time(),'healthy':False,'detail':'cuInit timeout'}
            health[key]=cached
        if not cached.get('healthy',False):
            unhealthy.add(row['gpu'])
            alerts.append(f'GPU {row["gpu"]} CUDA 初始化失败，禁止调度：{cached.get("detail","").strip()}')
    # Explicit user-requested recovery is consumed only once a permitted GPU is healthy.
    request=policy.get('cuda_recovery_request')
    if request and state.get('cuda_recovery_consumed')!=request and len(unhealthy)<len(rows):
        recovered=[]
        for key,job in state['jobs'].items():
            if job['status']!='failed': continue
            log=STATE/'logs'/f'{key}.log'
            if not log.exists(): continue
            with log.open('rb') as handle:
                handle.seek(max(0,log.stat().st_size-6000))
                tail=handle.read().decode(errors='replace')
            if 'CUDA unknown error' not in tail and 'unspecified launch failure' not in tail:
                continue
            job.setdefault('retry_history',[]).append({'request':request,'attempts':job['attempts'],
                                                      'recorded_at':time.time()})
            job.update(status='pending',attempts=0,retry_after=0)
            recovered.append(key)
        state['cuda_recovery_consumed']=request
        state.setdefault('cuda_recovery_events',[]).append({'request':request,'jobs':recovered})
    return rows,unhealthy,alerts


def launch(name,command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    shell=' '.join(shlex.quote(str(v)) for v in command)+' >> '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',name,'-c',str(ROOT),shell],check=True)


def update_report(state,gpus,active,alerts,specs):
    now=datetime.now(timezone.utc).isoformat()
    round1=sum((ROOT/'checkpoints/journal_search'/d/t/'complete.json').exists()
               for d in DATASETS for t in ['control','lower_lr','lower_prototype_weight'])
    snapshot={'updated_utc':now,'gpus':gpus,'active_sessions':sorted(active),
              'round1_training_trials_complete':round1,'round1_training_trials_total':30,
              'jobs':state['jobs'],'alerts':alerts}
    save_json(STATE/'status.json',snapshot)
    lines=['# 实验自动监控','',f'更新时间（UTC）：{now}', '',
           f'首轮训练及对应评测完成：{round1}/30 组训练配置。',
           '每 60 秒巡检；原实验保持运行，空卡调度后续任务。',
           '后续流程：验证集定向搜索 → 固定配置 → 完整预算、三种子测试。',
           '测试结果不用于重新选择参数；不会自动替换论文主表。','',
           '| GPU | 利用率 | 显存 MiB |','|---|---:|---:|']
    for row in gpus: lines.append(f'| {row["gpu"]} | {row["utilization"]}% | {row["memory_mib"]} |')
    extended=[job for key,job in state['jobs'].items() if key.startswith('journal_search_extended_seed')]
    if extended:
        lines.extend(['',f'新增探索轮：{sum(j["status"]=="complete" for j in extended)}/{len(extended)} 组训练完成；'
                      '固定种子 2031/2032/2033，按验证集三种子均值选择，不追加测试集选优。'])
    lines.extend(['','## 自动调度任务',''])
    for key,job in state['jobs'].items():
        lines.append(f'- {key}: {job["status"]}; GPU {job.get("gpu","—")}; 启动次数 {job.get("attempts",0)}')
    lines.extend(['','## 告警',''])
    lines.extend(['- '+alert for alert in alerts] or ['暂无检测到的告警。'])
    (ROOT/'EXPERIMENT_STATUS.md').write_text('\n'.join(lines)+'\n')
    # All entries here are frozen-choice, fresh held-out confirmation; not rank claims.
    lines=['# 固定配置三种子测试复核','',
           '仅比较同一新划分下的 control 与固定候选。不得与旧全数据主表直接作排名比较。',
           '候选按验证集预先固定，以下测试结果不反馈给搜索。每格均值 ± 样本标准差。','',
           '| 数据集 | 配置 | C-FID | KL | DS | PS | DTW-6 | DTW-8 | DTW-10 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=['Context-FID','KL','DS_mean','PS_mean','Segment-DTW-L6','Segment-DTW-L8','Segment-DTW-L10']
    for dataset in DATASETS:
        for trial in ['control','selected']:
            records=[]
            for seed in [2026,2027,2028]:
                path=ROOT/'checkpoints'/f'journal_confirmation_seed{seed}'/dataset/trial/'complete.json'
                if path.exists(): records.append(read(path)['records'][0])
            if len(records)==3:
                cells=[]
                for key in keys:
                    values=[r['metrics'][key] for r in records]
                    cells.append(f'{np.mean(values):.3f} ± {np.std(values,ddof=1):.3f}')
            else: cells=[f'待完成（{len(records)}/3）']*len(keys)
            lines.append('| '+dataset+' | '+trial+' | '+' | '.join(cells)+' |')
    (ROOT/'JOURNAL_CONFIRMATION_RESULTS.md').write_text('\n'.join(lines)+'\n')


def tick(state,no_launch):
    global PYTHON
    policy_path=ROOT/'Config/journal_runtime.yaml'
    policy=yaml.safe_load(policy_path.read_text()) if policy_path.exists() else {}
    if policy.get('experiments_paused', False):
        print(datetime.now(timezone.utc).isoformat(), 'experiments paused by user; no jobs launched', flush=True)
        return
    PYTHON=ROOT/policy.get('pytorch_python','.venv/bin/python')
    if not PYTHON.is_file():
        raise FileNotFoundError(f'Configured PyTorch interpreter is unavailable: {PYTHON}')
    active=sessions()
    gpus,occupied=gpu_state()
    gpus,unhealthy,alerts=runtime_health(state,gpus)
    occupied.update(unhealthy)
    # Explicit user request to start on cards already used by other projects.
    # Only launch a project job when there is ample free memory, and keep one
    # project job per card even though external processes may share it.
    share=set(policy.get('shared_gpu_launch_gpus',[]))
    project_busy={job.get('gpu') for group in ('jobs','uncertainty_jobs')
                  for key,job in state.get(group,{}).items()
                  if job.get('status')=='running' and
                  (('kpd_auto_'+key) if group=='jobs' else ('kpd_uncertainty_'+key)) in active}
    for row in gpus:
        gpu=row['gpu']
        if (gpu in share and gpu not in unhealthy and gpu not in project_busy
                and row['memory_mib']+policy.get('shared_gpu_min_free_mib',16000) <= 32607):
            occupied.discard(gpu)
    allowed={row['gpu'] for row in gpus}
    # Reserve cards even during startup / CPU evaluation, before CUDA initializes.
    for seed in [2026,2027,2028]:
        for method,gpu in [('gtgan',seed-2026),('timegan',seed-2023)]:
            name=f'kpd_{method}_{seed}'
            if name in active: occupied.add(gpu)
            elif not (ROOT/'OUTPUT/main_results_records'/f'fmri__{method}__seed{seed}.json').exists():
                alerts.append(f'{method} seed {seed}: 会话已退出但缺少完成记录')
                # Only TimeGAN has a tested mid-training resume path.
                counter=state.setdefault('baseline_retries',{}).get(name,0)
                if method=='timegan' and counter<2 and gpu in allowed and gpu not in occupied and not no_launch:
                    launch(name,['bash','scripts/run_tf_baseline.sh','--tf-xla','--method',method,
                                 '--seed',seed,'--gpu',gpu],ROOT/'OUTPUT/venv_logs'/f'{method}_{seed}.log')
                    state['baseline_retries'][name]=counter+1; occupied.add(gpu); active.add(name)
    for gpu,datasets in [(6,['fmri','traffic','etth','illness','stocks']),
                         (7,['energy','electricity','weather','exchange','eeg'])]:
        name=f'kpd_journal_search_gpu{gpu}'
        complete=all((ROOT/'checkpoints/journal_search'/d/t/'complete.json').exists()
                     for d in datasets for t in ['control','lower_lr','lower_prototype_weight'])
        if name in active: occupied.add(gpu)
        elif not complete:
            counter=state.setdefault('queue_retries',{}).get(name,0)
            if counter<2 and gpu in allowed and gpu not in occupied and not no_launch:
                launch(name,[PYTHON,'-u','scripts/journal_search.py','--gpu',gpu,'--datasets',*datasets],
                       ROOT/'OUTPUT/journal_search_logs'/f'worker_gpu{gpu}.log')
                state['queue_retries'][name]=counter+1; occupied.add(gpu); active.add(name)
            else: alerts.append(f'首轮 GPU {gpu} 队列未完成，重试次数 {counter}/2')
    search_enabled=policy.get('parameter_search_enabled',True)
    specs=(next_specs()+expanded_specs(ROOT)+refinement_specs(ROOT)) if search_enabled else []
    ready=[]
    for path,spec in specs:
        for trial in spec['training_trials']:
            key=f'{spec["namespace"]}_{spec["dataset"]}_{trial["name"]}'
            job=state['jobs'].setdefault(key,{'status':'pending','attempts':0})
            name='kpd_auto_'+key
            destination=artifact(spec,trial['name'])
            if (destination/'complete.json').exists(): job['status']='complete'; continue
            if name in active:
                job['status']='running'; occupied.add(job['gpu'])
                logs=[p for p in destination.rglob('*.log')]
                logs.append(STATE/'logs'/f'{key}.log')
                recent=max([p.stat().st_mtime for p in logs if p.exists()] or [time.time()])
                if time.time()-recent>3600: alerts.append(f'{key}: 日志超过 60 分钟未更新，保留进程待检查')
                continue
            if job['status']=='running':
                job['status']='retry_pending'; job['retry_after']=time.time()+300
            if job['attempts']>=3:
                job['status']='failed'; alerts.append(f'{key}: 连续失败，已停止该分支'); continue
            if time.time()<job.get('retry_after',0): continue
            if trial.get('depends_on') and not Path(trial['depends_on']).exists(): continue
            ready.append((key,name,path,spec,trial,job))
    baseline_gpus=set(policy.get('baseline_allowed_gpus',allowed))
    search_gpus=set(policy.get('search_allowed_gpus',allowed))
    if baseline_gpus & search_gpus:
        raise ValueError('Baseline and search GPU pools must be disjoint')
    if not baseline_gpus <= allowed or not search_gpus <= allowed:
        raise ValueError('GPU pools must be subsets of allowed_gpus')
    uncertainty_tick(ROOT,state,active,[row for row in gpus if row['gpu'] in baseline_gpus],
                     occupied,launch,no_launch,python_path=PYTHON)
    free=[row['gpu'] for row in gpus if row['gpu'] in search_gpus and row['gpu'] not in occupied]
    if not no_launch:
        for gpu,entry in zip(free,ready):
            key,name,path,spec,trial,job=entry
            launch(name,[PYTHON,'-u','scripts/journal_search.py','--gpu',gpu,'--datasets',spec['dataset'],
                         '--trial',trial['name'],'--spec-file',path],STATE/'logs'/f'{key}.log')
            job.update(status='running',gpu=gpu,attempts=job['attempts']+1,launched_at=time.time())
            save_json(STATE/'scheduler_state.json',state)
            active.add(name)
    # Surface failures from the legacy queues too, even before a worker exits.
    for path in (ROOT/'OUTPUT/journal_search_logs').glob('*.log'):
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-12000))
            tail=handle.read().decode(errors='replace')
        if 'Traceback (most recent call last)' in tail or 'Nonfinite' in tail:
            alerts.append(f'搜索日志含异常（需确认是否已恢复）：{path.name}')
    update_report(state,gpus,active,alerts,specs)
    save_json(STATE/'scheduler_state.json',state)
    print(datetime.now(timezone.utc).isoformat(),f'active={len(active)} ready={len(ready)} alerts={len(alerts)}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--no-launch',action='store_true')
    args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'watchdog.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=STATE/'scheduler_state.json'
        state=read(path) if path.exists() else {'jobs':{}}
        while True:
            try: tick(state,args.no_launch)
            except Exception as error:
                import traceback
                traceback.print_exc()
                (STATE/'watchdog_error.txt').write_text(f'{datetime.now(timezone.utc).isoformat()} {error!r}\n')
                if args.once: raise
            if args.once: return
            policy_path=ROOT/'Config/journal_runtime.yaml'
            policy=yaml.safe_load(policy_path.read_text()) if policy_path.exists() else {}
            time.sleep(max(10,policy.get('poll_seconds',60)))


if __name__=='__main__':
    main()
