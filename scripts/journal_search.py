"""Isolated, held-out hyperparameter screening; never updates publication tables."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def apply_options(config, options):
    result = copy.deepcopy(config)
    for key, value in options.items():
        node = result
        parts = key.split('.')
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return result


def architecture(config):
    p = config['model']['params']
    keys = ['seq_length', 'feature_size', 'n_layer_enc', 'n_layer_dec', 'd_model',
            'n_heads', 'mlp_hidden_times', 'kernel_size', 'padding_size', 'fdm_mode',
            'prototype_mode', 'prototype_scales', 'num_prototypes', 'sampling_mode']
    return {k: p.get(k) for k in keys}


def prepare_data(base, out, limit, evaluation_split='validation'):
    """Split raw time points first: 70% train, 15% validation, 15% sealed test.

    EEG retains the original loader's class-segment window filtering and layout,
    but estimates cleaning and normalization statistics on the training block.
    """
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat, arff
    from sklearn.preprocessing import MinMaxScaler
    params = base['dataloader']['train_dataset']['params']
    source = ROOT / params['data_root']
    window = params['window']
    is_eeg = source.suffix == '.arff'
    labels = None
    if is_eeg:
        frame = pd.DataFrame(arff.loadarff(source)[0])
        labels = frame['eyeDetection'].astype(int).to_numpy()
        raw = frame.drop(columns=['eyeDetection']).to_numpy(dtype=np.float64)
    elif source.is_dir():
        raw = loadmat(source / 'sim4.mat')['ts'].astype(np.float64)
    else:
        raw = pd.read_csv(source).select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    a, b = int(len(raw)*0.7), int(len(raw)*0.85)
    # Never create a window crossing a split boundary. The last block is not read
    # by screening, and is not normalized or used for configuration selection.
    if evaluation_split not in ['validation','test']:
        raise ValueError('Unknown evaluation split')
    left, right = (a,b) if evaluation_split=='validation' else (b,len(raw))
    blocks = [raw[:a].copy(), raw[left:right].copy()]
    if is_eeg:
        mu, sigma = blocks[0].mean(0), blocks[0].std(0).clip(1e-12)
        for index, block in enumerate(blocks):
            block[np.abs((block-mu)/sigma)>3] = np.nan
            blocks[index] = pd.DataFrame(block).interpolate(limit_direction='both').to_numpy()
        center = blocks[0].mean(0)
        span = np.ptp(blocks[0], axis=0).clip(1e-12)
        blocks = [2*(block-center)/span for block in blocks]
    def windows(block, state=None):
        if state is None:
            return np.lib.stride_tricks.sliding_window_view(block, window, axis=0).transpose(0,2,1)
        # Match the existing EEG __Classify__ segment selection / flatten layout.
        edges = np.diff(state, prepend=state[0])
        starts = np.flatnonzero(edges == 1)[:-1]
        ends = np.flatnonzero(edges == -1)
        pieces = []
        for lefts, rights in [(np.insert(ends,0,0), starts), (starts, ends)]:
            for left, right in zip(lefts, rights):
                for pos in range(int(left)+50, int(right)-window-50):
                    pieces.append(block[pos:pos+window].T.reshape(window,-1))
        if not pieces:
            raise ValueError('No valid EEG windows in chronological split; do not substitute test data')
        return np.stack(pieces)
    train = windows(blocks[0], labels[:a] if is_eeg else None)
    val = windows(blocks[1], labels[left:right] if is_eeg else None)
    scaler = MinMaxScaler().fit(train.reshape(-1,raw.shape[1]))
    train = scaler.transform(train.reshape(-1,raw.shape[1])).reshape(train.shape).astype(np.float32)
    val = scaler.transform(val.reshape(-1,raw.shape[1])).reshape(val.shape).astype(np.float32)
    # Out-of-training-range validation values remain un-clipped.
    indices = np.random.default_rng(1701).choice(len(val), min(limit,len(val)), replace=False)
    val = val[np.sort(indices)]
    if min(len(train),len(val)) < 16 or not np.isfinite(train).all() or not np.isfinite(val).all():
        raise ValueError('Insufficient or nonfinite split data')
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/'validation.npy', val)
    np.savez(out/'train_scaler.npz', scale=scaler.scale_, offset=scaler.min_)
    save_json(out/'split.json', {
        'source':str(source), 'source_sha256':hashlib.sha256((source/'sim4.mat' if source.is_dir() else source).read_bytes()).hexdigest(),
        'raw_train_range':[0,a], 'raw_validation_range':[a,b], 'sealed_test_range':[b,len(raw)],
        'evaluation_split':evaluation_split,'evaluation_range':[left,right],
        'window':window, 'train_windows':len(train), 'validation_windows':len(val),
        'validation_indices':np.sort(indices).tolist(), 'fit_scaler_on':'train_only',
        'eeg_layout':'original_channel_major_reshape' if is_eeg else None,
    })
    return train, val


def train_model(config, data, out, steps, seed, warm_start_from=None):
    import numpy as np
    import torch
    from ema_pytorch import EMA
    from Utils.io_utils import instantiate_from_config, seed_everything
    seed_everything(seed)
    model = instantiate_from_config(config['model']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['solver']['base_lr'], betas=(0.9,0.96))
    ema = EMA(model, beta=config['solver']['ema']['decay'], update_every=config['solver']['ema']['update_interval']).cuda()
    schedule = copy.deepcopy(config['solver']['scheduler'])
    schedule['params']['optimizer'] = optimizer
    scheduler = instantiate_from_config(schedule)
    tensor = torch.from_numpy(data*2-1)
    batch_size = min(config['dataloader']['batch_size'],len(tensor))
    accumulation = config['solver']['gradient_accumulate_every']
    checkpoint = out/'latest.pt'
    if not checkpoint.exists() and warm_start_from:
        shutil.copy2(ROOT/warm_start_from,out/'latest.pt.tmp')
        (out/'latest.pt.tmp').replace(checkpoint)
    signature = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    generator = torch.Generator().manual_seed(seed+9000)
    order, cursor, start = torch.randperm(len(tensor),generator=generator), 0, 0
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['config_signature'] != signature:
            raise ValueError('Configuration changed for an existing search checkpoint')
        if state.get('seed',seed) != seed:
            raise ValueError('Cannot resume another seed')
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        generator.set_state(state['sampler_rng'])
        order, cursor, start = state['order'], state['cursor'], state['step']
        print(f'RESUMED step={start}',flush=True)
    model.train()
    started = time.monotonic()
    for step in range(start,steps):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(accumulation):
            if cursor+batch_size>len(order):
                order, cursor = torch.randperm(len(tensor),generator=generator), 0
            batch = tensor[order[cursor:cursor+batch_size]].cuda()
            cursor += batch_size
            loss = model(batch,target=batch)/accumulation
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss at step {step}')
            loss.backward(); loss_value += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); scheduler.step(loss_value); ema.update()
        if step % 100 == 0:
            print(f'TRAIN step={step+1}/{steps} loss={loss_value:.6f} lr={optimizer.param_groups[0]["lr"]:.8f} elapsed={time.monotonic()-started:.1f}',flush=True)
        if (step+1)%500 == 0 or step+1 == steps:
            state = {'seed':seed,'config_signature':signature,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                     'scheduler':scheduler.state_dict(),'step':step+1,'torch_rng':torch.get_rng_state(),
                     'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),
                     'python_rng':random.getstate(),'sampler_rng':generator.get_state(),'order':order,'cursor':cursor}
            torch.save(state,out/'latest.pt.tmp'); (out/'latest.pt.tmp').replace(checkpoint)
    return ema.ema_model.eval()


def run_trial(spec, train_options, gpu, smoke):
    import numpy as np
    import torch
    import yaml
    from Utils.io_utils import seed_everything
    base = yaml.safe_load((ROOT/spec['base_config']).read_text())
    config = apply_options(base,train_options['overrides'])
    assert architecture(config) == architecture(base), 'Architecture change forbidden'
    dataset, trial = spec['dataset'], train_options['name']
    namespace=spec.get('namespace','journal_search')
    if not namespace.replace('_','').isalnum():
        raise ValueError('Invalid artifact namespace')
    out = ROOT/('OUTPUT/journal_search_smoke' if smoke else f'checkpoints/{namespace}')/dataset/trial
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        return
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        evaluation_split=spec.get('evaluation_split','validation')
        data, val = prepare_data(base,out,32 if smoke else spec['validation_samples'],evaluation_split)
        steps = 2 if smoke else train_options.get('train_steps',spec['screen_steps'])
        model = train_model(config,data,out,steps,spec['screen_seed'],train_options.get('warm_start_from'))
        records = []
        samplers=train_options.get('samplers',spec['samplers'])
        for sampler in samplers[:1] if smoke else samplers:
            destination = out/sampler['name']; destination.mkdir(exist_ok=True)
            if (destination/'record.json').exists():
                records.append(json.loads((destination/'record.json').read_text())); continue
            params = {**base['model']['params'],**sampler['params']}
            for key in sampler['params']:
                setattr(model,key,params[key])
            if smoke:
                model.adaptive_sampling_timesteps = 2
            seed_everything(spec['screen_seed']+10000)
            count = len(val)
            chunks = []
            with torch.no_grad():
                for offset in range(0,count,config['dataloader']['sample_size']):
                    size = min(config['dataloader']['sample_size'],count-offset)
                    chunks.append(model.generate_mts(batch_size=size).cpu().numpy())
            fake = (np.concatenate(chunks)+1)/2
            if fake.shape != val.shape or not np.isfinite(fake).all():
                raise ValueError('Invalid generated array')
            np.save(destination/'samples.npy',fake)
            if smoke:
                continue
            command = [sys.executable,str(ROOT/'scripts/eval_seeded.py'),'--root',str(destination),
                       '--ori_path',str(out/'validation.npy'),'--fake_path',str(destination/'samples.npy'),
                       '--n_iter','1','--seed',str(spec['screen_seed'])]
            with (destination/'evaluation.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics = dict((k,float(v)) for k,v in (line.split(':',1) for line in (destination/'metrics.txt').read_text().splitlines()))
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError('Nonfinite metrics')
            record = {'dataset':dataset,'trial':trial,'sampler':sampler['name'], 'seed':spec['screen_seed'],
                      'train_steps':steps,'metrics':metrics,'split':evaluation_split,'publication_eligible':False,
                      'exploratory':spec.get('exploratory',False),
                      'train_overrides':train_options['overrides'],'sampler_params':sampler['params']}
            save_json(destination/'record.json',record); records.append(record)
            print(f'EVALUATED {dataset}/{trial}/{sampler["name"]} {metrics}',flush=True)
        save_json(out/'complete.json',{'records':records,'smoke':smoke})
        print(f'COMPLETE {dataset}/{trial} smoke={smoke}',flush=True)


def select(spec):
    import numpy as np
    import yaml
    from scipy.stats import rankdata
    records = []
    if spec.get('evaluation_split','validation') != 'validation':
        raise ValueError('Never select configurations using test results')
    root = ROOT/'checkpoints'/spec.get('namespace','journal_search')/spec['dataset']
    for trial in spec['training_trials']:
        path = root/trial['name']/'complete.json'
        if not path.exists(): return
        records.extend(json.loads(path.read_text())['records'])
    values = []
    for rec in records:
        m = rec['metrics']
        values.append([m['Context-FID'],m['KL'],m['DS_mean'],m['PS_mean'],
                       np.mean([m[f'Segment-DTW-L{length}'] for length in [6,8,10]])])
    values = np.array(values)
    ranks = np.stack([rankdata(values[:,i],method='average') for i in range(5)],axis=1)
    winner = int(np.argmin(ranks.mean(axis=1)))
    selected = records[winner]
    save_json(root/'selection.json',{'rule':'equal-weight mean validation rank over five metric families',
              'selected':selected,'candidates':records,'ranks':ranks.tolist(),
              'status':'screening only; requires full-budget three-seed confirmation before publication'})
    base = yaml.safe_load((ROOT/spec['base_config']).read_text())
    chosen = apply_options(base,selected['train_overrides'])
    chosen['model']['params'].update(selected['sampler_params'])
    assert architecture(chosen)==architecture(base)
    chosen['solver']['results_folder']=f'./checkpoints/journal_candidates/{spec["dataset"]}'
    (root/'candidate.yaml').write_text(yaml.safe_dump(chosen,sort_keys=False))
    text = '# Validation screening: '+spec['dataset']+'\n\nNot publication results; all values are single-seed screening scores.\n\n'
    text += '| Candidate | C-FID | KL | DS | PS | Mean segment-DTW | Mean rank |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for rec,row,r in zip(records,values,ranks):
        text += '| '+rec['trial']+'/'+rec['sampler']+' | '+' | '.join(f'{x:.3f}' for x in [*row,r.mean()])+' |\n'
    (root/'VALIDATION_RESULTS.md').write_text(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--trial')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--spec-file')
    args=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2')
    import yaml
    if args.trial:
        spec_path=Path(args.spec_file) if args.spec_file else ROOT/'Config/journal_search'/f'{args.datasets[0]}.yaml'
        spec=yaml.safe_load(spec_path.read_text())
        trial=next(t for t in spec['training_trials'] if t['name']==args.trial)
        run_trial(spec,trial,args.gpu,args.smoke)
        return
    failures=[]
    for dataset in args.datasets:
        spec=yaml.safe_load((ROOT/'Config/journal_search'/f'{dataset}.yaml').read_text())
        for trial in spec['training_trials']:
            logs=ROOT/'OUTPUT/journal_search_logs'; logs.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--gpu',str(args.gpu),
                     '--datasets',dataset,'--trial',trial['name']]
            with (logs/f'{dataset}_{trial["name"]}.log').open('a') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                failures.append([dataset,trial['name'],result.returncode])
        select(spec)
    if failures:
        raise RuntimeError(f'Failed trials: {failures}')


if __name__=='__main__':
    main()
