"""Validation-only refinement around the completed extended-search winners."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import rankdata

from scripts.journal_search import architecture, apply_options, save_json

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2041, 2042, 2043]
METRICS = ['Context-FID', 'KL', 'DS_mean', 'PS_mean',
           'Segment-DTW-L6', 'Segment-DTW-L8', 'Segment-DTW-L10']


def initialize(root):
    destination = root/'Config/journal_search_refinement'
    destination.mkdir(parents=True, exist_ok=True)
    for selected_path in sorted((root/'checkpoints/journal_search_extended').glob('*/selection.json')):
        dataset = selected_path.parent.name
        target = destination/f'{dataset}.yaml'
        if target.exists():
            continue
        selected = json.loads(selected_path.read_text())['selected']['records'][0]
        incumbent = copy.deepcopy(selected['train_overrides'])
        plans = [
            {'name':'incumbent', 'train_steps':selected['train_steps'], 'overrides':incumbent},
            {'name':'prototype_005', 'train_steps':selected['train_steps'],
             'overrides':{**incumbent, 'model.params.prototype_loss_weight':0.005}},
            {'name':'prototype_020', 'train_steps':selected['train_steps'],
             'overrides':{**incumbent, 'model.params.prototype_loss_weight':0.02}},
        ]
        winner_sampler = selected['sampler_params']
        samplers = [
            {'name':'incumbent', 'params':winner_sampler},
            {'name':'refine150', 'params':{'adaptive_sampling_timesteps':150,
                                           'reflection_strength':0.03}},
            {'name':'refine300', 'params':{'adaptive_sampling_timesteps':300,
                                           'reflection_strength':0.01}},
        ]
        # Keep unique candidates while preserving their prospective names.
        unique=[]; seen=set()
        for item in samplers:
            signature=json.dumps(item['params'],sort_keys=True)
            if signature not in seen:
                seen.add(signature); unique.append(item)
        plan = {'dataset':dataset, 'base_config':f'Config/journal/{dataset}.yaml',
                'seeds':SEEDS, 'screen_steps':selected['train_steps'],
                'validation_samples':1024, 'evaluation_split':'validation',
                'exploratory':True, 'parent_selection':str(selected_path.relative_to(root)),
                'training_trials':plans, 'samplers':unique}
        target.write_text(yaml.safe_dump(plan,sort_keys=False))


def validate(plan, base):
    if plan['seeds'] != SEEDS or plan['evaluation_split'] != 'validation' or not plan['exploratory']:
        raise ValueError('Refinement requires fixed paired seeds and validation-only selection')
    for trial in plan['training_trials']:
        if architecture(apply_options(base,trial['overrides'])) != architecture(base):
            raise ValueError('Architecture changes are forbidden')
        if trial.get('warm_start_from'):
            raise ValueError('Every refinement seed must train from fresh initialization')
    permitted={'adaptive_sampling_timesteps','reflection_strength'}
    if any(set(s['params'])-permitted for s in plan['samplers']):
        raise ValueError('Unsupported sampling dimension')


def specs(root):
    initialize(root)
    result=[]
    for path in sorted((root/'Config/journal_search_refinement').glob('*.yaml')):
        plan=yaml.safe_load(path.read_text())
        base=yaml.safe_load((root/plan['base_config']).read_text())
        validate(plan,base)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        for seed in SEEDS:
            spec=copy.deepcopy(plan)
            spec.update(namespace=f'journal_search_refinement_seed{seed}',screen_seed=seed,
                        plan_sha256=digest)
            destination=root/'Config'/spec['namespace']/path.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists() and yaml.safe_load(destination.read_text()) != spec:
                raise ValueError(f'Cannot mutate frozen refinement spec: {destination}')
            if not destination.exists():
                destination.write_text(yaml.safe_dump(spec,sort_keys=False))
            result.append((destination,spec))
        aggregate(root,plan)
    return result


def aggregate(root, plan):
    output=root/'checkpoints/journal_search_refinement'/plan['dataset']
    output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists(): return
    summaries=[]; values=[]
    for trial in plan['training_trials']:
        for sampler in plan['samplers']:
            records=[]
            for seed in SEEDS:
                path=(root/'checkpoints'/f'journal_search_refinement_seed{seed}'/
                      plan['dataset']/trial['name']/sampler['name']/'record.json')
                if not path.exists(): return
                records.append(json.loads(path.read_text()))
            raw=np.asarray([[r['metrics'][key] for key in METRICS] for r in records])
            mean,std=raw.mean(0),raw.std(0,ddof=1)
            values.append([*mean[:4],mean[4:].mean()])
            summaries.append({'candidate':trial['name']+'/'+sampler['name'],
                              'mean':dict(zip(METRICS,mean)), 'std':dict(zip(METRICS,std)),
                              'records':records})
    ranks=np.stack([rankdata(np.asarray(values)[:,i],method='average') for i in range(5)],axis=1)
    for item,row in zip(summaries,ranks): item['mean_validation_rank']=float(row.mean())
    winner=summaries[int(np.argmin(ranks.mean(1)))]
    save_json(output/'selection.json',{'exploratory':True,'publication_eligible':False,
        'seeds':SEEDS,'selection_split':'validation','selected':winner,'candidates':summaries,
        'rule':'five-family mean rank of paired-three-seed means'})
    lines=[f'# Refinement validation: {plan["dataset"]}','','Fixed seeds 2041/2042/2043. Validation only.','',
           '| Candidate | C-FID | KL | DS | PS | DTW6 | DTW8 | DTW10 | Mean rank |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in summaries:
        cells=[f'{item["mean"][key]:.3f} ± {item["std"][key]:.3f}' for key in METRICS]
        lines.append('| '+item['candidate']+' | '+' | '.join(cells)+f' | {item["mean_validation_rank"]:.3f} |')
    (output/'VALIDATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
