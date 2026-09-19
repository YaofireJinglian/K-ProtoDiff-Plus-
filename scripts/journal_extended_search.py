"""Prospectively specified, paired-three-seed exploratory validation search."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import rankdata

from scripts.journal_search import architecture, apply_options, save_json

SEEDS=[2031,2032,2033]
METRICS=['Context-FID','KL','DS_mean','PS_mean',
         'Segment-DTW-L6','Segment-DTW-L8','Segment-DTW-L10']


def validate_plan(plan,base):
    if plan['seeds']!=SEEDS or plan['evaluation_split']!='validation':
        raise ValueError('Fixed paired seeds and validation-only selection are required')
    if not plan.get('exploratory'):
        raise ValueError('This round must remain labeled exploratory')
    for trial in plan['training_trials']:
        if architecture(apply_options(base,trial['overrides']))!=architecture(base):
            raise ValueError('Architecture changes are forbidden')
        if trial.get('warm_start_from'):
            raise ValueError('New seeds require fresh model initialization')
    for sampler in plan['samplers']:
        if set(sampler['params'])-{'adaptive_sampling_timesteps','reflection_strength'}:
            raise ValueError('Unsupported sampling search dimension')


def expanded_specs(root):
    result=[]
    for path in sorted((root/'Config/journal_search_extended').glob('*.yaml')):
        plan=yaml.safe_load(path.read_text())
        base=yaml.safe_load((root/plan['base_config']).read_text())
        validate_plan(plan,base)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        for seed in SEEDS:
            spec=copy.deepcopy(plan)
            spec.update(namespace=f'journal_search_extended_seed{seed}',screen_seed=seed,
                        plan_sha256=digest)
            destination=root/'Config'/spec['namespace']/path.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if yaml.safe_load(destination.read_text())!=spec:
                    raise ValueError(f'Cannot mutate frozen extended spec: {destination}')
            else:
                destination.write_text(yaml.safe_dump(spec,sort_keys=False))
            result.append((destination,spec))
        aggregate(root,plan)
    # The existing scheduler retains ownership of all running jobs.
    return result


def paired_summary(groups,expected_seeds=SEEDS):
    """Rank full-precision three-seed means; never pick individual seed results."""
    means=[]
    summaries=[]
    for name,records in groups.items():
        if sorted(r['seed'] for r in records)!=expected_seeds:
            raise ValueError('Every candidate must contain exactly the fixed three seeds')
        if any(r['split']!='validation' for r in records):
            raise ValueError('Test scores cannot enter extended selection')
        if len({json.dumps([r['train_overrides'],r['sampler_params'],r['train_steps']],sort_keys=True) for r in records})!=1:
            raise ValueError('Paired seeds must use the same hyperparameters and budget')
        values=np.asarray([[r['metrics'][key] for key in METRICS] for r in records])
        if not np.isfinite(values).all(): raise ValueError('Nonfinite candidate score')
        mean,std=values.mean(0),values.std(0,ddof=1)
        means.append([*mean[:4],mean[4:].mean()])
        summaries.append({'candidate':name,'mean':dict(zip(METRICS,mean)),
                          'std':dict(zip(METRICS,std)),'records':records})
    values=np.asarray(means)
    ranks=np.stack([rankdata(values[:,i],method='average') for i in range(5)],axis=1)
    for item,r in zip(summaries,ranks): item['mean_validation_rank']=float(r.mean())
    winner=int(np.argmin(ranks.mean(1)))
    return summaries,summaries[winner]


def aggregate(root,plan):
    destination=root/'checkpoints/journal_search_extended'/plan['dataset']
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'selection.json').exists(): return
    groups={}
    for trial in plan['training_trials']:
        for sampler in plan['samplers']:
            name=trial['name']+'/'+sampler['name']
            records=[]
            for seed in SEEDS:
                path=root/'checkpoints'/f'journal_search_extended_seed{seed}'/plan['dataset']/trial['name']/sampler['name']/'record.json'
                if not path.exists(): return
                records.append(json.loads(path.read_text()))
            groups[name]=records
    summaries,winner=paired_summary(groups)
    report={'exploratory':True,'publication_eligible':False,'seeds':SEEDS,
            'selection_split':'validation','rule':'five-family mean rank of paired-three-seed means',
            'selected':winner,'candidates':summaries,
            'note':'Existing test results have already been observed. No new independent-test or SOTA claim.'}
    save_json(destination/'selection.json',report)
    base=yaml.safe_load((root/plan['base_config']).read_text())
    record=winner['records'][0]
    chosen=apply_options(base,record['train_overrides'])
    chosen['model']['params'].update(record['sampler_params'])
    chosen['solver']['max_epochs']=record['train_steps']
    chosen['solver']['save_cycle']=max(1,record['train_steps']//10)
    chosen['solver']['results_folder']=f'./checkpoints/journal_extended_candidates/{plan["dataset"]}'
    (destination/'candidate.yaml').write_text(yaml.safe_dump(chosen,sort_keys=False))
    lines=[f'# Extended exploratory validation: {plan["dataset"]}','',
           'Fixed seeds 2031/2032/2033; all seeds included. Not independent final-test results.',
           'Longer-training and longer-sampling candidates receive more compute.','',
           '| Candidate | C-FID | KL | DS | PS | DTW-6 | DTW-8 | DTW-10 | Mean rank |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in summaries:
        cells=[f'{item["mean"][key]:.3f} ± {item["std"][key]:.3f}' for key in METRICS]
        lines.append('| '+item['candidate']+' | '+' | '.join(cells)+f' | {item["mean_validation_rank"]:.3f} |')
    (destination/'VALIDATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
