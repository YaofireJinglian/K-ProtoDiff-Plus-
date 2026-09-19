import json
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_main_results_table import aggregate, build_markdown
from build_segment_dtw_table import build
from baseline_reproduction_queue import entries, tick


def test_fixed_seed_inventory_and_configs():
    jobs = list(entries())
    assert len(jobs) == len(set(jobs)) == 191
    for dataset, method, seed in jobs:
        spec = yaml.safe_load((ROOT/'Config/baselines'/f'{dataset}.yaml').read_text())
        assert seed in spec['seeds'] == [2026,2027,2028]
        assert method in spec
        assert (ROOT/spec['real_path']).exists()
        if method == 'k_protodiff':
            assert spec[method]['config']['model']['params']['prototype_mode'] == 'point'


def records():
    return [dict(dataset='Stocks', method='TimeVAE', seed=seed,
                 metrics={m:float(i) for m in ['C-FID','KL','DS','PS','Seg-DTW-L6','Seg-DTW-L8','Seg-DTW-L10']})
            for i,seed in enumerate([2026,2027,2028],start=1)]


def test_replicated_mean_and_sd_replace_paper_together(tmp_path):
    rs = records()
    assert '2.000 ± 1.000 †' in build_markdown(aggregate(rs),3)
    assert ' † |' not in build_markdown(aggregate(rs[:2]),3)
    for r in rs:
        (tmp_path/f'{r["seed"]}.json').write_text(json.dumps(r))
    assert build(tmp_path,3).count('2.000 ± 1.000 †') == 3
    (tmp_path/'duplicate.json').write_text(json.dumps(rs[0]))
    with pytest.raises(ValueError,match='duplicate'):
        build(tmp_path,3)


def test_nonfinite_rejected():
    rs = records()
    rs[0]['metrics']['KL'] = float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        aggregate(rs)


def test_unhealthy_or_occupied_cards_never_launch(tmp_path):
    launches = []
    state = {}
    tick(tmp_path,state,set(),[{'gpu':0},{'gpu':1}],{0,1},lambda *a:launches.append(a),False)
    assert not launches
    tick(tmp_path,state,set(),[{'gpu':1}],set(),lambda *a:launches.append(a),False)
    assert len(launches)==1
    assert state['uncertainty_jobs']['fmri__gtgan__seed2026']['gpu']==1


def test_cuda13_interpreter_used_for_new_pytorch_jobs(tmp_path):
    launches = []
    python_path = tmp_path/'.venv-cu130/bin/python'
    tick(tmp_path,{},set(),[{'gpu':1}],set(),lambda *a:launches.append(a),False,
         python_path=python_path)
    assert launches[0][1][0] == python_path
