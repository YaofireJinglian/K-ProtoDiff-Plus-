import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_training_efficiency import report


def test_training_overhead_single_seed_only(tmp_path):
    output=tmp_path/'result.md'
    for seed in [2026,2027,2028]:
        for mode,scale in [('point',1),('multiscale',1.2)]:
            r=dict(dataset='stocks',seed=seed,mode=mode,effective_batch=256,
                   ms_per_update=100*scale,seconds=10*scale,samples_per_second=2560/scale,
                   peak_allocated_mib=200*scale,peak_reserved_mib=256,parameters_m=1)
            (tmp_path/f'{seed}_{mode}.json').write_text(json.dumps(r))
        report(tmp_path,output)
        assert '| stocks | multiscale | 1 |' in output.read_text()
        assert 'pending' not in output.read_text()
    assert '| 20.000 | 20.000 |' in output.read_text()
    assert '±' not in output.read_text()
