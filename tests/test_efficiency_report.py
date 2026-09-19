import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_sampling_efficiency import summarize


def test_efficiency_ratios_single_seed(tmp_path):
    records=[]
    for seed in [2026,2027,2028]:
        for mode,seconds,memory in [('legacy_full',10,100),('adaptive50',2,80)]:
            records.append(dict(dataset='stocks',seed=seed,mode=mode,measurements=[
                dict(seconds=seconds,samples_per_second=32/seconds,peak_allocated_mib=memory,
                     peak_reserved_mib=128,denoiser_calls=50)]*3))
    path=tmp_path/'table.md'
    summarize(records,path)
    text=path.read_text()
    assert '| 5.000 | 20.000 |' in text
    assert '| stocks | adaptive50 | 1 |' in text
    assert '±' not in text
