import json
from scripts.baseline_reproduction_queue import tick


def test_long_models_interleaved_and_recovery_only_once(tmp_path):
    config=tmp_path/'Config'
    config.mkdir()
    (config/'journal_runtime.yaml').write_text('baseline_interleave: true\nbaseline_recovery_request: test\n')
    logs=tmp_path/'OUTPUT/baseline_reproduction_logs'
    logs.mkdir(parents=True)
    key='fmri__gtgan__seed2026'
    (logs/f'{key}.log').write_text('CUDA error: unspecified launch failure')
    state={'uncertainty_jobs':{key:{'status':'failed_needs_review','attempts':1}}}
    calls=[]
    tick(tmp_path,state,set(),[{'gpu':i} for i in range(8)],set(),lambda *args:calls.append(args),False)
    methods=[args[1][args[1].index('--method')+1] for args in calls]
    assert len(set(methods))==7
    assert methods.count('gtgan')<=2
    job=state['uncertainty_jobs'][key]
    assert job['recovery_history'][0]['attempts']==1
    tick(tmp_path,state,set(),[],set(),lambda *args:None,True)
    assert state['uncertainty_jobs'][key]['status']=='failed_needs_review'
    assert len(job['recovery_history'])==1
