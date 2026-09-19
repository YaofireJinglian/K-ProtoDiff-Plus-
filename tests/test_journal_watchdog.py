import json
import yaml
import time
from scripts import journal_watchdog as watchdog


def test_followup_specs_are_validation_driven_and_frozen(tmp_path,monkeypatch):
    real_root=watchdog.ROOT
    spec=yaml.safe_load((real_root/'Config/journal_search/fmri.yaml').read_text())
    base=yaml.safe_load((real_root/'Config/journal/fmri.yaml').read_text())
    (tmp_path/'Config/journal_search').mkdir(parents=True)
    (tmp_path/'Config/journal').mkdir()
    (tmp_path/'Config/journal_search/fmri.yaml').write_text(yaml.safe_dump(spec))
    (tmp_path/'Config/journal/fmri.yaml').write_text(yaml.safe_dump(base))
    first=tmp_path/'checkpoints/journal_search/fmri'
    first.mkdir(parents=True)
    chosen={'trial':'control','train_overrides':{},'sampler_params':spec['samplers'][0]['params']}
    (first/'selection.json').write_text(json.dumps({'selected':chosen}))
    monkeypatch.setattr(watchdog,'ROOT',tmp_path)
    monkeypatch.setattr(watchdog,'DATASETS',['fmri'])
    planned=watchdog.next_specs()
    assert len(planned)==1
    path,round2=planned[0]
    assert round2['screen_steps']==6000
    assert round2.get('evaluation_split','validation')=='validation'
    for trial in round2['training_trials']:
        folder=watchdog.artifact(round2,trial['name'])
        folder.mkdir(parents=True)
        (folder/'complete.json').write_text('{}')
    watchdog.artifact(round2,'selection.json').write_text(json.dumps({'selected':chosen}))
    all_specs=watchdog.next_specs()
    assert len(all_specs)==4
    for path,confirmation in all_specs[1:]:
        assert confirmation['evaluation_split']=='test'
        assert confirmation['screen_steps']==base['solver']['max_epochs']
        assert confirmation['frozen_validation_choice']==chosen
        assert confirmation['screen_seed'] in [2026,2027,2028]
        assert confirmation['training_trials'][1]['depends_on'].endswith('control/complete.json')
    # Re-running the planner preserves already frozen specs byte-for-byte.
    before={path:path.read_bytes() for path,_ in all_specs}
    watchdog.next_specs()
    assert before=={path:path.read_bytes() for path,_ in all_specs}


def test_gpu_zero_excluded_and_cuda_failures_wait_for_recovery(tmp_path,monkeypatch):
    (tmp_path/'Config').mkdir()
    (tmp_path/'Config/journal_runtime.yaml').write_text(
        'allowed_gpus: [1]\ncuda_probe_interval_seconds: 300\ncuda_recovery_request: test_recovery\n')
    monitor=tmp_path/'monitor'
    (monitor/'logs').mkdir(parents=True)
    (monitor/'logs/job.log').write_text('RuntimeError: CUDA unknown error')
    monkeypatch.setattr(watchdog,'ROOT',tmp_path)
    monkeypatch.setattr(watchdog,'STATE',monitor)
    state={'jobs':{'job':{'status':'failed','attempts':3}},
           'cuda_health':{'one':{'checked_at':time.time(),'healthy':False,'detail':'999'}}}
    rows,bad,alerts=watchdog.runtime_health(state,[{'gpu':0,'uuid':'zero'},{'gpu':1,'uuid':'one'}])
    assert rows==[{'gpu':1,'uuid':'one'}] and bad=={1} and alerts
    assert state['jobs']['job']['attempts']==3
    assert 'cuda_recovery_consumed' not in state
    state['cuda_health']['one']['healthy']=True
    watchdog.runtime_health(state,rows)
    assert state['jobs']['job']['status']=='pending'
    assert state['jobs']['job']['attempts']==0
    assert state['jobs']['job']['retry_history'][0]['attempts']==3
    state['jobs']['job']['status']='failed'
    watchdog.runtime_health(state,rows)
    assert state['jobs']['job']['status']=='failed'  # recovery request is one-shot
