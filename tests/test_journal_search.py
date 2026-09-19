from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

import pytest
from scripts.journal_search import ROOT, apply_options, architecture, prepare_data, select


def test_search_specs_preserve_architecture():
    paths = list((ROOT/'Config/journal_search').glob('*.yaml'))
    assert len(paths) == 10
    for path in paths:
        spec = yaml.safe_load(path.read_text())
        base = yaml.safe_load((ROOT/spec['base_config']).read_text())
        assert len(spec['training_trials'])*len(spec['samplers']) == 9
        for trial in spec['training_trials']:
            changed = apply_options(base,trial['overrides'])
            assert architecture(changed) == architecture(base)


def test_raw_split_and_training_only_normalization(tmp_path):
    source = tmp_path/'series.csv'
    pd.DataFrame({'value':np.arange(1000,dtype=float)}).to_csv(source,index=False)
    base = {'dataloader':{'train_dataset':{'params':{'data_root':str(source),'window':24}}}}
    train, val = prepare_data(base,tmp_path/'split',1000)
    metadata = json.loads((tmp_path/'split/split.json').read_text())
    assert metadata['raw_train_range'] == [0,700]
    assert metadata['raw_validation_range'] == [700,850]
    assert metadata['sealed_test_range'] == [850,1000]
    assert train.shape == (677,24,1)
    assert val.shape == (127,24,1)
    assert np.isclose(train.min(),0) and np.isclose(train.max(),1)
    assert val.min() > 1  # No leakage / refitting on validation / clipping.


def test_options_do_not_mutate_base():
    base={'model':{'params':{'prototype_loss_weight':0.05}}}
    changed=apply_options(base,{'model.params.prototype_loss_weight':0.01})
    assert base['model']['params']['prototype_loss_weight']==0.05
    assert changed['model']['params']['prototype_loss_weight']==0.01


def test_never_select_on_test_scores():
    with pytest.raises(ValueError,match='Never select'):
        select({'evaluation_split':'test','dataset':'fmri'})


def test_confirmation_test_split_keeps_training_fixed(tmp_path):
    source=tmp_path/'series.csv'
    pd.DataFrame({'value':np.arange(1000,dtype=float)}).to_csv(source,index=False)
    base={'dataloader':{'train_dataset':{'params':{'data_root':str(source),'window':24}}}}
    training,val=prepare_data(base,tmp_path/'validation',1000)
    same_training,test=prepare_data(base,tmp_path/'test',1000,'test')
    np.testing.assert_array_equal(training,same_training)
    assert test.min()>val.max()
    assert json.loads((tmp_path/'test/split.json').read_text())['evaluation_range']==[850,1000]
