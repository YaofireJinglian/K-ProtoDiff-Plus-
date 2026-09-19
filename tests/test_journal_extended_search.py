import copy
import pytest
import yaml
from scripts.journal_search import ROOT
from scripts.journal_extended_search import SEEDS, METRICS, validate_plan, paired_summary, expanded_specs


def records(values,split='validation'):
    return [{'seed':seed,'split':split,'metrics':dict.fromkeys(METRICS,value),
             'train_overrides':{},'sampler_params':{},'train_steps':6000}
            for seed,value in zip(SEEDS,values)]


def test_all_extended_plans_share_seeds_and_preserve_architecture():
    paths=list((ROOT/'Config/journal_search_extended').glob('*.yaml'))
    assert len(paths)==10
    for path in paths:
        plan=yaml.safe_load(path.read_text())
        base=yaml.safe_load((ROOT/plan['base_config']).read_text())
        validate_plan(plan,base)
        assert len(plan['training_trials'])==5
        assert len(plan['samplers'])==3
        assert not any(t.get('warm_start_from') for t in plan['training_trials'])


def test_selection_uses_all_seeds_not_best_seed():
    summary,winner=paired_summary({'lucky_one':records([0.01,9,9]),'consistent':records([2,2,2])})
    assert winner['candidate']=='consistent'
    assert summary[0]['mean']['KL']>6


def test_partial_or_test_results_cannot_select():
    with pytest.raises(ValueError,match='fixed three seeds'):
        paired_summary({'partial':records([1,2])})
    with pytest.raises(ValueError,match='Test scores'):
        paired_summary({'test':records([1,2,3],split='test')})


def test_seeds_and_specs_are_frozen(tmp_path):
    source=ROOT/'Config/journal_search_extended/fmri.yaml'
    (tmp_path/'Config/journal_search_extended').mkdir(parents=True)
    (tmp_path/'Config/journal').mkdir()
    destination=tmp_path/'Config/journal_search_extended/fmri.yaml'
    destination.write_text(source.read_text())
    (tmp_path/'Config/journal/fmri.yaml').write_text((ROOT/'Config/journal/fmri.yaml').read_text())
    specs=expanded_specs(tmp_path)
    assert len(specs)==3
    assert [s['screen_seed'] for _,s in specs]==SEEDS
    plan=yaml.safe_load(destination.read_text())
    plan['screen_steps']=7000
    destination.write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError,match='frozen extended spec'):
        expanded_specs(tmp_path)
