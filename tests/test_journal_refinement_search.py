import yaml
from scripts.journal_refinement_search import SEEDS, initialize, validate
from scripts.journal_search import architecture, apply_options


def test_refinement_is_paired_validation_only_and_architecture_fixed(tmp_path):
    # The real completed selections are the immutable input to generated plans.
    from scripts.journal_refinement_search import ROOT
    initialize(ROOT)
    plans=list((ROOT/'Config/journal_search_refinement').glob('*.yaml'))
    assert len(plans)==10
    for path in plans:
        plan=yaml.safe_load(path.read_text())
        base=yaml.safe_load((ROOT/plan['base_config']).read_text())
        validate(plan,base)
        assert plan['seeds']==SEEDS and plan['evaluation_split']=='validation'
        assert len(plan['training_trials'])==3
        assert 2 <= len(plan['samplers']) <= 3
        for trial in plan['training_trials']:
            assert architecture(apply_options(base,trial['overrides']))==architecture(base)
