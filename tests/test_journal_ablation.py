import sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from journal_ablation import DATASETS,VARIANTS,samplers,groups,ROOT


def test_fixed_inventory_and_comparisons():
    assert len(DATASETS)*len(VARIANTS)==44
    assert len(DATASETS)*sum(len(samplers(v)) for v in VARIANTS)==92
    for pairs in groups().values():
        for variant,sampler in pairs: assert sampler in samplers(variant)
    for dataset in DATASETS:
        spec=yaml.safe_load((ROOT/f'Config/journal_ablation/{dataset}.yaml').read_text())
        base=yaml.safe_load((ROOT/spec['base_config']).read_text())
        assert spec['seed']==2026
        assert spec['training_steps']==base['solver']['max_epochs']
        assert spec['evaluation_split']=='validation'


def test_scale_count_control_and_loss_ablation():
    for v in ['full','scale4','scale8','scale16','scale4_8']:
        p={'prototype_scales':[4,8,16],'num_prototypes':8,**VARIANTS[v]}
        assert len(p['prototype_scales'])*p['num_prototypes']==24
    for v in ['no_aux','no_balance','no_diversity','no_consistency']:
        assert len(VARIANTS[v])==1
        assert next(iter(VARIANTS[v].values()))==0
