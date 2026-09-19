import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_main_results_table import highlight_rankings


def test_displayed_ties_second_distinct_and_missing():
    entries = [('TimeVAE', '0.001 ± 0.002 †'),
               ('TimeGAN', '0.001 ± 0.000'),
               ('GT-GAN', 'OOM'),
               ('PaD-TS', '0.002 ± 0.001'),
               ('Diffusion-TS', '0.003'),
               ('K-ProtoDiff', '待运行（1/3）')]
    lines = ['| KL | ' + method + ' | ' + ' | '.join([cell] * 10) + ' |'
             for method, cell in entries]
    result = highlight_rankings(lines)
    assert result[0].count('**0.001 ± 0.002 †**') == 10
    assert result[1].count('**0.001 ± 0.000**') == 10
    assert result[3].count('<u>0.002 ± 0.001</u>') == 10
    assert '**' not in result[2] + result[4] + result[5]


def test_all_missing():
    line = '| KL | TimeVAE | ' + ' | '.join(['OOM'] * 10) + ' |'
    assert highlight_rankings([line]) == [line]
