import importlib.util
from pathlib import Path
import pytest
import torch

source = Path(__file__).resolve().parents[1]/'baselines/PaD-TS/eval_utils/MMD.py'
spec = importlib.util.spec_from_file_location('pad_mmd', source)
mmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mmd)


@pytest.mark.parametrize('kernel', ['rbf', 'multiscale'])
def test_chunked_values_and_both_gradients_match_dense(kernel):
    torch.manual_seed(52)
    # Cross the chunk boundary with a non-full final chunk.
    x = torch.randn(1031, 32, 1, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    dense = mmd._BMMD_dense(x, y, kernel)
    gx, gy = torch.autograd.grad(dense.mean(), (x, y))
    chunked = mmd.BMMD(x, y, kernel)
    cx, cy = torch.autograd.grad(chunked.mean(), (x, y))
    torch.testing.assert_close(chunked, dense)
    torch.testing.assert_close(cx, gx)
    torch.testing.assert_close(cy, gy)
