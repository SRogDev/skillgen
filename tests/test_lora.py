"""Tests for LoRALinear (src/skillgen/training/fundamentals.py).

Written BEFORE the implementation (TDD): at init B is zeros, so the adapter
must be an exact no-op; only A and B may receive gradients; the trainable
parameter count must match r * (in_features + out_features).
"""

import torch
from torch import nn

from skillgen.training.fundamentals import LoRALinear, lora_parameter_count


def make_lora(in_features=8, out_features=4, r=2, alpha=4, **kwargs):
    torch.manual_seed(0)
    base = nn.Linear(in_features, out_features)
    return LoRALinear(base, r=r, alpha=alpha, **kwargs)


def test_output_shape_matches_wrapped_linear():
    layer = make_lora()
    x = torch.randn(3, 8)
    assert layer(x).shape == (3, 4)


def test_at_init_output_equals_base_output_exactly():
    # B starts at zero => delta W = 0 => LoRA output == base output, exactly.
    layer = make_lora()
    x = torch.randn(5, 8)
    with torch.no_grad():
        expected = layer.base(x)
    assert torch.equal(layer(x), expected)


def test_base_frozen_and_only_lora_gets_gradients():
    layer = make_lora()
    assert layer.base.weight.requires_grad is False
    if layer.base.bias is not None:
        assert layer.base.bias.requires_grad is False

    loss = layer(torch.randn(4, 8)).sum()
    loss.backward()

    assert layer.base.weight.grad is None
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None


def test_trainable_parameter_count_matches_formula():
    for in_f, out_f, r in [(8, 4, 2), (16, 16, 4), (32, 8, 1)]:
        layer = make_lora(in_features=in_f, out_features=out_f, r=r)
        expected = r * (in_f + out_f)
        assert lora_parameter_count(in_f, out_f, r) == expected
        assert layer.trainable_parameter_count() == expected
        # Cross-check against the actual tensors PyTorch would optimise.
        actual = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        assert actual == expected


def test_output_changes_after_one_optimizer_step_on_lora_only():
    layer = make_lora()
    x = torch.randn(4, 8)
    with torch.no_grad():
        before = layer(x)

    trainable = [p for p in layer.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    # A non-trivial target so B (and then A) receive useful gradients.
    loss = (layer(x) - 1.0).pow(2).mean()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        after = layer(x)
    assert not torch.allclose(before, after)
