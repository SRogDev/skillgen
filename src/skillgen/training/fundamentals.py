"""PyTorch fundamentals for SkillGen: LoRA implemented from scratch.

LoRA (Low-Rank Adaptation) freezes a pretrained weight matrix W and learns
a small update instead:

    W' = W + delta_W,   where delta_W = (alpha / r) * B @ A

A has shape (r, in_features) and B has shape (out_features, r), with rank
r much smaller than either dimension. Only A and B are trained, so the
number of trainable parameters drops from in*out to r*(in + out).

This module is deliberately minimal and readable: it is the version Roger
explains in an interview before reaching for a library such as PEFT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def lora_parameter_count(in_features: int, out_features: int, r: int) -> int:
    """Trainable parameters in one LoRA adapter: A is r*in, B is out*r."""
    return r * (in_features + out_features)


class LoRALinear(nn.Module):
    """Wrap an existing ``nn.Linear``, freeze it, and add a LoRA adapter.

    Shapes (for an input ``x`` of shape ``(..., in_features)``):

    - ``base.weight``: ``(out_features, in_features)`` — frozen.
    - ``lora_A``: ``(r, in_features)`` — kaiming-initialised.
    - ``lora_B``: ``(out_features, r)`` — zero-initialised, so at the start
      ``delta_W = B @ A = 0`` and the wrapped layer behaves *exactly* like
      the original Linear. Training then moves B (and A) away from zero.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.base = base
        self.r = r
        self.alpha = alpha
        # The update is scaled so that changing r does not silently change
        # the effective step size of the adapter.
        self.scaling = alpha / r
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Freeze the pretrained layer: no gradients, no optimiser state.
        for param in self.base.parameters():
            param.requires_grad = False

        # A breaks symmetry with a kaiming init (same family of init that
        # nn.Linear uses); B starts at zero so delta_W starts at exactly 0.
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x @ A.T -> (..., r);  (x @ A.T) @ B.T -> (..., out_features)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

    def trainable_parameter_count(self) -> int:
        """Number of parameters the optimiser will actually update (A + B)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"
        )
