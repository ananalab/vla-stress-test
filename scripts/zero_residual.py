"""Write an untrained residual checkpoint (last actor layer = 0, so the correction is exactly 0).

Evaluating it must reproduce the VLA-alone results episode by episode; this checks the
whole residual plumbing before any training.
    python scripts/zero_residual.py runs/zero/final.pt
"""

import sys
from pathlib import Path

import torch

from vla_stress.residual_rl.policy import ActorCritic

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
model = ActorCritic()
assert all((p == 0).all() for p in model.actor[-1].parameters())
torch.save({"model": model.state_dict(), "step": 0, "update": 0, "args": {"alpha": 0.2, "hidden": 128}}, out)
print("saved", out)
