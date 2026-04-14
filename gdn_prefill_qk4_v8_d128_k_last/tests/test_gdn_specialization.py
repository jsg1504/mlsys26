import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
from gdn_blackwell.gdn import GDN


assert GDN.can_implement(
    (1, 16, 4, 128),
    (1, 16, 8, 128),
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
)

assert not GDN.can_implement(
    (1, 16, 2, 128),
    (1, 16, 8, 128),
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
), "Task 5 runtime should reject q/k head counts other than 4"

assert not GDN.can_implement(
    (1, 16, 4, 128),
    (1, 16, 12, 128),
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
), "Task 5 runtime should reject v head counts other than 8"

assert not GDN.can_implement(
    (1, 16, 4, 128),
    (1, 16, 8, 128),
    torch.float16,
    torch.float16,
    torch.float32,
), "Task 5 runtime should reject float16 input/output dtypes"
