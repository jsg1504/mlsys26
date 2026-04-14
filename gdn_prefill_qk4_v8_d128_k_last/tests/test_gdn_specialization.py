import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))
from gdn_blackwell.gdn import GDN

try:
    GDN(is_persistent=True)
except ValueError as exc:
    assert "non-persistent" in str(exc)
else:
    raise AssertionError("Task 5 runtime should reject persistent mode construction")

try:
    GDN(chunk_size=64)
except ValueError as exc:
    assert "chunk_size" in str(exc)
else:
    raise AssertionError("Task 5 runtime should reject non-128 chunk_size")

try:
    GDN(head_dim=64)
except ValueError as exc:
    assert "head_dim" in str(exc)
else:
    raise AssertionError("Task 5 runtime should reject non-128 head_dim")


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
    (2, 16, 4, 128),
    (2, 16, 8, 128),
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
), "Task 5 runtime should reject batch sizes other than 1"

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
