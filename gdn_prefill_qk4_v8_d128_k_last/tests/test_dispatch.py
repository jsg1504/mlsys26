import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))

from gdn_blackwell.dispatch import choose_path, make_cache_key


assert choose_path(total_seq_len=64, num_seqs=1) == "small"
assert choose_path(total_seq_len=1024, num_seqs=8) == "small"
assert choose_path(total_seq_len=1025, num_seqs=8) == "large"
assert choose_path(total_seq_len=1024, num_seqs=9) == "large"
assert choose_path(total_seq_len=8192, num_seqs=32) == "large"
assert make_cache_key("small", "bf16", True, True)[0] == "small"
assert make_cache_key("large", "bf16", True, True)[0] == "large"

try:
    make_cache_key("bogus", "bf16", True, True)
except ValueError as exc:
    assert "Unsupported path_name" in str(exc)
else:
    raise AssertionError("Expected invalid path_name rejection")
