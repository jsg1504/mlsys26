import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "python"))

from gdn_blackwell.dispatch import choose_path, make_cache_key


assert choose_path(total_seq_len=64, num_seqs=1) == "small"
assert choose_path(total_seq_len=8192, num_seqs=32) == "large"
assert make_cache_key("small", "bf16", True, True)[0] == "small"
assert make_cache_key("large", "bf16", True, True)[0] == "large"
