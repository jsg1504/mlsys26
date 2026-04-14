SMALL_TOTAL_SEQ_LEN_THRESHOLD = 1024
SMALL_NUM_SEQS_THRESHOLD = 8
VALID_PATH_NAMES = frozenset({"small", "large"})


def choose_path(*, total_seq_len, num_seqs):
    if total_seq_len <= SMALL_TOTAL_SEQ_LEN_THRESHOLD and num_seqs <= SMALL_NUM_SEQS_THRESHOLD:
        return "small"
    return "large"


def make_cache_key(path_name, dtype_name, has_initial_state, has_output_state):
    if path_name not in VALID_PATH_NAMES:
        raise ValueError(f"Unsupported path_name: {path_name}")
    return (path_name, dtype_name, has_initial_state, has_output_state)
