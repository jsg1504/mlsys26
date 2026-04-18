import torch
import weakref

_CU_SEQLENS_METADATA_CACHE = {}


def get_cu_seqlens_metadata(cu_seqlens):
    cache_key = id(cu_seqlens)
    version = getattr(cu_seqlens, "_version", None)
    cached = _CU_SEQLENS_METADATA_CACHE.get(cache_key)
    if cached is not None:
        cached_ref, cached_version, cached_metadata = cached
        if cached_ref() is cu_seqlens and cached_version == version:
            return cached_metadata

    host_values = tuple(int(v) for v in cu_seqlens.detach().cpu().tolist())
    num_seqs = len(host_values) - 1
    max_s_q = max(
        (host_values[i + 1] - host_values[i] for i in range(num_seqs)),
        default=0,
    )
    metadata = {
        "num_seqs": num_seqs,
        "max_s_q": max_s_q,
        "sum_s_q": host_values[-1] if host_values else 0,
        "first": host_values[0] if host_values else None,
        "last": host_values[-1] if host_values else None,
        "nondecreasing": all(
            host_values[i + 1] >= host_values[i] for i in range(len(host_values) - 1)
        ),
    }

    def _remove_stale_entry(_ref, *, cache_key=cache_key):
        _CU_SEQLENS_METADATA_CACHE.pop(cache_key, None)

    _CU_SEQLENS_METADATA_CACHE[cache_key] = (
        weakref.ref(cu_seqlens, _remove_stale_entry),
        version,
        metadata,
    )
    return metadata
