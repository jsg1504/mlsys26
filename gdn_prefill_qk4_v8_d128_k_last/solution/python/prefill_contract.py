import torch
import torch.nn.functional as F


def prepare_g_beta(A_log, a, dt_bias, b):
    if not isinstance(A_log, torch.Tensor):
        raise TypeError(f"A_log must be a torch.Tensor, got {type(A_log).__name__}")
    if not isinstance(a, torch.Tensor):
        raise TypeError(f"a must be a torch.Tensor, got {type(a).__name__}")
    if not isinstance(dt_bias, torch.Tensor):
        raise TypeError(f"dt_bias must be a torch.Tensor, got {type(dt_bias).__name__}")
    if not isinstance(b, torch.Tensor):
        raise TypeError(f"b must be a torch.Tensor, got {type(b).__name__}")
    if A_log.dtype != torch.float32:
        raise TypeError(f"A_log.dtype must be torch.float32, got {A_log.dtype}")
    if a.dtype != torch.bfloat16:
        raise TypeError(f"a.dtype must be torch.bfloat16, got {a.dtype}")
    if dt_bias.dtype != torch.float32:
        raise TypeError(f"dt_bias.dtype must be torch.float32, got {dt_bias.dtype}")
    if b.dtype != torch.bfloat16:
        raise TypeError(f"b.dtype must be torch.bfloat16, got {b.dtype}")
    if A_log.shape != (8,):
        raise ValueError(f"A_log.shape must be (8,), got {tuple(A_log.shape)}")
    if a.ndim != 2 or a.shape[1] != 8:
        raise ValueError(f"a.shape must be (T, 8), got {tuple(a.shape)}")
    if dt_bias.shape != (8,):
        raise ValueError(f"dt_bias.shape must be (8,), got {tuple(dt_bias.shape)}")
    if b.ndim != 2 or b.shape[1] != 8:
        raise ValueError(f"b.shape must be (T, 8), got {tuple(b.shape)}")
    if a.shape != b.shape:
        raise ValueError(f"a.shape and b.shape must match, got {tuple(a.shape)} and {tuple(b.shape)}")

    x = a.float() + dt_bias.float()
    gate_log = -torch.exp(A_log.float()) * F.softplus(x)
    beta = torch.sigmoid(b.float())
    return gate_log, beta


def validate_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens):
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("state", state),
        ("A_log", A_log),
        ("a", a),
        ("dt_bias", dt_bias),
        ("b", b),
        ("cu_seqlens", cu_seqlens),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")

    reference_device = q.device
    for name, tensor in (
        ("k", k),
        ("v", v),
        ("state", state),
        ("A_log", A_log),
        ("a", a),
        ("dt_bias", dt_bias),
        ("b", b),
        ("cu_seqlens", cu_seqlens),
    ):
        if tensor.device != reference_device:
            raise ValueError(
                f"{name}.device must match q.device, got {tensor.device} and {reference_device}"
            )

    if q.shape != k.shape:
        raise ValueError(f"q.shape and k.shape must match, got {tuple(q.shape)} and {tuple(k.shape)}")
    if v.shape[0] != q.shape[0]:
        raise ValueError(f"v.shape[0] must match q.shape[0], got {v.shape[0]} and {q.shape[0]}")
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q.dtype must be torch.bfloat16, got {q.dtype}")
    if k.dtype != torch.bfloat16:
        raise TypeError(f"k.dtype must be torch.bfloat16, got {k.dtype}")
    if v.dtype != torch.bfloat16:
        raise TypeError(f"v.dtype must be torch.bfloat16, got {v.dtype}")
    if tuple(q.shape[1:]) != (4, 128):
        raise ValueError(f"q.shape must be (T, 4, 128), got {tuple(q.shape)}")
    if tuple(k.shape[1:]) != (4, 128):
        raise ValueError(f"k.shape must be (T, 4, 128), got {tuple(k.shape)}")
    if tuple(v.shape[1:]) != (8, 128):
        raise ValueError(f"v.shape must be (T, 8, 128), got {tuple(v.shape)}")
    if state.ndim != 4 or tuple(state.shape[1:]) != (8, 128, 128):
        raise ValueError(f"state.shape must be (N, 8, 128, 128), got {tuple(state.shape)}")
    if state.dtype != torch.float32:
        raise TypeError(f"state.dtype must be torch.float32, got {state.dtype}")
    if A_log.shape != (8,) or A_log.dtype != torch.float32:
        raise ValueError(f"A_log must have shape (8,) and dtype torch.float32, got {tuple(A_log.shape)} and {A_log.dtype}")
    if a.shape != (q.shape[0], 8) or a.dtype != torch.bfloat16:
        raise ValueError(f"a must have shape (T, 8) and dtype torch.bfloat16, got {tuple(a.shape)} and {a.dtype}")
    if dt_bias.shape != (8,) or dt_bias.dtype != torch.float32:
        raise ValueError(f"dt_bias must have shape (8,) and dtype torch.float32, got {tuple(dt_bias.shape)} and {dt_bias.dtype}")
    if b.shape != (q.shape[0], 8) or b.dtype != torch.bfloat16:
        raise ValueError(f"b must have shape (T, 8) and dtype torch.bfloat16, got {tuple(b.shape)} and {b.dtype}")
    if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2 or cu_seqlens.dtype != torch.int64:
        raise ValueError(
            f"cu_seqlens must have shape (N + 1,) with N >= 1 and dtype torch.int64, got {tuple(cu_seqlens.shape)} and {cu_seqlens.dtype}"
        )
    if state.shape[0] != cu_seqlens.shape[0] - 1:
        raise ValueError(
            f"state.shape[0] must match len(cu_seqlens) - 1, got {state.shape[0]} and {cu_seqlens.shape[0] - 1}"
        )
    if cu_seqlens[0].item() != 0 or cu_seqlens[-1].item() != q.shape[0]:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at T, got {cu_seqlens.tolist()} and T={q.shape[0]}"
        )
    if torch.any(cu_seqlens[1:] < cu_seqlens[:-1]):
        raise ValueError(
            f"cu_seqlens must be non-decreasing, got {cu_seqlens.tolist()}"
        )
