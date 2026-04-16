"""NVRTC compile, cache, and launch utilities for custom CUDA kernels."""
import ctypes
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import cuda.bindings.driver as drv
import cuda.bindings.nvrtc as nvrtc

_module_cache: Dict[str, Tuple] = {}

def _check(err):
    if isinstance(err, tuple):
        err = err[0]
    if err != 0:
        raise RuntimeError(f"CUDA/NVRTC error: {err}")

def compile_and_load(source: str, kernel_names: list[str], arch: str = "sm_100a") -> dict:
    src_hash = hashlib.md5(source.encode()).hexdigest()
    if src_hash in _module_cache:
        return _module_cache[src_hash][1]

    err, prog = nvrtc.nvrtcCreateProgram(source.encode(), b"kernel.cu", 0, [], [])
    _check(err)

    opts = [f"--gpu-architecture={arch}".encode(), b"--use_fast_math"]
    compile_err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if compile_err[0] != 0:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC compile failed:\n{log.decode()}")

    _, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptx_size
    _check(nvrtc.nvrtcGetPTX(prog, ptx))
    nvrtc.nvrtcDestroyProgram(prog)

    err, module = drv.cuModuleLoadData(ptx)
    _check(err)

    functions = {}
    for name in kernel_names:
        err, func = drv.cuModuleGetFunction(module, name.encode())
        _check(err)
        functions[name] = func

    _module_cache[src_hash] = (module, functions)
    return functions


def launch(func, grid: tuple, block: tuple, args: list,
           shared_mem: int = 0, stream=None):
    arg_ptrs = []
    arg_values = []
    for arg in args:
        if isinstance(arg, int):
            val = ctypes.c_int64(arg)
        elif isinstance(arg, float):
            val = ctypes.c_float(arg)
        else:
            val = arg
        arg_values.append(val)
        arg_ptrs.append(ctypes.cast(ctypes.pointer(val), ctypes.c_void_p))

    arr = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
    cu_stream = stream if stream is not None else drv.CUstream(0)

    _check(drv.cuLaunchKernel(
        func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        shared_mem,
        cu_stream,
        arr, 0
    ))
