"""
Profile GDN decode or prefill kernel with NCU on Modal B200.

Usage:
  modal run scripts/profile_kernel.py   (profiles both decode and prefill)

Workloads are derived from the actual contest dataset:
  - Decode: B in {1, 4, 8, 16, 32, 48, 64}
  - Prefill: representative (num_seqs, total_seq_len) pairs covering
    the full distribution from mlsys26-contest workloads

Edit PROFILE_CONFIG below to customize.
"""

import modal
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ──────────── Configuration (from contest workloads) ──────────
PROFILE_CONFIG = {
    "decode": {
        # All 7 batch sizes from contest dataset
        "batch_sizes": [1, 4, 8, 16, 32, 48, 64],
    },
    "prefill": {
        # Representative (num_seqs, total_seq_len) from actual workloads.
        # Covers: small/large T, low/high N, each split_factor tier.
        # Script creates uniform seq lengths: each seq gets T_total/N tokens.
        "workloads": [
            (1, 16),      # N=1 short  (split=8, 64 blocks)
            (1, 134),     # N=1 medium (split=8, 64 blocks)
            (1, 2107),    # N=1 long   (split=8, 64 blocks)
            (2, 983),     # N=2        (split=8, 128 blocks)
            (3, 2857),    # N=3 long   (split=8, 192 blocks)
            (4, 959),     # N=4        (split=8, 256 blocks)
            (20, 8192),   # N=20       (split=4, 640 blocks)  -- different split tier
            (43, 8192),   # N=43       (split=1, 344 blocks)  -- split=1
            (57, 8192),   # N=57 max   (split=1, 456 blocks)
        ],
    },
    "sections": ["SpeedOfLight", "MemoryWorkloadAnalysis", "LaunchStats", "Occupancy"],
}
# ──────────────────────────────────────────────────────────────

app = modal.App("ncu-profile")

image = modal.Image.from_registry(
    "nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12"
)

DECODE_MAIN_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <curand.h>

using bf16 = __nv_bfloat16;
constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_K_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int V_PER_Q = NUM_V_HEADS / NUM_Q_HEADS;

__device__ __forceinline__ float softplus(float x) { return log1pf(expf(x)); }
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

$$KERNEL$$

void fill_random_bf16(bf16* d_ptr, int n) {
    float* tmp; cudaMalloc(&tmp, n * sizeof(float));
    curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, tmp, n, 0.0f, 1.0f);
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, tmp, n * sizeof(float), cudaMemcpyDeviceToHost);
    bf16* hb = (bf16*)malloc(n * sizeof(bf16));
    for (int i = 0; i < n; i++) hb[i] = __float2bfloat16(h[i]);
    cudaMemcpy(d_ptr, hb, n * sizeof(bf16), cudaMemcpyHostToDevice);
    free(h); free(hb); cudaFree(tmp); curandDestroyGenerator(gen);
}
void fill_random_f32(float* d_ptr, int n) {
    curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, d_ptr, n, 0.0f, 1.0f);
    curandDestroyGenerator(gen);
}

int main(int argc, char** argv) {
    int B = atoi(argv[1]);
    printf("Profiling DECODE kernel B=%d\n", B);

    bf16 *q, *k, *v, *a_gate, *b_gate, *output;
    float *state, *A_log, *dt_bias, *new_state;
    cudaMalloc(&q, B * NUM_Q_HEADS * HEAD_DIM * sizeof(bf16));
    cudaMalloc(&k, B * NUM_K_HEADS * HEAD_DIM * sizeof(bf16));
    cudaMalloc(&v, B * NUM_V_HEADS * HEAD_DIM * sizeof(bf16));
    cudaMalloc(&state, B * NUM_V_HEADS * HEAD_DIM * HEAD_DIM * sizeof(float));
    cudaMalloc(&A_log, NUM_V_HEADS * sizeof(float));
    cudaMalloc(&a_gate, B * NUM_V_HEADS * sizeof(bf16));
    cudaMalloc(&dt_bias, NUM_V_HEADS * sizeof(float));
    cudaMalloc(&b_gate, B * NUM_V_HEADS * sizeof(bf16));
    cudaMalloc(&output, B * NUM_V_HEADS * HEAD_DIM * sizeof(bf16));
    cudaMalloc(&new_state, B * NUM_V_HEADS * HEAD_DIM * HEAD_DIM * sizeof(float));

    fill_random_bf16(q, B*NUM_Q_HEADS*HEAD_DIM);
    fill_random_bf16(k, B*NUM_K_HEADS*HEAD_DIM);
    fill_random_bf16(v, B*NUM_V_HEADS*HEAD_DIM);
    fill_random_f32(state, B*NUM_V_HEADS*HEAD_DIM*HEAD_DIM);
    fill_random_f32(A_log, NUM_V_HEADS);
    fill_random_bf16(a_gate, B*NUM_V_HEADS);
    fill_random_f32(dt_bias, NUM_V_HEADS);
    fill_random_bf16(b_gate, B*NUM_V_HEADS);
    float scale = 0.08838834764831843f;
    cudaDeviceSynchronize();

    int sf;
    if (B <= 2) sf = 8;
    else if (B <= 4) sf = 4;
    else if (B <= 16) sf = 2;
    else sf = 1;

    dim3 grid(B * NUM_V_HEADS * sf);
    dim3 block(HEAD_DIM);

    for (int i = 0; i < 3; i++)
        gdn_decode_kernel<<<grid, block>>>(q,k,v,state,A_log,a_gate,dt_bias,b_gate,scale,output,new_state,B,sf);
    cudaDeviceSynchronize();

    gdn_decode_kernel<<<grid, block>>>(q,k,v,state,A_log,a_gate,dt_bias,b_gate,scale,output,new_state,B,sf);
    cudaDeviceSynchronize();

    printf("Done.\n");
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(state);
    cudaFree(A_log); cudaFree(a_gate); cudaFree(dt_bias); cudaFree(b_gate);
    cudaFree(output); cudaFree(new_state);
    return 0;
}
"""

PREFILL_MAIN_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <curand.h>

using bf16 = __nv_bfloat16;
constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_K_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int V_PER_Q = NUM_V_HEADS / NUM_Q_HEADS;
constexpr int NUM_WARPS = HEAD_DIM / 32;

__device__ __forceinline__ float softplus(float x) { return log1pf(expf(x)); }

$$KERNEL$$

void fill_random_bf16(bf16* d_ptr, int n) {
    float* tmp; cudaMalloc(&tmp, n * sizeof(float));
    curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, tmp, n, 0.0f, 1.0f);
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, tmp, n * sizeof(float), cudaMemcpyDeviceToHost);
    bf16* hb = (bf16*)malloc(n * sizeof(bf16));
    for (int i = 0; i < n; i++) hb[i] = __float2bfloat16(h[i]);
    cudaMemcpy(d_ptr, hb, n * sizeof(bf16), cudaMemcpyHostToDevice);
    free(h); free(hb); cudaFree(tmp); curandDestroyGenerator(gen);
}
void fill_random_f32(float* d_ptr, int n) {
    curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, d_ptr, n, 0.0f, 1.0f);
    curandDestroyGenerator(gen);
}

int main(int argc, char** argv) {
    int N = atoi(argv[1]);   // num_seqs
    int T_per = atoi(argv[2]); // seq_len per sequence
    int T = N * T_per;       // total tokens
    printf("Profiling PREFILL kernel N=%d, T_per=%d, T_total=%d\n", N, T_per, T);

    // Build cu_seqlens: [0, T_per, 2*T_per, ..., N*T_per]
    int64_t* h_cu = (int64_t*)malloc((N + 1) * sizeof(int64_t));
    for (int i = 0; i <= N; i++) h_cu[i] = (int64_t)i * T_per;
    int64_t* cu_seqlens;
    cudaMalloc(&cu_seqlens, (N + 1) * sizeof(int64_t));
    cudaMemcpy(cu_seqlens, h_cu, (N + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    free(h_cu);

    bf16 *q, *k, *v, *a_gate, *b_gate, *output;
    float *state, *A_log, *dt_bias, *new_state;
    cudaMalloc(&q, T * NUM_Q_HEADS * HEAD_DIM * sizeof(bf16));
    cudaMalloc(&k, T * NUM_K_HEADS * HEAD_DIM * sizeof(bf16));
    cudaMalloc(&v, T * NUM_V_HEADS * HEAD_DIM * sizeof(bf16));
    cudaMalloc(&state, N * NUM_V_HEADS * HEAD_DIM * HEAD_DIM * sizeof(float));
    cudaMalloc(&A_log, NUM_V_HEADS * sizeof(float));
    cudaMalloc(&a_gate, T * NUM_V_HEADS * sizeof(bf16));
    cudaMalloc(&dt_bias, NUM_V_HEADS * sizeof(float));
    cudaMalloc(&b_gate, T * NUM_V_HEADS * sizeof(bf16));
    cudaMalloc(&output, T * NUM_V_HEADS * HEAD_DIM * sizeof(bf16));
    cudaMalloc(&new_state, N * NUM_V_HEADS * HEAD_DIM * HEAD_DIM * sizeof(float));

    fill_random_bf16(q, T*NUM_Q_HEADS*HEAD_DIM);
    fill_random_bf16(k, T*NUM_K_HEADS*HEAD_DIM);
    fill_random_bf16(v, T*NUM_V_HEADS*HEAD_DIM);
    fill_random_f32(state, N*NUM_V_HEADS*HEAD_DIM*HEAD_DIM);
    fill_random_f32(A_log, NUM_V_HEADS);
    fill_random_bf16(a_gate, T*NUM_V_HEADS);
    fill_random_f32(dt_bias, NUM_V_HEADS);
    fill_random_bf16(b_gate, T*NUM_V_HEADS);
    float scale = 0.08838834764831843f;
    cudaDeviceSynchronize();

$$PREFILL_LAUNCH_CONFIG$$

$$PREFILL_LAUNCH_HELPER$$

    // Warmup
$$PREFILL_WARMUP$$
    cudaDeviceSynchronize();

    // Profiled run
$$PREFILL_PROFILED$$
    cudaDeviceSynchronize();

    printf("Done.\n");
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(state);
    cudaFree(A_log); cudaFree(a_gate); cudaFree(dt_bias); cudaFree(b_gate);
    cudaFree(cu_seqlens); cudaFree(output); cudaFree(new_state);
    return 0;
}
"""


def extract_kernel(src: str, kernel_type: str) -> str:
    """Extract kernel function(s) from source, stripping TVM FFI wrapper."""
    lines = src.split("\n")
    result = []
    skip_func = False
    brace_depth = 0

    for line in lines:
        # Skip TVM includes
        if "#include <tvm/" in line:
            continue
        # Skip using/constexpr (already in template)
        if line.startswith("using bf16") or line.startswith("constexpr int NUM_"):
            continue
        if line.startswith("constexpr int HEAD_DIM") or line.startswith("constexpr int V_PER_Q"):
            continue
        if line.startswith("constexpr int NUM_WARPS"):
            continue
        # Skip softplus (already in template)
        if "__device__ __forceinline__ float softplus" in line:
            skip_func = True
            brace_depth = 0
        if "__device__ __forceinline__ float warp_reduce_sum" in line:
            skip_func = True
            brace_depth = 0
        if skip_func:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0 and "{" in "".join(result[-5:] if result else []):
                skip_func = False
            if line.strip() == "}" and brace_depth <= 0:
                skip_func = False
            continue
        # Stop at host wrapper
        if kernel_type == "decode" and "void gdn_decode(" in line and "__global__" not in line:
            break
        if kernel_type == "prefill" and "void gdn_prefill(" in line and "__global__" not in line:
            break
        if "TVM_FFI_DLL_EXPORT" in line:
            continue
        result.append(line)

    return "\n".join(result)


def build_prefill_main(prefill_kernel: str) -> str:
    """Build a profiling harness that matches the current prefill kernel shape."""
    templated_kernel = (
        "template <" in prefill_kernel or "template<" in prefill_kernel
    ) and "gdn_prefill_kernel" in prefill_kernel
    uses_dynamic_smem = "extern __shared__" in prefill_kernel

    if templated_kernel:
        launch_config = """
    int sf;
    if (N <= 8) sf = 8;
    else if (N <= 16) sf = 4;
    else if (N <= 32) sf = 2;
    else sf = 1;

    dim3 grid(N * NUM_V_HEADS * sf);
    dim3 block(HEAD_DIM);
"""
        launch_helper = """
    auto launch = [&](auto kernel_fn) {
        kernel_fn<<<grid, block>>>(
            q, k, v, state, A_log, a_gate, dt_bias, b_gate,
            cu_seqlens, scale, output, new_state, N);
    };
"""
        warmup = """
    for (int i = 0; i < 2; i++) {
        switch (sf) {
            case 8:  launch(gdn_prefill_kernel<8>); break;
            case 4:  launch(gdn_prefill_kernel<4>); break;
            case 2:  launch(gdn_prefill_kernel<2>); break;
            default: launch(gdn_prefill_kernel<1>); break;
        }
    }
"""
        profiled = """
    switch (sf) {
        case 8:  launch(gdn_prefill_kernel<8>); break;
        case 4:  launch(gdn_prefill_kernel<4>); break;
        case 2:  launch(gdn_prefill_kernel<2>); break;
        default: launch(gdn_prefill_kernel<1>); break;
    }
"""
    else:
        if uses_dynamic_smem:
            launch_config = """
    dim3 grid(N * NUM_V_HEADS);
    dim3 block(HEAD_DIM);
    size_t smem_bytes =
        (HEAD_DIM * HEAD_DIM + NUM_WARPS * ROW_TILE + ROW_TILE + HEAD_DIM) * sizeof(float);
    cudaFuncSetAttribute(
        gdn_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
"""
            launch_helper = """
    auto launch = [&]() {
        gdn_prefill_kernel<<<grid, block, smem_bytes>>>(
            q, k, v, state, A_log, a_gate, dt_bias, b_gate,
            cu_seqlens, scale, output, new_state, N);
    };
"""
        else:
            launch_config = """
    dim3 grid(N * NUM_V_HEADS);
    dim3 block(HEAD_DIM);
"""
            launch_helper = """
    auto launch = [&]() {
        gdn_prefill_kernel<<<grid, block>>>(
            q, k, v, state, A_log, a_gate, dt_bias, b_gate,
            cu_seqlens, scale, output, new_state, N);
    };
"""
        warmup = """
    for (int i = 0; i < 2; i++) {
        launch();
    }
"""
        profiled = """
    launch();
"""

    return (
        PREFILL_MAIN_TEMPLATE.replace("$$KERNEL$$", prefill_kernel)
        .replace("$$PREFILL_LAUNCH_CONFIG$$", launch_config.rstrip())
        .replace("$$PREFILL_LAUNCH_HELPER$$", launch_helper.rstrip())
        .replace("$$PREFILL_WARMUP$$", warmup.rstrip())
        .replace("$$PREFILL_PROFILED$$", profiled.rstrip())
    )


@app.function(image=image, gpu="B200:1", timeout=600)
def profile(decode_src: str, prefill_src: str, config: dict):
    import subprocess, os, tempfile

    build_dir = tempfile.mkdtemp()
    sections = config["sections"]
    section_args = []
    for s in sections:
        section_args.extend(["--section", s])

    all_results = {}

    # ──── DECODE ────
    if "decode" in config and config["decode"].get("batch_sizes"):
        decode_kernel = extract_kernel(decode_src, "decode")
        main_src = DECODE_MAIN_TEMPLATE.replace("$$KERNEL$$", decode_kernel)
        main_path = os.path.join(build_dir, "decode_main.cu")
        with open(main_path, "w") as f:
            f.write(main_src)

        print("=== Compiling DECODE ===")
        comp = subprocess.run(
            ["nvcc", "-arch=sm_100a", "-O3", "-lineinfo",
             "-lcurand", "-o", f"{build_dir}/profile_decode", main_path],
            capture_output=True, text=True
        )
        if comp.returncode != 0:
            print(f"DECODE COMPILE FAILED:\n{comp.stderr}")
        else:
            print("Decode compilation OK")
            for B in config["decode"]["batch_sizes"]:
                print(f"\n{'='*60}")
                print(f"  DECODE  B={B}")
                print(f"{'='*60}")
                prof = subprocess.run(
                    ["ncu", "--kernel-name", "gdn_decode_kernel",
                     "--launch-skip", "3", "--launch-count", "1"]
                    + section_args +
                    [f"{build_dir}/profile_decode", str(B)],
                    capture_output=True, text=True, timeout=120
                )
                print(prof.stdout)
                if prof.stderr:
                    print(f"STDERR: {prof.stderr[:300]}")
                all_results[f"decode_B{B}"] = prof.stdout

    # ──── PREFILL ────
    if "prefill" in config and config["prefill"].get("workloads"):
        prefill_kernel = extract_kernel(prefill_src, "prefill")
        main_src = build_prefill_main(prefill_kernel)
        main_path = os.path.join(build_dir, "prefill_main.cu")
        with open(main_path, "w") as f:
            f.write(main_src)

        print("\n=== Compiling PREFILL ===")
        comp = subprocess.run(
            ["nvcc", "-arch=sm_100a", "-O3", "-lineinfo",
             "-lcurand", "-o", f"{build_dir}/profile_prefill", main_path],
            capture_output=True, text=True
        )
        if comp.returncode != 0:
            print(f"PREFILL COMPILE FAILED:\n{comp.stderr}")
        else:
            print("Prefill compilation OK")
            for (N, T_per) in config["prefill"]["workloads"]:
                print(f"\n{'='*60}")
                print(f"  PREFILL  N={N}, T={T_per}")
                print(f"{'='*60}")
                prof = subprocess.run(
                    ["ncu", "--kernel-name", "gdn_prefill_kernel",
                     "--launch-skip", "2", "--launch-count", "1"]
                    + section_args +
                    [f"{build_dir}/profile_prefill", str(N), str(T_per)],
                    capture_output=True, text=True, timeout=300
                )
                print(prof.stdout)
                if prof.stderr:
                    print(f"STDERR: {prof.stderr[:300]}")
                all_results[f"prefill_N{N}_T{T_per}"] = prof.stdout

    return all_results


@app.local_entrypoint()
def main(kernel: str = "both"):
    """Profile GDN kernels. Use --kernel decode|prefill|both."""
    decode_path = PROJECT_ROOT / "gdn_decode_qk4_v8_d128_k_last" / "solution" / "cuda" / "kernel.cu"
    prefill_path = PROJECT_ROOT / "gdn_prefill_qk4_v8_d128_k_last" / "solution" / "cuda" / "kernel.cu"

    config = {"sections": PROFILE_CONFIG["sections"]}

    if kernel in ("decode", "both"):
        config["decode"] = PROFILE_CONFIG["decode"]
    if kernel in ("prefill", "both"):
        config["prefill"] = PROFILE_CONFIG["prefill"]

    decode_src = decode_path.read_text() if "decode" in config else ""
    prefill_src = prefill_path.read_text() if "prefill" in config else ""

    print(f"Profiling: {kernel}")
    if "decode" in config:
        print(f"  Decode batch sizes: {config['decode']['batch_sizes']}")
    if "prefill" in config:
        print(f"  Prefill workloads:  {config['prefill']['workloads']}")
    print(f"  NCU sections:       {config['sections']}")

    results = profile.remote(decode_src, prefill_src, config)
    if not results:
        print("No results returned!")
