# GDN Prefill Optimization Survey

Date: 2026-04-07

## Scope

Target: `gdn_prefill_qk4_v8_d128_k_last`

Goal:
- Analyze the current project state in detail
- Review current prefill baseline and existing optimization logs
- Research B200-relevant optimization directions
- Analyze GDN prefill workload and dataflow
- List optimization methodologies that are both theoretically promising and practically applicable to the current kernel

## Current Project Status

Code and logs reviewed:
- `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- `mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json`
- `mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl`
- `mlsys26-contest/solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/flashinfer_wrapper_123ca6.json`
- `logs/prefill/bench_history.jsonl`
- `logs/prefill/optimization_log.md`
- `EVALUATION.md`
- `FAQ.md`

Current measured prefill result:
- Timestamp: 2026-04-06T13:52:45.201275+00:00
- Mean speedup: `6.37`
- Mean latency: `34.28 ms`
- Correctness: pass, matched ratio `1.0`
- Note: initial TVM FFI baseline with dynamic shared memory

Submission-path status:
- The active experimental prefill kernel lives under `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- The repository root submission path is still a starter template:
  - `config.toml` still points to the generic template configuration
  - `solution/cuda/kernel.cu` is not wired to the GDN prefill implementation
- This means optimization work is currently happening in the track subfolder, but the repo root is not yet ready as a direct submission target

Optimization log state:
- `logs/prefill/optimization_log.md` is effectively empty except for the title/template header
- There is no evidence of prior iterative prefill optimization beyond the initial benchmark entry

## Agent Team

Suggested split for future iterations:
- Kernel Analyst: current CUDA kernel structure, synchronization, occupancy, memory traffic
- Hardware Analyst: B200 / Blackwell tuning features and constraints
- Workload Analyst: GDN math, workload distribution, sequence-length behavior
- Integration Lead: prioritize methods, choose implementation order, maintain optimization log

For this survey, those roles were simulated through parallel repo analysis and Blackwell documentation review.

## GDN Prefill Workload Characterization

Definition facts:
- `q`: `[T, 4, 128]` bf16
- `k`: `[T, 4, 128]` bf16
- `v`: `[T, 8, 128]` bf16
- `state`: `[N, 8, 128, 128]` fp32, k-last layout `[N, H, V, K]`
- `A_log`: `[8]` fp32
- `a`: `[T, 8]` bf16
- `dt_bias`: `[8]` fp32
- `b`: `[T, 8]` bf16
- `output`: `[T, 8, 128]` bf16
- `new_state`: `[N, 8, 128, 128]` fp32

Head mapping:
- `num_q_heads = 4`
- `num_k_heads = 4`
- `num_v_heads = 8`
- grouped value attention ratio: `V_PER_Q = 2`
- Each q/k head services two v-heads

Per-token recurrence:
- `g = exp(-exp(A_log) * softplus(a + dt_bias))`
- `beta = sigmoid(b)`
- `temp = g * state`
- `kdot = k @ temp`
- `residual = beta * (v - kdot)`
- `state_new = temp + k^T @ residual`
- `output = scale * (q @ state_new)`

Equivalent interpretation:
- Each token performs one decay over the full `128x128` fp32 state, one row-wise dot against `k`, one rank-1 update with `k^T @ residual`, and one output projection with `q @ state_new`
- This is a sequential state-space style recurrence, not a standard fully parallel attention prefill

State size and reuse:
- State per `(seq, v_head)` is `128 * 128 * 4 = 64 KB`
- State is reused across every token of the same `(seq, v_head)`
- Because of the recurrence, tokens inside one sequence/head cannot be freely reordered

Workload distribution from `jsonl`:
- Total workloads: `100`
- `total_seq_len`: min `6`, p50 `139`, p90 `8192`, max `8192`, mean `1830.65`
- `num_seqs`: min `1`, p50 `2`, p90 `35`, max `57`, mean `7.97`
- Average sequence length (`total_seq_len / num_seqs`):
  - `< 32`: `22` cases
  - `32-63`: `19` cases
  - `64-127`: `15` cases
  - `128-255`: `20` cases
  - `256-511`: `11` cases
  - `512-1023`: `9` cases
  - `1024+`: `4` cases

Implication:
- The benchmark mix is bimodal enough that a single strategy is unlikely to be optimal
- Many cases are very short or moderately short per-sequence, while the largest latency contribution will likely come from long-sequence workloads near `8192`

## Current Kernel Analysis

Current mapping:
- One block per `(seq, v_head)`
- `128` threads per block, one thread per `K` dimension
- Entire `64 KB` fp32 state is copied into dynamic shared memory
- Tokens are processed sequentially inside the block

Important structural properties:
- Shared memory footprint per block is about `66 KB`:
  - `state`: `64 KB`
  - reductions and staging buffers: about `1 KB`
- For each token:
  - First loop over `vi in [0, 128)` computes `k @ temp`
  - Second loop over `vi in [0, 128)` updates state and computes output
  - Both loops use warp reductions followed by full-block `__syncthreads()`

Primary bottlenecks:
- Excessive block-wide synchronization:
  - roughly two full-block synchronizations per `vi`, twice per token
  - this dominates when average sequence length is short
- Low residency:
  - `~66 KB` dynamic shared memory per block restricts active blocks per SM
  - with B200 `228 KB` shared memory per SM, this is likely at most `3` such blocks per SM before other limits
- Scalarized computation:
  - the kernel performs many dot products and rank-1 updates using scalar math and warp shuffles
  - it does not use tensor-core style matrix pipelines
- No overlap between copy and compute:
  - state load, token loads, compute, and writeback are serialized
- No dynamic load balancing:
  - long sequences pin blocks much longer than short sequences
- GVA duplication is not exploited:
  - two v-heads sharing the same q/k head still load q/k independently in separate blocks

Strengths of the current design:
- The state stays on chip for the full sequence/head
- Correctness already passes across all workloads
- Layout aligns well with row-wise updates over `V=128`

## B200 / Blackwell Findings Relevant Here

The points below are from NVIDIA documentation for Blackwell / CUDA and are directly relevant to this kernel class.

Confirmed hardware/software facts:
- Official evaluation runs on bare-metal B200 with locked clocks
- Modal B200 is documented in `FAQ.md` as `sm100`
- Blackwell tuning guide identifies B200 as compute capability `10.0`
- B200 supports up to `228 KB` shared memory per SM and up to `227 KB` per thread block with explicit opt-in
- Blackwell retains thread block clusters and distributed shared memory
- On B200, nonportable cluster size `16` is supported if opted in
- CUDA 13.2 programming guide adds cluster launch control on Blackwell (`cc 10.0`) for work stealing
- CUDA programming guide documents TMA for bulk asynchronous copies and pipelines for overlap

What matters for this kernel:
- Large shared-memory kernels are viable on B200
  - This kernel already benefits from the `227 KB` per-block ceiling
  - But using more shared memory blindly can collapse residency
- Thread block clusters + distributed shared memory are potentially useful when one logical state tile exceeds a good per-block footprint
  - For this kernel, clusters are more interesting for cooperation or dynamic scheduling than for raw capacity
- Cluster launch control is relevant because per-block work is highly imbalanced across variable sequence lengths
  - This is unusually well aligned with the GDN prefill workload
- TMA and pipelined async copies are relevant if the design becomes chunked/tiled
  - Current kernel does not have enough structured staging to benefit yet
- Tensor-core / warpgroup async paths matter only if the math is restructured as chunk GEMMs or triangular block updates
  - The current scalar recurrent implementation is not in that form

Important constraint:
- Blackwell’s extra tensor-core capability does not automatically accelerate scalar recurrence kernels
- To benefit from Blackwell-specific math pipelines, the algorithm must be reformulated into chunked matrix operations

## Baseline FlashInfer Solution Signal

The provided FlashInfer baseline is not a token-serial scalar kernel.

Observed from baseline source bundle:
- It calls `chunk_gated_delta_rule`
- The Blackwell implementation is chunk-based
- It supports persistent and non-persistent modes
- It uses CuTe / CUTLASS Blackwell-specific machinery
- The documented chunk size is `128`

Interpretation:
- The strongest available prior signal is that high-performance Blackwell GDN prefill is chunked, not per-token scalar
- This does not prove the exact best design for this contest workload, but it is strong evidence for the main optimization direction

## Optimization Methodologies

### 1. Chunked reformulation of the recurrence

Idea:
- Process each sequence in token chunks, ideally `chunk_size = 128` first
- Convert token-by-token recurrence into chunk-local matrix operations plus inter-chunk state propagation

Why it matters:
- Makes the computation look like dense `128x128` tile math instead of repeated scalar dot products
- Opens the door to tensor-core or at least tile-optimized pipelines
- Greatly reduces synchronization overhead relative to the current per-`vi` scheme

Applicability to current kernel:
- Not a small patch
- Requires redesign of the algorithm and block decomposition
- High upside, highest engineering cost

Assessment:
- `Priority: P1`
- `Applicability now: yes, but as a rewrite`

### 2. Persistent kernel or work-stealing scheduler for variable-length imbalance

Idea:
- Replace static one-block-per-`(seq, vh)` scheduling with persistent CTAs
- Use Blackwell cluster launch control or a simpler persistent work queue to steal remaining long-sequence work

Why it matters:
- Work per block is proportional to sequence length
- Current workload distribution has very uneven lengths
- Static scheduling leaves SMs idle once short sequences finish

Applicability to current kernel:
- Possible without changing the math
- Can be added to both scalar and chunked versions
- Cluster launch control is Blackwell-specific and directly relevant

Assessment:
- `Priority: P1`
- `Applicability now: yes`

### 3. Two-path kernel family: short-sequence kernel and long-sequence kernel

Idea:
- Dispatch different kernels by average sequence length or per-sequence length bucket
- Example split:
  - short path: keep simple per-token recurrence, lower setup cost
  - long path: chunked / persistent / more aggressive staging

Why it matters:
- More than half the workloads have average sequence length below `128`
- A sophisticated chunk pipeline can lose on short cases due to setup overhead

Applicability to current kernel:
- Straightforward at the host dispatch level
- Requires implementing at least one additional kernel

Assessment:
- `Priority: P1`
- `Applicability now: yes`

### 4. Fuse the two `vi` loops to reduce synchronizations

Idea:
- Reduce the current `128 + 128` loop structure and the per-iteration full-block barriers
- Example direction:
  - compute `kdot` in tiles
  - keep a tile of `s_state` and `v` in registers/shared memory
  - update and emit output with fewer global block barriers

Why it matters:
- The current kernel pays synchronization cost far too often
- This is the clearest bottleneck visible from code inspection

Applicability to current kernel:
- Yes
- Moderate redesign, but still within the current scalar-kernel approach

Assessment:
- `Priority: P1`
- `Applicability now: yes`

### 5. Register tiling over `vi`

Idea:
- Each thread or warp owns a small tile of `vi` rows instead of handling one row at a time
- Hold several `s_state[vi][tid]` values in registers

Why it matters:
- Improves instruction-level parallelism
- Cuts repeated shared-memory traffic
- Reduces synchronization frequency if paired with tiled reduction

Applicability to current kernel:
- Yes
- Best combined with methodology 4

Assessment:
- `Priority: P2`
- `Applicability now: yes`

### 6. Warp-specialized reduction/output pipeline

Idea:
- Split roles across warps:
  - some warps compute `k @ temp`
  - some update state
  - some handle output accumulation / epilogue

Why it matters:
- Current kernel makes all warps wait at every stage
- Specialization can reduce idle lanes and allow pipelining

Applicability to current kernel:
- Possible, but awkward in the current one-thread-per-K mapping
- More natural after a tile redesign

Assessment:
- `Priority: P2`
- `Applicability now: partial`

### 7. Reuse q/k loads across the two v-heads that share one q/k head

Idea:
- Since `vh 0/1` share q/k head `0`, `vh 2/3` share q/k head `1`, etc., reuse q/k data across the paired v-head blocks
- Candidate mechanisms:
  - one block handles both paired v-heads
  - cluster/shared-memory cooperation across paired blocks

Why it matters:
- Current kernel duplicates q/k loads across v-head pairs
- This is a structural inefficiency specific to GVA

Applicability to current kernel:
- Yes, but easiest if one CTA handles a v-head pair
- Less compelling than synchronization and scheduling fixes, because q/k tensors are small relative to state work

Assessment:
- `Priority: P2`
- `Applicability now: yes`

### 8. TMA-based async staging for chunked kernels

Idea:
- Use Tensor Memory Accelerator to stage chunk tiles of q/k/v/g/beta or state tiles into shared memory
- Overlap asynchronous copy with compute using barriers/pipelines

Why it matters:
- TMA is best for bulk multidimensional transfers
- It becomes valuable once the kernel is chunked and tile-regular

Applicability to current kernel:
- Not worth applying to the current scalar token loop
- Valuable only after chunking or major tiling

Assessment:
- `Priority: P2`
- `Applicability now: not for current baseline, yes for chunked rewrite`

### 9. Thread block clusters + distributed shared memory

Idea:
- Use clusters when a logical tile benefits from multi-CTA cooperation
- Possible use cases:
  - two or more CTAs cooperatively hold state tiles
  - paired v-head processing
  - chunk pipeline stages split across CTAs

Why it matters:
- B200 supports DSMEM and cluster sizes above the portable default
- Could help reduce global-memory exchanges between cooperating CTAs

Applicability to current kernel:
- Overkill for the current baseline
- Plausible only after a chunked/tiled redesign

Assessment:
- `Priority: P3`
- `Applicability now: low`

### 10. L2 persistence hints for read-mostly tensors

Idea:
- Mark long-lived read-mostly data like `A_log`, `dt_bias`, and potentially sequence metadata for L2 persistence

Why it matters:
- Blackwell supports L2 persistence controls
- However, these tensors are tiny and not the dominant bottleneck

Applicability to current kernel:
- Yes, but low expected return

Assessment:
- `Priority: P4`
- `Applicability now: yes, low impact`

## Applicability Matrix For The Current Kernel

| Method | Can apply to current code path? | Expected upside | Cost |
|---|---|---:|---:|
| Chunked recurrence rewrite | Yes, as rewrite | Very high | Very high |
| Persistent / work stealing scheduler | Yes | High | Medium |
| Dual-path dispatch | Yes | High | Medium |
| Fuse `vi` loops / reduce barriers | Yes | High | Medium |
| Register tiling over `vi` | Yes | Medium | Medium |
| Warp specialization | Partial | Medium | Medium-high |
| GVA q/k reuse across 2 v-heads | Yes | Medium | Medium |
| TMA async staging | Not on current scalar design | High after rewrite | High |
| Clusters / DSMEM | Low on current design | Medium after rewrite | High |
| L2 persistence hints | Yes | Low | Low |

## Recommended Order

Recommended execution order, balancing upside and implementation risk:

1. Reduce synchronization in the existing scalar kernel
- Fuse or tile the current two `vi` loops
- Add register tiling
- Keep correctness constant

2. Introduce persistent scheduling or work stealing
- First try a simple persistent work queue
- If environment/toolchain allows, evaluate Blackwell cluster launch control

3. Split kernels by sequence regime
- Short-path kernel for small average sequence length
- Long-path kernel for large average sequence length

4. Rewrite the long-path kernel around chunked recurrence
- Start with `chunk_size = 128`
- Make the design compatible with TMA/pipelining

5. Only after chunking, consider deeper Blackwell features
- TMA
- thread block clusters / DSMEM
- more explicit tensor-core style scheduling

## Concrete Read On The Current Baseline

What is clearly worth trying now:
- Fewer `__syncthreads()` through tiled `vi` processing
- Register tiling
- Persistent scheduling for long-tail imbalance
- Separate short and long code paths

What is probably not worth trying first:
- Micro-optimizing `softplus` or `sigmoid`
- Small cache hints without changing the main compute structure
- Cluster/DSMEM before the kernel is tiled/chunked

What likely has the highest ceiling:
- Moving from token-serial scalar update to chunked block-linear recurrence

## Implementation Readiness Check

Applicable to the current scalar kernel with moderate refactor:
- Reduce full-block synchronization count
- Add persistent CTA scheduling or a global work queue
- Split short-sequence and long-sequence paths
- Pair two v-heads inside one CTA to reuse q/k loads

Applicable only after a more substantial redesign:
- TMA async staging
- cluster-level cooperation / DSMEM
- tensor-core-oriented chunked recurrence

Blocked at repo integration level:
- Even if the kernel in the track subfolder improves, the repo root submission entry still needs to be aligned before final packaging/submission

## Risks and Correctness Notes

Numerical risks:
- State is fp32 for a reason; moving state accumulation to bf16 is risky
- Gating terms involve `exp`, `softplus`, and `sigmoid`; approximate fast-math changes can shift recurrence behavior over long sequences
- The current correctness tolerance is relatively loose (`atol=1`, `rtol=0.3`, matched ratio `>= 0.9`), but regressions can still hide until the longest workloads

Algorithmic risks:
- Chunked reformulations must preserve exact state propagation semantics across chunk boundaries
- GVA head expansion must remain consistent with output head ordering
- Variable-length handling must preserve `cu_seqlens` semantics exactly

Performance risks:
- An aggressive long-sequence kernel can underperform on the many short workloads
- A large-shared-memory design can lose badly if it lowers active blocks too much

## Bottom Line

The current prefill kernel is a correct scalar baseline with on-chip state reuse, but its main bottlenecks are structural:
- too many full-block synchronizations
- static scheduling under variable-length imbalance
- no chunking, pipelining, or tensor-core-friendly reformulation

Best next actions:
- Near term: reduce synchronization and add persistent scheduling
- Mid term: split short and long sequence paths
- High-upside path: rewrite the long path as a chunked recurrence kernel, then exploit Blackwell features such as TMA and possibly cluster-level scheduling

## Sources

Repo sources:
- `/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- `/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json`
- `/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl`
- `/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/mlsys26-contest/solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/flashinfer_wrapper_123ca6.json`
- `/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/logs/prefill/bench_history.jsonl`
- `/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/logs/prefill/optimization_log.md`
- `/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/EVALUATION.md`
- `/home/jsg1504/workdir/KimJangMaengShin/codex-mlsys26/FAQ.md`

External sources:
- NVIDIA Blackwell Tuning Guide: https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html
- CUDA Programming Guide 13.2: https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf
- NVIDIA Blackwell architecture page: https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- CUDA Toolkit 12.8 / Blackwell support blog: https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support
