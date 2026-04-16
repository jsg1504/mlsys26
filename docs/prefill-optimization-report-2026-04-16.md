# GDN Prefill Kernel Optimization Report

**Date:** 2026-04-16
**Target:** `gdn_prefill_qk4_v8_d128_k_last` on NVIDIA B200 (sm_100a)
**Result:** Mean latency **0.343 ms → 0.284 ms** (17.2% improvement)

---

## 1. Benchmark History

| # | Date | Latency (ms) | Speedup | Git Hash | Description |
|---|------|-------------|---------|----------|-------------|
| 1 | 04-06 | 34.28 | 6.4x | HEAD | Initial TVM FFI baseline (sequential CUDA) |
| 2 | 04-15 | 0.343 | 488x | 4660730 | CuTe/CUTLASS Blackwell migration |
| 3 | 04-16 | 3.363 | 66x | 14f8b9e | CUDA V-parallel (shared memory state) |
| 4 | 04-16 | 1.109 | 166x | 14f8b9e | CUDA V-parallel (register-cached state) |
| 5 | 04-16 | 0.334 | 558x | 14f8b9e | CuTe-DSL + optimized Python wrapper |
| 6 | 04-16 | 0.313 | 533x | 3d9442e | Hybrid dispatch (THRESHOLD=512) |
| 7 | 04-16 | **0.284** | **540x** | 4f6a958 | Hybrid dispatch (THRESHOLD=128, tuned) |

**Trend:** Improving. Best achieved: #7 (0.284 ms, 540x speedup).

---

## 2. What Was Done

### Phase 1: NCU Profiling & Bottleneck Analysis

프로파일링 스크립트(`scripts/profile_kernel.py`)를 수정하여 CuTe-DSL JIT 커널(`kernel_cutlass_kernel_fib_pyt...`)을 정확히 캡처했습니다.

**NCU 프로파일 결과 (5개 대표 워크로드):**

| Workload | T | N | Grid | Duration | Compute% | Occupancy |
|----------|------|-----|------|----------|----------|-----------|
| Short | 6 | 1 | 8 | 64 μs | 0.4% | 10.5% |
| Medium | 134 | 1 | 8 | 113 μs | 0.5% | 10.8% |
| Long | 8192 | 32 | 256 | 787 μs | 20.2% | 12.2% |

**발견한 병목:**
1. CuTe-DSL 커널 최소 비용 64 μs (227 KB shared mem, TMA setup, 8-warp pipeline)
2. Gate 연산에 PyTorch 커널 10회 launch (~50-80 μs 오버헤드)
3. `cu_seqlens` CPU 동기화 (`.cpu().tolist()`)
4. 짧은 시퀀스(60%)에서 GPU 활용도 < 1% (148 SM 중 8개만 사용)

### Phase 1: Hybrid Dispatch 구현

**아키텍처:**
```
main.py::run()
  ├── T ≤ 128 → NVRTC sequential CUDA kernel (fused gates)
  └── T > 128 → NVRTC fused gate kernel + CuTe-DSL chunked kernel
```

**구현한 파일:**

| 파일 | 역할 |
|------|------|
| `nvrtc_loader.py` | NVRTC 컴파일/캐시/실행 유틸리티 (cuda-python) |
| `nvrtc_kernels/sequential_kernel.cu` | Register-cached V-parallel GDN 커널 (fused gates) |
| `nvrtc_kernels/fused_gate_kernel.cu` | Gate 연산 퓨전 커널 (10 PyTorch 커널 → 1) |
| `main.py` | Hybrid dispatch 로직 |

**Sequential CUDA Kernel 핵심 설계:**
- 128 threads/block, 각 thread가 V dimension 1개 담당
- State column (128 fp32) 전체를 레지스터에 캐시 (`__launch_bounds__(128, 1)`)
- Gate 연산 (softplus, exp, sigmoid) 커널 내부에서 직접 계산
- Shared memory: k[128] + q[128]만 사용 (broadcast)
- `__fmaf_rn` intrinsics로 state update/output 정밀도 유지

**Fused Gate Kernel:**
- `gate_log[t,h] = -exp(A_log[h]) * softplus(a[t,h] + dt_bias[h])`
- `beta[t,h] = sigmoid(b[t,h])`
- 단일 커널, ~5 μs 실행 시간

### Phase 1: THRESHOLD 튜닝

| THRESHOLD | Mean Latency | Median |
|-----------|-------------|--------|
| 64 | 0.284 ms | 0.186 ms |
| **128** | **0.284 ms** | **0.188 ms** |
| 256 | 0.294 ms | 0.215 ms |
| 512 | 0.313 ms | 0.233 ms |
| 1024 | 0.390 ms | 0.282 ms |

THRESHOLD=128 선택 (64와 동일 성능, 더 안정적).

### Phase 2: CuTe-DSL 오버헤드 추가 최적화

- `chunk_gated_delta_rule`에 `precomputed_max_s_q` 파라미터 추가
- `num_seqs == 1`일 때 `max_s_q = T`로 GPU sync 스킵
- `.item()` sync 위치 최적화 시도

**결과:** 추가 개선 미미 (~0.001 ms). CuTe-DSL 커널 실행 시간 자체가 병목으로 확인됨.

---

## 3. Key Learnings

### 성공한 최적화
1. **Register caching**: Shared memory → register로 state 이동 시 3x 속도 향상 (3.36 ms → 1.11 ms)
2. **Gate fusion**: 10개 PyTorch 커널을 1개 NVRTC CUDA 커널로 퓨전, ~50 μs 절감
3. **Hybrid dispatch**: 짧은 시퀀스에 경량 CUDA 커널 사용으로 CuTe-DSL의 64 μs 최소 비용 회피
4. **NVRTC**: torch 의존성 없이 cuda-python으로 커널 컴파일/실행 가능

### 실패한/효과 없었던 시도
1. **Dual v-head kernel (256 threads)**: 레지스터 압력 증가로 오히려 느려짐 (1.52 ms vs 1.11 ms)
2. **THRESHOLD 올리기 (1024)**: CUDA sequential이 중간 길이에서 CuTe-DSL보다 느려 역효과
3. **`__fmaf_rn` intrinsics**: 컴파일러가 이미 `--use_fast_math`로 FMA 생성 중, 효과 없음
4. **`.item()` overlap**: gate kernel과 max_s_q sync가 같은 stream에서 순차 실행, overlap 불가

### CuTe-DSL 커널의 벽
- 227 KB shared memory + 240 registers → SM당 1 블록 제한
- Theoretical occupancy 12.5% (실제 10-12%)
- Blackwell WGMMA (128x256x16 per instruction)로 이미 최적화됨
- Python 오버헤드를 모두 제거해도 커널 실행 시간이 500-1000 μs로 잔존

---

## 4. 현재 아키텍처

```
gdn_prefill_qk4_v8_d128_k_last/solution/python/
├── main.py                          # Hybrid dispatch (THRESHOLD=128)
├── nvrtc_loader.py                  # NVRTC compile/cache/launch
├── nvrtc_kernels/
│   ├── sequential_kernel.cu         # Short path (T ≤ 128)
│   └── fused_gate_kernel.cu         # Gate fusion (long path)
├── gdn_blackwell/
│   ├── gdn.py                       # CuTe-DSL chunked kernel (long path)
│   ├── dispatch.py                  # Path selection
│   ├── gdn_helpers.py               # SMEM layout helpers
│   └── gdn_tile_scheduler.py        # Tile scheduling
└── prefill_contract.py              # Validation (unused in hot path)
```

---

## 5. 추가 최적화 방향 (미구현)

| 방향 | 예상 효과 | 난이도 | 설명 |
|------|----------|--------|------|
| Chunked CUDA + wmma | -30% (medium) | 높음 | Forward substitution + 16x16 tensor core matmul |
| CuTe-DSL chunk_size 64 | -10~20% | 중간 | Shared mem 감소 → occupancy 2배 |
| CuTe-DSL persistent mode | -5~10% | 중간 | Launch overhead 감소 |
| Blackwell WGMMA PTX assembly | 최적 | 매우 높음 | CuTe-DSL 수준 성능의 순수 CUDA 구현 |
