// buggy_matmul_fixture.cuh - intentionally broken kernel, used ONLY to test
// scripts/amp_diagnose.py's static analyzer (see tests/test_diagnose.py).
// This recreates a known bug class: shared-memory dimension mismatch
// plus a missing __syncthreads().
#pragma once

template<int BM, int BN, int BK>
__global__ void buggy_matmul_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    int ty = threadIdx.y, tx = threadIdx.x;
    float acc = 0.0f;

    for (int kb = 0; kb < K; kb += BK) {
        sA[ty][tx] = A[ty * K + kb + tx];
        sB[ty][tx] = B[(kb + ty) * N + tx];
        // BUG 1: no __syncthreads() here before the read loop below —
        // some threads may read sA/sB before others finished writing it.
        for (int k = 0; k < BN; ++k) acc += sA[ty][k] * sB[k][tx];
        // BUG 2 (dimension mismatch): loop bound is BN, but sA/sB's second/
        // first dimension is declared BK — out-of-bounds read if BN != BK.
    }
    C[ty * N + tx] = acc;
}
