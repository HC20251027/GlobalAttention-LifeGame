#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
void BlockAttentionUpdateCuda(float* d_current, float* d_next, float* d_heatMap,
    float* d_blockRep, float* d_blockAttn, float* d_blockQ, float* d_blockK, float* d_blockV,
    int w, int h, float dt, bool paused, float decay,
    float lambda_distance, float lambda_state, float lambda_block, float sigma, int blockSize);
#ifdef __cplusplus
}
#endif
