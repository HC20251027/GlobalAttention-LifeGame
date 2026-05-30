#pragma once
#include <cuda_runtime.h>
#include "AttentionParams.h"
#ifdef __cplusplus
extern "C" {
#endif
void FullAttentionUpdateCuda(float* d_current, float* d_next, float* d_heatMap,
    float* d_Q, float* d_K, float* d_V, float* d_attnScores, float* d_attnOut,
    int w, int h, float dt, bool paused, float decay,
    const AttentionParams& params);
#ifdef __cplusplus
}
#endif
