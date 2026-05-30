#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
void GlobalStatsUpdateCuda(float* d_current, float* d_next, float* d_heatMap,
    float* d_globalStats, int w, int h, float dt, bool paused, float decay,
    float w_local, float w_global, float threshold);
#ifdef __cplusplus
}
#endif
