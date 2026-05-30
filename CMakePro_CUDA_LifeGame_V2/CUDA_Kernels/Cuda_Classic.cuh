#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// 经典生命游戏演化（float 版本）
void ClassicUpdateCuda(float* d_current, float* d_next, float* d_heatMap,
    int w, int h, float deltaTime, bool paused, float decay, int b_mask, int s_mask);

// 随机撒种（float 版本）
void SeedCudaLifeFloat(float* d_world, int w, int h, float density);

// 鼠标绘制（float 版本）
void MousePaintCudaFloat(float* d_world, float* d_heat, int w, int h,
    int mx, int my, int radius, bool erase);

// 人口统计（float 版本）
int GetPopulationCudaFloat(float* d_world, int w, int h);

#ifdef __cplusplus
}
#endif
