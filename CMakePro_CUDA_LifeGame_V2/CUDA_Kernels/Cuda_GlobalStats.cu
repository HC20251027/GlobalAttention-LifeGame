#include "Cuda_GlobalStats.cuh"
#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cmath>

// ============================================================================
// 辅助 Kernel：计算状态方差
// ============================================================================
__global__ void kComputeVariance(const float* state, float mean, float* variance, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = state[idx] - mean;
        variance[idx] = diff * diff;
    }
}

// ============================================================================
// 辅助 Kernel：计算空间加权中心
// ============================================================================
__global__ void kComputeCenter(const float* state, float* cx_out, float* cy_out, int w, int h) {
    // 使用单个 block 做规约
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w * h;

    // 累加 x * state
    sdata[tid] = (idx < n) ? ((float)(idx % w) * state[idx]) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(cx_out, sdata[0]);

    // 累加 y * state
    sdata[tid] = (idx < n) ? ((float)(idx / w) * state[idx]) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(cy_out, sdata[0]);
}

// ============================================================================
// 核心 Kernel：全局统计量注入更新
// ============================================================================
__global__ void kGlobalStatsUpdate(
    const float* current, float* next, float* heatMap,
    const float* globalStats, // [total, mean, variance, cx, cy]
    int w, int h, float dt, bool paused, float decay,
    float w_local, float w_global, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;

    if (!paused) {
        // --- 局部邻居和（与经典模式一致）---
        float localSum = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = y + dy;
            if (ny < 0) ny = h - 1;
            else if (ny >= h) ny = 0;
            int rowOffset = ny * w;
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx;
                if (nx < 0) nx = w - 1;
                else if (nx >= w) nx = 0;
                localSum += current[rowOffset + nx];
            }
        }

        // --- 全局上下文计算 ---
        float total   = globalStats[0];
        float mean    = globalStats[1];
        float variance = globalStats[2];
        float cx      = globalStats[3];
        float cy      = globalStats[4];

        // 到存活质心的归一化距离
        float dx_center = (float)x - cx;
        float dy_center = (float)y - cy;
        float dist_to_center = sqrtf(dx_center * dx_center + dy_center * dy_center);
        float maxDist = sqrtf((float)(w * w + h * h));
        float normDist = dist_to_center / maxDist;

        // 全局上下文信号：结合密度、方差和空间位置
        float globalContext = mean * (1.0f - normDist) + sqrtf(variance) * normDist * 0.1f;

        // --- 加权融合决策 ---
        float z = w_local * localSum + w_global * globalContext * 8.0f; // ×8 缩放到与 localSum 同量级

        // sigmoid 激活
        float sigmoid_z = 1.0f / (1.0f + expf(-(z - threshold * 8.0f)));

        next[idx] = (sigmoid_z > 0.5f) ? 1.0f : 0.0f;
    }

    // 热力图（与经典模式一致）
    float hVal = heatMap[idx];
    if (next[idx] > 0.5f) {
        hVal = 1.0f;
    } else {
        hVal *= decay;
        if (hVal < 0.005f) hVal = 0.0f;
    }
    heatMap[idx] = hVal;
}

// ============================================================================
// Host 函数：计算全局统计量并执行更新
// ============================================================================
extern "C" void GlobalStatsUpdateCuda(
    float* d_current, float* d_next, float* d_heatMap,
    float* d_globalStats, int w, int h, float dt, bool paused, float decay,
    float w_local, float w_global, float threshold)
{
    int n = w * h;

    if (!paused) {
        // --- 阶段 1: 计算全局统计量 ---

        // 1a. 总和
        thrust::device_ptr<float> ptr(d_current);
        float total = thrust::reduce(thrust::device, ptr, ptr + n, 0.0f, thrust::plus<float>());
        float mean = total / (float)n;

        // 1b. 方差
        float* d_variance = nullptr;
        cudaMalloc(&d_variance, n * sizeof(float));
        int threads1d = 256;
        int blocks1d = (n + threads1d - 1) / threads1d;
        kComputeVariance<<<blocks1d, threads1d>>>(d_current, mean, d_variance, n);
        thrust::device_ptr<float> varPtr(d_variance);
        float variance = thrust::reduce(thrust::device, varPtr, varPtr + n, 0.0f, thrust::plus<float>()) / (float)n;
        cudaFree(d_variance);

        // 1c. 空间加权中心
        float* d_cx = nullptr;
        float* d_cy = nullptr;
        cudaMalloc(&d_cx, sizeof(float));
        cudaMalloc(&d_cy, sizeof(float));
        cudaMemset(d_cx, 0, sizeof(float));
        cudaMemset(d_cy, 0, sizeof(float));

        int reduceThreads = 512;
        int reduceBlocks = (n + reduceThreads - 1) / reduceThreads;
        kComputeCenter<<<reduceBlocks, reduceThreads, reduceThreads * sizeof(float)>>>(
            d_current, d_cx, d_cy, w, h);
        cudaDeviceSynchronize();

        float cx_host = 0.0f, cy_host = 0.0f;
        cudaMemcpy(&cx_host, d_cx, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cy_host, d_cy, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_cx);
        cudaFree(d_cy);

        // 归一化质心（total > 0 时）
        if (total > 0.0f) {
            cx_host /= total;
            cy_host /= total;
        }

        // 1d. 写入全局统计量缓冲区
        float h_stats[5] = { total, mean, variance, cx_host, cy_host };
        cudaMemcpy(d_globalStats, h_stats, 5 * sizeof(float), cudaMemcpyHostToDevice);
    }

    // --- 阶段 2: 执行更新 ---
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    kGlobalStatsUpdate<<<grid, block>>>(
        d_current, d_next, d_heatMap, d_globalStats,
        w, h, dt, paused, decay, w_local, w_global, threshold);
}
