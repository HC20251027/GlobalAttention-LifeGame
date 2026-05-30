#include "Cuda_Classic.cuh"
#include <device_launch_parameters.h>
#include <time.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// ============================================================================
// 无状态哈希随机数引擎（与原项目一致）
// ============================================================================
__device__ __forceinline__ float GetRandomFloatStateless(unsigned int index, unsigned int seed) {
    unsigned int state = index * 747796405U + 2891336453U + seed;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    unsigned int result = (word >> 22u) ^ word;
    return (float)result / 4294967295.0f;
}

// ============================================================================
// 随机撒种 Kernel（float 版本）
// ============================================================================
__global__ void kSeedKernelFloat(float* world, unsigned int seed, float density, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < w * h) {
        float r = GetRandomFloatStateless(idx, seed);
        world[idx] = (r < density) ? 1.0f : 0.0f;
    }
}

extern "C" void SeedCudaLifeFloat(float* d_world, int w, int h, float density) {
    int n = w * h;
    unsigned int seed = (unsigned int)time(NULL);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kSeedKernelFloat<<<blocks, threads>>>(d_world, seed, density, w, h);
    cudaDeviceSynchronize();
}

// ============================================================================
// 经典生命游戏演化 Kernel（float 版本，逻辑与原 kLifeUpdate 完全一致）
// ============================================================================
__global__ void kClassicUpdate(const float* current, float* next, float* heatMap,
    int w, int h, float dt, bool paused, float decay, int b_mask, int s_mask)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;

    if (!paused) {
        int neighbors = 0;

        // 环面拓扑边界包裹（GPU 分支替代模运算）
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

                // float 版本：> 0.5f 视为存活
                neighbors += (current[rowOffset + nx] > 0.5f) ? 1 : 0;
            }
        }

        float alive = current[idx];
        if (alive > 0.5f) {
            next[idx] = (s_mask & (1 << neighbors)) ? 1.0f : 0.0f;
        } else {
            next[idx] = (b_mask & (1 << neighbors)) ? 1.0f : 0.0f;
        }
    }

    // 热力图计算（与原项目一致）
    float hVal = heatMap[idx];
    if (next[idx] > 0.5f) {
        hVal = 1.0f;
    } else {
        hVal *= decay;
        if (hVal < 0.005f) hVal = 0.0f;
    }
    heatMap[idx] = hVal;
}

extern "C" void ClassicUpdateCuda(float* d_current, float* d_next, float* d_heatMap,
    int w, int h, float deltaTime, bool paused, float decay, int b_mask, int s_mask)
{
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    kClassicUpdate<<<grid, block>>>(d_current, d_next, d_heatMap, w, h, deltaTime, paused, decay, b_mask, s_mask);
}

// ============================================================================
// 鼠标绘制 Kernel（float 版本）
// ============================================================================
__global__ void kMousePaintFloat(float* world, float* heatMap, int w, int h,
    int mx, int my, int radius, bool erase)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int dx = x - mx;
    int dy = y - my;
    if (dx * dx + dy * dy < radius * radius) {
        int idx = y * w + x;
        world[idx] = erase ? 0.0f : 1.0f;
        if (!erase) heatMap[idx] = 1.0f;
    }
}

extern "C" void MousePaintCudaFloat(float* d_world, float* d_heat, int w, int h,
    int mx, int my, int radius, bool erase)
{
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    kMousePaintFloat<<<grid, block>>>(d_world, d_heat, w, h, mx, my, radius, erase);
}

// ============================================================================
// 人口统计（float 版本，Thrust 规约）
// ============================================================================
__global__ void kCountAlive(const float* world, int* out, int n) {
    // 每个线程处理一个元素，将 float 转为 int（>0.5 → 1, 否则 0）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (world[idx] > 0.5f) ? 1 : 0;
    }
}

extern "C" int GetPopulationCudaFloat(float* d_world, int w, int h) {
    int n = w * h;
    int* d_int = nullptr;
    cudaMalloc(&d_int, n * sizeof(int));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kCountAlive<<<blocks, threads>>>(d_world, d_int, n);

    thrust::device_ptr<int> ptr(d_int);
    int result = thrust::reduce(thrust::device, ptr, ptr + n, (int)0, thrust::plus<int>());

    cudaFree(d_int);
    return result;
}
