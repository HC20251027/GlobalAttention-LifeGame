#include "Cuda_FullAttention.cuh"
#include <device_launch_parameters.h>
#include <cmath>

// ============================================================================
// 辅助 Kernel：计算 Q/K/V（每个细胞一个 d 维向量）
// ============================================================================
__global__ void kComputeQKV(
    const float* state,
    float* Q, float* K, float* V,
    const float* W_q, const float* W_k, const float* W_v,
    int n, int w, int h, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float s = state[idx];
    float px = (float)(idx % w) / (float)w;  // 归一化位置 x
    float py = (float)(idx / w) / (float)h;  // 归一化位置 y
    float input[3] = { s, px, py };

    for (int j = 0; j < d; j++) {
        Q[idx * d + j] = W_q[0 * d + j] * input[0] + W_q[1 * d + j] * input[1] + W_q[2 * d + j] * input[2];
        K[idx * d + j] = W_k[0 * d + j] * input[0] + W_k[1 * d + j] * input[1] + W_k[2 * d + j] * input[2];
        V[idx * d + j] = W_v[0 * d + j] * input[0] + W_v[1 * d + j] * input[1] + W_v[2 * d + j] * input[2];
    }
}

// ============================================================================
// 辅助 Kernel：分块计算注意力得分（每个 query 块 vs 所有 keys）
// ============================================================================
__global__ void kAttentionScoresBlock(
    const float* Q, const float* K, float* scores,
    int n, int d, int qStart, int qEnd, float scale)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x + qStart;
    int ki = blockIdx.y * blockDim.y + threadIdx.y;
    if (qi >= qEnd || ki >= n) return;

    float dot = 0.0f;
    for (int j = 0; j < d; j++) {
        dot += Q[qi * d + j] * K[ki * d + j];
    }
    scores[(qi - qStart) * n + ki] = dot * scale;
}

// ============================================================================
// 辅助 Kernel：行级 Softmax（分块版本）
// ============================================================================
__global__ void kSoftmaxRowsBlock(float* scores, int qSize, int n) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= qSize) return;

    float* row = scores + qi * n;

    float maxVal = row[0];
    for (int j = 1; j < n; j++) {
        if (row[j] > maxVal) maxVal = row[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        row[j] = expf(row[j] - maxVal);
        sum += row[j];
    }

    if (sum > 0.0f) {
        float invSum = 1.0f / sum;
        for (int j = 0; j < n; j++) {
            row[j] *= invSum;
        }
    }
}

// ============================================================================
// 辅助 Kernel：加权聚合（每个 query 块 vs 所有 values）
// ============================================================================
__global__ void kAttentionAggregateBlock(
    const float* scores, const float* V, float* out,
    int n, int d, int qStart, int qEnd)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x + qStart;
    int dj = blockIdx.y * blockDim.y + threadIdx.y;
    if (qi >= qEnd || dj >= d) return;

    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += scores[(qi - qStart) * n + k] * V[k * d + dj];
    }
    out[qi * d + dj] = sum;
}

// ============================================================================
// 辅助 Kernel：输出投影 + sigmoid 激活
// ============================================================================
__global__ void kOutputProjection(
    const float* attnOut, float* next, float* heatMap,
    const float* W_o, float bias,
    int n, int d, int w, int h, float dt, bool paused, float decay)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!paused) {
        // 投影到标量
        float z = bias;
        for (int j = 0; j < d; j++) {
            z += W_o[j] * attnOut[idx * d + j];
        }

        // sigmoid
        float sigmoid_z = 1.0f / (1.0f + expf(-z));
        next[idx] = (sigmoid_z > 0.5f) ? 1.0f : 0.0f;
    }

    // 热力图
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
// Host 函数：精确全局注意力更新（分块计算以控制显存）
// ============================================================================
extern "C" void FullAttentionUpdateCuda(
    float* d_current, float* d_next, float* d_heatMap,
    float* d_Q, float* d_K, float* d_V, float* d_attnScores, float* d_attnOut,
    int w, int h, float dt, bool paused, float decay,
    const AttentionParams& params)
{
    int n = w * h;
    int d = params.featureDim;
    float scale = 1.0f / sqrtf((float)d);

    if (!paused) {
        // --- 阶段 1: 计算 Q/K/V ---
        int threads1d = 256;
        int blocks1d = (n + threads1d - 1) / threads1d;
        kComputeQKV<<<blocks1d, threads1d>>>(
            d_current, d_Q, d_K, d_V,
            params.W_q, params.W_k, params.W_v, n, w, h, d);

        // --- 阶段 2: 分块计算注意力得分 + Softmax + 聚合 ---
        // 分块大小：每次处理 QUERY_BLOCK_SIZE 个 query
        const int QUERY_BLOCK_SIZE = 256;

        for (int qStart = 0; qStart < n; qStart += QUERY_BLOCK_SIZE) {
            int qEnd = min(qStart + QUERY_BLOCK_SIZE, n);
            int qSize = qEnd - qStart;

            // 2a. 注意力得分
            dim3 scoreGrid((qSize + 15) / 16, (n + 15) / 16);
            kAttentionScoresBlock<<<scoreGrid, dim3(16, 16)>>>(
                d_Q, d_K, d_attnScores, n, d, qStart, qEnd, scale);

            // 2b. Softmax
            int smBlocks = (qSize + 255) / 256;
            kSoftmaxRowsBlock<<<smBlocks, 256>>>(d_attnScores, qSize, n);

            // 2c. 加权聚合
            dim3 aggGrid((qSize + 15) / 16, (d + 15) / 16);
            kAttentionAggregateBlock<<<aggGrid, dim3(16, 16)>>>(
                d_attnScores, d_V, d_attnOut, n, d, qStart, qEnd);

            cudaDeviceSynchronize();
        }

        // --- 阶段 3: 输出投影 + 激活 ---
        kOutputProjection<<<blocks1d, threads1d>>>(
            d_attnOut, d_next, d_heatMap,
            params.W_o, params.bias, n, d, w, h, dt, paused, decay);
    } else {
        // paused 时只更新热力图
        int threads1d = 256;
        int blocks1d = (n + threads1d - 1) / threads1d;
        // 直接调用热力图更新（复用 kOutputProjection 的 paused 路径）
        kOutputProjection<<<blocks1d, threads1d>>>(
            d_attnOut, d_next, d_heatMap,
            params.W_o, params.bias, n, d, w, h, dt, true, decay);
    }
}
