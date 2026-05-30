#include "Cuda_BlockAttention.cuh"
#include <device_launch_parameters.h>
#include <cmath>

// ============================================================================
// 辅助 Kernel：计算每个块的代表向量（块内状态加权平均）
// ============================================================================
__global__ void kComputeBlockRep(const float* state, float* blockRep,
    int w, int h, int blockW, int blockH, int numBlocksX, int numBlocksY)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    if (bx >= numBlocksX || by >= numBlocksY) return;

    int start_x = bx * blockW;
    int start_y = by * blockH;
    int end_x = min(start_x + blockW, w);
    int end_y = min(start_y + blockH, h);

    float sum = 0.0f;
    int count = 0;
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            sum += state[y * w + x];
            count++;
        }
    }

    int blockIdx = by * numBlocksX + bx;
    blockRep[blockIdx] = (count > 0) ? (sum / (float)count) : 0.0f;
}

// ============================================================================
// 辅助 Kernel：计算块间注意力得分矩阵
// ============================================================================
__global__ void kBlockAttentionScores(const float* blockRep, float* attnScores,
    int numBlocks, float lambda_distance, float sigma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= numBlocks || j >= numBlocks) return;

    // 块代表向量的相似度（标量情况）
    float rep_i = blockRep[i];
    float rep_j = blockRep[j];

    // 简化的距离衰减（基于块索引距离）
    int bi = i; int bj = j;
    int numBlocksPerRow = (int)sqrtf((float)numBlocks + 0.5f);
    int ix = bi % numBlocksPerRow;
    int iy = bi / numBlocksPerRow;
    int jx = bj % numBlocksPerRow;
    int jy = bj / numBlocksPerRow;
    float dist_sq = (float)(ix - jx) * (ix - jx) + (float)(iy - jy) * (iy - jy);

    float score = rep_i * rep_j * expf(-dist_sq / (2.0f * sigma * sigma));
    attnScores[i * numBlocks + j] = score;
}

// ============================================================================
// 辅助 Kernel：对注意力得分做行级 Softmax
// ============================================================================
__global__ void kSoftmaxRows(float* scores, int numRows, int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    float* rowPtr = scores + row * numCols;

    // 找最大值
    float maxVal = rowPtr[0];
    for (int j = 1; j < numCols; j++) {
        if (rowPtr[j] > maxVal) maxVal = rowPtr[j];
    }

    // 减去最大值后 exp 并求和
    float sum = 0.0f;
    for (int j = 0; j < numCols; j++) {
        rowPtr[j] = expf(rowPtr[j] - maxVal);
        sum += rowPtr[j];
    }

    // 归一化
    if (sum > 0.0f) {
        for (int j = 0; j < numCols; j++) {
            rowPtr[j] /= sum;
        }
    }
}

// ============================================================================
// 核心 Kernel：分块注意力更新
// ============================================================================
__global__ void kBlockAttentionUpdate(
    const float* current, float* next, float* heatMap,
    const float* blockRep, const float* blockAttn,
    int w, int h, float dt, bool paused, float decay,
    float lambda_distance, float lambda_state, float lambda_block, float sigma,
    int blockW, int blockH, int numBlocksX, int numBlocksY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;

    if (!paused) {
        // --- 块内注意力（局部邻居 + 块内加权）---
        int bx = x / blockW;
        int by = y / blockH;
        int localBlockIdx = by * numBlocksX + bx;

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

        // --- 块间注意力贡献 ---
        float blockInfluence = 0.0f;
        int numBlocks = numBlocksX * numBlocksY;
        const float* myAttnRow = blockAttn + localBlockIdx * numBlocks;
        for (int b = 0; b < numBlocks; b++) {
            blockInfluence += myAttnRow[b] * blockRep[b];
        }

        // --- 融合决策 ---
        float z = lambda_distance * localSum + lambda_state * blockInfluence * 8.0f;

        // 阈值激活
        float threshold = 0.5f * 8.0f; // 缩放到与 localSum 同量级
        next[idx] = (z > threshold) ? 1.0f : 0.0f;
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
// Host 函数：分块注意力更新
// ============================================================================
extern "C" void BlockAttentionUpdateCuda(
    float* d_current, float* d_next, float* d_heatMap,
    float* d_blockRep, float* d_blockAttn, float* d_blockQ, float* d_blockK, float* d_blockV,
    int w, int h, float dt, bool paused, float decay,
    float lambda_distance, float lambda_state, float lambda_block, float sigma, int blockSize)
{
    int numBlocksX = (w + blockSize - 1) / blockSize;
    int numBlocksY = (h + blockSize - 1) / blockSize;
    int numBlocks = numBlocksX * numBlocksY;

    if (!paused) {
        // --- 阶段 1: 计算块代表向量 ---
        dim3 blockGrid(numBlocksX, numBlocksY);
        kComputeBlockRep<<<blockGrid, 1>>>(d_current, d_blockRep, w, h, blockSize, blockSize, numBlocksX, numBlocksY);

        // --- 阶段 2: 计算块间注意力得分矩阵 ---
        int attnThreads = 16;
        dim3 attnGrid((numBlocks + attnThreads - 1) / attnThreads, (numBlocks + attnThreads - 1) / attnThreads);
        kBlockAttentionScores<<<attnGrid, dim3(attnThreads, attnThreads)>>>(
            d_blockRep, d_blockAttn, numBlocks, lambda_distance, sigma);

        // --- 阶段 3: Softmax 归一化 ---
        int smThreads = 256;
        int smBlocks = (numBlocks + smThreads - 1) / smThreads;
        kSoftmaxRows<<<smBlocks, smThreads>>>(d_blockAttn, numBlocks, numBlocks);

        cudaDeviceSynchronize();
    }

    // --- 阶段 4: 执行细胞更新 ---
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    kBlockAttentionUpdate<<<grid, block>>>(
        d_current, d_next, d_heatMap,
        d_blockRep, d_blockAttn,
        w, h, dt, paused, decay,
        lambda_distance, lambda_state, lambda_block, sigma,
        blockSize, blockSize, numBlocksX, numBlocksY);
}
