#pragma once

// ===================================================================
// 可学习参数定义与预设值
// ===================================================================
struct AttentionParams {
    // --- 模式 2: 全局统计量注入 ---
    float w_local = 0.7f;       // 局部邻居权重
    float w_global = 0.3f;      // 全局上下文权重
    float threshold = 0.5f;      // sigmoid 激活阈值

    // --- 模式 3: 分块窗口注意力 ---
    float lambda_distance = 1.0f;  // 距离衰减权重
    float lambda_state = 0.5f;     // 状态相似度权重
    float lambda_block = 0.3f;     // 块间注意力权重
    float sigma = 2.0f;            // 距离衰减标准差
    int blockSize = 32;             // 块大小 B

    // --- 模式 4: 精确全局注意力 ---
    static constexpr int MAX_FEATURE_DIM = 16;
    int featureDim = 4;            // Q/K/V 特征维度 d
    float W_q[48];  // 3 × MAX_FEATURE_DIM (输入: state, pos_x, pos_y)
    float W_k[48];
    float W_v[48];
    float W_o[16];  // MAX_FEATURE_DIM
    float bias = 0.0f;

    // 构造函数：初始化预设值
    AttentionParams() {
        // 使用简单的伪随机正交初始化（预设值）
        unsigned int seed = 42;
        auto pseudoRandom = [&seed]() -> float {
            seed = seed * 747796405U + 2891336453U;
            return ((float)(seed & 0xFFFF) / 65536.0f) - 0.5f;
        };
        for (int i = 0; i < 48; i++) {
            W_q[i] = pseudoRandom() * 0.1f;
            W_k[i] = pseudoRandom() * 0.1f;
            W_v[i] = pseudoRandom() * 0.1f;
        }
        for (int i = 0; i < 16; i++) {
            W_o[i] = pseudoRandom() * 0.1f;
        }
    }
};

// 模式规模限制配置
struct SimModeLimits {
    int maxW;
    int maxH;
    const char* description;
};
