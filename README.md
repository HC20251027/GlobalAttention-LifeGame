# 🧬 全局注意力生命游戏 (Global Attention Life Game)

> 基于 **CUDA + OpenGL 零拷贝互操作** 架构的高性能细胞自动机仿真系统  
> 从经典康威生命游戏到 Transformer 式全局注意力机制的进化

---

## 📑 文档索引

| 章节 | 说明 |
|---|---|
| [🚀 项目简介](#-项目简介) | 全局注意力改造概述与核心特性 |
| [🎮 四种演化模式](#-四种演化模式) | Classic / GlobalStats / BlockAttention / FullAttention 详解 |
| [🖥️ 运行环境](#️-运行环境) | 硬件要求与软件依赖 |
| [🛠️ 构建指南](#️-构建指南) | Windows / Linux 编译步骤 |
| [🎮 操作指南](#-操作指南-1) | 基础操作、模式切换、RLE 导入 |
| [📊 性能参考](#-性能参考) | RTX 4060 实测数据 |
| [🔬 技术架构](#-技术架构) | 数据流图与设计决策 |
| [📁 项目结构](#-项目结构) | 文件组织说明 |
| [🎯 推荐实验](#-推荐实验) | 4 种有趣实验建议 |
| [⚠️ 已知限制](#️-已知限制) | 使用注意事项 |
| [📖 原项目文档](#-原项目文档) | CUDA 极速生命游戏 V1.2 原始 README |

---

## 🚀 项目简介

本项目是 [CMakePro_LifeGame](https://github.com/XiaoYu-1111/CMakePro_LifeGame) 的深度改造版本，在保留原有极致性能的基础上，引入了 **全局注意力机制**，打破了经典生命游戏"仅依赖 8 邻居"的局部限制。每个细胞现在可以"感知"和"影响"整个网格的状态，演化出更复杂、协同性更强的全局模式。

### 核心特性

| 特性 | 说明 |
|---|---|
| **4 种演化模式** | 经典 / 全局统计 / 分块注意 / 精确全局注意 |
| **统一 float 数据** | 所有模式共享 float 双缓冲，无缝切换 |
| **可学习参数** | 注意力权重计算包含可调参数，预设值即开即用 |
| **规模自适应** | 不同模式自动适配推荐网格规模 |
| **实时交互** | 鼠标绘制、RLE 导入、参数实时调节 |

---

## 🎮 四种演化模式

### 1. 经典生命游戏 (Classic)
- **规则**: 传统 B3/S23，仅依赖 8 邻居
- **规模**: 无限制（最大 32768×32768）
- **计算**: O(N)，GPU 并行邻居计数

### 2. 全局统计量注入 (Global Stats)
- **机制**: 每帧计算全局统计量（总存活数、均值、方差、空间质心）
- **决策**: 局部邻居和 + 全局上下文加权融合
- **规模**: 无限制（Thrust 规约 O(N)）
- **参数**: `w_local`, `w_global`, `threshold`

### 3. 分块窗口注意力 (Block Attention)
- **机制**: 网格分块，块内精确注意力 + 块间 Softmax 注意力
- **决策**: 距离衰减 + 状态相似度 + 块间注意力加权
- **规模**: 推荐 ≤ 4096×4096（块间矩阵显存可控）
- **参数**: `λ_distance`, `λ_state`, `λ_block`, `σ`, `blockSize`

### 4. 精确全局注意力 (Full Attention)
- **机制**: 完整 Transformer 式注意力，每个细胞与所有细胞交互
- **计算**: Q/K/V 投影 → 注意力得分 → Softmax → 加权聚合 → 输出投影
- **规模**: 强制 ≤ 256×256（注意力矩阵 N² 显存开销）
- **参数**: `featureDim(d)`, `W_q/k/v/o`, `bias`

---

## 🖥️ 运行环境

### 硬件要求
- **显卡**: NVIDIA GPU（支持 CUDA 的计算能力 5.0+）
- **显存**: 建议 4GB+（4K 网格约需 1GB，8K 约需 4GB）
- **显示器**: 支持 OpenGL 4.3+

### 软件依赖
- **CUDA Toolkit**: 12.8+
- **CMake**: 3.18+
- **编译器**: MSVC 2022 (Windows) / GCC 11+ (Linux)
- **第三方库**:
  - GLFW 3.4
  - GLEW 2.1.0
  - SDL2 2.30.9
  - Dear ImGui (Docking 分支)
  - ImPlot

---

## 🛠️ 构建指南

### Windows (Visual Studio 2022)

```bash
# 1. 克隆仓库
git clone https://github.com/YourRepo/GlobalAttentionLifeGame.git
cd GlobalAttentionLifeGame/CMakePro_CUDA_LifeGame_V2

# 2. 修改 CMakeLists.txt 中的第三方库路径
# 编辑第 24-27 行，指向你的库安装位置

# 3. 创建构建目录
mkdir build && cd build

# 4. 生成项目
cmake .. -G "Visual Studio 17 2022" -A x64

# 5. 编译
cmake --build . --config Release

# 6. 运行
./Release/CMakePro_CUDA_LifeGame_V2.exe
```

### Linux

```bash
# 1. 安装依赖 (Ubuntu/Debian)
sudo apt-get install libglfw3-dev libglew-dev libsdl2-dev zenity

# 2. 克隆并构建
git clone https://github.com/YourRepo/GlobalAttentionLifeGame.git
cd GlobalAttentionLifeGame/CMakePro_CUDA_LifeGame_V2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. 运行
./CMakePro_CUDA_LifeGame_V2
```

---

## 🎮 操作指南

### 基础操作

| 操作 | 功能 |
|---|---|
| `鼠标中键拖拽` | 平移视角（像素级跟手） |
| `鼠标滚轮` | 缩放视图（向指针中心聚焦） |
| `R 键` | 重置视口（回归 1.0× 默认居中） |
| `左键拖拽` | 绘制存活细胞 |
| `右键拖拽` | 擦除存活细胞 |
| `空格键` | 暂停/继续演化 |
| `Tab 键` | 切换屏幕 |

### 模式切换与参数调节

1. **打开控制面板**（默认右侧悬浮窗）
2. **选择演化模式**: `SIMULATION_MODE` 下拉框
3. **调节参数**: 根据当前模式显示对应的滑动条
   - 全局统计量: 局部/全局权重、阈值
   - 分块注意力: 距离/状态/块间权重、衰减 σ、块大小
   - 精确注意力: 特征维度、偏置、重新初始化权重按钮
4. **规模限制**: 切换模式时自动检查并调整网格大小

### RLE 图案导入

1. 点击 `PATTERN_&_IMPORT` 展开面板
2. 点击 `BROWSE` 选择 `.rle` 文件
3. 设置导入位置 `(offsetX, offsetY)`
4. 对于超大图案（如 Caterpillar），使用裁剪加载
5. 点击 `LOAD` 导入

---

## 📊 性能参考

在 RTX 4060 (8GB VRAM) 上的实测数据：

| 模式 | 网格大小 | 显存占用 | 帧率 |
|---|---|---|---|
| Classic | 1920×1080 | ~50 MB | 60 FPS |
| Classic | 8192×8192 | ~800 MB | 60 FPS |
| Classic | 20480×20480 | ~4 GB | 60 FPS |
| GlobalStats | 8192×8192 | ~850 MB | 60 FPS |
| BlockAttention | 4096×4096 | ~1.2 GB | 60 FPS |
| FullAttention | 256×256 | ~150 MB | 60 FPS |

*注：帧率受 VSync 限制，实际计算耗时远低于 16.6ms*

---

## 🔬 技术架构

### 数据流

```
┌─────────────────────────────────────────────┐
│              ImGui UI 层                      │
│  [模式选择] [参数调节] [规模限制] [统计面板]    │
├─────────────────────────────────────────────┤
│           统一渲染管线 (OpenGL)                │
│  热力图纹理 → 片段着色器 → 屏幕               │
├─────────────────────────────────────────────┤
│           CUDA Kernel 调度层                  │
│  switch(mode): Classic/Global/Block/Full    │
├─────────────────────────────────────────────┤
│           统一显存布局 (float)                 │
│  d_current + d_next + d_heat + 模式专用缓冲   │
└─────────────────────────────────────────────┘
```

### 关键设计决策

1. **统一 float 双缓冲**: 所有模式共享 `float*` 缓冲区，避免模式切换时的数据转换开销
2. **模式专用缓冲区延迟分配**: 仅在切换到对应模式时分配显存，Classic 模式零额外开销
3. **Thrust 规约**: 全局统计量模式使用 `thrust::reduce` 硬件级并行
4. **分块计算**: FullAttention 模式采用 QUERY_BLOCK_SIZE 分块，控制显存峰值
5. **预设参数**: 所有可学习参数带有合理预设值，无需训练即可体验不同模式

---

## 📁 项目结构

```
CMakePro_CUDA_LifeGame_V2/
├── CMakeLists.txt                          # 构建配置
├── CUDA_Kernels/                           # CUDA 核函数
│   ├── Cuda_Check.cu/cuh                   # 原硬件检测 + 旧接口
│   ├── Cuda_Classic.cu/cuh                 # 经典模式 (float)
│   ├── Cuda_GlobalStats.cu/cuh             # 全局统计量注入
│   ├── Cuda_BlockAttention.cu/cuh          # 分块窗口注意力
│   └── Cuda_FullAttention.cu/cuh           # 精确全局注意力
├── Main_header/                            # 头文件
│   ├── AttentionParams.h                   # 可学习参数定义
│   ├── Common.h                            # SimMode 枚举 + GLHandles
│   └── CMakePro_cuda.h                     # UI 渲染 + 调度逻辑
├── Main_cpp/                               # 主程序
│   └── CMakePro_cuda.cpp                   # 入口 + 初始化
└── resources_LifeGame_V2/                  # 资源文件
    ├── shader/                             # GLSL 着色器
    ├── font/                               # 字体
    └── images/                             # 图标
```

---

## 🎯 推荐实验

### 1. 模式对比观察
- 在同一初始图案（如 Gosper Gun）下切换不同模式
- 观察全局注意力如何影响图案的传播速度和稳定性

### 2. 参数敏感性测试
- 在 GlobalStats 模式下调节 `w_global` 从 0 到 1
- 观察全局上下文对局部结构的抑制/增强作用

### 3. 大规模分块注意力
- 设置 4096×4096 网格，BlockAttention 模式
- 加载多个分散的图案，观察块间注意力如何促进远距离交互

### 4. 精确注意力的涌现行为
- 256×256 网格，FullAttention 模式
- 尝试不同的 `featureDim` 和随机权重初始化
- 观察是否会出现经典模式无法产生的涌现结构

---

## ⚠️ 已知限制

1. **FullAttention 规模限制**: 由于注意力矩阵 N² 的显存开销，该模式强制限制在 256×256。这是物理限制，非实现问题。

2. **BlockAttention 块大小**: 块大小必须是网格尺寸的约数，否则边缘块会有少量浪费。推荐块大小 32（CUDA warp 大小的整数倍）。

3. **首次模式切换延迟**: 切换到 Block/Full 模式时会有短暂卡顿（显存分配），这是正常现象。

4. **Windows Defender**: 未签名程序可能被拦截，请添加白名单。

---

## 📖 原项目文档

<details>
<summary>点击展开：CUDA 极速生命游戏 V1.2 原始 README</summary>

---

# 🚀 CUDA GPU-Accelerated Game of Life (CUDA 极速生命游戏)

欢迎来到基于 **CUDA + OpenGL 零拷贝互操作** 架构的高性能生命游戏仿真系统！

本程序通过 NVIDIA CUDA 技术，将原本由 CPU 密集计算的细胞自动机演化完全交由 GPU 并行处理。在经历了一系列深度的 GPU 显存级和指令级优化后，目前已成功实现在主流消费级显卡（如 RTX 4060）上流畅运行 **超 4 亿网格点（$20480 \times 20480$）** 的宇宙级模拟尺度。

---

## 💡 版本 1.2 重大技术更新说明 (Git 独家版)

在最新版本中，我们对整个仿真和交互系统进行了底层重构，主要实现了以下核心突破：

### 1. 极致显存瘦身：无状态 PCG 哈希随机数引擎
* **技术突破**：彻底抛弃了传统的、在显存中为每个像素常驻一个 $48$ 字节 `curandState` 的设计。我们在 GPU 寄存器内部实现了一套高随机度的**无状态 PCG 32 哈希随机数算法**。
* **显存缩减**：单像素显存开销从原本的 $58$ 字节**暴降至 $10$ 字节**（优化了 **5.8 倍**）。在 $8192 \times 8192$ 尺度下，**瞬间为您抠出 $3.22\text{ GB}$ 显存开销**。

### 2. 动态分辨率与"面积预算限制"约束
* **技术突破**：打破了过去死板的硬编码比例限制。系统升级为"**总面积预算（Max Area）限制 $\le 4.19$ 亿像素**"的弹性约束，并将单边硬件安全限制放宽至 **`32768 px`**。
* **空间置换**：现在，只要您将高度（HEIGHT）调小（如标准的 $1080$ px），宽度上限便会动态拔高到 `32768 px` 的硬件极限，让显存利用效率最大化。

### 3. GPU 显存实时监控进度条 (VRAM Monitor)
* **技术突破**：利用 CUDA 原生 API `cudaMemGetInfo` 建立了一个 0.5s 定时刷新数据源，配合 ImGui 渲染了实时的 **GPU 显存占用进度条**。
* **安全警示**：进度条会根据当前网格配置实时计算并可视化显存健康度。处于安全范围时显示青绿色，占用超过 $70\%$ 时显示橙色，超过 $85\%$ 时显示红色高亮警告，防止发生 OOM 崩溃。

### 4. 跨平台 RLE 导入与"极速裁剪部署"
* **技术突破**：集成了系统原生跨平台文件选择对话框（Windows/Linux 双模），可自由选择外部 `.rle` 格式的飞船或机枪。
* **纵向裁剪**：针对类似 *Caterpillar（履带者号）* 这种尺寸高达 $4195 \times 330721$ 的太空奇观，开发了 `ParseRleFileCropped` 极速裁剪读取器（在行数超限时会提前 break 退出），能够直接切下巨型飞船的局部身子（如 $4195 \times 32768$）并通过 `cudaMemcpy2D` 零拷贝载入 GPU。

### 5. 系统级视口方向映射体系 (一键 90° 旋转与镜像)
* **技术突破**：引入了 `rotate90`、`flipX`、`flipY` 三个控制标记，并在 **OpenGL 着色器端** 和 **CPU 端鼠标涂抹逻辑** 进行了同步映射变换。
* **完美对齐**：当您加载 Caterpillar 等垂直细长条飞船时，可以勾选 `ROTATE_90_DEG` **将垂直画面平放到水平视口中运行**。此时：
  * 顶点着色器会自动交换尺寸并重构正确的宽高比，**彻底消除画面挤压与压扁现象**。
  * 鼠标中键拖拽平移的速度、以及画笔涂抹像素的物理位置依然保持 **100% 贴合跟手**，无任何漂移或粘滞。

### 6. 5 大荧光主题预设 (Theme Presets)
* 集成了 5 套经过精心色彩调试的荧光主题一键切换：`CLASSIC`（经典极客绿）、`CYBER`（赛博霓虹）、`MAGMA`（熔岩火红）、`OCEAN`（冰海深蓝）以及 `MATRIX`（黑客帝国）。

---

## 💻 运行环境
*   **显卡需求**：必须使用支持 CUDA 的 **NVIDIA 显卡**。
*   **驱动要求**：请确保显卡驱动已更新至较新版本（以便支持最新的 CUDA Runtime）。
*   **系统**：Windows 10 / 11 (64-bit)，或配置了 Zenity 组件的 Linux 桌面系统。

---

#### 🕹️ 核心操作快捷键
*   **[鼠标中键拖拽]**：平移视角。无论视口分辨率有多么极端（如 $32768 \times 1080$），拖拽均能保证 **100% 像素跟手**。
*   **[鼠标滚轮]**：缩放视图（向指针中心聚焦）。
*   **[R 键]**：**一键重置视口缩放与平移（回归 $1.0\times$ 默认居中视角）**。
*   **[左键点击/拖拽]**：绘制/注入存活细胞。
*   **[右键点击/拖拽]**：擦除存活细胞。
*   **[控制面板 PAUSE / RESUME]**：暂停或继续物理演化（可在常驻底部的全局状态栏中一键控制）。

---

## 🌟 推荐测试的巨型 RLE 图案

为了体验 GPU 在并行计算上的霸权级处理优势，强烈建议您在 **`PATTERN_&_IMPORT`** 中载入以下经典巨型图案：

1.  **OTCA Metapixel（元像素套娃）**：
    *   一个由 $2048 \times 2048$ 细胞组成的宏观像素。您可以在 $16384 \times 16384$ 网格下铺设一个 **$8 \times 8$ 的元像素矩阵**。当拉近时，是数亿个并行演化的闪烁矩阵；当拉远时，这些元像素组合成了一个全新的、宏观层面的二阶生命游戏！
2.  **Spacefiller（空间填充器）**：
    *   以 $O(t^2)$ 级数无限疯狂分裂的结构。在 CPU 上跑几百代就会因为细胞数量爆表而卡死；而在您的 GPU 上，无论屏幕上有 1 个细胞还是 4 亿个细胞，计算复杂度恒定，**帧率将自始至终稳定满帧**。
3.  **Caterpillar（履带者号）**：
    *   生命游戏最著名的巨型工程飞船。勾选 `USE_CROP_LOADING` 开启裁剪，将坐标设置为 `(cropX: 0, cropY: 0, cropW: 4195, cropH: 32768)`。加载后，勾选 `ROTATE_90_DEG` 和 `FLIP_VERTICAL`，您将看到这艘宏伟的宇宙战舰以完美的比例横在屏幕中，并极其流畅地自左向右飞行。

---

## ⚠️ 运行注意事项
1.  **杀毒软件误报**：由于程序采用底层原生 C++/CUDA 编译且未经过昂贵的商业数字签名，Windows Defender 可能会拦截。请点击"仍要运行"或添加白名单。
2.  **缺少 DLL 文件**：如果提示缺少 `cudart64_xx.dll`，请确保您的电脑已经正确安装了 NVIDIA 显卡驱动，或将 CUDA 运行库放置在 exe 同级目录下。

---

## 🛠️ 项目技术栈
*   **并行计算**：CUDA Kernel & Thrust (Thrust::Reduce 硬件级并行人口统计)
*   **图形互操作**：CUDA-OpenGL Interop (摒弃 CPU 作为中介，显存数据直接零拷贝映射至纹理)
*   **图形渲染**：OpenGL 4.3 Core Profile + 自适应抗锯齿网格 Shader (搭载 CRT 扫描线与暗角后期特效)
*   **用户界面**：Dear ImGui + ImPlot (实现高度自定义的磨砂玻璃悬浮窗与实时人口曲线)

</details>

---

## 🙏 致谢

- 原项目 [CMakePro_LifeGame](https://github.com/XiaoYu-1111/CMakePro_LifeGame) 提供了优秀的 CUDA-OpenGL 互操作基础
- [Dear ImGui](https://github.com/ocornut/imgui) 提供了极致的即时模式 GUI 体验
- [Conway's Game of Life](https://conwaylife.com/) 社区提供了丰富的 RLE 图案资源

---

## 📜 许可证

本项目基于原项目的开源协议发布。

---

**项目地址**: [Your GitHub Repo Link]  
**问题反馈**: 欢迎在 Issues 中提交 bug 报告和功能建议！
