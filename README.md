<p align="center">
  <a href="assets/banner.png"><img src="assets/banner.svg" alt="LoRA Optimizer" width="100%"></a>
</p>

<p align="center">
  <b>简体中文</b> · <a href="README_EN.md">English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Custom_Nodes-2563eb?style=flat-square" alt="ComfyUI Custom Nodes">
  <img src="https://img.shields.io/badge/Aaalice-Adaptive_VRAM-e94560?style=flat-square" alt="Aaalice Adaptive VRAM">
  <img src="https://img.shields.io/badge/TIES_%7C_DARE_%7C_DELLA-Merging-8b5cf6?style=flat-square" alt="Merge Algorithms">
  <img src="https://img.shields.io/badge/AutoTuner-Parameter_Sweep-f59e0b?style=flat-square" alt="AutoTuner">
  <img src="https://img.shields.io/badge/License-GPL--3.0-22c55e?style=flat-square" alt="GPL-3.0">
</p>

# 🧩 ComfyUI-LoRA-Optimizer-Aaalice

面向 ComfyUI 的多 LoRA 分析、冲突处理与合并节点包。

它不会把多个 LoRA 直接相加后交给模型，而是先解析它们实际命中的模型权重，逐层分析方向、强度、符号冲突和子空间重叠，再为不同层选择合适的合并方式。普通用户可以直接使用默认设置；需要精细控制时，也可以接入 Settings、AutoTuner、Estimator、Conflict Editor 或 Merge Formula。

> 本仓库是 [ethanfel/ComfyUI-LoRA-Optimizer](https://github.com/ethanfel/ComfyUI-LoRA-Optimizer) 的 Aaalice 分支。保留上游完整功能，并额外维护低显存设备上的自适应内存调度。

<p align="center"><img src="assets/comparison.png" alt="普通 LoRA 堆叠与优化合并对比" width="100%"></p>

## ✨ 与上游的区别

当前分支与 `upstream/main` 保持同步，核心功能、节点 ID、工作流格式和合并策略均兼容上游；Aaalice 分支额外提供以下生产级改进：

| 项目 | 上游当前行为 | Aaalice 分支 |
|---|---|---|
| 大层执行 | CUDA 可用时主要按完整目标层计算 | 每个 target 自动选择 **full GPU → tiled GPU → CPU**；大层按输出行在 GPU 分块，不再同时常驻多份完整 dense diff |
| 显存规划 | 主要依赖用户手动调整内存选项 | 每层按实时空闲显存、因子、参与 LoRA 数量和策略工作区重算计划，并保留固定安全余量 |
| 算法覆盖 | 完整 tensor 路径 | weighted、SLERP、TIES、DARE、DELLA、consensus、STAR/TAME、refinement 与压缩均保留原数学语义，不因低显存静默换算法 |
| 原生取消 | 长循环可能无法及时响应队列取消 | Pass 1、Pass 2、AutoTuner、CPU worker、缓存等待、SVD 和保存前均响应 ComfyUI 原生取消；取消不应用半成品 patch |
| 多语言 | 节点显示以英文为主 | 所有节点名称、输入、输出、选项和说明跟随 ComfyUI 语言在简体中文与英文之间自动切换 |
| 8GB 显存体验 | Krea、Flux、Wan 等多 LoRA 组合可能在大层 OOM | 小层保持完整 GPU 快速路径，大层走有界 GPU tile；只有无 GPU、显式 CPU 或最小 tile 仍不可用时才回退 CPU |

分块路径限制的是临时 GPU 工作集，不会降低策略等级。不同归约顺序可能产生极小浮点差异，但算法、目标 rank、输出结构和质量门槛保持不变。

## 🎯 适合解决什么问题

- 多个风格、角色、细节 LoRA 同时使用时出现过曝、偏色、糊脸或特征互相覆盖
- 不同训练器导出的 LoRA key 命名不一致，普通堆叠无法正确识别重叠层
- 想按层选择 TIES、共识、球面插值或加权合并，而不是全模型固定一种算法
- 想把最终结果保存为标准 `.safetensors`，或转换成 conditioning hook
- 已有一长串普通 `Load LoRA` 节点，不想重新搭建 LoRA Stack
- 想让 AutoTuner 搜索参数，或用社区结果和 k-NN Estimator 快速预测配置
- 大模型或多 LoRA 合并时，想限制 RAM/VRAM 峰值并避免显存崩溃

不建议把它当作所有 LoRA 的统一加载器。蒸馏、DPO、Lightning、LCM、Turbo、Hyper 和编辑模型 LoRA 通常依赖精确权重，优先使用普通 `Load LoRA` 以加法方式加载；确需合并时使用 `additive`，并关闭稀疏化。

## 🚀 三步快速开始

1. 添加 **LoRA Stack (Dynamic)**，选择要合并的 LoRA 并设置各自强度。
2. 添加 **LoRA Optimizer**，连接基础模型、`lora_stack`，图像模型可选连接 `CLIP`。
3. 将优化器输出的 `MODEL` / `CLIP` 接到采样流程。

```text
LoRA Stack (Dynamic) ───────────────► LoRA Optimizer ───► KSampler
                                           ▲
Load Checkpoint ───► MODEL / CLIP ─────────┘
```

第一次使用时**不要连接 Settings 节点**。`LoRA Optimizer` 的内置默认值已经启用：

- `auto_strength = enabled`
- `optimization_mode = per_prefix`
- `patch_compression = smart`
- `svd_device = gpu`
- `normalize_keys = enabled`
- `strategy_set = full`
- `sparsification = disabled`

将 `analysis_report` 接到任意 Show Text 节点，可以查看每个 LoRA 的统计、冲突关系、逐层策略和最终合并摘要。

### 🌐 节点语言

节点名称、输入、输出、枚举选项、说明和自定义右键菜单会跟随 ComfyUI 的 `Locale` 设置自动切换。目前完整提供简体中文与英文；切换语言后按 ComfyUI 的提示刷新页面即可，无需修改工作流，底层 socket 名和序列化值保持不变。

## 📦 安装、替换与更新

### 🔌 使用 ComfyUI Manager

在 ComfyUI Manager 中选择 **Install via Git URL**，输入：

```text
https://github.com/Aaalice233/ComfyUI-LoRA-Optimizer-Aaalice.git
```

安装完成后重启 ComfyUI。

### 🛠️ 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Aaalice233/ComfyUI-LoRA-Optimizer-Aaalice.git
```

本项目依赖 `scikit-learn>=1.3`。如果当前 ComfyUI 环境没有自动安装依赖，请使用运行 ComfyUI 的同一个 Python 环境执行：

```bash
python -m pip install "scikit-learn>=1.3"
```

然后重启 ComfyUI。节点位于 `LoRA Optimizer` 分类。

### ♻️ 从上游版切换

上游版与 Aaalice 版注册相同的节点 ID，**不能同时安装或同时启用**。切换前关闭 ComfyUI，移走或禁用原来的 `ComfyUI-LoRA-Optimizer` 文件夹，再安装本仓库。

### ⬆️ 更新

```bash
cd ComfyUI/custom_nodes/ComfyUI-LoRA-Optimizer-Aaalice
git pull --ff-only
```

更新代码后必须重启 ComfyUI。

## 🧭 应该选择哪条工作流

| 你的情况 | 推荐入口 |
|---|---|
| 新工作流，希望简单稳定 | `LoRA Stack (Dynamic)` → `LoRA Optimizer` |
| 已有普通 Load LoRA 链 | `LoRA Optimizer (Inline Chain)` |
| 不知道参数怎么选，愿意等待搜索 | `LoRA AutoTuner` 或 `LoRA AutoTuner Settings` → `LoRA Optimizer` |
| 想快速复用相似组合经验 | `LoRA Merge Estimator` → `TUNER_DATA` → `LoRA Optimizer` |
| 已知 LoRA 冲突，需要人工指定 | `LoRA Conflict Editor` → `LoRA Optimizer Settings` |
| 需要先合并 1+2，再与 3 合并 | `LoRA Merge Formula` |
| 使用 kijai WanVideoWrapper | 标记为 `(WIP)` 的 WanVideo 节点，谨慎测试 |

### 🔗 普通 Stack 工作流

```text
Load Checkpoint ──► LoRA Optimizer ──► Sampler
                         ▲
LoRA Stack (Dynamic) ────┘
```

### 🔗 Inline Chain 工作流

```text
Load Checkpoint ─► Load LoRA #1 ─► Load LoRA #2 ─► LoRA Optimizer (Inline Chain) ─► Sampler
                                                          ▲
                                  LoRA Inline Chain Options（可选）
```

Inline 节点会读取并移除上游加载器留下的 LoRA patch，使用同一套优化器重新合并，再应用到模型。Stock `Load LoRA`、`Load LoRA (Model Only)` 和 rgthree Power Lora Loader 可以恢复真实文件名；没有写入来源标记的第三方加载器会显示为 `chain lora #N`。

### 🔗 AutoTuner 工作流

```text
LoRA Merge Settings（可选）
          │
LoRA AutoTuner Settings ─► LoRA Optimizer.settings
LoRA Stack ──────────────► LoRA Optimizer.lora_stack
```

`LoRA Optimizer` 会根据接入的 Settings 类型自动进入高级模式或 AutoTuner 模式，不必更换主节点。

### 🔗 Estimator 工作流

```text
LoRA Stack ─► LoRA Merge Estimator ─► TUNER_DATA ─► LoRA Optimizer
                   ▲                                  ▲
              MODEL / CLIP                       MODEL / CLIP
```

Estimator 第一次运行会显式下载社区缓存并在 `ComfyUI/models/estimator/` 构建本地 k-NN 索引；后续直接复用。没有相近样本时，以 AutoTuner 的完整搜索为准。

## 🧱 节点总览

### 🧰 构建与合并

| 节点 | 用途 |
|---|---|
| **LoRA Stack** | 固定槽位 LoRA 列表 |
| **LoRA Stack (Dynamic)** | 动态增减 LoRA，推荐新工作流使用 |
| **LoRA Optimizer** | 推荐主节点，默认即可运行，也接受 Settings 或 `TUNER_DATA` |
| **LoRA Optimizer (Legacy)** | 所有参数集中在一个节点，主要用于旧工作流兼容 |
| **LoRA Optimizer (Inline Chain)** | 优化普通 Load LoRA 链留下的 patch |
| **LoRA Inline Chain Options** | 为 Inline 链逐个设置启用、强度、冲突、过滤和保留选项 |
| **LoRA Merge Formula** | 定义 `(1+2)+3` 一类的分层合并顺序 |
| **LoRA Conflict Editor** | 查看冲突并手动指定冲突模式或全局策略 |

### ⚙️ 设置与自动搜索

| 节点 | 用途 |
|---|---|
| **LoRA Merge Settings** | Optimizer 与 AutoTuner 共用的架构、缓存、平滑、强度下限和显存设置 |
| **LoRA Optimizer Settings** | 高级合并策略、稀疏化、压缩和 SVD 设置 |
| **LoRA AutoTuner Settings** | 参数搜索、评分、缓存、持久记忆和社区缓存设置 |
| **LoRA AutoTuner** | 独立 AutoTuner 节点，搜索并输出排名结果 |
| **LoRA Merge Estimator** | 根据社区数据用 k-NN 预测候选配置 |
| **Merge Selector** | 从 `TUNER_DATA` 中选择第 N 名配置 |
| **Save / Load Tuner Data** | 保存或恢复 `.tuner` / `.json` 搜索结果 |
| **Build AutoTuner Python Evaluator** | 接入自定义 Python 质量评估器 |

### 🔍 分析、转换与输出

| 节点 | 用途 |
|---|---|
| **LoRA Compatibility Analyzer** | 合并前查看 LoRA 兼容性 |
| **LoRA Metadata Reader** | 读取 `.safetensors` 中的描述、提示词和合并元数据 |
| **LoRA Extract from Model** | 对基础模型与微调模型做差并提取 LoRA |
| **LoRA Combination Generator** | 生成 2/3 路组合，用于 AutoTuner 数据采集 |
| **Save Merged LoRA** | 保存为标准 `.safetensors` LoRA |
| **Merged LoRA to Hook** | 转为 ComfyUI conditioning hook |
| **(WIP) WanVideo LoRA Optimizer** | kijai WanVideoWrapper 专用实验节点 |
| **(WIP) Merged LoRA → WanVideo** | 将合并结果转换到 WanVideoWrapper 路径 |

## ⚙️ 推荐设置

### ✅ 通用推荐

| 设置 | 推荐值 | 原因 |
|---|---|---|
| `auto_strength` | `enabled` | 防止多个 LoRA 叠加后整体能量过高 |
| `optimization_mode` | `per_prefix` | 每个目标层独立选择策略 |
| `merge_refinement` | `none` | 默认成本最低；发现干扰再尝试 `refine` |
| `sparsification` | `disabled` | 不应无目的丢弃权重 |
| `patch_compression` | `smart` | 在线性合并路径保持无损并显著降低 patch 内存 |
| `svd_device` | `gpu` | 普通层与可分块大层优先使用 GPU；显式选择 `cpu` 时才固定使用 CPU SVD |
| `free_vram_between_passes` | `enabled`（8GB 建议） | Pass 1 与 Pass 2 之间释放缓存 |
| `strategy_set` | `full` | 允许优化器使用完整策略集合 |
| `vram_budget` | `0` 起步 | 合并结果先放系统 RAM，最稳妥 |
| `cache_patches` | 图像模型 `enabled`；大型视频模型 `disabled` | 在重复执行速度和系统 RAM 之间取舍 |

### 🧯 8GB 显存

不要把 `global + aggressive + basic + CPU SVD` 当作防 OOM 的固定方案。这样会限制策略并可能损失质量。建议仍使用：

```text
auto_strength          enabled
optimization_mode      per_prefix
merge_refinement       none
sparsification         disabled
patch_compression      smart
svd_device             gpu
free_vram_between_passes enabled
strategy_set           full
vram_budget            0
```

Aaalice 分支会自动识别放不下的完整目标层并转入 GPU 分块。日志首次出现 tiled GPU 代表保护逻辑正在工作；只有能力限制下才会明确记录 CPU fallback 原因。无需手动设置 tile 大小。

### 🎨 输出过曝或“炸色”

1. 保持 `auto_strength = enabled`。
2. 将 `output_strength` 设为 `-1` 让优化器自动建议总强度，或手动降低到 `0.6–0.9`。
3. 尝试 `merge_refinement = refine`。
4. 对少数异常热层设置 `tame_layers = 0.5`，以 `tame_threshold = 0.3` 起步。
5. 只有明确存在冲突时，再尝试 `della_conflict` 或 `dare_conflict`。

### 🧪 多个强冲突 LoRA

- `strategy_set = full`
- `merge_refinement = refine`；仍有问题再尝试 `full`
- `sparsification = della_conflict`
- `sparsification_density = 0.7`
- `star_eta = 60–80` 仅适合数量多且冲突明显的组合；2–3 个相互独立的 LoRA 保持 `100`（关闭）

## 🧠 工作原理

<p align="center"><a href="assets/optimizer-pipeline.png"><img src="assets/optimizer-pipeline.svg" alt="LoRA Optimizer Pipeline" width="100%"></a></p>

### 🔬 Pass 1：分析

1. 识别 Standard LoRA、LoCon 及兼容的 trainer 变体。
2. 将 Kohya、AI-Toolkit、LyCORIS、diffusers/PEFT 等命名归一化到模型实际 key。
3. 将同一 LoRA 内映射到同一目标权重的 alias 聚合。
4. 计算逐 LoRA 范数、两两 cosine、符号冲突、幅度分布和低秩子空间重叠。
5. 只保留轻量统计，释放当前目标层的 dense diff。

### 🧬 Pass 2：合并

1. 按目标层重新构造必要 diff。
2. 根据该层统计选择策略。
3. 应用可选的 DARE/DELLA、STAR、方向正交化、TALL-mask 或 KnOTS。
4. 在线性路径尽量保留精确低秩表示；需要时做 SVD patch 压缩。
5. 应用到 `MODEL` / `CLIP`，同时输出报告与可保存的 `LORA_DATA`。

### 🧠 逐层策略

| 条件 | 常用策略 |
|---|---|
| 只有一个 LoRA 命中 | `weighted_sum`，保留完整贡献 |
| 多个 LoRA 基本独立 | `weighted_average` |
| 方向一致、冲突较低 | `consensus` |
| 冲突明显且子空间重叠 | `ties` |
| 需要方向平滑插值 | `slerp` |
| 结构、蒸馏或编辑 LoRA | 手动使用 `additive` 更安全 |

<p align="center"><a href="assets/merge-strategies.png"><img src="assets/merge-strategies.svg" alt="LoRA Merge Strategies" width="100%"></a></p>

### 💪 Auto-Strength

Auto-Strength 根据每个分支的精确范数和两两内积估算组合能量，只做统一向下缩放，不改变原始强度比例、不翻转正负号，也不会主动放大。相同方向的 LoRA 会缩得更多；相互独立的 LoRA 缩放较温和；方向相反的 LoRA 因为本身抵消，通常不需要大幅降低。

### 🔑 Key 归一化

支持对 SD/SDXL、Flux、Wan、Z-Image、LTX、ACE-Step、Ideogram 4、Anima、Qwen-Image 等常见结构做架构感知映射。Z-Image 的 fused QKV 会在分析时拆分，合并后再恢复到 ComfyUI 原生布局。

### 🗜️ Patch 压缩

| 模式 | 行为 | 质量 |
|---|---|---|
| `smart` | 只压缩可保持线性合并信息的路径 | 推荐，线性路径无损 |
| `aggressive` | 包括非线性结果在内都压缩 | 更省内存，但可能损失细节 |
| `disabled` | 保留 dense patch | 无压缩损失，但 RAM 占用最高 |

## 💾 内存与显存说明

Aaalice 分支按**每个 resolved target group、每个执行阶段**自动规划：

1. 小层完整峰值低于安全预算的 80%：走 `full_gpu` 快速路径。
2. 完整层放不下但来源可按行重建：走 `tiled_gpu`，Pass 1 统计、合并、随机 SVD 和压缩都只保留当前 tile。
3. 无可用 GPU、用户显式选择 CPU、未知第三方 payload，或连最小 tile 都无法容纳：走 `cpu`，并记录明确原因。
4. 每个 tile 前检查 ComfyUI 原生取消；取消后丢弃当前 target，不写最终缓存、不应用任何 patch。
5. 完成 target 后释放因子 staging 与临时 tensor，再按实时空闲显存规划下一层。

默认安全规则：预留 `max(512 MiB, 总显存的 10%)`；单份 dense tile 不超过 128 MiB；总 tile 工作区不超过 512 MiB。`LORA_OPTIMIZER_TILE_MB=16..512` 仅供开发和诊断覆盖，普通用户无需设置。

可复用基准工具：

```bash
python tools/benchmark_chunked_merge.py --mode tiled_gpu --rows 8192 --cols 8192 --rank 32 --loras 9
python tools/benchmark_chunked_merge.py --mode tiled_gpu --real-lora-dir <LoRA目录> --loras 9
```

它会输出分析/合并耗时、tile 数、GPU 峰值、CPU RSS、数值误差与取消延迟；第二条命令直接使用真实 LoRA 因子。

几个容易混淆的设置：

| 设置 | 实际控制内容 | 不控制什么 |
|---|---|---|
| `vram_budget` | 合并完成后，有多少 patch 可常驻 GPU | 不会强行突破实时空闲显存与安全余量 |
| `svd_device` | 首选 GPU 或显式 CPU 压缩 | 不需要用户手工为大层选择 tile |
| `free_vram_between_passes` | Pass 1 和 Pass 2 之间清理 GPU cache | 单层峰值由 tiled planner 独立控制 |
| `cache_patches` | 是否缓存最终 patch 以便快速重跑 | 不缓存原始 LoRA 文件本身 |
| `diff_cache_mode` | AutoTuner 是否复用候选配置间的 diff | 只影响 AutoTuner，不是普通优化器显存上限 |

未压缩的 nonlinear 结果仍需要一份最终 CPU patch，这是输出本身的必要内存。若出现能力型 CPU fallback 后系统内存压力很大：

- 关闭不必要程序并确保系统分页文件可用。
- 大型视频模型将 `cache_patches` 设为 `disabled`。
- AutoTuner 使用 `diff_cache_mode = auto` 或 `disabled`；`disk` 仅在临时盘空间充足时使用。
- 减少一次参与合并的 LoRA 数量，先保存中间结果，再进行下一阶段。

## 🧪 AutoTuner、缓存与 Estimator

### 🏁 AutoTuner

AutoTuner 复用一次 Pass 1 分析，评分大量参数组合，再真实合并排名靠前的候选。常用默认值：

- `top_n = 3`
- `scoring_speed = turbo`
- `scoring_svd = disabled`
- `scoring_device = gpu`
- `scoring_formula = v2`
- `diff_cache_mode = auto`
- `memory_mode = auto`

`TUNER_DATA` 可接入 **Merge Selector**、**Save Tuner Data** 或普通 **LoRA Optimizer**。

### 💽 Diff Cache

| 模式 | 行为 |
|---|---|
| `disabled` | 每个候选重新计算，内存最低 |
| `auto` | 在 `diff_cache_ram_pct` 范围内使用 RAM，超出部分按需重算 |
| `ram` | 尽量全部留在 RAM，最快但占用最高 |
| `disk` | 使用临时文件和 memory map，节省 RAM 但需要大量磁盘空间 |

### 🌐 Community Cache

默认 `community_cache = disabled`，不会进行社区缓存交互。只有用户显式选择 `upload_only` 或 `upload_and_download` 时才会访问 Hugging Face；上传需要具有写权限的 `HF_TOKEN`。身份基于 LoRA 文件内容 SHA256，不依赖文件名或目录。

公共数据集：[`ethanfel/lora-optimizer-community-cache`](https://huggingface.co/datasets/ethanfel/lora-optimizer-community-cache)

## 🧬 兼容性

### ✅ 模型家族

- SD 1.5 / SDXL
- Flux
- Z-Image / Lumina2
- Wan 2.1 / 2.2
- LTX Video
- ACE-Step
- Ideogram 4
- Anima / Cosmos-Predict2 DiT
- Qwen-Image
- 其他由 ComfyUI 支持、且 LoRA 可归一化到标准目标权重的架构

### ✅ LoRA 与训练器

- Standard LoRA
- LoCon 与可还原为 up/down(/mid) adapter 的兼容变体
- Kohya
- AI-Toolkit
- LyCORIS
- Musubi Tuner
- diffusers / PEFT 常见命名
- Efficiency Nodes、Comfyroll 等输出的标准 tuple stack

不同架构和第三方格式变化很快。遇到未知 key 或跳过项时，以 `analysis_report` 与 ComfyUI 日志为准，不要假设所有 LyCORIS 变体都能无损转换。

## 🎛️ 每 LoRA 控制

Dynamic Stack 与 Inline Options 支持按 LoRA 设置：

- 模型与 CLIP 独立强度
- `conflict_mode`
- `preserve`
- `key_filter`

`key_filter` 常用值：

| 值 | 行为 |
|---|---|
| `all` | 使用全部目标层 |
| `shared_only` | 只保留两个及以上 LoRA 共同命中的层 |
| `unique_only` | 只保留该 LoRA 独有的层 |
| `audio_only` | 只保留音频相关层，适用于 LTX-2 / ACE-Step 等 |
| `no_audio` | 排除音频层，只保留视频或其他部分 |

这可用于拆分 T2V/I2V/VACE 特有层，或在音视频 LoRA 中选择声音与画面来源。

## 📤 保存与复用

**Save Merged LoRA** 将 `LORA_DATA` 保存为标准 `.safetensors`：

- `save_rank = 0`：使用合并结果已有 rank。
- `bake_strength = enabled`：保存后的 LoRA 在强度 `1.0` 时复现当前合并效果。
- 支持保存到配置过的 LoRA 目录及其子目录。

**Merged LoRA to Hook** 将结果转换为 ComfyUI `HOOKS`，适合按 conditioning、采样阶段或区域应用，而不全局修改模型。

**LoRA Extract from Model** 通过基础模型与微调模型做差后 SVD 分解生成 LoRA。基础模型和微调模型必须严格对应，否则提取结果没有意义。

## ⚠️ 使用边界

- 不要默认把 Lightning、LCM、Turbo、Hyper、DPO、蒸馏或编辑 LoRA 与风格 LoRA一起做 TIES/稀疏化。
- 普通 Optimizer 只分析自己 `lora_stack` 内的 LoRA；上游已经加载的 patch 不会被它自动纳入。需要捕获普通加载链时使用 Inline Chain。
- Fully baked checkpoint 中的变化无法与基础权重自动区分；使用 Extract from Model 时必须提供对应基础模型。
- Inline Chain 对没有来源标记的第三方 loader 只能使用捕获权重身份；报告会明确显示回退名称。
- `(WIP)` WanVideo 节点仍属于实验路径，生产工作流请先保存副本并验证输出。

## 🧯 常见问题

<details>
<summary><b>🔥 日志显示在 Pass 1 爆显存，调 vram_budget 或 svd_device 没用</b></summary>

`vram_budget` 管最终 patch，`svd_device` 管压缩首选设备，都不等于临时 tile 上限。Aaalice 分支会自动把危险层切换到 GPU 分块；确认已安装最新版并重启 ComfyUI。若仍回退 CPU，日志会给出无 GPU、显式 CPU、未知 payload 或最小 tile 不可用等具体原因。

</details>

<details>
<summary><b>🐢 出现 CPU 回退后速度变慢</b></summary>

正常的大层应优先进入 tiled GPU，而不是整层 CPU。CPU 仅用于能力型回退。检查是否把 `svd_device` 显式设为 `cpu`、是否存在未知第三方 payload，以及日志中的 fallback 原因；不要仅为防 OOM 主动降低策略。

</details>

<details>
<summary><b>🎭 合并后角色或风格变弱</b></summary>

先检查 `auto_strength` 和 `output_strength`。尝试 `output_strength = -1`，或逐步提高到 `1.0–1.2`。保持 `smart` 而不是 `aggressive` 压缩，并确认没有误启用标准 DARE/DELLA。

</details>

<details>
<summary><b>🧱 有些层没有参与合并</b></summary>

检查 `analysis_report` 中的 unknown key、shape mismatch、key filter 和架构识别结果。混合训练器 LoRA 时保持 `normalize_keys = enabled`；HunyuanVideo 等自动检测覆盖不足的架构可手动选 `architecture_preset = dit`。

</details>

<details>
<summary><b>🔁 安装后出现重复节点或导入异常</b></summary>

检查 `custom_nodes` 下是否同时存在上游版和 Aaalice 版。两者节点 ID 相同，只保留一个，然后重启 ComfyUI。

</details>

<details>
<summary><b>⛔ 误触后如何停止长时间合并</b></summary>

直接使用 ComfyUI 原生取消按钮。Optimizer、Inline、Merge Formula、AutoTuner、CPU worker、缓存等待和分块 SVD 共用同一个取消信号；常规 tiled 阶段通常在一个 tile 内停止。取消不会应用部分 patch，也不会把半成品写入最终缓存或保存文件。

</details>

## 📚 文档与示例

- [节点完整参考](docs/wiki/Nodes.md)
- [配置指南](docs/wiki/Configuration-Guide.md)
- [工作流说明](docs/wiki/Workflows.md)
- [算法说明](docs/wiki/Merge-Algorithms.md)
- [原理与实现](docs/wiki/How-It-Works.md)
- [故障排查](docs/wiki/Tips-and-Troubleshooting.md)
- [技术报告](docs/technical-report.md)
- [示例工作流目录](example_workflows/)

仓库内 Wiki 文档主要继承自上游，部分页面仍为英文；本文件是 Aaalice 分支的默认中文入口。

## 🤝 上游、研究与致谢

- 上游项目：[ethanfel/ComfyUI-LoRA-Optimizer](https://github.com/ethanfel/ComfyUI-LoRA-Optimizer)
- Aaalice 分支：[Aaalice233/ComfyUI-LoRA-Optimizer-Aaalice](https://github.com/Aaalice233/ComfyUI-LoRA-Optimizer-Aaalice)
- 原始基础：[ComfyUI-ZImage-LoRA-Merger](https://github.com/DanrisiUA/ComfyUI-ZImage-LoRA-Merger)
- TIES-Merging：[Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)
- DARE：[Yu et al., ICML 2024](https://arxiv.org/abs/2311.03099)
- DELLA：[Deep et al., 2024](https://arxiv.org/abs/2406.11617)
- KnOTS：[Ramé et al., 2024](https://arxiv.org/abs/2407.09095)
- TALL-masks：[Wang et al., 2024](https://arxiv.org/abs/2406.12832)
- STAR：[Spectral Truncation And Rescale, NAACL 2025](https://arxiv.org/abs/2502.10339)

完整来源与算法说明见上游文档和本仓库技术报告。感谢所有上游贡献者、测试者和社区数据贡献者。

## 📄 许可证

本项目使用 [GPL-3.0](LICENSE) 许可证。分发修改版本时请遵守相同许可证要求，并保留上游版权与来源说明。
