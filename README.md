# ComfyUI LoRA Optimizer Aaalice

[English](README_EN.md)

一个面向多 LoRA 合并的 GPU 优先优化器：由 **LoRA Manager** 提供 LoRA 堆栈，本项目负责分析、优化、缓存、应用和保存最终结果。

## ✨ 核心特点

- 对每个目标层分析重叠、方向、冲突、范数、秩和子空间关系。
- 按层自动选择 Weighted、SLERP、TIES、Consensus 等合并策略。
- 自动执行 `full GPU → tiled GPU → CPU` 三级调度，大层不会因完整 dense diff 撑爆显存。
- 支持 LoRA、LoCon、LoHa、LoKr、融合 QKV、MODEL 和 CLIP 权重。
- 支持 ComfyUI 原生取消和分块进度，不会应用被中断的半成品补丁。
- 支持内存缓存及可选的跨重启本地持久缓存。
- 不下载模型、不上传数据、不发送遥测。

## 🧩 节点

项目只保留 3 个正式节点：

### LoRA Optimizer

接收 LoRA Manager 或其他兼容节点输出的 `LORA_STACK`，完成分析、合并、缓存并把结果应用到 MODEL/CLIP。

输出：

- `MODEL`：已应用合并补丁的模型。
- `CLIP`：已应用文本编码器补丁的 CLIP；未连接输入时为空。
- `analysis_report`：策略、冲突、强度、显存调度和缓存信息。
- `LORA_DATA`：供 Save Merged LoRA 使用的完整补丁数据。

### LoRA Optimizer Settings

集中配置合并策略、自动强度、稀疏化、压缩、显存、STAR/TAME、内存缓存和持久缓存；不连接时使用推荐默认值。

### Save Merged LoRA

将 `LORA_DATA` 保存为标准 `.safetensors` LoRA，可选择自动秩或指定 SVD 压缩秩，并可烘焙当前输出强度。

## 🔗 推荐工作流

```text
LoRA Manager ── LORA_STACK ──> LoRA Optimizer ──> MODEL / CLIP
                                      │
LoRA Optimizer Settings ──────────────┘
                                      │
                                      └── LORA_DATA ──> Save Merged LoRA
```

本项目不再提供 LoRA Stack 节点，避免和 LoRA Manager 重复维护同一份 LoRA 列表。

## 💾 跨重启持久缓存

在 **LoRA Optimizer Settings** 中设置：

- `cache_patches`：当前 ComfyUI 进程内的内存缓存，重启后失效。
- `persistent_cache`：本地磁盘缓存，重启 ComfyUI 后仍可复用。

持久缓存默认启用。不希望在本地保存缓存文件时，将 `persistent_cache` 设为 `disabled`；此后不会读取或写入持久缓存，已有文件仍需按下文方式手动清理。

`persistent_cache=enabled` 时，第一次完整合并结束后会把最终 MODEL/CLIP patch 原子写入：

```text
<ComfyUI user directory>/lora_optimizer_cache/
```

下次输入 LoRA、强度、模型结构和全部数学设置一致时，Optimizer 会直接读取并应用最终 patch，跳过 LoRA 文件加载、Pass 1 分析和 Pass 2 合并。缓存命中会显示在日志和 `analysis_report` 中。

缓存安全规则：

- LoRA 文件使用 SHA-256 内容指纹；文件改变后自动失效。
- 模型/CLIP 结构、输入强度、策略和高级设置均计入缓存键。
- 使用 `.safetensors`，不反序列化可执行 Python 对象。
- 写入采用临时文件和原子替换；取消或异常不会留下可命中的半成品。
- 默认最多使用 20 GiB，超过后按最近使用时间清理旧条目。
- 写入前至少保留 512 MiB 磁盘空间。
- `persistent_cache=disabled` 时不会读取或写入持久缓存文件。
- TAME 依赖基础模型实际权重，因此启用 TAME 时不会使用跨模型持久缓存。

开发者可通过 `LORA_OPTIMIZER_CACHE_GB` 将上限设置为 1–200 GiB，例如：

```powershell
$env:LORA_OPTIMIZER_CACHE_GB = "40"
```

如需手动清空缓存，关闭 ComfyUI 后删除 `lora_optimizer_cache` 目录即可。

## ⚙️ GPU 分块调度

每个目标层都会独立规划：

1. **full GPU**：预计峰值显存安全时走完整快速路径。
2. **tiled GPU**：大层按输出行分块重建、分析、合并和压缩。
3. **CPU**：仅用于无 CUDA/ROCm、明确选择 CPU SVD、未知 payload 或最小分块也无法安全执行的情况。

默认会预留 `max(512 MiB, 总显存的 10%)`，单个 dense tile 不超过 128 MiB。诊断时可用 `LORA_OPTIMIZER_TILE_MB=16..512` 覆盖分块工作集预算，普通用户无需调整。

## 📦 安装与更新

在 `ComfyUI/custom_nodes` 下克隆：

```bash
git clone https://github.com/Aaalice2333/ComfyUI-LoRA-Optimizer-Aaalice.git
```

更新：

```bash
cd ComfyUI-LoRA-Optimizer-Aaalice
git pull
```

安装或更新后必须重启 ComfyUI。

## 🔄 工作流结构

使用 LoRA Manager 生成 `LORA_STACK`，连接 LoRA Optimizer；需要高级参数时连接 LoRA Optimizer Settings，需要导出文件时再连接 Save Merged LoRA。

## 🧪 验证

```bash
E:/ComfyUI-aki-v3/python/python.exe -m unittest discover -s tests -t . -p "test_*.py"
```

GPU 基准工具：

```bash
python tools/benchmark_chunked_merge.py --help
```

## 📄 许可证

GPL-3.0-only，详见 [LICENSE](LICENSE)。
