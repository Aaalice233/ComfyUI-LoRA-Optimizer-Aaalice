# ComfyUI LoRA Optimizer Aaalice

[中文](README.md)

A GPU-first optimizer for merging multiple LoRAs. **LoRA Manager** owns the stack; this project analyzes, optimizes, caches, applies, and saves the final merge.

## ✨ Highlights

- Analyzes overlap, direction, conflict, norm, rank, and subspace relationships per target.
- Automatically selects Weighted, SLERP, TIES, Consensus, and related strategies per layer.
- Uses `full GPU → tiled GPU → CPU` scheduling so large dense diffs do not exhaust VRAM.
- Supports LoRA, LoCon, LoHa, LoKr, fused QKV, MODEL, and CLIP weights.
- Supports native ComfyUI cancellation and tiled progress without applying interrupted partial patches.
- Provides in-memory caching and optional persistent local caching across ComfyUI restarts.
- Does not download models, upload data, send telemetry, or make background network requests.

## 🧩 Nodes

The project exposes exactly three nodes.

### LoRA Optimizer

Consumes a compatible `LORA_STACK` from LoRA Manager, analyzes and merges it, caches the completed patches, and applies them to MODEL/CLIP.

Outputs the patched MODEL, optional patched CLIP, an analysis report, and `LORA_DATA` for saving.

### LoRA Optimizer Settings

Centralizes merge strategy, automatic strength, sparsification, compression, VRAM policy, STAR/TAME, memory cache, and persistent cache settings. Recommended defaults are used when disconnected.

### Save Merged LoRA

Writes `LORA_DATA` as a standard `.safetensors` LoRA with automatic or fixed-rank SVD compression and optional strength baking.

## 🔗 Recommended workflow

```text
LoRA Manager ── LORA_STACK ──> LoRA Optimizer ──> MODEL / CLIP
                                      │
LoRA Optimizer Settings ──────────────┘
                                      │
                                      └── LORA_DATA ──> Save Merged LoRA
```

The project no longer ships a second LoRA stack implementation.

## 💾 Persistent cache across restarts

LoRA Optimizer Settings provides two independent switches:

- `cache_patches`: process-local RAM cache; cleared when ComfyUI exits.
- `persistent_cache`: local disk cache; reusable after ComfyUI restarts.

Persistent caching is enabled by default. Set `persistent_cache=disabled` if no local cache files should be read or written; existing entries remain until manually removed as described below.

When `persistent_cache=enabled`, a completed merge is atomically stored under:

```text
<ComfyUI user directory>/lora_optimizer_cache/
```

When the LoRA files, strengths, model structure, and mathematical settings match, the optimizer loads and applies the final MODEL/CLIP patches directly. LoRA loading, Pass 1 analysis, and Pass 2 merging are skipped. Hits are reported in both the log and `analysis_report`.

Safety and invalidation rules:

- LoRA files use SHA-256 content fingerprints and invalidate automatically after modification.
- Model/CLIP structure, strengths, strategy, and all result-affecting settings are included in the key.
- Cache entries use `.safetensors`; no executable Python objects are deserialized.
- Writes use a temporary file plus atomic replacement; cancellation cannot commit partial entries.
- The default LRU budget is 20 GiB.
- At least 512 MiB of free disk space is retained before a write.
- `persistent_cache=disabled` performs no persistent-cache reads or writes.
- TAME depends on base-model tensor values, so persistent reuse is disabled when TAME is active.

Developers can set a 1–200 GiB budget with `LORA_OPTIMIZER_CACHE_GB`:

```powershell
$env:LORA_OPTIMIZER_CACHE_GB = "40"
```

Delete the `lora_optimizer_cache` directory while ComfyUI is stopped to clear it manually.

## ⚙️ GPU scheduling

Each target is planned independently:

1. **full GPU** when the estimated peak safely fits.
2. **tiled GPU** for row-streamed reconstruction, analysis, merging, and compression.
3. **CPU** only without CUDA/ROCm, for explicit CPU SVD, unsupported payloads, or when even the minimum tile cannot run safely.

The planner reserves `max(512 MiB, 10% of total VRAM)` and caps a dense tile at 128 MiB. `LORA_OPTIMIZER_TILE_MB=16..512` is available for diagnostics; normal users do not need to tune it.

## 📦 Installation

Clone under `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/Aaalice2333/ComfyUI-LoRA-Optimizer-Aaalice.git
```

Update with:

```bash
cd ComfyUI-LoRA-Optimizer-Aaalice
git pull
```

Restart ComfyUI after installation or update.

## 🔄 Workflow structure

Build `LORA_STACK` with LoRA Manager and connect it to LoRA Optimizer. Connect LoRA Optimizer Settings only when advanced controls are needed, and connect Save Merged LoRA only when exporting a file.

## 🧪 Validation

```bash
E:/ComfyUI-aki-v3/python/python.exe -m unittest discover -s tests -t . -p "test_*.py"
python tools/benchmark_chunked_merge.py --help
```

## 📄 License

GPL-3.0-only. See [LICENSE](LICENSE).
