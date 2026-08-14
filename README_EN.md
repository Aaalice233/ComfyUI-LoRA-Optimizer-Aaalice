<p align="center">
  <a href="assets/banner.png"><img src="assets/banner.svg" alt="LoRA Optimizer" width="100%"></a>
</p>

<p align="center">
  <a href="README.md">简体中文</a> · <b>English</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Custom_Nodes-2563eb?style=flat-square" alt="ComfyUI Custom Nodes">
  <img src="https://img.shields.io/badge/Aaalice-Adaptive_VRAM-e94560?style=flat-square" alt="Aaalice Adaptive VRAM">
  <img src="https://img.shields.io/badge/TIES_%7C_DARE_%7C_DELLA-Merging-8b5cf6?style=flat-square" alt="Merge Algorithms">
  <img src="https://img.shields.io/badge/AutoTuner-Parameter_Sweep-f59e0b?style=flat-square" alt="AutoTuner">
  <img src="https://img.shields.io/badge/License-GPL--3.0-22c55e?style=flat-square" alt="GPL-3.0">
</p>

# 🧩 ComfyUI-LoRA-Optimizer-Aaalice

A ComfyUI node pack for multi-LoRA analysis, conflict handling, and merging.

Instead of blindly adding LoRA patches, the optimizer resolves the model weights they actually target, analyzes direction, magnitude, sign conflict, and subspace overlap layer by layer, then selects an appropriate merge strategy for each target group. The main node works with safe defaults, while Settings, AutoTuner, Estimator, Conflict Editor, and Merge Formula nodes provide deeper control when needed.

> This repository is the Aaalice fork of [ethanfel/ComfyUI-LoRA-Optimizer](https://github.com/ethanfel/ComfyUI-LoRA-Optimizer). It retains the complete upstream feature set and adds adaptive memory scheduling for low-VRAM systems.

<p align="center"><img src="assets/comparison.png" alt="Regular LoRA stacking compared with optimized merging" width="100%"></p>

## ✨ Differences from upstream

The current branch stays aligned with `upstream/main`. Node IDs, workflow formats, merge algorithms, and public behavior remain upstream-compatible. The Aaalice fork adds these production-grade improvements:

| Area | Current upstream behavior | Aaalice fork |
|---|---|---|
| Large-layer execution | Primarily computes complete target layers when CUDA is available | Automatically selects **full GPU → tiled GPU → CPU** per target; oversized layers are streamed by output rows instead of retaining many complete dense diffs |
| VRAM planning | Users often need to tune memory options manually | Replans from live free VRAM, factors, contributor count, and strategy workspaces while preserving a fixed safety reserve |
| Algorithm coverage | Complete-tensor path | Preserves weighted, SLERP, TIES, DARE, DELLA, consensus, STAR/TAME, refinement, and compression semantics without silently replacing an algorithm |
| Native cancellation | Long loops may not promptly observe queue cancellation | Pass 1, Pass 2, AutoTuner, CPU workers, cache waits, SVD, and final saves observe ComfyUI's native interrupt; interrupted runs never apply partial patches |
| Localization | Node UI is primarily English | Every node title, input, output, option, description, and custom menu follows ComfyUI's locale in English or Simplified Chinese |
| 8 GB VRAM behavior | Large Krea, Flux, or Wan stacks may OOM on a large target | Small targets retain the full-GPU fast path; large targets use bounded GPU tiles, with CPU reserved for explicit or capability-based fallback |

Tiling limits the temporary GPU workset without lowering the selected strategy. Reduction order may produce negligible floating-point differences, while algorithms, target ranks, patch structures, and quality gates remain unchanged.

## 🎯 What it is for

- Fix oversaturation, color shifts, lost identity, or detail loss when stacking style, character, and detail LoRAs
- Normalize different trainer key conventions so overlapping layers are detected correctly
- Pick TIES, consensus, SLERP, weighted average, or weighted sum per layer instead of globally
- Save an optimized result as a standard `.safetensors` LoRA or a conditioning hook
- Optimize an existing chain of regular `Load LoRA` nodes without rebuilding it around a stack
- Search settings with AutoTuner or predict configurations from community data with the k-NN Estimator
- Control RAM and VRAM peaks while merging many LoRAs or large model families

It is not a universal replacement for every LoRA loader. Distillation, DPO, Lightning, LCM, Turbo, Hyper, and edit-model LoRAs often rely on precisely calibrated weights. Prefer a regular `Load LoRA` node for those. If they must be merged, use `additive` mode and disable sparsification.

## 🚀 Three-step quick start

1. Add **LoRA Stack (Dynamic)** and choose the LoRAs and strengths to merge.
2. Add **LoRA Optimizer**, then connect the base model, `lora_stack`, and optionally `CLIP` for image models.
3. Connect the optimizer's `MODEL` / `CLIP` outputs to the sampling workflow.

```text
LoRA Stack (Dynamic) ───────────────► LoRA Optimizer ───► KSampler
                                           ▲
Load Checkpoint ───► MODEL / CLIP ─────────┘
```

For the first run, **do not connect a Settings node**. The main optimizer already uses these defaults:

- `auto_strength = enabled`
- `optimization_mode = per_prefix`
- `patch_compression = smart`
- `svd_device = gpu`
- `normalize_keys = enabled`
- `strategy_set = full`
- `sparsification = disabled`

Connect `analysis_report` to any Show Text node to inspect per-LoRA statistics, pair conflicts, per-layer strategy decisions, and the final merge summary.

### 🌐 Node language

Node titles, inputs, outputs, enum options, descriptions, and custom context menus automatically follow ComfyUI's `Locale` setting. Complete English and Simplified Chinese localizations are included. Refresh the page when ComfyUI requests it after changing locale; workflow socket names and serialized values remain unchanged.

## 📦 Installation, replacement, and updates

### 🔌 ComfyUI Manager

In ComfyUI Manager, choose **Install via Git URL** and enter:

```text
https://github.com/Aaalice233/ComfyUI-LoRA-Optimizer-Aaalice.git
```

Restart ComfyUI after installation.

### 🛠️ Manual installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Aaalice233/ComfyUI-LoRA-Optimizer-Aaalice.git
```

The project requires `scikit-learn>=1.3`. If your ComfyUI environment does not install it automatically, run the following with the same Python environment that launches ComfyUI:

```bash
python -m pip install "scikit-learn>=1.3"
```

Restart ComfyUI. The nodes are registered under the `LoRA Optimizer` category.

### ♻️ Replacing the upstream version

The upstream and Aaalice repositories register the same node IDs. **Do not install or enable both at the same time.** Stop ComfyUI, move or disable the old `ComfyUI-LoRA-Optimizer` folder, then install this repository.

### ⬆️ Updating

```bash
cd ComfyUI/custom_nodes/ComfyUI-LoRA-Optimizer-Aaalice
git pull --ff-only
```

Restart ComfyUI after updating.

## 🧭 Choosing a workflow

| Situation | Recommended entry point |
|---|---|
| New workflow; simple and stable | `LoRA Stack (Dynamic)` → `LoRA Optimizer` |
| Existing regular Load LoRA chain | `LoRA Optimizer (Inline Chain)` |
| Unsure about settings and willing to run a search | `LoRA AutoTuner` or `LoRA AutoTuner Settings` → `LoRA Optimizer` |
| Quickly reuse experience from similar combinations | `LoRA Merge Estimator` → `TUNER_DATA` → `LoRA Optimizer` |
| Known conflict requiring manual control | `LoRA Conflict Editor` → `LoRA Optimizer Settings` |
| Merge 1+2 before blending with 3 | `LoRA Merge Formula` |
| kijai WanVideoWrapper | The `(WIP)` WanVideo nodes; test carefully |

### 🔗 Standard Stack workflow

```text
Load Checkpoint ──► LoRA Optimizer ──► Sampler
                         ▲
LoRA Stack (Dynamic) ────┘
```

### 🔗 Inline Chain workflow

```text
Load Checkpoint ─► Load LoRA #1 ─► Load LoRA #2 ─► LoRA Optimizer (Inline Chain) ─► Sampler
                                                          ▲
                                       LoRA Inline Chain Options (optional)
```

The Inline node reads and removes LoRA patches left by upstream loaders, merges them through the same optimizer engine, and applies the optimized result. Stock `Load LoRA`, `Load LoRA (Model Only)`, and rgthree Power Lora Loader chains expose their real filenames. Unstamped third-party loaders fall back to labels such as `chain lora #N`.

### 🔗 AutoTuner workflow

```text
LoRA Merge Settings (optional)
          │
LoRA AutoTuner Settings ─► LoRA Optimizer.settings
LoRA Stack ──────────────► LoRA Optimizer.lora_stack
```

The main `LoRA Optimizer` automatically enters advanced or AutoTuner mode based on the connected Settings object; the main node does not need to be replaced.

### 🔗 Estimator workflow

```text
LoRA Stack ─► LoRA Merge Estimator ─► TUNER_DATA ─► LoRA Optimizer
                   ▲                                  ▲
              MODEL / CLIP                       MODEL / CLIP
```

On its first explicit run, the Estimator downloads the community cache and builds a local k-NN index under `ComfyUI/models/estimator/`. Later runs reuse the index. Use a full AutoTuner sweep when no close neighbors are found.

## 🧱 Node overview

### 🧰 Building and merging

| Node | Purpose |
|---|---|
| **LoRA Stack** | Fixed-slot LoRA list |
| **LoRA Stack (Dynamic)** | Dynamically add or remove LoRAs; recommended for new workflows |
| **LoRA Optimizer** | Recommended main node; runs with defaults or accepts Settings / `TUNER_DATA` |
| **LoRA Optimizer (Legacy)** | All parameters on one node for older workflow compatibility |
| **LoRA Optimizer (Inline Chain)** | Optimizes patches left by a regular Load LoRA chain |
| **LoRA Inline Chain Options** | Per-inline-LoRA enable, strength, conflict, filtering, and preserve controls |
| **LoRA Merge Formula** | Defines hierarchical merge orders such as `(1+2)+3` |
| **LoRA Conflict Editor** | Inspects conflicts and manually overrides conflict modes or strategy |

### ⚙️ Settings and automatic search

| Node | Purpose |
|---|---|
| **LoRA Merge Settings** | Architecture, caching, smoothing, strength floor, and VRAM settings shared by Optimizer and AutoTuner |
| **LoRA Optimizer Settings** | Advanced strategies, sparsification, compression, and SVD options |
| **LoRA AutoTuner Settings** | Search, scoring, cache, persistent memory, and community options |
| **LoRA AutoTuner** | Standalone search node that ranks and merges configurations |
| **LoRA Merge Estimator** | k-NN prediction from community data |
| **Merge Selector** | Applies rank N from `TUNER_DATA` |
| **Save / Load Tuner Data** | Saves and restores `.tuner` / `.json` results |
| **Build AutoTuner Python Evaluator** | Connects a custom Python quality evaluator |

### 🔍 Analysis, conversion, and output

| Node | Purpose |
|---|---|
| **LoRA Compatibility Analyzer** | Reports compatibility before merging |
| **LoRA Metadata Reader** | Reads descriptions, prompts, and merge metadata from `.safetensors` |
| **LoRA Extract from Model** | Diffs a base and fine-tuned model, then extracts a LoRA |
| **LoRA Combination Generator** | Produces two- and three-way combinations for AutoTuner dataset collection |
| **Save Merged LoRA** | Exports a standard `.safetensors` LoRA |
| **Merged LoRA to Hook** | Converts the result into ComfyUI conditioning hooks |
| **(WIP) WanVideo LoRA Optimizer** | Experimental kijai WanVideoWrapper path |
| **(WIP) Merged LoRA → WanVideo** | Converts merged data for the WanVideoWrapper path |

## ⚙️ Recommended settings

### ✅ General recommendation

| Setting | Recommended value | Why |
|---|---|---|
| `auto_strength` | `enabled` | Prevents excessive combined energy |
| `optimization_mode` | `per_prefix` | Chooses a strategy independently for each target group |
| `merge_refinement` | `none` | Lowest default cost; try `refine` only when interference remains |
| `sparsification` | `disabled` | Do not discard weights without a specific reason |
| `patch_compression` | `smart` | Preserves linear-merge information while reducing patch memory |
| `svd_device` | `gpu` | Uses GPU for ordinary and chunkable large targets; choose `cpu` only when CPU SVD is explicitly required |
| `free_vram_between_passes` | `enabled` on 8 GB cards | Releases cache between Pass 1 and Pass 2 |
| `strategy_set` | `full` | Makes the complete strategy set available |
| `vram_budget` | Start at `0` | Keeps final patches in system RAM for the safest baseline |
| `cache_patches` | `enabled` for image models; `disabled` for large video models | Trades system RAM for fast reruns |

### 🧯 8 GB VRAM

Do not use `global + aggressive + basic + CPU SVD` as a permanent OOM workaround. It restricts the optimizer and can reduce quality. Keep the regular quality settings:

```text
auto_strength            enabled
optimization_mode        per_prefix
merge_refinement         none
sparsification           disabled
patch_compression        smart
svd_device               gpu
free_vram_between_passes enabled
strategy_set             full
vram_budget              0
```

The Aaalice fork automatically moves complete targets that do not fit into tiled GPU execution. The first tiled-GPU log line means the protection is active; CPU fallback is reserved for explicit CPU selection or a reported capability limit. No manual tile-size setting is required.

### 🎨 Oversaturated or fried output

1. Keep `auto_strength = enabled`.
2. Set `output_strength = -1` for an automatic recommendation, or reduce it to `0.6–0.9`.
3. Try `merge_refinement = refine`.
4. For a few abnormally hot layers, start with `tame_layers = 0.5` and `tame_threshold = 0.3`.
5. Try `della_conflict` or `dare_conflict` only when the report shows genuine conflict.

### 🧪 Many strongly conflicting LoRAs

- `strategy_set = full`
- Start with `merge_refinement = refine`; use `full` only if necessary
- `sparsification = della_conflict`
- `sparsification_density = 0.7`
- `star_eta = 60–80` only for large, clearly conflicting stacks; keep `100` (off) for two or three mostly independent LoRAs

## 🧠 How it works

<p align="center"><a href="assets/optimizer-pipeline.png"><img src="assets/optimizer-pipeline.svg" alt="LoRA Optimizer Pipeline" width="100%"></a></p>

### 🔬 Pass 1: analysis

1. Detect Standard LoRA, LoCon, and compatible trainer variants.
2. Normalize Kohya, AI-Toolkit, LyCORIS, diffusers/PEFT, and related names to real model keys.
3. Aggregate aliases from the same LoRA that resolve to the same target weight.
4. Measure per-LoRA norms, pairwise cosine, sign conflict, magnitude distribution, and low-rank subspace overlap.
5. Retain lightweight statistics and release the current target group's dense diffs.

### 🧬 Pass 2: merge

1. Reconstruct the required diffs one target group at a time.
2. Select a strategy from the local statistics.
3. Optionally apply DARE/DELLA, STAR, direction orthogonalization, TALL-masks, or KnOTS.
4. Preserve exact low-rank representations on linear paths where possible; run SVD patch compression when needed.
5. Apply patches to `MODEL` / `CLIP` and emit the report plus reusable `LORA_DATA`.

### 🧠 Per-group strategy selection

| Condition | Typical strategy |
|---|---|
| One LoRA touches the group | `weighted_sum`, preserving the full contribution |
| Multiple mostly independent LoRAs | `weighted_average` |
| Aligned directions with little conflict | `consensus` |
| High conflict with overlapping subspaces | `ties` |
| Smooth directional interpolation is useful | `slerp` |
| Structural, distillation, or edit LoRAs | Manually selected `additive` is safer |

<p align="center"><a href="assets/merge-strategies.png"><img src="assets/merge-strategies.svg" alt="LoRA Merge Strategies" width="100%"></a></p>

### 💪 Auto-Strength

Auto-Strength estimates the combined energy from exact branch norms and pairwise dot products. It applies one uniform downward scale, preserves original strength ratios and signs, and never boosts strengths. Aligned LoRAs are reduced more, independent LoRAs less, and opposing LoRAs usually need little reduction because they already cancel.

### 🔑 Key normalization

Architecture-aware mapping covers common SD/SDXL, Flux, Wan, Z-Image, LTX, ACE-Step, Ideogram 4, Anima, and Qwen-Image naming conventions. Z-Image fused QKV is split for component-level analysis and restored to ComfyUI's native layout after merging.

### 🗜️ Patch compression

| Mode | Behavior | Quality |
|---|---|---|
| `smart` | Compresses paths where linear merge information can be retained | Recommended; lossless on linear paths |
| `aggressive` | Compresses nonlinear results as well | Lower memory, but may lose detail |
| `disabled` | Keeps dense patches | No compression loss, highest RAM usage |

## 💾 RAM and VRAM behavior

The Aaalice fork plans **each resolved target group and execution stage** automatically:

1. If the complete target peak is below 80% of the safe budget, use the `full_gpu` fast path.
2. If the full target does not fit but its source is row-addressable, use `tiled_gpu`; Pass 1 statistics, merge math, randomized SVD, and compression retain only the current tile.
3. Use `cpu` only without a usable GPU, after explicit CPU selection, for an unknown third-party payload, or when even the minimum tile cannot fit.
4. Poll ComfyUI's native interrupt before every tile. An interrupted target is discarded without final cache writes or patch application.
5. Release factor staging and temporary tensors after every target, then replan from current free VRAM.

Default safety rules reserve `max(512 MiB, 10% of total VRAM)`, cap one dense tile at 128 MiB, and cap the complete tiled workset at 512 MiB. `LORA_OPTIMIZER_TILE_MB=16..512` is a developer diagnostic override; regular users should not set it.

Reusable benchmark:

```bash
python tools/benchmark_chunked_merge.py --mode tiled_gpu --rows 8192 --cols 8192 --rank 32 --loras 9
python tools/benchmark_chunked_merge.py --mode tiled_gpu --real-lora-dir <LoRA-directory> --loras 9
```

It reports analysis/merge timings, tile count, peak GPU memory, CPU RSS, numerical error, and cancellation latency. The second command uses real LoRA factors.

Commonly confused options:

| Setting | What it controls | What it does not control |
|---|---|---|
| `vram_budget` | How many final merged patches may remain on GPU | Live free VRAM and the safety reserve are still enforced |
| `svd_device` | Preferred GPU or explicit CPU compression | Users do not select tile size manually |
| `free_vram_between_passes` | GPU cache cleanup between Pass 1 and Pass 2 | Per-target peaks are independently bounded by the tiled planner |
| `cache_patches` | Whether final patches are cached for quick reruns | Source LoRA file caching |
| `diff_cache_mode` | Reusing diffs between AutoTuner candidates | The regular optimizer's VRAM limit |

An uncompressed nonlinear result still needs one final CPU patch; that is output memory, not duplicate workspace. If a capability-based CPU fallback creates system-memory pressure:

- Close unnecessary applications and keep the system page file available.
- Disable `cache_patches` for large video models.
- Use AutoTuner `diff_cache_mode = auto` or `disabled`; use `disk` only with ample temporary-drive space.
- Merge fewer LoRAs at once, save an intermediate LoRA, then run the next stage.

## 🧪 AutoTuner, caches, and Estimator

### 🏁 AutoTuner

AutoTuner reuses one Pass 1 analysis, scores many parameter combinations, then performs real merges for the highest-ranked candidates. Useful defaults are:

- `top_n = 3`
- `scoring_speed = turbo`
- `scoring_svd = disabled`
- `scoring_device = gpu`
- `scoring_formula = v2`
- `diff_cache_mode = auto`
- `memory_mode = auto`

`TUNER_DATA` can be connected to **Merge Selector**, **Save Tuner Data**, or the regular **LoRA Optimizer**.

### 💽 Diff Cache

| Mode | Behavior |
|---|---|
| `disabled` | Recomputes each candidate; lowest memory usage |
| `auto` | Uses RAM up to `diff_cache_ram_pct`, then recomputes overflow on demand |
| `ram` | Keeps as much as possible in RAM; fastest and largest |
| `disk` | Uses temporary files and memory mapping; saves RAM but needs substantial disk space |

### 🌐 Community Cache

`community_cache = disabled` by default, so no community-cache request is made. Hugging Face is accessed only after the user explicitly selects `upload_only` or `upload_and_download`. Uploading requires an `HF_TOKEN` with write access. Identity is based on the LoRA file's SHA256 content hash, not its filename or directory.

Public dataset: [`ethanfel/lora-optimizer-community-cache`](https://huggingface.co/datasets/ethanfel/lora-optimizer-community-cache)

## 🧬 Compatibility

### ✅ Model families

- SD 1.5 / SDXL
- Flux
- Z-Image / Lumina2
- Wan 2.1 / 2.2
- LTX Video
- ACE-Step
- Ideogram 4
- Anima / Cosmos-Predict2 DiT
- Qwen-Image
- Other ComfyUI-supported architectures whose LoRA keys can be resolved to standard target weights

### ✅ LoRA formats and trainers

- Standard LoRA
- LoCon and compatible adapters reducible to up/down(/mid) factors
- Kohya
- AI-Toolkit
- LyCORIS
- Musubi Tuner
- Common diffusers / PEFT naming
- Standard tuple stacks from Efficiency Nodes, Comfyroll, and similar packs

Third-party formats evolve quickly. If the report shows unknown keys or skipped entries, trust `analysis_report` and the ComfyUI log rather than assuming every LyCORIS variant is losslessly convertible.

## 🎛️ Per-LoRA controls

Dynamic Stack and Inline Options support per-LoRA:

- Separate model and CLIP strengths
- `conflict_mode`
- `preserve`
- `key_filter`

Common `key_filter` values:

| Value | Behavior |
|---|---|
| `all` | Use every target group |
| `shared_only` | Use only groups touched by at least two LoRAs |
| `unique_only` | Use only groups unique to this LoRA |
| `audio_only` | Keep audio-related layers for LTX-2, ACE-Step, and similar models |
| `no_audio` | Exclude audio layers and keep video or other branches |

This is useful for separating T2V/I2V/VACE-specific layers or choosing audio and video sources independently in multimodal LoRAs.

## 📤 Saving and reuse

**Save Merged LoRA** writes `LORA_DATA` as a standard `.safetensors` LoRA:

- `save_rank = 0` keeps the ranks already present in the merged result.
- `bake_strength = enabled` makes strength `1.0` reproduce the current merge.
- Configured LoRA folders and their subdirectories are supported.

**Merged LoRA to Hook** converts the result into ComfyUI `HOOKS` for conditioning-, schedule-, or region-specific application without globally patching the model.

**LoRA Extract from Model** diffs a base model against a fine-tuned model and SVD-decomposes the result. The base and fine-tuned models must correspond exactly or the extracted LoRA is not meaningful.

## ⚠️ Boundaries and caveats

- Do not automatically apply TIES or sparsification to Lightning, LCM, Turbo, Hyper, DPO, distillation, or edit LoRAs together with style LoRAs.
- The regular Optimizer analyzes only its own `lora_stack`. Upstream patches are not included unless the Inline Chain node captures them.
- Changes baked into a checkpoint cannot be separated from base weights automatically. Extract from Model requires the matching base model.
- Inline Chain can only use captured-weight identity for unstamped third-party loaders; the report clearly shows fallback labels.
- `(WIP)` WanVideo nodes are experimental. Keep workflow copies and validate results before production use.

## 🧯 Troubleshooting

<details>
<summary><b>🔥 Pass 1 OOMs and changing vram_budget or svd_device does nothing</b></summary>

`vram_budget` controls final patches and `svd_device` controls the preferred compression device; neither is the temporary tile limit. The Aaalice fork automatically switches unsafe complete targets to tiled GPU. If it still uses CPU, the log reports the explicit CPU choice, unknown payload, unavailable GPU, or minimum-tile limitation.

</details>

<details>
<summary><b>🐢 The run becomes slower after a CPU fallback</b></summary>

A normal oversized target should use tiled GPU instead of whole-target CPU. CPU is a capability fallback. Check whether `svd_device = cpu` was explicitly selected, whether the stack contains an unknown third-party payload, and the logged fallback reason; do not reduce strategy quality merely to avoid OOM.

</details>

<details>
<summary><b>🎭 Character identity or style becomes too weak</b></summary>

Check `auto_strength` and `output_strength` first. Try `output_strength = -1`, or raise it gradually toward `1.0–1.2`. Keep `smart` instead of `aggressive` compression and verify that standard DARE/DELLA was not enabled accidentally.

</details>

<details>
<summary><b>🧱 Some layers are not merged</b></summary>

Inspect unknown keys, shape mismatches, key filters, and architecture detection in `analysis_report`. Keep `normalize_keys = enabled` for mixed trainers. For under-detected architectures such as HunyuanVideo, manually choose `architecture_preset = dit`.

</details>

<details>
<summary><b>🔁 Duplicate nodes or import failures after installation</b></summary>

Check whether both the upstream and Aaalice folders exist under `custom_nodes`. They share node IDs. Keep only one and restart ComfyUI.

</details>

<details>
<summary><b>⛔ How do I stop an accidental long merge?</b></summary>

Use ComfyUI's native cancel button. Optimizer, Inline, Merge Formula, AutoTuner, CPU workers, cache waits, and chunked SVD share one interrupt signal; an ordinary tiled stage normally stops within one tile. Cancellation does not apply partial patches or write incomplete final cache/save files.

</details>

## 📚 Documentation and examples

- [Complete node reference](docs/wiki/Nodes.md)
- [Configuration guide](docs/wiki/Configuration-Guide.md)
- [Workflows](docs/wiki/Workflows.md)
- [Merge algorithms](docs/wiki/Merge-Algorithms.md)
- [How it works](docs/wiki/How-It-Works.md)
- [Tips and troubleshooting](docs/wiki/Tips-and-Troubleshooting.md)
- [Technical report](docs/technical-report.md)
- [Example workflows](example_workflows/)

The Wiki pages are inherited primarily from upstream. `README.md` is the default Simplified Chinese entry point, while this file is the standalone English overview.

## 🤝 Upstream, research, and credits

- Upstream project: [ethanfel/ComfyUI-LoRA-Optimizer](https://github.com/ethanfel/ComfyUI-LoRA-Optimizer)
- Aaalice fork: [Aaalice233/ComfyUI-LoRA-Optimizer-Aaalice](https://github.com/Aaalice233/ComfyUI-LoRA-Optimizer-Aaalice)
- Original foundation: [ComfyUI-ZImage-LoRA-Merger](https://github.com/DanrisiUA/ComfyUI-ZImage-LoRA-Merger)
- TIES-Merging: [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)
- DARE: [Yu et al., ICML 2024](https://arxiv.org/abs/2311.03099)
- DELLA: [Deep et al., 2024](https://arxiv.org/abs/2406.11617)
- KnOTS: [Ramé et al., 2024](https://arxiv.org/abs/2407.09095)
- TALL-masks: [Wang et al., 2024](https://arxiv.org/abs/2406.12832)
- STAR: [Spectral Truncation And Rescale, NAACL 2025](https://arxiv.org/abs/2502.10339)

See the upstream documentation and this repository's technical report for the full source and algorithm references. Thanks to all upstream contributors, testers, and community-data contributors.

## 📄 License

This project is licensed under [GPL-3.0](LICENSE). Modified distributions must comply with the same license and preserve upstream copyright and attribution notices.
