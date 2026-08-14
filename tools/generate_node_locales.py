from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("lora_optimizer_locale_source", ROOT / "lora_optimizer.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

ZH_NAMES = {
    "model": "模型", "lora_stack": "LoRA 堆栈", "output_strength": "输出强度",
    "clip": "CLIP", "clip_strength_multiplier": "CLIP 强度倍率", "settings": "设置",
    "analysis_report": "分析报告", "lora_data": "LoRA 数据", "auto_strength": "自动强度",
    "optimization_mode": "优化模式", "merge_refinement": "合并精炼", "sparsification": "稀疏化",
    "sparsification_density": "稀疏密度", "dare_dampening": "DARE 阻尼",
    "patch_compression": "补丁压缩", "svd_device": "SVD 设备",
    "free_vram_between_passes": "阶段间释放显存", "strategy_set": "策略集",
    "normalize_keys": "规范化键名", "architecture_preset": "架构预设",
    "auto_strength_floor_mode": "强度下限模式", "auto_strength_floor": "自动强度下限",
    "decision_smoothing": "决策平滑", "smooth_slerp_gate": "平滑 SLERP 门控",
    "vram_budget": "显存预算", "cache_patches": "内存缓存", "persistent_cache": "跨重启持久缓存",
    "star_eta": "STAR 保留率", "tame_layers": "TAME 强度", "tame_threshold": "TAME 阈值",
    "save_folder": "保存文件夹", "filename": "文件名", "save_rank": "保存秩",
    "bake_strength": "烘焙强度", "prompt": "提示词", "description": "描述", "file_path": "文件路径",
}

ZH_TOOLTIPS = {
    "model": "连接要应用合并 LoRA 的基础模型。",
    "lora_stack": "连接 LoRA Manager 或其他兼容节点输出的 LORA_STACK。",
    "output_strength": "合并结果的总强度；-1 使用自动建议值。",
    "clip": "可选文本编码器；连接后同时合并并应用 CLIP LoRA。",
    "clip_strength_multiplier": "CLIP 相对模型输出强度的倍率。",
    "settings": "连接 LoRA Optimizer Settings；未连接时使用推荐默认值。",
    "auto_strength": "合并多个 LoRA 时自动降低单项强度，减少过饱和和失真。",
    "optimization_mode": "按层自动决策、全局统一决策，或直接加法合并。",
    "merge_refinement": "none 直接合并；refine 增加正交化和 TALL；full 再增加 Procrustes 对齐。",
    "sparsification": "使用 DARE/DELLA 在全部权重或仅冲突区域降低 LoRA 干扰。",
    "sparsification_density": "保留权重的比例；数值越低，丢弃越多。",
    "dare_dampening": "降低 DARE 重缩放在低密度下的噪声放大。",
    "patch_compression": "智能压缩安全层、激进压缩全部层，或禁用压缩。",
    "svd_device": "选择压缩 SVD 的计算设备；GPU 通常更快。",
    "free_vram_between_passes": "在分析与合并阶段之间释放可回收显存。",
    "strategy_set": "限制自动决策可使用的合并策略范围。",
    "normalize_keys": "统一不同训练器生成的 LoRA 键名，并拆分融合 QKV 以便准确分析。",
    "architecture_preset": "按模型架构调整阈值；auto 会从 LoRA 键名自动识别。",
    "auto_strength_floor_mode": "自动使用架构感知下限，或手动使用下方滑块。",
    "auto_strength_floor": "手动模式下，限制自动强度最多可缩小到的倍率。",
    "decision_smoothing": "将单层指标向同块平均值平滑，减少噪声导致的策略跳变。",
    "smooth_slerp_gate": "使用平滑余弦值判断是否进入 SLERP。",
    "vram_budget": "最终补丁可占用的空闲显存比例；实时显存不足时仍会安全放在内存。",
    "cache_patches": "仅在当前 ComfyUI 进程的内存中缓存完整合并结果；重启后失效。",
    "persistent_cache": "将完整合并补丁安全写入本地磁盘，重启 ComfyUI 后可直接复用；关闭时不会创建持久缓存文件。",
    "star_eta": "每个 LoRA 的 STAR 频谱保留率；100 表示关闭。",
    "tame_layers": "限制异常高能量层的强度；0 表示关闭。",
    "tame_threshold": "相对基础权重范数超过此比例时将层视为过热。",
    "lora_data": "连接 LoRA Optimizer 输出的可保存 LoRA 数据。",
    "save_folder": "选择 ComfyUI 配置中的 LoRA 保存目录。",
    "filename": "保存文件名，可包含子目录；自动添加 .safetensors。",
    "save_rank": "0 自动使用输入总秩；非零值用 SVD 压缩到指定秩。",
    "bake_strength": "启用后，将当前合并强度烘焙到保存文件中。",
    "prompt": "写入文件元数据的示例提示词或触发词。",
    "description": "写入文件元数据的说明或备注。",
}

ZH_OPTIONS = {
    "enabled": "启用", "disabled": "禁用", "per_prefix": "按层", "global": "全局",
    "additive": "直接加法", "none": "无", "refine": "精炼", "full": "完整",
    "dare": "DARE", "della": "DELLA", "dare_conflict": "DARE（仅冲突）",
    "della_conflict": "DELLA（仅冲突）", "smart": "智能", "aggressive": "激进",
    "gpu": "GPU", "cpu": "CPU", "no_slerp": "排除 SLERP", "basic": "基础",
    "auto": "自动", "sd_unet": "SD/SDXL UNet", "dit": "DiT",
    "acestep_dit": "ACE-Step DiT", "llm": "LLM", "manual": "手动",
}

ZH_TITLES = {
    "LoRAOptimizerSimple": "LoRA 优化器",
    "LoRAOptimizerSettings": "LoRA 优化器设置",
    "SaveMergedLoRA": "保存合并 LoRA",
}

ZH_DESCRIPTIONS = {
    "LoRAOptimizerSimple": "分析、优化、合并、缓存并应用 LoRA Manager 提供的 LoRA 堆栈。",
    "LoRAOptimizerSettings": "集中配置合并策略、显存、缓存、压缩和高级优化行为。",
    "SaveMergedLoRA": "将优化器输出保存为可复用的 safetensors LoRA。",
}

EN_DESCRIPTIONS = {
    "LoRAOptimizerSimple": "Analyze, optimize, merge, cache, and apply a LoRA Manager stack.",
    "LoRAOptimizerSettings": "Configure merge strategy, memory, caching, compression, and advanced optimization.",
    "SaveMergedLoRA": "Save optimizer output as a reusable safetensors LoRA.",
}

ACRONYMS = {
    "cpu": "CPU", "dare": "DARE", "della": "DELLA", "gpu": "GPU", "lora": "LoRA",
    "slerp": "SLERP", "star": "STAR", "svd": "SVD", "tame": "TAME", "vram": "VRAM",
}


def pretty_name(name: str) -> str:
    return " ".join(ACRONYMS.get(part.lower(), part.capitalize()) for part in name.split("_"))


def localized_name(name: str, language: str) -> str:
    return ZH_NAMES.get(name, pretty_name(name)) if language == "zh" else pretty_name(name)


def option_labels(name: str, values: object, language: str) -> dict[str, str] | None:
    if name == "save_folder" or not isinstance(values, (list, tuple)) or not values or len(values) > 50:
        return None
    labels = {}
    for value in values:
        if isinstance(value, (str, int, float, bool)):
            raw = str(value)
            labels[raw] = ZH_OPTIONS.get(raw, pretty_name(raw)) if language == "zh" else pretty_name(raw)
    return labels or None


def build(language: str) -> dict[str, object]:
    result = {}
    for node_id, node_class in module.NODE_CLASS_MAPPINGS.items():
        inputs = {}
        for section in ("required", "optional"):
            for name, schema in node_class.INPUT_TYPES().get(section, {}).items():
                metadata = schema[1] if isinstance(schema, tuple) and len(schema) > 1 and isinstance(schema[1], dict) else {}
                entry = {"name": localized_name(name, language)}
                tooltip = metadata.get("tooltip") if language == "en" else ZH_TOOLTIPS.get(name)
                if tooltip:
                    entry["tooltip"] = tooltip
                values = schema[0] if isinstance(schema, tuple) and schema else None
                options = option_labels(name, values, language)
                if options:
                    entry["options"] = options
                inputs[name] = entry
        outputs = {
            str(index): {"name": localized_name(name, language)}
            for index, name in enumerate(getattr(node_class, "RETURN_NAMES", ()))
        }
        display_name = module.NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        result[node_id] = {
            "display_name": ZH_TITLES.get(node_id, display_name) if language == "zh" else display_name,
            "description": ZH_DESCRIPTIONS[node_id] if language == "zh" else EN_DESCRIPTIONS[node_id],
            "inputs": inputs,
            "outputs": outputs,
        }
    return result


for language in ("en", "zh"):
    destination = ROOT / "locales" / language / "nodeDefs.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build(language), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination.relative_to(ROOT))
