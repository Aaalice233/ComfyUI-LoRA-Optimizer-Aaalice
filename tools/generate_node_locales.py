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
    "analysis_report": "分析报告", "architecture_preset": "架构预设", "auto_strength": "自动强度",
    "auto_strength_floor": "自动强度下限", "auto_strength_floor_mode": "强度下限模式",
    "bake_strength": "烘焙强度", "base_clip": "基础 CLIP", "base_model": "基础模型",
    "base_model_filter": "基础模型筛选", "cache_patches": "缓存补丁", "callable_name": "函数名",
    "chain_options": "链式选项", "clip": "CLIP", "clip_strength": "CLIP 强度",
    "clip_strength_multiplier": "CLIP 强度倍率", "combine_mode": "组合模式", "combo_info": "组合信息",
    "combo_size": "组合大小", "community_cache": "社区缓存", "compatibility_map": "兼容性映射",
    "conflict_mode": "冲突模式", "context_json": "上下文 JSON", "create_nodes": "创建节点",
    "dare_dampening": "DARE 阻尼", "decision_smoothing": "决策平滑", "description": "描述",
    "diff_cache_mode": "差分缓存模式", "diff_cache_ram_pct": "差分缓存内存占比", "enabled": "启用",
    "energy_threshold": "能量阈值", "estimator_report": "估算报告", "evaluator": "评估器",
    "file_path": "文件路径", "filename": "文件名", "finetuned_clip": "微调 CLIP",
    "finetuned_model": "微调模型", "folder_filter": "文件夹筛选", "formula": "合并公式",
    "free_vram_between_passes": "阶段间释放显存", "hooks": "Hooks", "input_mode": "输入模式",
    "k": "K 值", "key_filter": "键筛选", "lora_count": "LoRA 数量", "lora_data": "LoRA 数据",
    "lora_name": "LoRA 名称", "lora_name_text": "LoRA 名称（文本）", "lora_stack": "LoRA 堆栈",
    "memory_mode": "记忆模式", "merge_refinement": "合并精炼", "merge_settings": "合并设置",
    "merge_strategy": "合并策略", "merge_strategy_override": "覆盖合并策略", "metadata_info": "元数据信息",
    "model": "模型", "model_strength": "模型强度", "module_path": "模块路径",
    "normalize_keys": "规范化键名", "optimization_mode": "优化模式", "output_mode": "输出模式",
    "output_strength": "输出强度", "overwrite": "覆盖文件", "patch_compression": "补丁压缩",
    "preserve": "保护", "prev_hooks": "前置 Hooks", "prompt": "提示词", "rank": "秩",
    "rank_mode": "秩模式", "rebuild_index": "重建索引", "record_dataset": "记录数据集",
    "report": "报告", "rerun_mode": "重跑模式", "rerun_source": "重跑来源",
    "save_folder": "保存文件夹", "save_rank": "保存秩", "scoring_device": "评分设备",
    "scoring_formula": "评分公式", "scoring_speed": "评分速度", "scoring_svd": "SVD 评分",
    "selection": "选择名次", "settings": "设置", "settings_source": "设置来源",
    "settings_visibility": "设置显示", "shuffle_order": "打乱种子", "smooth_slerp_gate": "平滑 SLERP 门控",
    "sparsification": "稀疏化", "sparsification_density": "稀疏密度", "star_eta": "STAR 保留率",
    "strategy_set": "策略集", "strength": "强度", "svd_device": "SVD 设备",
    "tame_layers": "TAME 强度", "tame_threshold": "TAME 阈值", "top_n": "候选数量",
    "top_n_output": "输出数量", "tuner_data": "调优数据", "tuner_data_file": "调优数据文件",
    "vram_budget": "显存预算", "wan_model": "WanVideo 模型", "weight": "权重",
}

ZH_TOOLTIPS = {
    "lora_name": "选择要加入堆栈的 LoRA 文件。",
    "strength": "设置此 LoRA 对最终结果的影响强度；1.0 为完整强度。",
    "conflict_mode": "控制此 LoRA 在冲突权重中的参与范围：全部、低冲突或高冲突。",
    "key_filter": "限制此 LoRA 参与的权重键，可筛选共享、独有、音频或非音频层。",
    "preserve": "将此 LoRA 作为受保护的风格层叠加，不参与 TIES 符号淘汰和稀疏裁剪。",
    "lora_stack": "连接另一个 LoRA Stack 以继续扩展堆栈。",
    "settings_visibility": "简单模式只显示单一强度；高级模式显示模型、CLIP、冲突和筛选选项。",
    "input_mode": "下拉模式从文件列表选择；文本模式可输入短名称、相对路径或连接文本节点。",
    "lora_count": "设置当前显示并参与处理的 LoRA 槽位数量。",
    "enabled": "临时启用或停用此项，不会从列表中删除它。",
    "lora_name_text": "输入 LoRA 文件名、相对路径或 None；可连接文本节点。",
    "model_strength": "设置 LoRA 对图像生成模型的影响强度。",
    "clip_strength": "设置 LoRA 对文本编码器和提示词理解的影响强度。",
    "base_model_filter": "按基础模型类型筛选 LoRA 下拉列表，需要 ComfyUI-Lora-Manager。",
    "model": "连接要应用合并 LoRA 的基础模型。",
    "output_strength": "合并结果的总强度；-1 使用自动建议值。",
    "clip": "可选文本编码器；连接后同时合并并应用 CLIP LoRA。",
    "clip_strength_multiplier": "CLIP 相对模型输出强度的倍率。",
    "auto_strength": "合并多个 LoRA 时自动降低单项强度，减少过饱和和失真。",
    "auto_strength_floor": "限制自动强度最多可缩小到的倍率；-1 使用架构感知默认值。",
    "auto_strength_floor_mode": "自动使用架构感知下限，或手动使用上方滑块。",
    "free_vram_between_passes": "在分析与合并阶段之间释放可回收显存。",
    "vram_budget": "最终补丁可占用的空闲显存比例；实时显存不足时仍会安全放在内存。",
    "optimization_mode": "按层自动决策、全局统一决策，或直接加法合并。",
    "cache_patches": "在内存中保留完整合并结果以加速重复执行；关闭可节省内存。",
    "patch_compression": "智能压缩安全层、激进压缩全部层，或禁用压缩。",
    "svd_device": "选择压缩 SVD 的计算设备；GPU 通常更快。",
    "normalize_keys": "统一不同训练器生成的 LoRA 键名，并拆分融合 QKV 以便准确分析。",
    "sparsification": "使用 DARE/DELLA 在全部权重或仅冲突区域降低 LoRA 干扰。",
    "sparsification_density": "保留权重的比例；数值越低，丢弃越多。",
    "dare_dampening": "降低 DARE 重缩放在低密度下的噪声放大。",
    "merge_refinement": "none 直接合并；refine 增加正交化和 TALL；full 再增加 Procrustes 对齐。",
    "strategy_set": "限制自动决策可使用的合并策略范围。",
    "architecture_preset": "按模型架构调整阈值；auto 会从 LoRA 键名自动识别。",
    "merge_strategy_override": "连接冲突编辑器输出以覆盖自动选择的合并策略。",
    "settings_source": "选择手动设置、实时 AutoTuner 设置或已加载调优数据。",
    "tuner_data": "连接 AutoTuner 或已加载的调优数据。",
    "decision_smoothing": "将单层指标向同块平均值平滑，减少噪声导致的策略跳变。",
    "smooth_slerp_gate": "使用平滑余弦值判断是否进入 SLERP。",
    "settings": "连接设置节点以集中控制高级选项。",
    "chain_options": "按 Inline 链顺序设置每个捕获 LoRA 的独立选项。",
    "lora_data": "连接 LoRA Optimizer 输出的可保存或可转换 LoRA 数据。",
    "save_folder": "选择 ComfyUI 配置中的 LoRA 保存目录。",
    "filename": "保存文件名，可包含子目录；自动添加 .safetensors。",
    "save_rank": "0 自动使用输入总秩；非零值用 SVD 压缩到指定秩。",
    "bake_strength": "启用后，将当前合并强度烘焙到保存文件中。",
    "prompt": "写入文件元数据的示例提示词或触发词。",
    "description": "写入文件元数据的说明或备注。",
    "module_path": "包含外部评估函数的 Python 文件路径或可导入模块名。",
    "callable_name": "要导入并由 AutoTuner 调用的函数名。",
    "combine_mode": "控制内置评分与外部评估器评分的组合方式。",
    "weight": "blend 模式下外部评估器的权重。",
    "context_json": "传给外部评估器的可选 JSON 对象。",
    "merge_strategy": "选择自动、TIES、Consensus、SLERP、加权平均或直接加法。",
    "prev_hooks": "可选连接已有 Hook 链。",
    "wan_model": "连接 WanVideo 模型。",
    "top_n": "进入真实合并评估的最高排名候选数量。",
    "scoring_svd": "控制评分阶段是否使用合并质量或有效秩 SVD 指标。",
    "scoring_device": "选择评分计算设备。",
    "evaluator": "可选连接外部评估器，与内置指标共同评分。",
    "community_cache": "控制 Hugging Face 社区缓存的上传和下载行为。",
    "diff_cache_mode": "控制候选之间的 LoRA 差分缓存位置和策略。",
    "diff_cache_ram_pct": "auto 模式可用于差分缓存的空闲系统内存比例。",
    "scoring_speed": "控制每个候选评分的目标层采样密度。",
    "scoring_formula": "选择 AutoTuner 第二阶段使用的评分公式版本。",
    "output_mode": "merge 输出最佳合并；tuning_only 仅输出设置并透传基础模型。",
    "memory_mode": "跨会话复用、只读或清除 AutoTuner 的持久结果。",
    "selection": "选择要应用的候选排名，1 表示第一名。",
    "record_dataset": "将完整分析与候选评分记录为阈值研究数据集。",
    "overwrite": "允许覆盖已有文件；关闭时自动追加序号。",
    "tuner_data_file": "选择之前保存的调优数据文件。",
    "create_nodes": "根据兼容性分析结果自动创建 LoRA Stack 和加载节点。",
    "star_eta": "每个 LoRA 的 STAR 频谱保留率；100 表示关闭。",
    "tame_layers": "限制异常高能量层的强度；0 表示关闭。",
    "tame_threshold": "相对基础权重范数超过此比例时将层视为过热。",
    "merge_settings": "连接共享合并设置；未连接时使用推荐默认值。",
    "formula": "使用 1 开始的 LoRA 序号、+、括号和可选权重定义分层合并顺序。",
    "base_model": "LoRA 微调前的原始基础模型。",
    "finetuned_model": "已经烘焙微调差分的模型。",
    "rank": "SVD 分解的最大秩。",
    "rank_mode": "auto 按能量选择秩；fixed 固定使用指定秩。",
    "energy_threshold": "auto 秩模式需要保留的差分能量比例。",
    "base_clip": "可选基础 CLIP；与微调 CLIP 同时连接可提取文本编码器差分。",
    "finetuned_clip": "微调后的 CLIP，必须与基础 CLIP 同时连接。",
    "shuffle_order": "组合顺序的确定性随机种子。",
    "combo_size": "生成两两组合、三项组合或同时生成两者。",
    "folder_filter": "用逗号分隔的相对路径前缀筛选 LoRA。",
    "rerun_mode": "将组合重新运行到独立进度文件，供一次性回填使用。",
    "rerun_source": "选择全部打乱组合或仅重放原进度文件中的组合。",
    "k": "近邻估算使用的 K 值。",
    "rebuild_index": "控制组合索引自动、强制重建或跳过重建。",
    "top_n_output": "输出最高排名组合的数量。",
}

ZH_OPTIONS = {
    "all": "全部", "low_conflict": "低冲突", "high_conflict": "高冲突",
    "shared_only": "仅共享键", "unique_only": "仅独有键", "audio_only": "仅音频层", "no_audio": "仅非音频层",
    "simple": "简单", "advanced": "高级", "dropdown": "下拉选择", "text": "文本输入", "All": "全部",
    "enabled": "启用", "disabled": "禁用", "per_prefix": "按层", "global": "全局", "additive": "直接加法",
    "smart": "智能", "aggressive": "激进", "gpu": "GPU", "cpu": "CPU", "dare": "DARE",
    "della": "DELLA", "dare_conflict": "DARE（仅冲突）", "della_conflict": "DELLA（仅冲突）",
    "none": "无", "refine": "精炼", "full": "完整", "no_slerp": "排除 SLERP", "basic": "基础",
    "auto": "自动", "sd_unet": "SD/SDXL UNet", "dit": "DiT", "acestep_dit": "ACE-Step DiT", "llm": "LLM",
    "manual": "手动", "from_autotuner": "来自 AutoTuner", "from_tuner_data": "来自调优数据",
    "blend": "混合", "external_only": "仅外部评分", "multiply": "相乘", "ties": "TIES",
    "consensus": "Consensus", "slerp": "SLERP", "weighted_average": "加权平均", "weighted_sum": "直接加法",
    "merge_quality": "合并质量", "lora_rank": "LoRA 有效秩", "upload_only": "仅上传",
    "upload_and_download": "上传并下载", "ram": "内存", "disk": "磁盘", "fast": "快速",
    "turbo": "极速", "turbo+": "极速+", "v2": "v2（推荐）", "v1": "v1（旧版）",
    "merge": "执行合并", "tuning_only": "仅调优", "auto_ignore_strength": "自动（忽略强度）",
    "read_only": "只读", "clear_and_run": "清除并重跑", "fixed": "固定", "2": "两项",
    "3": "三项", "2_and_3": "两项和三项", "shuffle": "打乱顺序", "original_progress": "原进度记录",
    "force": "强制重建", "skip": "跳过", "None": "无",
}

ZH_TITLES = {
    "LoRAStack": "LoRA 堆栈", "LoRAStackDynamic": "LoRA 堆栈（动态）",
    "LoRAOptimizer": "LoRA 优化器（旧版）", "LoRAOptimizerSimple": "LoRA 优化器",
    "LoRAInlineChainOptions": "LoRA Inline 链式选项", "LoRAOptimizerInline": "LoRA 优化器（Inline 链）",
    "SaveMergedLoRA": "保存合并 LoRA", "BuildAutoTunerPythonEvaluator": "构建 AutoTuner Python 评估器",
    "LoRAConflictEditor": "LoRA 冲突编辑器", "MergedLoRAToHook": "合并 LoRA 转 Hook",
    "MergedLoRAToWanVideo": "（开发中）合并 LoRA 转 WanVideo", "WanVideoLoRAOptimizer": "（开发中）WanVideo LoRA 优化器",
    "LoRAAutoTuner": "LoRA 自动调优器", "LoRAMergeSelector": "LoRA 合并选择器",
    "SaveTunerData": "保存调优数据", "LoadTunerData": "加载调优数据",
    "LoRACompatibilityAnalyzer": "LoRA 兼容性分析器", "LoRAMergeSettings": "LoRA 合并设置",
    "LoRAOptimizerSettings": "LoRA 优化器设置", "LoRAAutoTunerSettings": "LoRA 自动调优设置",
    "LoRAMetadataReader": "LoRA 元数据读取器", "LoRAMergeFormula": "LoRA 合并公式",
    "LoRAExtractFromModel": "从模型提取 LoRA", "LoRACombinationGenerator": "LoRA 组合生成器",
    "LoRAEstimator": "LoRA 合并估算器",
}

ZH_DESCRIPTIONS = {
    "LoRAStack": "将一个 LoRA 及其强度、冲突和筛选规则加入堆栈。",
    "LoRAStackDynamic": "在单个节点中配置最多十个 LoRA，并支持简单/高级和下拉/文本输入。",
    "LoRAOptimizer": "使用全部高级参数分析并合并 LoRA 堆栈。",
    "LoRAOptimizerSimple": "使用推荐默认值和可选设置节点自动分析、合并并应用 LoRA。",
    "LoRAInlineChainOptions": "按 Inline 加载链顺序为每个捕获的 LoRA 设置独立选项。",
    "LoRAOptimizerInline": "捕获已应用到模型的 LoRA 链，重新优化后一次性应用。",
    "SaveMergedLoRA": "将优化器输出保存为可复用的 safetensors LoRA。",
    "BuildAutoTunerPythonEvaluator": "接入本地 Python 评估函数，为 AutoTuner 补充外部评分。",
    "LoRAConflictEditor": "手动覆盖每个 LoRA 的冲突规则和目标合并策略。",
    "MergedLoRAToHook": "将合并 LoRA 数据转换为 ComfyUI Hook。",
    "MergedLoRAToWanVideo": "将合并 LoRA 数据应用到 WanVideo 模型。",
    "WanVideoLoRAOptimizer": "为 WanVideo 模型提供旧版高级优化入口。",
    "LoRAAutoTuner": "扫描候选设置、评分并输出最佳合并或调优数据。",
    "LoRAMergeSelector": "从已有 AutoTuner 结果中选择并应用指定排名配置。",
    "SaveTunerData": "把 AutoTuner 结果保存到用户目录。",
    "LoadTunerData": "加载已保存的 AutoTuner 结果和元数据。",
    "LoRACompatibilityAnalyzer": "分析 LoRA 之间的架构、键覆盖和冲突兼容性，并可自动创建分组节点。",
    "LoRAMergeSettings": "集中配置多个优化器共享的合并基础参数。",
    "LoRAOptimizerSettings": "集中配置 LoRA Optimizer 的高级行为。",
    "LoRAAutoTunerSettings": "集中配置 AutoTuner 的搜索、评分、缓存与输出。",
    "LoRAMetadataReader": "读取堆栈内 LoRA 的提示词、描述和元数据。",
    "LoRAMergeFormula": "用公式定义 LoRA 的分层合并顺序和分组权重。",
    "LoRAExtractFromModel": "比较基础模型与微调模型，并用 SVD 提取 LoRA。",
    "LoRACombinationGenerator": "按确定性顺序生成 LoRA 两项或三项组合。",
    "LoRAEstimator": "用索引和近邻指标快速估算 LoRA 组合排名。",
}

EN_DESCRIPTIONS = {
    key: value for key, value in {
        "LoRAStack": "Add one LoRA and its strength, conflict, and key-filter rules to a stack.",
        "LoRAStackDynamic": "Configure up to ten LoRAs in one node with simple/advanced and dropdown/text modes.",
        "LoRAOptimizer": "Analyze and merge a LoRA stack with all legacy advanced controls exposed.",
        "LoRAOptimizerSimple": "Analyze, merge, and apply a LoRA stack using recommended defaults and optional settings nodes.",
        "LoRAInlineChainOptions": "Configure each captured LoRA by its position in an inline loader chain.",
        "LoRAOptimizerInline": "Capture an already-applied LoRA chain, optimize it, and apply the merged result once.",
        "SaveMergedLoRA": "Save optimizer output as a reusable safetensors LoRA.",
        "BuildAutoTunerPythonEvaluator": "Connect a local Python evaluator to supplement AutoTuner scoring.",
        "LoRAConflictEditor": "Override per-LoRA conflict rules and the target merge strategy.",
        "MergedLoRAToHook": "Convert merged LoRA data to a ComfyUI hook.",
        "MergedLoRAToWanVideo": "Apply merged LoRA data to a WanVideo model.",
        "WanVideoLoRAOptimizer": "Legacy advanced optimizer entry point for WanVideo models.",
        "LoRAAutoTuner": "Sweep candidate settings, score them, and output the best merge or tuning data.",
        "LoRAMergeSelector": "Select and apply a ranked configuration from existing AutoTuner data.",
        "SaveTunerData": "Save AutoTuner results to the user directory.",
        "LoadTunerData": "Load saved AutoTuner results and metadata.",
        "LoRACompatibilityAnalyzer": "Analyze architecture, key overlap, and conflict compatibility, with optional node creation.",
        "LoRAMergeSettings": "Share common merge parameters across optimizer nodes.",
        "LoRAOptimizerSettings": "Configure advanced LoRA Optimizer behavior.",
        "LoRAAutoTunerSettings": "Configure AutoTuner search, scoring, caching, and output behavior.",
        "LoRAMetadataReader": "Read prompts, descriptions, and metadata from a LoRA stack.",
        "LoRAMergeFormula": "Define hierarchical LoRA merge order and group weights with a formula.",
        "LoRAExtractFromModel": "Compare a base and finetuned model and extract their difference as a LoRA with SVD.",
        "LoRACombinationGenerator": "Generate deterministic two- or three-LoRA combinations.",
        "LoRAEstimator": "Estimate LoRA combination rankings with an index and nearest-neighbor metrics.",
    }.items()
}


_ACRONYMS = {
    "cpu": "CPU", "dare": "DARE", "della": "DELLA", "gpu": "GPU",
    "hf": "HF", "id": "ID", "json": "JSON", "lora": "LoRA",
    "qkv": "QKV", "slerp": "SLERP", "star": "STAR", "svd": "SVD",
    "tall": "TALL", "tame": "TAME", "ties": "TIES", "vram": "VRAM",
}


def pretty_name(name: str) -> str:
    return " ".join(_ACRONYMS.get(part.lower(), part.capitalize())
                    for part in name.split("_"))


def localized_name(name: str, language: str) -> str:
    match = re.match(r"(.+?)_(\d+)$", name)
    base = match.group(1) if match else name
    index = match.group(2) if match else None
    value = ZH_NAMES.get(base, pretty_name(base)) if language == "zh" else pretty_name(base)
    return f"{value} #{index}" if index else value


def localized_tooltip(name: str, english: str | None, language: str) -> str | None:
    match = re.match(r"(.+?)_(\d+)$", name)
    base = match.group(1) if match else name
    index = match.group(2) if match else None
    if language == "en":
        return english
    tooltip = ZH_TOOLTIPS.get(base)
    if tooltip and index:
        return f"LoRA #{index}：{tooltip}"
    return tooltip or (f"设置{localized_name(name, language)}。" if english else None)


def option_labels(name: str, values: object, language: str) -> dict[str, str] | None:
    if re.sub(r"_\d+$", "", name) in {"lora_name", "tuner_data_file", "save_folder"}:
        return None
    if not isinstance(values, (list, tuple)) or not values or len(values) > 50:
        return None
    labels = {}
    for value in values:
        if not isinstance(value, (str, int, float, bool)):
            continue
        raw = str(value)
        labels[raw] = ZH_OPTIONS.get(raw, pretty_name(raw)) if language == "zh" else pretty_name(raw)
    return labels or None


def build(language: str) -> dict[str, object]:
    result = {}
    for node_id, node_class in module.NODE_CLASS_MAPPINGS.items():
        input_types = node_class.INPUT_TYPES()
        inputs = {}
        for section in ("required", "optional"):
            for name, schema in input_types.get(section, {}).items():
                metadata = schema[1] if isinstance(schema, tuple) and len(schema) > 1 and isinstance(schema[1], dict) else {}
                entry = {"name": localized_name(name, language)}
                tooltip = localized_tooltip(name, metadata.get("tooltip"), language)
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
            "description": ZH_DESCRIPTIONS.get(node_id, "") if language == "zh" else EN_DESCRIPTIONS.get(node_id, ""),
            "inputs": inputs,
            "outputs": outputs,
        }
    return result


for language in ("en", "zh"):
    destination = ROOT / "locales" / language / "nodeDefs.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build(language), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination.relative_to(ROOT))
