import { app } from "/scripts/app.js";

const MESSAGES = {
    en: {
        swapToAutoTuner: "Swap to LoRA AutoTuner",
        swapToOptimizer: "Swap to LoRA Optimizer",
        removeLora: "Remove LoRA #{index}",
        moveLoraUp: "Move LoRA #{index} up",
        analyzerSolo: "Solo (Analyzer)",
        analyzerGroup: "Group {index} (Analyzer)",
    },
    zh: {
        swapToAutoTuner: "切换为 LoRA 自动调优器",
        swapToOptimizer: "切换为 LoRA 优化器",
        removeLora: "移除 LoRA #{index}",
        moveLoraUp: "上移 LoRA #{index}",
        analyzerSolo: "独立组（兼容性分析）",
        analyzerGroup: "合并组 {index}（兼容性分析）",
    },
};

export function getLocale() {
    const value = app.ui?.settings?.getSettingValue?.("Comfy.Locale") || "en";
    return String(value).toLowerCase().startsWith("zh") ? "zh" : "en";
}

export function t(key, params = {}) {
    const template = MESSAGES[getLocale()]?.[key] || MESSAGES.en[key] || key;
    return template.replace(/\{(\w+)\}/g, (_, name) => String(params[name] ?? `{${name}}`));
}
