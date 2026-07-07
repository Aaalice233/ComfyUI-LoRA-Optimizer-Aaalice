import { app } from "/scripts/app.js";

const HIDDEN_TAG = "loraopt_hidden";
const origProps = {};

const MAX_LORAS = 10;

// Per-slot widgets, in INPUT_TYPES declaration order.
const SLOT_WIDGET_NAMES = [
    "enabled",
    "strength",
    "model_strength",
    "clip_strength",
    "conflict_mode",
    "key_filter",
    "preserve",
];

// --- Widget layout (NO migration function — nothing to migrate yet) ---
// This node shipped with the layout below, so no legacy workflows exist. BUT:
// the litegraph workflow format restores widgets_values POSITIONALLY, so any
// FUTURE per-slot widget addition/removal/reorder WILL desync every saved
// workflow (values land in the wrong widgets) unless a configure() migration
// re-pads old saves to the new layout — see migrateWidgetsValues() in
// lora_stack_dynamic.js for the pattern. Baseline layout constants for that
// future migration:
//   CONTROL_WIDGET_COUNT = 2   // settings_visibility, lora_count
//   PER_SLOT = 7               // SLOT_WIDGET_NAMES above, x 10 slots
//   TRAILING_WIDGET_COUNT = 0
// (This node is pure widgets — chain_options is its OUTPUT, not a widget.)

function toggleWidget(node, widget, show, suffix = "") {
    if (!widget) return;

    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
        };
    }

    widget.hidden = !show;
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show
        ? origProps[widget.name].origComputeSize
        : () => [0, -4];

    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            toggleWidget(node, w, show, ":" + widget.name);
        }
    }
}

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function interceptWidgetValue(widget, onChange) {
    let widgetValue = widget.value;
    const desc =
        Object.getOwnPropertyDescriptor(widget, "value") ||
        Object.getOwnPropertyDescriptor(
            Object.getPrototypeOf(widget),
            "value"
        );

    Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get() {
            return desc?.get ? desc.get.call(widget) : widgetValue;
        },
        set(newVal) {
            if (desc?.set) {
                desc.set.call(widget, newVal);
            } else {
                widgetValue = newVal;
            }
            onChange(newVal);
        },
    });
}

function updateVisibility(node) {
    const settingsVisWidget = findWidget(node, "settings_visibility");
    const countWidget = findWidget(node, "lora_count");
    if (!settingsVisWidget || !countWidget) return;

    const isSimple = settingsVisWidget.value === "simple";
    const count = countWidget.value;

    for (let i = 1; i <= MAX_LORAS; i++) {
        const visible = i <= count;
        for (const base of SLOT_WIDGET_NAMES) {
            let show;
            if (base === "enabled") {
                show = visible; // always shown for active slots
            } else if (base === "strength") {
                show = visible && isSimple; // simple: single multiplier
            } else {
                show = visible && !isSimple; // advanced: split multipliers + knobs
            }
            toggleWidget(node, findWidget(node, `${base}_${i}`), show);
        }
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
    app.canvas?.setDirty?.(true, true);
}

// --- Node Registration ---

app.registerExtension({
    name: "LoRAOptimizer.LoRAInlineChainOptions",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAInlineChainOptions") return;

        // Re-apply visibility after workflow restore: configure() overwrites
        // settings_visibility / lora_count with the saved values.
        const origConfigure = node.configure;
        node.configure = function (info) {
            const r = origConfigure
                ? origConfigure.apply(this, arguments)
                : undefined;
            updateVisibility(this);
            return r;
        };

        // Intercept settings_visibility and lora_count changes to update visibility
        for (const w of node.widgets || []) {
            if (w.name === "settings_visibility" || w.name === "lora_count") {
                interceptWidgetValue(w, () => updateVisibility(node));
            }
        }

        // Initial visibility update — delay to ensure widgets are fully initialized
        setTimeout(() => updateVisibility(node), 100);
    },
});
