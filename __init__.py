"""ComfyUI-LoRA-Optimizer — auto-optimizer for multi-LoRA merging."""

try:
    from .lora_optimizer import (NODE_CLASS_MAPPINGS,
                                 NODE_DISPLAY_NAME_MAPPINGS,
                                 _install_lora_name_stamp)
except ImportError:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    _install_lora_name_stamp = None

# Wrap nodes.LoraLoader.load_lora so stock + rgthree LoRA loads stamp their
# real filenames onto the model/clip — the inline optimizer reads them back
# to show real names, attribute per-LoRA, and reconcile with the file-based
# community dataset. Fail-safe: never break node registration if comfy's
# `nodes` module isn't importable yet (idempotent — safe to call repeatedly).
if _install_lora_name_stamp is not None:
    try:
        _install_lora_name_stamp()
    except Exception:
        pass

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
