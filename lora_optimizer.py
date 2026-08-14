"""
ComfyUI-LoRA-Optimizer
Auto-optimizer node for combining multiple LoRAs via diff-based merging
with TIES conflict resolution and automatic parameter selection.
"""

import torch
import logging
import math
import os
import sys
import json
import hashlib
import zlib
import functools
import weakref
import time
import re
import gc
import importlib
import importlib.util
import concurrent.futures
import threading
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
import comfy.model_management
from comfy.weight_adapter.lora import LoRAAdapter
try:
    from comfy.weight_adapter.lokr import LoKrAdapter
except Exception:
    class LoKrAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights

try:
    from comfy.weight_adapter.loha import LoHaAdapter
except Exception:
    class LoHaAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights
from safetensors import safe_open
from safetensors.torch import save_file
try:
    from .chunked_merge import ExecutionPlanner, InterruptController, chunked_randomized_svd
    from .chunked_optimizer import ChunkedOptimizerMixin
    from .persistent_cache import PersistentCacheUnsupported, PersistentPatchCache
except ImportError:
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    if _MODULE_DIR not in sys.path:
        sys.path.insert(0, _MODULE_DIR)
    from chunked_merge import ExecutionPlanner, InterruptController, chunked_randomized_svd
    from chunked_optimizer import ChunkedOptimizerMixin
    from persistent_cache import PersistentCacheUnsupported, PersistentPatchCache

# --- Triton SVD kernel (optional) ---
# Set LORA_OPTIMIZER_DISABLE_TRITON=1 to skip the bundled Triton SVD kernel and
# fall back to torch.linalg. The kernel can hard-crash the process (e.g.
# "__triton_launcher.c" abort, no Python traceback) on some GPU/driver/torch
# combinations — this is the escape hatch when that happens.
_kernel_path = None
_DISABLE_TRITON = os.environ.get("LORA_OPTIMIZER_DISABLE_TRITON", "").strip().lower() in ("1", "true", "yes", "on")
if _DISABLE_TRITON:
    _batched_svd = None
    _batched_procrustes = None
    _HAS_SVD_KERNEL = False
    _HAS_TRITON = False
    logging.info("[LoRA Optimizer] Triton SVD kernel disabled via LORA_OPTIMIZER_DISABLE_TRITON; using torch.linalg fallback")
else:
    try:
        _kernel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")
        _kernel_spec = importlib.util.spec_from_file_location("lora_optimizer_kernel", _kernel_path)
        _kernel_mod = importlib.util.module_from_spec(_kernel_spec)
        _kernel_spec.loader.exec_module(_kernel_mod)
        _batched_svd = _kernel_mod.batched_svd
        _batched_procrustes = _kernel_mod.batched_procrustes
        _HAS_SVD_KERNEL = True
        _HAS_TRITON = _kernel_mod.HAS_TRITON
        logging.info(f"[LoRA Optimizer] SVD kernel loaded (Triton={_HAS_TRITON})")
    except Exception as e:
        _batched_svd = None
        _batched_procrustes = None
        _HAS_SVD_KERNEL = False
        _HAS_TRITON = False
        if _kernel_path and os.path.exists(_kernel_path):
            logging.warning(f"[LoRA Optimizer] kernel.py found but failed to load: {e}")


# Pass-2 profiling is diagnostic-only because CUDA synchronization adds overhead.
_PROFILE_MERGE = os.environ.get(
    "LORA_OPTIMIZER_PROFILE_MERGE", "").strip().lower() in (
        "1", "true", "yes", "on")


def _merge_profiling_enabled():
    return _PROFILE_MERGE


ANALYSIS_CACHE_VERSION = "1.9.0"


def _throw_if_processing_interrupted():
    checker = getattr(
        comfy.model_management,
        "throw_exception_if_processing_interrupted",
        None)
    if checker is not None:
        checker()


def _read_safetensors_metadata(filepath):
    """Read metadata header from a safetensors file without loading tensors."""
    try:
        if not filepath.endswith(".safetensors"):
            return {}
        with safe_open(filepath, framework="pt") as f:
            return dict(f.metadata()) if f.metadata() else {}
    except Exception:
        return {}


def _resolve_safe_output_path(base_dir, filename, suffix, label):
    """Resolve a save path under `base_dir`, allowing subdirectories but blocking traversal."""
    if filename is None:
        raise ValueError(f"[{label}] Filename is required.")
    base_dir_real = os.path.realpath(base_dir)
    base_name = filename if filename.endswith(suffix) else f"{filename}{suffix}"
    candidate = os.path.realpath(os.path.join(base_dir_real, base_name))
    try:
        if os.path.commonpath([base_dir_real, candidate]) != base_dir_real:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"[{label}] Path escapes base directory: {filename}") from exc
    return candidate


# ---------------------------------------------------------------------------
# Architecture-aware threshold presets
# ---------------------------------------------------------------------------
# Each preset contains numeric thresholds used by density estimation, conflict
# detection, auto-strength, and scoring heuristics.  The preset is selected
# based on detected model architecture (or manual override).
_ARCH_PRESETS = {
    "sd_unet": {
        "density_noise_floor_ratio": 0.1,
        "density_clamp_min": 0.1,
        "density_clamp_max": 0.9,
        "dare_ideal_density": 0.7,
        "consensus_cos_sim_min": 0.5,
        "consensus_conflict_max": 0.15,
        "orthogonal_cos_sim_max": 0.25,
        "orthogonal_conflict_max": 0.60,
        "ties_conflict_threshold": 0.25,
        "magnitude_ratio_total_sign": 2.0,
        "alignment_threshold": 0.1,
        "suggested_max_strength_cap": 3.0,
        "auto_strength_orthogonal_floor": 0.85,
        "display_name": "SD/SDXL UNet",
        "full_rank": {
            "rank_threshold": 512,
            "disable_slerp_upgrade": True,
            "prefer_sum_orthogonal": True,
            "auto_strength_floor": 1.0,
        },
    },
    "dit": {
        "density_noise_floor_ratio": 0.05,
        "density_clamp_min": 0.4,
        "density_clamp_max": 0.95,
        "dare_ideal_density": 0.8,
        "consensus_cos_sim_min": 0.5,
        "consensus_conflict_max": 0.15,
        "orthogonal_cos_sim_max": 0.25,
        "orthogonal_conflict_max": 0.60,
        "ties_conflict_threshold": 0.25,
        "magnitude_ratio_total_sign": 2.0,
        "alignment_threshold": 0.1,
        "suggested_max_strength_cap": 5.0,
        "auto_strength_orthogonal_floor": 0.85,
        # Per-prefix max/min Frobenius ratio above which the weighted_average→SLERP
        # upgrade is suppressed (SLERP washes out the dominant LoRA on imbalanced
        # orthogonal pairs). Orthogonal pairs above this route to additive instead.
        "slerp_imbalance_ratio": 2.0,
        "display_name": "DiT (Flux/WAN/Z-Image/LTX/Ideogram-4/Krea-2/HunyuanVideo)",
        "full_rank": {
            "rank_threshold": 512,
            "disable_slerp_upgrade": True,
            "prefer_sum_orthogonal": True,
            "auto_strength_floor": 1.0,
        },
    },
    "llm": {
        "density_noise_floor_ratio": 0.15,
        "density_clamp_min": 0.1,
        "density_clamp_max": 0.8,
        "dare_ideal_density": 0.5,
        "consensus_cos_sim_min": 0.5,
        "consensus_conflict_max": 0.15,
        "orthogonal_cos_sim_max": 0.25,
        "orthogonal_conflict_max": 0.60,
        "ties_conflict_threshold": 0.25,
        "magnitude_ratio_total_sign": 2.0,
        "alignment_threshold": 0.1,
        "suggested_max_strength_cap": 3.0,
        "auto_strength_orthogonal_floor": 0.9,
        "display_name": "LLM (Qwen/LLaMA)",
        "full_rank": {
            "rank_threshold": 512,
            "disable_slerp_upgrade": True,
            "prefer_sum_orthogonal": True,
            "auto_strength_floor": 1.0,
        },
    },
    # ACE-Step v1.5: 24-block Linear DiT for music generation.
    # Cross-attention is the primary voice-identity pathway — it always uses
    # full/global attention (never sliding window) to attend to concatenated
    # [text + lyrics + timbre] conditioning.  Self-attention alternates:
    # odd layers = sliding window (local timbre/transients), even layers =
    # global GQA (long-range musical structure).
    # Vocal + music LoRAs typically produce orthogonal updates in cross-attn
    # (encoding voice vs genre conditioning), so TIES trimming is destructive.
    # Tuned for: wider orthogonal band (→ more SLERP), higher TIES threshold
    # (→ less aggressive trimming), full magnitude preservation.
    "acestep_dit": {
        "density_noise_floor_ratio": 0.05,
        "density_clamp_min": 0.4,
        "density_clamp_max": 0.95,
        "dare_ideal_density": 0.85,
        "consensus_cos_sim_min": 0.5,
        "consensus_conflict_max": 0.15,
        "orthogonal_cos_sim_max": 0.30,
        "orthogonal_conflict_max": 0.65,
        "ties_conflict_threshold": 0.35,
        "magnitude_ratio_total_sign": 2.0,
        "alignment_threshold": 0.1,
        "suggested_max_strength_cap": 5.0,
        "auto_strength_orthogonal_floor": 1.0,
        "display_name": "ACE-Step (Music DiT)",
        "full_rank": {
            "rank_threshold": 512,
            "disable_slerp_upgrade": True,
            "prefer_sum_orthogonal": True,
            "auto_strength_floor": 1.0,
        },
    },
}

_VIDEO_ARCH_ORTHOGONAL_FLOOR = {"wan": 1.0, "ltx": 1.0, "acestep": 1.0}

_ARCH_TO_PRESET = {
    "sdxl": "sd_unet", "sd15": "sd_unet", "unknown": "sd_unet",
    "flux": "dit", "wan": "dit", "zimage": "dit", "ltx": "dit",
    "ideogram4": "dit", "anima": "dit", "krea2": "dit",
    "acestep": "acestep_dit",
    "qwen_image": "llm",
}

# comfy model class (type(model.model).__name__) -> this repo's arch name.
# Used ONLY for virtual adapter payloads: architecture cannot always be recovered
# from model-space keys alone (attention-only Qwen-Image and ACE-Step v1.0 both
# surface as transformer_blocks.N.attn.to_q). The caller supplies the real MODEL
# object, and ComfyUI assigns each supported architecture a distinct
# model_base class, so its class name is authoritative. Verified against
# comfy/model_base.py (BaseModel subclasses). Only base classes are listed —
# subclasses (WAN22, WAN21_Vace, Flux2, LongCatImage, ZImagePixelSpace, …)
# resolve through their MRO in _model_class_arch. Unmapped classes -> None ->
# key-based fallback. Note: Z-Image loads as comfy's Lumina2 / ZImagePixelSpace
# (there is no separate Z-Image class); both share NextDiT with real Lumina2
# and resolve to the same 'dit' preset, so mapping Lumina2 -> zimage is safe.
# SD 1.5 is intentionally omitted: comfy loads it on BaseModel itself, which is
# every model's base — mapping it would mis-tag everything.
_MODEL_CLASS_TO_ARCH = {
    "QwenImage": "qwen_image",
    "ACEStep": "acestep",
    "ACEStep15": "acestep",
    "LTXV": "ltx",
    "LTXAV": "ltx",
    "Flux": "flux",
    "WAN21": "wan",
    "Lumina2": "zimage",
    "Ideogram4": "ideogram4",
    "Anima": "anima",
    "Krea2": "krea2",
    "SDXL": "sdxl",
}


def _resolve_arch_preset(arch_override, detected_arch):
    """Resolve architecture preset from override or detected architecture."""
    if arch_override and arch_override != "auto" and arch_override in _ARCH_PRESETS:
        key = arch_override
    else:
        key = _ARCH_TO_PRESET.get(detected_arch, "sd_unet")
    return key, _ARCH_PRESETS[key]


class _LoRAMergeBase(ChunkedOptimizerMixin):
    """
    Base class for diff-based LoRA merging.

    Computes full weight diffs (Up @ Down x alpha) for LoRAs of any rank,
    then merges the diffs. Supports TIES-Merging (NeurIPS 2023) for
    resolving sign conflicts.

    Not a registered ComfyUI node — subclassed by _LoRAOptimizerEngine.
    """

    def __init__(self):
        self.loaded_loras = {}
        self._lora_format_cache = {}  # id(lora_dict) -> format_index (0-3)
        self._cpu_fallback_reported = False
        self._tiled_gpu_reported = False
        self._execution_planner = ExecutionPlanner()
        self._interrupt_controller = None
        self._execution_stats = None
        self._progress_state = None

    def _progress_update(self, stage, target_key, fraction):
        state = self._progress_state
        if state is None:
            return
        key = (stage, repr(target_key))
        fraction = max(0.0, min(1.0, float(fraction)))
        with state["lock"]:
            previous = state["targets"].get(key, 0.0)
            if fraction <= previous:
                return
            state["targets"][key] = fraction
            units = int(round((fraction - previous) * 1000))
            if units > 0:
                state["value"] += units
                state["bar"].update(units)

    def _progress_decision(self):
        state = self._progress_state
        if state is None:
            return
        with state["lock"]:
            if not state["decision"]:
                state["decision"] = True
                state["value"] += 1000
                state["bar"].update(1000)

    def _progress_finish(self):
        state = self._progress_state
        if state is None:
            return
        with state["lock"]:
            remaining = state["total"] - state["value"]
            if remaining > 0:
                state["bar"].update(remaining)
                state["value"] = state["total"]

    def _interrupt_check(self):
        if self._interrupt_controller is None:
            self._interrupt_controller = InterruptController()
        self._interrupt_controller.check()

    def _track_model_identity(self, model, clip=None):
        """
        Returns True when the cached model/clip identity no longer matches the
        given objects, then records the new identity. Uses weakrefs on top of
        id(): a bare id() comparison can alias a NEW object that happens to
        reuse a freed object's address, silently serving stale cached merges.
        """
        current_mid = id(model) if model is not None else None
        current_cid = id(clip) if clip is not None else None
        prev_mid = getattr(self, '_cached_model_id', None)
        prev_cid = getattr(self, '_cached_clip_id', None)
        prev_mref = getattr(self, '_cached_model_ref', None)
        prev_cref = getattr(self, '_cached_clip_ref', None)

        changed = False
        if prev_mid is not None:
            if current_mid != prev_mid:
                changed = True
            elif prev_mref is not None and prev_mref() is not model:
                changed = True  # same id, old object gone — address reuse
        if not changed and prev_cid is not None:
            if current_cid != prev_cid:
                changed = True
            elif prev_cref is not None and prev_cref() is not clip:
                changed = True

        def _ref(obj):
            if obj is None:
                return None
            try:
                return weakref.ref(obj)
            except TypeError:
                return None

        self._cached_model_id = current_mid
        self._cached_clip_id = current_cid
        self._cached_model_ref = _ref(model)
        self._cached_clip_ref = _ref(clip)
        return changed

    @staticmethod
    def _get_compute_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _select_group_compute_device(self, device, target_shape, n_diffs):
        """Keep oversized dense target groups off the GPU before they can OOM."""
        if device is None or device.type == "cpu" or not target_shape:
            return device

        dense_bytes = math.prod(target_shape) * torch.float32.itemsize
        # Analysis retains every contributor. Multi-LoRA SLERP may additionally
        # need a packed copy plus working vectors, so budget for its true peak.
        required_bytes = dense_bytes * max(n_diffs + 2, 2 * n_diffs + 3)
        free_bytes = comfy.model_management.get_free_memory(device)
        if required_bytes <= free_bytes * 0.85:
            return device

        if not self._cpu_fallback_reported:
            logging.warning(
                "[LoRA Optimizer] Oversized target groups exceed available VRAM "
                "(estimated %.1f GiB, free %.1f GiB); processing them on CPU.",
                required_bytes / (1024 ** 3), free_bytes / (1024 ** 3))
            self._cpu_fallback_reported = True
        return torch.device("cpu")

    @staticmethod
    def _model_class_arch(model):
        """Authoritative architecture from the comfy MODEL class, for
        virtual adapter payloads only. type(model.model).__name__ (walking the
        MRO so subclasses resolve via their base) maps to this repo's arch name
        via _MODEL_CLASS_TO_ARCH. Disambiguates architectures that model-space
        keys cannot (attention-only Qwen-Image vs ACE-Step v1.0). Returns None
        for a missing model, a model without an inner .model, or an unmapped
        class — the caller then falls back to key-based detection. Never raises.
        """
        try:
            inner = getattr(model, "model", None)
            if inner is None:
                return None
            for cls in type(inner).__mro__:
                arch = _MODEL_CLASS_TO_ARCH.get(cls.__name__)
                if arch is not None:
                    return arch
        except Exception:
            return None
        return None

    @staticmethod
    def _detect_architecture(lora_sd):
        """
        Detect model architecture from LoRA key patterns.
        Returns: 'zimage', 'flux', 'wan', 'acestep', 'sdxl', 'sd15', 'ltx',
        'qwen_image', 'ideogram4', 'anima', 'krea2', or 'unknown'.
        """
        keys = list(lora_sd.keys())
        keys_str = ' '.join(k.lower() for k in keys)

        # Ideogram 4: NextDiT-family single-stream DiT with the same
        # layers.N.attention.qkv pattern as Z-Image (Lumina2) — MUST be
        # checked first. Discriminators: attention output proj is
        # "attention.o" (Z-Image: "attention.out"), modulation is lowercase
        # "adaln_modulation" (Z-Image: "adaLN_modulation"), the fal trainer
        # uses a conditional_transformer. prefix, and the fused qkv up/B
        # matrix has 13824 rows (3x4608; Z-Image Turbo: 6912 = 3x2304).
        if any('conditional_transformer.layers.' in k for k in keys):
            return 'ideogram4'
        if any(re.search(r'layers[._]\d+[._]attention[._]o(?=[._])', k) for k in keys):
            return 'ideogram4'
        if (any('adaln_modulation' in k for k in keys)
                and any('feed_forward' in k for k in keys)):
            return 'ideogram4'
        for k in keys:
            if (re.search(r'layers[._]\d+[._]attention[._]qkv', k)
                    and ('lora_B' in k or 'lora_up' in k or '.lora.up.' in k)):
                shape = getattr(lora_sd.get(k), 'shape', None)
                if shape is not None and len(shape) >= 1 and shape[0] == 13824:
                    return 'ideogram4'

        # Krea 2 (krea/Krea-2) — from-scratch single-stream image DiT (NOT the
        # older FLUX.1-Krea finetune). The unique marker across every trainer
        # form is a per-attention sigmoid GATE projection — attn.to_gate (the
        # "krea_2" trainer + diffusers forms) / attn.gate / attn.w{q,k,v,o} (the
        # Comfy-Org native form). No other supported arch has an attention gate,
        # so this MUST run before the qwen (transformer.transformer_blocks), flux,
        # and ACE-Step (transformer_blocks.N.attn.to_q) branches would otherwise
        # claim its diffusers-style keys. Verified vs 3 real LoRA forms:
        #   diffusion_model.blocks.N.attn.wq            (Comfy-Org native)
        #   diffusion_model.transformer_blocks.N.attn.to_gate  (krea_2 trainer)
        #   transformer.transformer_blocks.N.attn.to_gate + text_fusion (diffusers)
        # The gate/proj is always a leaf module followed by the LoRA suffix '.'
        # (.lora_down/.lora_A/.lokr_w1/.alpha). Require the '.' so this does NOT
        # false-match LTX-2's '*_attn.to_gate_logits' (where '_logits' follows) —
        # that hijacked LTX-2 LoRAs as krea2 and broke their merge.
        if any(re.search(r'attn[._](?:to_gate|gate|w[qkvo])(?=\.)', k) for k in keys):
            return 'krea2'

        # Z-Image Turbo (Lumina2): layers.N with attention patterns
        # Handles: diffusion_model.layers.N, single_transformer_blocks.N (non-FLUX),
        #          lora_unet_layers_N (Musubi Tuner)
        if any('diffusion_model.layers.' in k and ('attention' in k or 'adaln' in k.lower())
               for k in keys):
            return 'zimage'
        if any('lora_unet_layers_' in k and 'attention' in k.lower() for k in keys):
            return 'zimage'
        # single_transformer_blocks WITHOUT transformer. or Kohya transformer_ prefix = Z-Image
        if any('single_transformer_blocks' in k
               and 'transformer.single_transformer_blocks' not in k
               and 'transformer_single_transformer_blocks' not in k
               for k in keys):
            return 'zimage'

        # Qwen-Image: transformer_blocks with img_mlp/txt_mlp/img_mod/txt_mod
        # Must be checked BEFORE FLUX — both use transformer.transformer_blocks
        # but Qwen has dual-stream markers (img_mlp, txt_mlp, img_mod, txt_mod, add_q_proj).
        # Also detect Qwen LoRAs that only target attention (to_q/to_k/to_v) without
        # dual-stream markers — these have transformer.transformer_blocks but lack
        # FLUX-specific double_blocks/single_blocks patterns.
        _has_qwen_markers = any('transformer_blocks' in k and
               any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod', 'add_q_proj'])
               for k in keys)
        if _has_qwen_markers:
            return 'qwen_image'
        # Qwen attention-only LoRAs: transformer.transformer_blocks with to_q/to_k/to_v
        # but NO double_blocks/single_blocks (which would indicate FLUX)
        _has_transformer_blocks = any('transformer.transformer_blocks' in k for k in keys)
        _has_flux_blocks = any('double_blocks' in k or 'single_blocks' in k for k in keys)
        if _has_transformer_blocks and not _has_flux_blocks:
            # transformer.transformer_blocks without FLUX block patterns = Qwen-Image
            return 'qwen_image'

        # FLUX: double/single blocks in various trainer formats
        if any('transformer.single_transformer_blocks' in k or 'transformer.transformer_blocks' in k
               for k in keys):
            return 'flux'
        if any('transformer_single_transformer_blocks' in k or 'transformer_double_blocks' in k
               for k in keys):
            return 'flux'
        if any('double_blocks' in k or 'single_blocks' in k for k in keys):
            return 'flux'

        # Anima (CircleStone Labs) — a Cosmos-Predict2 DiT with SPLIT QKV. MUST be
        # checked before ACE-Step / Wan / LTX: its blocks.N.self_attn.q_proj and
        # its Qwen3 text encoder (lora_te_layers_N_*) otherwise match those.
        # Unambiguous discriminators: the unique `llm_adapter`, the Cosmos triple
        # `adaln_modulation_{self_attn,cross_attn,mlp}`, the GPT2 `mlp.layer1/2`,
        # and `{self_attn,cross_attn}.output_proj`. (The 'lora_unet' prefix some
        # trainers emit is a convention — Anima is a DiT, not a UNet.)
        if any('llm_adapter' in k for k in keys):
            return 'anima'
        _anima_block = any(('blocks.' in k or 'blocks_' in k)
                           and ('self_attn' in k or 'cross_attn' in k) for k in keys)
        if _anima_block and any(
                'adaln_modulation_self_attn' in k or 'adaln_modulation_cross_attn' in k
                or 'mlp.layer1' in k or 'mlp.layer2' in k
                or '_mlp_layer1' in k or '_mlp_layer2' in k
                or 'self_attn.output_proj' in k or 'cross_attn.output_proj' in k
                or '_self_attn_output_proj' in k or '_cross_attn_output_proj' in k
                for k in keys):
            return 'anima'

        # ACE-Step v1.5: layers.N with self_attn/cross_attn using q_proj/k_proj/v_proj
        if any('layers.' in k and ('self_attn' in k or 'cross_attn' in k)
               and any(x in k for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
               for k in keys):
            return 'acestep'

        # ACE-Step v1.0: transformer_blocks.N with attn/cross_attn and to_q/to_k/to_v,
        # or unique keys like speaker_embedder / lyric_encoder
        if any('speaker_embedder' in k or 'lyric_encoder' in k for k in keys):
            return 'acestep'
        if any(re.search(r'transformer_blocks\.\d+\.(?:attn|cross_attn)\.to_(?:q|k|v|out)', k)
               for k in keys):
            return 'acestep'

        # Wan: blocks.N with self_attn/cross_attn/ffn (exclude transformer_blocks which is ACE-Step v1.0)
        def _is_wan_key(k):
            if 'transformer_blocks' in k:
                return False
            return ('blocks.' in k or 'blocks_' in k) and any(
                x in k for x in ['self_attn', 'cross_attn', 'ffn']
            )
        if any(_is_wan_key(k) for k in keys):
            return 'wan'

        # LTX Video: transformer_blocks with attn1/attn2 and adaln_single
        if any('adaln_single' in k for k in keys):
            return 'ltx'
        has_img_mlp = any('img_mlp' in k for k in keys)
        if not has_img_mlp and any(
                'transformer_blocks' in k and ('attn1' in k or 'attn2' in k)
                and 'input_blocks' not in k and 'output_blocks' not in k
                and 'down_blocks' not in k and 'up_blocks' not in k
                and 'middle_block' not in k and 'mid_block' not in k
                for k in keys):
            return 'ltx'

        # SDXL: dual text encoders (lora_te1/te2 or text_encoder_2)
        if 'lora_te1_' in keys_str or 'lora_te2_' in keys_str:
            return 'sdxl'
        if any('text_encoder_2.' in k for k in keys):
            return 'sdxl'

        # SD 1.5: single text encoder or UNet block patterns without dual TE
        if 'lora_te_' in keys_str:
            return 'sd15'
        if any('input_blocks' in k or 'output_blocks' in k for k in keys):
            return 'sd15'
        if any('down_blocks' in k or 'up_blocks' in k for k in keys):
            return 'sd15'

        return 'unknown'

    @staticmethod
    def _normalize_keys_zimage(lora_sd):
        """
        Normalize Z-Image Turbo (Lumina2) LoRA keys to a canonical format.

        Handles:
        1. Split fused QKV (attention.qkv) into separate to_q/to_k/to_v
           for per-component conflict analysis during merge.
        2. Remap attention.out -> attention.to_out.0 (diffusers convention).
        3. Standardize Musubi Tuner format (lora_unet_layers_N_...).
        4. Ensure diffusion_model.layers.N prefix.

        Returns new dict with normalized keys. Original dict is not modified.
        """
        normalized = {}
        processed = set()

        # First pass: standardize prefixes (Musubi Tuner -> canonical)
        prefix_fixed = {}
        for k, v in lora_sd.items():
            new_k = k
            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]
            # Musubi Tuner: lora_unet_layers_N_attention_... -> diffusion_model.layers.N.attention...
            if new_k.startswith('lora_unet_'):
                new_k = new_k.replace('lora_unet_', 'diffusion_model.')
                # Convert underscore-separated to dot-separated for known patterns
                new_k = re.sub(r'layers_(\d+)_', r'layers.\1.', new_k)
                new_k = re.sub(r'attention_', 'attention.', new_k)
                new_k = re.sub(r'feed_forward_', 'feed_forward.', new_k)
            # Ensure diffusion_model. prefix
            if new_k.startswith('layers.'):
                new_k = 'diffusion_model.' + new_k
            prefix_fixed[new_k] = v

        # Second pass: split fused QKV and remap output projection
        # Find all layer indices
        layer_pattern = re.compile(r'((?:diffusion_model\.)?layers\.(\d+)\.attention)\.')
        layers_seen = set()
        for k in prefix_fixed:
            m = layer_pattern.search(k)
            if m:
                layers_seen.add((m.group(1), int(m.group(2))))

        for base, layer_idx in layers_seen:
            # --- Split fused QKV ---
            # In fused QKV LoRAs: lora_down/A is [rank, in_features] (shared),
            # lora_up/B is [3*out_features, rank] (fused). Only the up/B
            # matrix needs splitting; the down/A matrix is copied to all three.
            for lora_fmt in [('.lora_A.weight', '.lora_B.weight'),
                             ('.lora_down.weight', '.lora_up.weight'),
                             ('.lora.down.weight', '.lora.up.weight')]:
                down_suffix, up_suffix = lora_fmt
                qkv_down_key = f"{base}.qkv{down_suffix}"
                qkv_up_key = f"{base}.qkv{up_suffix}"

                if qkv_down_key in prefix_fixed and qkv_up_key in prefix_fixed:
                    qkv_down = prefix_fixed[qkv_down_key]
                    qkv_up = prefix_fixed[qkv_up_key]

                    # Only the up/B matrix is fused [3*out, rank] — split it
                    out_dim = qkv_up.shape[0]
                    if out_dim % 3 != 0:
                        continue  # Not valid fused QKV, try next format
                    q_up, k_up, v_up = torch.chunk(qkv_up, 3, dim=0)

                    # Down/A is shared [rank, in] — copy to all three
                    for comp, comp_up in [('to_q', q_up),
                                          ('to_k', k_up),
                                          ('to_v', v_up)]:
                        normalized[f"{base}.{comp}{down_suffix}"] = qkv_down
                        normalized[f"{base}.{comp}{up_suffix}"] = comp_up

                    # Copy alpha (same for all three components)
                    alpha_key = f"{base}.qkv.alpha"
                    if alpha_key in prefix_fixed:
                        for comp in ('to_q', 'to_k', 'to_v'):
                            normalized[f"{base}.{comp}.alpha"] = prefix_fixed[alpha_key]
                        processed.add(alpha_key)

                    processed.add(qkv_down_key)
                    processed.add(qkv_up_key)
                    break  # Found QKV format, don't try others

            # --- Remap output projection: attention.out -> attention.to_out.0 ---
            for lora_fmt in [('.lora_A.weight', '.lora_B.weight'),
                             ('.lora_up.weight', '.lora_down.weight'),
                             ('.lora_B.weight', '.lora_A.weight'),
                             ('.lora.up.weight', '.lora.down.weight')]:
                sfx_a, sfx_b = lora_fmt
                out_a = f"{base}.out{sfx_a}"
                out_b = f"{base}.out{sfx_b}"
                if out_a in prefix_fixed and out_b in prefix_fixed:
                    normalized[f"{base}.to_out.0{sfx_a}"] = prefix_fixed[out_a]
                    normalized[f"{base}.to_out.0{sfx_b}"] = prefix_fixed[out_b]
                    processed.add(out_a)
                    processed.add(out_b)

                    out_alpha = f"{base}.out.alpha"
                    if out_alpha in prefix_fixed:
                        normalized[f"{base}.to_out.0.alpha"] = prefix_fixed[out_alpha]
                        processed.add(out_alpha)
                    break

        # Pass through all unprocessed keys
        for k, v in prefix_fixed.items():
            if k not in processed:
                normalized[k] = v

        return normalized

    # Compound component names in FLUX models where underscores must be preserved
    _FLUX_COMPOUND_NAMES = sorted([
        'img_attn', 'txt_attn', 'img_mlp', 'txt_mlp',
        'img_mod', 'txt_mod', 'img_norm1', 'img_norm2',
        'txt_norm1', 'txt_norm2', 'query_norm', 'key_norm',
        'lora_up', 'lora_down', 'lora_A', 'lora_B',
        'lora_mid', 'redux_up', 'redux_down',
    ], key=len, reverse=True)  # longest first to avoid partial matches

    @classmethod
    def _flux_kohya_underscore_to_dot(cls, rest):
        """Convert Kohya underscore-separated key to dot-separated,
        preserving compound component names like img_attn, lora_up."""
        protected = []
        for i, name in enumerate(cls._FLUX_COMPOUND_NAMES):
            placeholder = f'\x00{i}\x00'
            if name in rest:
                rest = rest.replace(name, placeholder)
                protected.append((placeholder, name))
        rest = rest.replace('_', '.')
        for placeholder, name in protected:
            rest = rest.replace(placeholder, name)
        return rest

    @classmethod
    def _normalize_keys_flux(cls, lora_sd):
        """
        Normalize FLUX LoRA keys from various trainer formats to canonical format.

        Canonical format: diffusion_model.double_blocks.N.* / diffusion_model.single_blocks.N.*

        Handles:
        - AI-Toolkit: transformer.transformer_blocks.N -> double_blocks.N
                      transformer.single_transformer_blocks.N -> single_blocks.N
        - Kohya: lora_transformer_double_blocks_N -> double_blocks.N
                 lora_transformer_single_transformer_blocks_N -> single_blocks.N
        - Standard: double_blocks.N / single_blocks.N (ensure prefix)
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # AI-Toolkit format
            # transformer.single_transformer_blocks.N -> diffusion_model.single_blocks.N
            new_k = re.sub(
                r'^transformer\.single_transformer_blocks\.(\d+)\.',
                r'diffusion_model.single_blocks.\1.', new_k)
            # transformer.transformer_blocks.N -> diffusion_model.double_blocks.N
            new_k = re.sub(
                r'^transformer\.transformer_blocks\.(\d+)\.',
                r'diffusion_model.double_blocks.\1.', new_k)

            # Kohya underscore format — smart replacement preserving compound names
            m = re.match(r'^lora_transformer_single_transformer_blocks_(\d+)_(.*)', new_k)
            if m:
                block_num = m.group(1)
                rest = cls._flux_kohya_underscore_to_dot(m.group(2))
                new_k = f"diffusion_model.single_blocks.{block_num}.{rest}"
            m = re.match(r'^lora_transformer_double_blocks_(\d+)_(.*)', new_k)
            if m:
                block_num = m.group(1)
                rest = cls._flux_kohya_underscore_to_dot(m.group(2))
                new_k = f"diffusion_model.double_blocks.{block_num}.{rest}"

            # Ensure diffusion_model. prefix for standard format
            if new_k.startswith('double_blocks.') or new_k.startswith('single_blocks.'):
                new_k = 'diffusion_model.' + new_k

            # Generic transformer. prefix -> diffusion_model.
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)

            normalized[new_k] = v
        return normalized

    @staticmethod
    def _normalize_keys_wan(lora_sd):
        """
        Normalize Wan LoRA keys from various trainer formats to canonical format.

        Canonical format: diffusion_model.blocks.N.{self_attn,cross_attn,ffn}.*

        Handles LyCORIS, diffusers, Fun LoRA, finetrainer formats.

        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # LyCORIS/aitoolkit format
            if new_k.startswith('lycoris_blocks_'):
                new_k = new_k.replace('lycoris_blocks_', 'blocks.')
                # Add dot separator after block number
                new_k = re.sub(r'^blocks\.(\d+)_', r'blocks.\1.', new_k)
                # Use regex to match both underscore and dot as leading separator
                # (dot comes from the block number fix above)
                new_k = re.sub(r'[._]cross_attn[._]', '.cross_attn.', new_k)
                new_k = re.sub(r'[._]self_attn[._]', '.self_attn.', new_k)
                new_k = re.sub(r'[._]ffn_net_0_proj', '.ffn.0', new_k)
                new_k = re.sub(r'[._]ffn_net_2', '.ffn.2', new_k)
                new_k = new_k.replace('to_out_0', 'o')

            # Diffusers format prefixes
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)
            if new_k.startswith('pipe.dit.'):
                new_k = new_k.replace('pipe.dit.', 'diffusion_model.', 1)
            if new_k.startswith('blocks.'):
                new_k = 'diffusion_model.' + new_k
            if new_k.startswith('vace_blocks.'):
                new_k = 'diffusion_model.' + new_k

            # Common diffusers cleanup
            new_k = new_k.replace('.default.', '.')
            new_k = new_k.replace('.diff_m', '.modulation.diff')

            # Fun LoRA format: lora_unet__blocks_N_...
            if new_k.startswith('lora_unet__'):
                parts = new_k.split('.')
                main_part = parts[0]
                weight_type = '.'.join(parts[1:]) if len(parts) > 1 else None

                if 'blocks_' in main_part:
                    components = main_part[len('lora_unet__'):].split('_')
                    rebuilt = 'diffusion_model'

                    if components[0] == 'blocks':
                        rebuilt += f".blocks.{components[1]}"
                        idx = 2
                        if idx < len(components):
                            if (components[idx] == 'self' and idx + 1 < len(components)
                                    and components[idx + 1] == 'attn'):
                                rebuilt += '.self_attn'
                                idx += 2
                            elif (components[idx] == 'cross' and idx + 1 < len(components)
                                  and components[idx + 1] == 'attn'):
                                rebuilt += '.cross_attn'
                                idx += 2
                            elif components[idx] == 'ffn':
                                rebuilt += '.ffn'
                                idx += 1
                        if idx < len(components):
                            component = components[idx]
                            idx += 1
                            if idx < len(components) and components[idx] == 'img':
                                component += '_img'
                                idx += 1
                            rebuilt += f'.{component}'
                        # Append any remaining components with dot separators
                        while idx < len(components):
                            rebuilt += f'.{components[idx]}'
                            idx += 1

                    if weight_type:
                        if weight_type == 'alpha':
                            rebuilt += '.alpha'
                        elif weight_type in ('lora_down.weight', 'lora_down'):
                            rebuilt += '.lora_A.weight'
                        elif weight_type in ('lora_up.weight', 'lora_up'):
                            rebuilt += '.lora_B.weight'
                        else:
                            rebuilt += f'.{weight_type}'
                            if not rebuilt.endswith('.weight'):
                                rebuilt += '.weight'
                    new_k = rebuilt
                else:
                    new_k = main_part.replace('lora_unet__', 'diffusion_model.')
                    new_k = new_k.replace('_', '.')
                    if weight_type:
                        if weight_type == 'alpha':
                            new_k += '.alpha'
                        elif weight_type in ('lora_down.weight', 'lora_down'):
                            new_k += '.lora_A.weight'
                        elif weight_type in ('lora_up.weight', 'lora_up'):
                            new_k += '.lora_B.weight'
                        else:
                            new_k += f'.{weight_type}'
                            if not new_k.endswith('.weight'):
                                new_k += '.weight'

            # Finetrainer format
            new_k = new_k.replace('.attn1.to_q.', '.self_attn.q.')
            new_k = new_k.replace('.attn1.to_k.', '.self_attn.k.')
            new_k = new_k.replace('.attn1.to_v.', '.self_attn.v.')
            new_k = new_k.replace('.attn1.to_out.0.', '.self_attn.o.')
            new_k = new_k.replace('.attn2.to_q.', '.cross_attn.q.')
            new_k = new_k.replace('.attn2.to_k.', '.cross_attn.k.')
            new_k = new_k.replace('.attn2.to_v.', '.cross_attn.v.')
            new_k = new_k.replace('.attn2.to_out.0.', '.cross_attn.o.')

            normalized[new_k] = v

        # Note: RS-LoRA compensation removed. RS-LoRA files omit alpha and
        # rely on sqrt(rank) scaling, but we can't distinguish them from
        # standard PEFT LoRAs that also omit alpha. False positives cause
        # ~4x weight amplification (rank 16) to ~5.66x (rank 32). When alpha
        # is missing, _get_lora_key_info defaults alpha=rank (scale=1.0),
        # which is correct for standard LoRAs and only slightly weak for
        # RS-LoRA.

        return normalized

    # LoRA weight suffixes (longest-first to avoid partial matches)
    _LORA_KEY_SUFFIXES = [
        ".lora_up.weight", ".lora_down.weight",
        "_lora.up.weight", "_lora.down.weight",
        ".lora_B.weight", ".lora_A.weight",
        ".lora.up.weight", ".lora.down.weight",
        ".lokr_w1_a", ".lokr_w1_b",
        ".lokr_w2_a", ".lokr_w2_b",
        ".lokr_w1", ".lokr_w2",
        ".lokr_t2",
        ".hada_w1_a", ".hada_w1_b",
        ".hada_w2_a", ".hada_w2_b",
        ".hada_t1", ".hada_t2",
        ".alpha",
    ]

    @classmethod
    def _split_lora_suffix(cls, key):
        """Split a LoRA key into (prefix, suffix), preserving the suffix intact."""
        for suffix in cls._LORA_KEY_SUFFIXES:
            if key.endswith(suffix):
                return key[:-len(suffix)], suffix
        return key, ""

    @classmethod
    def _normalize_keys_sdxl(cls, lora_sd):
        """
        Normalize SDXL LoRA keys to canonical format.

        Canonical format: lora_unet_* / lora_te1_* / lora_te2_* (Kohya convention).

        Handles diffusers-format keys (down_blocks.N, up_blocks.N, mid_block).
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # Split off LoRA suffix to avoid dot-to-underscore mangling
            stem, suffix = cls._split_lora_suffix(new_k)

            # Diffusers format: text_encoder.* -> lora_te1_*, text_encoder_2.* -> lora_te2_*
            if stem.startswith('text_encoder_2.'):
                stem = 'lora_te2_' + stem[len('text_encoder_2.'):].replace('.', '_')
            elif stem.startswith('text_encoder.'):
                stem = 'lora_te1_' + stem[len('text_encoder.'):].replace('.', '_')

            # Diffusers UNet: unet.* -> lora_unet_*
            if stem.startswith('unet.'):
                stem = 'lora_unet_' + stem[len('unet.'):].replace('.', '_')

            normalized[stem + suffix] = v
        return normalized

    @classmethod
    def _normalize_keys_sd15(cls, lora_sd):
        """
        Normalize SD 1.5 LoRA keys to canonical format.

        Canonical format: lora_unet_* / lora_te_* (Kohya convention).

        Same as SDXL but single text encoder: text_encoder.* -> lora_te_*.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # Split off LoRA suffix to avoid dot-to-underscore mangling
            stem, suffix = cls._split_lora_suffix(new_k)

            # Diffusers format: text_encoder.* -> lora_te_* (single TE)
            if stem.startswith('text_encoder.'):
                stem = 'lora_te_' + stem[len('text_encoder.'):].replace('.', '_')

            # Diffusers UNet: unet.* -> lora_unet_*
            if stem.startswith('unet.'):
                stem = 'lora_unet_' + stem[len('unet.'):].replace('.', '_')

            normalized[stem + suffix] = v
        return normalized

    @staticmethod
    def _normalize_keys_ltx(lora_sd):
        """
        Normalize LTX Video LoRA keys to canonical format.

        Canonical format: diffusion_model.transformer_blocks.N.attn1/attn2.to_q/to_k/to_v.*

        LTX uses standard separate Q/K/V — only prefix standardization needed.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # Kohya format: lora_unet_transformer_blocks_N_... -> diffusion_model.transformer_blocks.N...
            if new_k.startswith('lora_unet_'):
                new_k = new_k.replace('lora_unet_', 'diffusion_model.')
                # Convert underscores back to dots for known structural segments
                new_k = re.sub(r'transformer_blocks_(\d+)_', r'transformer_blocks.\1.', new_k)
                new_k = re.sub(r'attn(\d)_', r'attn\1.', new_k)
                new_k = re.sub(r'to_(q|k|v|out)_', r'to_\1.', new_k)
                new_k = re.sub(r'ff_net_(\d+)_', r'ff.net.\1.', new_k)

            # Diffusers format: unet.* -> diffusion_model.*
            if new_k.startswith('unet.'):
                new_k = new_k.replace('unet.', 'diffusion_model.', 1)

            # Ensure diffusion_model. prefix
            if new_k.startswith('transformer_blocks.'):
                new_k = 'diffusion_model.' + new_k

            normalized[new_k] = v
        return normalized

    @staticmethod
    def _normalize_keys_anima(lora_sd):
        """
        Normalize Anima (CircleStone Labs / Cosmos-Predict2 DiT) LoRA keys to the
        canonical ComfyUI format:
            diffusion_model.blocks.N.{self_attn,cross_attn}.{q,k,v,output}_proj
            diffusion_model.blocks.N.mlp.layer1 / layer2
            diffusion_model.blocks.N.adaln_modulation_{self_attn,cross_attn,mlp}.N
            diffusion_model.llm_adapter.blocks.N.*
            diffusion_model.final_layer.{linear, adaln_modulation.N}

        Anima uses SPLIT QKV (separate q/k/v) — no fuse/refuse handling needed.
        Converts:
          - diffusion-pipe / ComfyUI  (diffusion_model.blocks.N.*)  — already canonical
          - Kohya sd-scripts          (lora_unet_blocks_N_*)        — underscore→dot restore
          - Diffusers Cosmos          (transformer_blocks.N.attn1/attn2.to_*) — best effort
        Text-encoder (Qwen3) lora_te_* keys pass through unchanged.
        """
        normalized = {}
        for k, v in lora_sd.items():
            nk = k
            if nk.startswith('base_model.model.'):
                nk = nk[len('base_model.model.'):]
            if nk.startswith('transformer.') and 'transformer_blocks' not in nk[:13]:
                nk = nk[len('transformer.'):]

            if nk.startswith('lora_unet_'):
                # Kohya: dots were flattened to underscores; restore structural dots.
                nk = 'diffusion_model.' + nk[len('lora_unet_'):]
                nk = re.sub(r'^diffusion_model\.llm_adapter_blocks_(\d+)_',
                            r'diffusion_model.llm_adapter.blocks.\1.', nk)
                nk = re.sub(r'^diffusion_model\.blocks_(\d+)_',
                            r'diffusion_model.blocks.\1.', nk)
                nk = re.sub(r'(self_attn|cross_attn)_(q|k|v|output)_proj', r'\1.\2_proj', nk)
                nk = re.sub(r'(self_attn|cross_attn)_(q|k)_norm', r'\1.\2_norm', nk)
                nk = re.sub(r'mlp_(layer\d)', r'mlp.\1', nk)
                nk = re.sub(r'adaln_modulation_(self_attn|cross_attn|mlp)_(\d+)',
                            r'adaln_modulation_\1.\2', nk)
                nk = re.sub(r'final_layer_(linear|adaln_modulation)', r'final_layer.\1', nk)
                nk = re.sub(r'(final_layer\.adaln_modulation)_(\d+)', r'\1.\2', nk)
            elif nk.startswith('transformer_blocks.'):
                # Diffusers Cosmos -> canonical native names.
                nk = nk.replace('transformer_blocks.', 'diffusion_model.blocks.', 1)
                nk = nk.replace('.attn1.', '.self_attn.').replace('.attn2.', '.cross_attn.')
                nk = nk.replace('.to_out.0', '.output_proj')
                nk = (nk.replace('.to_q', '.q_proj').replace('.to_k', '.k_proj')
                        .replace('.to_v', '.v_proj'))
                nk = nk.replace('.norm_q', '.q_norm').replace('.norm_k', '.k_norm')
                nk = nk.replace('.ff.net.0.proj', '.mlp.layer1').replace('.ff.net.2', '.mlp.layer2')
            elif nk.startswith(('blocks.', 'llm_adapter.', 'final_layer.', 'net.')):
                if nk.startswith('net.'):
                    nk = nk[len('net.'):]
                nk = 'diffusion_model.' + nk

            # Collapse the internal 'net.' root if a diffusion_model.net.* key slipped through.
            nk = nk.replace('diffusion_model.net.', 'diffusion_model.')
            normalized[nk] = v
        return normalized

    @staticmethod
    def _normalize_keys_qwen_image(lora_sd):
        """
        Normalize Qwen-Image LoRA keys to canonical format.

        Canonical format: diffusion_model.transformer_blocks.N.*

        Qwen-Image uses separate Q/K/V with dual-stream (image+text) attention.
        Supports transformer.*, lycoris_*, and direct key formats.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # LyCORIS format: lycoris_transformer_blocks_N_... -> diffusion_model.transformer_blocks.N...
            if new_k.startswith('lycoris_'):
                new_k = new_k.replace('lycoris_', 'diffusion_model.')
                new_k = re.sub(r'transformer_blocks_(\d+)_', r'transformer_blocks.\1.', new_k)
                # Restore dots for known component names
                for comp in ['to_q', 'to_k', 'to_v', 'add_q_proj', 'add_k_proj', 'add_v_proj',
                             'img_mlp', 'txt_mlp', 'img_mod', 'txt_mod', 'img_norm1', 'img_norm2',
                             'txt_norm1', 'txt_norm2']:
                    new_k = new_k.replace(f'_{comp}_', f'.{comp}.')
                    if new_k.endswith(f'_{comp}'):
                        new_k = new_k[:-len(f'_{comp}')] + f'.{comp}'

            # transformer.* -> diffusion_model.*
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)

            # Ensure diffusion_model. prefix
            if new_k.startswith('transformer_blocks.'):
                new_k = 'diffusion_model.' + new_k

            normalized[new_k] = v
        return normalized

    @classmethod
    def _normalize_keys_ideogram4(cls, lora_sd):
        """
        Normalize Ideogram 4 LoRA keys to the canonical ComfyUI form:
        diffusion_model.layers.N.{attention.{qkv,o}, feed_forward.{w1,w2,w3},
        adaln_modulation}.

        Handles:
        - ai-toolkit native: diffusion_model.layers.N.... (passthrough)
        - fal trainer: conditional_transformer.layers.N....
        - PEFT/diffusers-style prefixes: base_model.model., transformer.,
          bare layers.
        - Kohya-style underscores: lora_unet_layers_N_attention_qkv

        Unlike Z-Image, the native ComfyUI weight layout keeps qkv FUSED and
        every known trainer targets the fused weight — no split/re-fuse
        needed. Split-attention (to_q/to_k/to_v) diffusers-layout LoRAs are
        passed through unchanged: none exist in the wild and ComfyUI itself
        has no key mapping for them on this model.

        Returns new dict with normalized keys. Original dict is not modified.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix
            if new_k.startswith("base_model.model."):
                new_k = new_k[len("base_model.model."):]

            # Kohya underscore format -> dotted
            if new_k.startswith("lora_unet_"):
                rest = new_k[len("lora_unet_"):]
                rest = re.sub(r"^layers_(\d+)_", r"layers.\1.", rest)
                rest = re.sub(r"attention_(qkv|o)(?=[._])", r"attention.\1", rest)
                rest = re.sub(r"feed_forward_(w\d)(?=[._])", r"feed_forward.\1", rest)
                new_k = f"diffusion_model.{rest}"

            # fal trainer wraps the conditional model; PEFT exports use
            # transformer. — both map to ComfyUI's diffusion_model. prefix
            new_k = re.sub(r"^conditional_transformer\.", "diffusion_model.", new_k)
            new_k = re.sub(r"^transformer\.", "diffusion_model.", new_k)
            if new_k.startswith("layers."):
                new_k = "diffusion_model." + new_k

            normalized[new_k] = v
        return normalized

    _ACESTEP_COMPOUND_NAMES = sorted([
        "self_attn", "cross_attn",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lora_up", "lora_down", "lora_A", "lora_B", "lora_mid",
        "speaker_embedder", "lyric_encoder",
        "linear_q", "linear_k", "linear_v",
        "to_out", "to_q", "to_k", "to_v",
    ], key=len, reverse=True)

    @classmethod
    def _acestep_underscore_to_dot(cls, rest):
        """Convert Kohya-style ACE-Step keys to dotted form while preserving
        compound component names like self_attn and q_proj."""
        protected = []
        for i, name in enumerate(cls._ACESTEP_COMPOUND_NAMES):
            placeholder = f"\x01{i}\x01"
            if name in rest:
                rest = rest.replace(name, placeholder)
                protected.append((placeholder, name))
        rest = rest.replace("_", ".")
        for placeholder, name in protected:
            rest = rest.replace(placeholder, name)
        return rest

    @classmethod
    def _normalize_keys_acestep(cls, lora_sd):
        """
        Normalize ACE-Step LoRA keys to canonical
        diffusion_model.layers.N.{self,cross}_attn.{q,k,v,o}_proj form.

        Handles:
        - v1.5 PEFT: base_model.model.layers.N.{self,cross}_attn.{q,k,v,o}_proj
        - v1.5 Kohya: lora_unet_layers_N_self_attn_q_proj
        - v1.0 diffusers: transformer_blocks.N.{attn,cross_attn}.to_{q,k,v}
        - v1.0 special: speaker_embedder, lyric_encoder.encoders.N.self_attn.linear_{q,k,v}
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix (v1.5)
            if new_k.startswith("base_model.model."):
                new_k = new_k[len("base_model.model."):]

            # Kohya underscore format → dotted
            if new_k.startswith("lora_unet_"):
                rest = new_k[len("lora_unet_"):]
                rest = cls._acestep_underscore_to_dot(rest)
                new_k = f"diffusion_model.{rest}"

            # Common prefix normalization
            new_k = re.sub(r"^transformer\.", "diffusion_model.", new_k)
            new_k = re.sub(r"^model\.", "diffusion_model.", new_k)

            if new_k.startswith("layers."):
                new_k = "diffusion_model." + new_k
            if new_k.startswith("transformer_blocks."):
                new_k = "diffusion_model." + new_k

            # v1.0 → v1.5: transformer_blocks.N → layers.N
            new_k = re.sub(
                r"^diffusion_model\.transformer_blocks\.(\d+)\.",
                r"diffusion_model.layers.\1.", new_k
            )

            # v1.0 → v1.5: bare .attn. → .self_attn. (cross_attn already correct)
            # Safe: .cross_attn. and .self_attn. have _ before attn, not .
            new_k = re.sub(r"\.attn\.", ".self_attn.", new_k)

            # v1.0 → v1.5: to_q/to_k/to_v → q_proj/k_proj/v_proj
            if new_k.startswith("diffusion_model.layers."):
                new_k = re.sub(
                    r"\.to_(q|k|v)\.", lambda m: f".{m.group(1)}_proj.", new_k
                )
                new_k = re.sub(r"\.to_out\.0\.", ".o_proj.", new_k)
                new_k = re.sub(r"\.to_out\.", ".o_proj.", new_k)

            # v1.0 lyric_encoder: linear_q/k/v → q_proj/k_proj/v_proj
            if "lyric_encoder" in new_k:
                new_k = new_k.replace(".linear_q.", ".q_proj.")
                new_k = new_k.replace(".linear_k.", ".k_proj.")
                new_k = new_k.replace(".linear_v.", ".v_proj.")
                if not new_k.startswith("diffusion_model."):
                    new_k = "diffusion_model." + new_k

            # v1.0 speaker_embedder: keep as-is but add prefix
            if new_k.startswith("speaker_embedder"):
                new_k = "diffusion_model." + new_k

            normalized[new_k] = v
        return normalized

    @staticmethod
    def _normalize_keys_krea2(lora_sd):
        """Normalize Krea 2 LoRA keys to the canonical ComfyUI-native
        diffusion_model.* form that matches the actual Krea 2 model weights.

        Two trainer forms are handled, both verified key-by-key (name AND shape)
        against the official krea2_turbo model — 224/224 and 264/264 modules map:
          - the "krea_2" trainer:  diffusion_model.transformer_blocks.N.attn.to_q ...
          - the diffusers form:     transformer.transformer_blocks.N.attn.to_q ...,
            transformer.text_fusion.{layerwise,refiner}_blocks.N.*, and the non-block
            transformer.{img_in,txt_in,final_layer,time_embed,time_mod_proj}.
        Without this, the diffusers `transformer.*` keys are mis-routed by ComfyUI's
        FLUX branch (flux_to_diffusers) to non-existent double_blocks QKV tuple-offset
        targets: the LoRA is silently dropped AND the bogus offset target can corrupt
        the merge. The Krea 2 model uses single-stream blocks.N.attn.{wq,wk,wv,wo,gate}
        + mlp.{gate,up,down}, txtfusion.{layerwise,refiner}_blocks, and named
        projections first/last/tmlp/tproj/txtmlp.
        """
        def _norm_module(mod):
            s = mod
            s = re.sub(r'^transformer\.', '', s)
            s = re.sub(r'^diffusion_model\.', '', s)
            s = re.sub(r'^lora_unet_', '', s)
            s = s.replace('transformer_blocks.', 'blocks.')
            s = s.replace('text_fusion.', 'txtfusion.')
            # attention projections (diffusers -> krea native)
            s = re.sub(r'\battn\.to_q\b', 'attn.wq', s)
            s = re.sub(r'\battn\.to_k\b', 'attn.wk', s)
            s = re.sub(r'\battn\.to_v\b', 'attn.wv', s)
            s = re.sub(r'\battn\.to_out\.0\b', 'attn.wo', s)
            s = re.sub(r'\battn\.to_gate\b', 'attn.gate', s)
            # SwiGLU feed-forward -> mlp
            s = s.replace('ff.gate', 'mlp.gate').replace('ff.up', 'mlp.up').replace('ff.down', 'mlp.down')
            # named non-block projections (verified by shape)
            s = re.sub(r'^img_in$', 'first', s)
            s = re.sub(r'^final_layer\.linear$', 'last.linear', s)
            s = re.sub(r'^time_mod_proj$', 'tproj.1', s)
            s = re.sub(r'^time_embed\.linear_1$', 'tmlp.0', s)
            s = re.sub(r'^time_embed\.linear_2$', 'tmlp.2', s)
            s = re.sub(r'^txt_in\.linear_1$', 'txtmlp.1', s)
            s = re.sub(r'^txt_in\.linear_2$', 'txtmlp.3', s)
            return 'diffusion_model.' + s

        # Suffix-first match so '.lora_A.weight' wins over '.lora_A'
        suffixes = ('.lora_A.weight', '.lora_B.weight', '.lora_down.weight',
                    '.lora_up.weight', '.lora_A', '.lora_B', '.lora_down',
                    '.lora_up', '.alpha', '.diff_b', '.diff')
        normalized = {}
        for k, v in lora_sd.items():
            matched = None
            for sfx in suffixes:
                if k.endswith(sfx):
                    matched = sfx
                    break
            if matched is None:
                normalized[k] = v  # unrecognized form — leave untouched
                continue
            module = k[:-len(matched)]
            normalized[_norm_module(module) + matched] = v
        return normalized

    @classmethod
    def _normalize_keys(cls, lora_sd, architecture):
        """
        Dispatch to architecture-specific key normalizer.
        Returns a new dict with normalized keys.
        """
        if architecture == 'zimage':
            return cls._normalize_keys_zimage(lora_sd)
        elif architecture == 'flux':
            return cls._normalize_keys_flux(lora_sd)
        elif architecture == 'wan':
            return cls._normalize_keys_wan(lora_sd)
        elif architecture == 'acestep':
            return cls._normalize_keys_acestep(lora_sd)
        elif architecture == 'sdxl':
            return cls._normalize_keys_sdxl(lora_sd)
        elif architecture == 'sd15':
            return cls._normalize_keys_sd15(lora_sd)
        elif architecture == 'ltx':
            return cls._normalize_keys_ltx(lora_sd)
        elif architecture == 'qwen_image':
            return cls._normalize_keys_qwen_image(lora_sd)
        elif architecture == 'ideogram4':
            return cls._normalize_keys_ideogram4(lora_sd)
        elif architecture == 'anima':
            return cls._normalize_keys_anima(lora_sd)
        elif architecture == 'krea2':
            return cls._normalize_keys_krea2(lora_sd)
        return lora_sd  # unknown — pass through unchanged

    @staticmethod
    def _refuse_zimage_patches(patches):
        """
        Re-fuse split to_q/to_k/to_v patches back into fused QKV patches
        for Z-Image Turbo models. Also remaps to_out.0 -> out.

        Called after merging, before applying patches to the model.
        Returns a new dict with fused patches.
        """
        fused = {}
        qkv_groups = {}  # base -> {comp: (key, patch)}

        for key, patch in patches.items():
            # Handle both string keys and tuple keys
            if isinstance(key, tuple):
                key_str = key[0]
            else:
                key_str = key

            # Detect to_q/to_k/to_v patterns in the key
            m = re.search(r'(layers\.\d+\.attention)\.to_(q|k|v)(?:\.|$)', key_str)
            if m:
                base = m.group(1)
                comp = m.group(2)
                if base not in qkv_groups:
                    qkv_groups[base] = {}
                qkv_groups[base][comp] = (key, patch)
                continue

            # Detect to_out.0 -> out remap
            m_out = re.search(r'(layers\.\d+\.attention)\.to_out\.0(?:\.|$)', key_str)
            if m_out:
                new_key_str = key_str.replace('.to_out.0', '.out')
                if isinstance(key, tuple):
                    new_key = (new_key_str,) + key[1:]
                else:
                    new_key = new_key_str
                fused[new_key] = patch
                continue

            # Not a QKV or out key — pass through
            fused[key] = patch

        # Fuse QKV groups
        for base, comps in qkv_groups.items():
            if len(comps) == 3 and 'q' in comps and 'k' in comps and 'v' in comps:
                # All three components present — fuse
                q_key, q_patch = comps['q']
                k_key, k_patch = comps['k']
                v_key, v_patch = comps['v']

                # Build the fused key name
                if isinstance(q_key, tuple):
                    fused_key_str = re.sub(r'\.to_q(?=\.|$)', '.qkv', q_key[0])
                    fused_key = (fused_key_str,) + q_key[1:]
                else:
                    fused_key_str = re.sub(r'\.to_q(?=\.|$)', '.qkv', q_key)
                    fused_key = fused_key_str

                # Handle different patch formats
                if all(isinstance(p, tuple) and p[0] == "diff" for p in [q_patch, k_patch, v_patch]):
                    # Full-rank diff patch: ("diff", (tensor,))
                    q_diff = q_patch[1][0]
                    k_diff = k_patch[1][0]
                    v_diff = v_patch[1][0]
                    store_dtype = q_diff.dtype
                    # Components can sit on different devices (GPU-deferred
                    # score-during-merge patches next to CPU ones) — unify
                    if k_diff.device != q_diff.device:
                        k_diff = k_diff.to(q_diff.device)
                    if v_diff.device != q_diff.device:
                        v_diff = v_diff.to(q_diff.device)
                    fused_diff = torch.cat([q_diff, k_diff, v_diff], dim=0)
                    if store_dtype not in (torch.float32, torch.float64):
                        fused_diff = fused_diff.to(store_dtype)
                    fused[fused_key] = ("diff", (fused_diff,))
                elif all(isinstance(p, LoRAAdapter) for p in [q_patch, k_patch, v_patch]):
                    q_data = q_patch.weights
                    k_data = k_patch.weights
                    v_data = v_patch.weights
                    # weights = (mat_up, mat_down, alpha, mid, dora_scale, reshape)

                    # Check if down matrices are shared (true for original LoRA,
                    # false for independently SVD-compressed patches)
                    if q_data[1] is k_data[1] and k_data[1] is v_data[1]:
                        # Shared down: concatenate ups, keep one down copy
                        fused_up = torch.cat([q_data[0], k_data[0], v_data[0]], dim=0)
                        fused_down = q_data[1]
                        fused_alpha = q_data[2]
                        fused_patch = LoRAAdapter(set(), (fused_up, fused_down, fused_alpha, None, None, None))
                        fused[fused_key] = fused_patch
                    else:
                        # Independent decompositions (e.g., SVD-compressed) —
                        # expand to full-rank diffs, then fuse as a single diff patch
                        parts = []
                        store_dtype = q_data[0].dtype
                        for comp_data in [q_data, k_data, v_data]:
                            alpha = comp_data[2] if comp_data[2] is not None else float(comp_data[1].shape[0])
                            rank = comp_data[1].shape[0]
                            diff = torch.mm(comp_data[0].float(), comp_data[1].float()) * (alpha / rank)
                            parts.append(diff)
                        fused_diff = torch.cat(parts, dim=0)
                        if store_dtype not in (torch.float32, torch.float64):
                            fused_diff = fused_diff.to(store_dtype)
                        fused[fused_key] = ("diff", (fused_diff,))
                elif any(hasattr(p, "weights") for p in [q_patch, k_patch, v_patch]):
                    store_dtype = torch.float16
                    for candidate in [q_patch, k_patch, v_patch]:
                        if hasattr(candidate, "weights") and candidate.weights[0] is not None:
                            dtype = candidate.weights[0].dtype
                            if dtype not in (torch.float32, torch.float64):
                                store_dtype = dtype
                                break
                    parts = [_LoRAMergeBase._expand_patch_to_diff(comp_patch) for comp_patch in [q_patch, k_patch, v_patch]]
                    # Unify devices (GPU-deferred diffs can mix with CPU adapters)
                    parts = [p if p.device == parts[0].device else p.to(parts[0].device)
                             for p in parts]
                    fused_diff = torch.cat(parts, dim=0)
                    if store_dtype not in (torch.float32, torch.float64):
                        fused_diff = fused_diff.to(store_dtype)
                    fused[fused_key] = ("diff", (fused_diff,))
                else:
                    # Unknown patch format — pass through unfused
                    for comp_key, comp_patch in [(q_key, q_patch), (k_key, k_patch), (v_key, v_patch)]:
                        fused[comp_key] = comp_patch
            else:
                # Incomplete QKV group — pass through individual components
                for comp, (comp_key, comp_patch) in comps.items():
                    fused[comp_key] = comp_patch

        return fused

    def _load_lora(self, lora_name):
        """Loads LoRA file with caching"""
        if lora_name == "None" or lora_name is None:
            return None
        if lora_name in self.loaded_loras:
            return self.loaded_loras[lora_name]
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_loras[lora_name] = lora
        return lora
    
    def _get_model_keys(self, model):
        """Get LoRA prefix → target key mapping for the model."""
        if model is None:
            return {}
        return comfy.lora.model_lora_keys_unet(model.model, {})

    @staticmethod
    def _payload_rank(payload):
        """LoRA rank of a captured payload, or None for dense/exotic payloads.

        Used by _prepare_group_diffs' virtual-item rank accounting. tuple/list
        dense diffs and LoKr (whose weights[1] is the
        w2 Kronecker factor, not a rank) return None; a real LoRA/LoHa adapter
        returns mat_down.shape[0]."""
        try:
            if isinstance(payload, (tuple, list)):
                return None  # dense diff — rank not meaningful
            if isinstance(payload, LoKrAdapter):
                # weights[1] is the w2 Kronecker factor, not a rank —
                # reporting its shape as "rank N" would be a lie.
                return None
            down = payload.weights[1]
            return int(down.shape[0])
        except Exception:
            return None

    @staticmethod
    def _is_plain_additive_payload(payload):
        """True when ``payload`` can be detached from its ModelPatcher entry
        and represented as a base-independent additive delta.

        DoRA adapters are base-weight dependent, while a LoRA ``reshape``
        request pads/replaces the working weight shape before applying the
        delta.  Neither operation can be preserved by expanding the adapter to
        a bare diff and re-appending it later.  Keep those payloads on the
        incoming patcher instead of silently turning them into ordinary LoRA.
        """
        if isinstance(payload, tuple):
            # Only the exact ("diff", (tensor,)) shape is a plain dense delta.
            return (len(payload) == 2 and payload[0] == "diff"
                    and isinstance(payload[1], tuple) and len(payload[1]) == 1
                    and isinstance(payload[1][0], torch.Tensor))
        if isinstance(payload, list):
            return False  # nested compositions are order/base dependent

        weights = getattr(payload, "weights", None)
        if not isinstance(weights, (tuple, list)):
            return False
        if isinstance(payload, LoKrAdapter):
            # Current comfy LoKr schema: (..., t2, dora_scale).
            return len(weights) in (8, 9) and (len(weights) == 8 or weights[8] is None)
        if isinstance(payload, LoHaAdapter):
            # Current comfy LoHa schema: (..., t1, t2, dora_scale).
            return len(weights) in (7, 8) and (len(weights) == 7 or weights[7] is None)
        if isinstance(payload, LoRAAdapter):
            # (up, down, alpha, mid, dora_scale, reshape).  Older comfy
            # versions may expose a shorter tuple; absent optional fields are
            # equivalent to None.
            if not 4 <= len(weights) <= 6:
                return False
            dora_scale = weights[4] if len(weights) > 4 else None
            reshape = weights[5] if len(weights) > 5 else None
            return dora_scale is None and reshape is None
        return False

    @staticmethod
    def _collect_lora_prefixes(active_loras):
        """Collect all LoRA key prefixes from a stack in deterministic order."""
        all_lora_prefixes = set()
        suffixes = [
            ".lora_up.weight", ".lora_down.weight",
            "_lora.up.weight", "_lora.down.weight",
            ".lora_B.weight", ".lora_A.weight",
            ".lora.up.weight", ".lora.down.weight",
            # LoKr (Kronecker)
            ".lokr_w1", ".lokr_w2",
            ".lokr_w1_a", ".lokr_w1_b",
            ".lokr_w2_a", ".lokr_w2_b",
            ".lokr_t2",
            # LoHa (Hadamard)
            ".hada_w1_a", ".hada_w1_b",
            ".hada_w2_a", ".hada_w2_b",
            ".hada_t1", ".hada_t2",
            ".alpha",
        ]
        for item in active_loras:
            if item.get("_precomputed_diffs"):
                for key in item["lora"]:
                    all_lora_prefixes.add(key)
                continue
            for key in item["lora"].keys():
                for suffix in suffixes:
                    if key.endswith(suffix):
                        all_lora_prefixes.add(key[:-len(suffix)])
                        break
        return sorted(all_lora_prefixes, key=lambda x: (str(x),))

    @staticmethod
    def _resolve_target_key(lora_prefix, model_keys, clip_keys):
        """Resolve a LoRA prefix to a model/CLIP target key."""
        if lora_prefix in model_keys:
            return (model_keys[lora_prefix], False)
        if lora_prefix in clip_keys:
            return (clip_keys[lora_prefix], True)
        return (None, False)

    @staticmethod
    def _make_target_group_id(target_key, is_clip):
        return (bool(is_clip), target_key)

    @staticmethod
    def _choose_canonical_prefix(aliases):
        """Pick a stable human-readable label for an alias group."""
        if not aliases:
            return ""
        def _sort_key(alias):
            return (
                alias.startswith("lora_"),
                alias.count("_"),
                len(alias),
                alias,
            )
        return min(sorted(set(aliases)), key=_sort_key)

    def _build_target_groups(self, all_lora_prefixes, model_keys, clip_keys):
        """
        Group aliases by resolved (is_clip, target_key) so analysis/merge operates
        on actual model weights rather than raw trainer-specific prefixes.
        """
        grouped = {}
        # Build reverse lookup: target_key → True for fast virtual-key detection
        model_target_keys = set()
        for v in (model_keys.values() if model_keys else []):
            model_target_keys.add(v)
            if isinstance(v, tuple):
                model_target_keys.add(v[0])
        clip_target_keys = set()
        for v in (clip_keys.values() if clip_keys else []):
            clip_target_keys.add(v)
            if isinstance(v, tuple):
                clip_target_keys.add(v[0])
        dropped = []
        for prefix in all_lora_prefixes:
            target_key, is_clip = self._resolve_target_key(prefix, model_keys, clip_keys)
            if target_key is None:
                # Virtual LoRA keys are already target keys — map to themselves
                if prefix in model_target_keys:
                    target_key, is_clip = prefix, False
                elif prefix in clip_target_keys:
                    target_key, is_clip = prefix, True
                else:
                    dropped.append(prefix)
                    continue
            group_id = self._make_target_group_id(target_key, is_clip)
            entry = grouped.setdefault(group_id, {
                "target_key": target_key,
                "is_clip": is_clip,
                "aliases": [],
            })
            entry["aliases"].append(prefix)

        if dropped:
            sample = ", ".join(sorted(dropped)[:8])
            logging.warning(
                f"[LoRA Optimizer] {len(dropped)} LoRA key(s) did not map to any model/CLIP "
                f"weight and were SKIPPED (not in the merge). Sample: {sample}. "
                f"If these are expected layers (e.g. LTX-2 audio), either the model doesn't "
                f"expose them to LoRA (a plain Load LoRA would skip them too) or the key "
                f"format isn't recognized by key normalization.")

        ordered = {}
        prepared = []
        for entry in grouped.values():
            aliases = sorted(a for a in set(entry["aliases"]) if isinstance(a, str))
            if not aliases:
                canonical = str(entry["target_key"])
                aliases = [canonical]
            else:
                canonical = self._choose_canonical_prefix(aliases)
            prepared.append((entry["is_clip"], canonical, {
                "target_key": entry["target_key"],
                "is_clip": entry["is_clip"],
                "aliases": aliases,
                "label_prefix": canonical,
            }))
        for _is_clip, canonical, entry in sorted(prepared, key=lambda item: (item[0], item[1])):
            ordered[canonical] = entry
        return ordered

    @staticmethod
    def _group_target_key(target_group):
        return target_group["target_key"]

    @staticmethod
    def _target_is_audio(target_group):
        """True if this target group is an audio layer (LTX-2 / ACE-Step style).

        Heuristic: the substring 'audio' appears in the LoRA prefix or the resolved
        model key. Covers LTX-2's audio_embeddings_connector, audio_adaln_single,
        audio_patchify_proj, audio_proj_out, av_ca_audio_* and the per-block audio
        sublayers. Used by the `audio_only` / `no_audio` key_filter modes.
        """
        tk = target_group.get("target_key")
        if isinstance(tk, tuple):
            tk = tk[0]
        parts = [target_group.get("label_prefix", ""), str(tk)]
        parts.extend(a for a in target_group.get("aliases", []) if isinstance(a, str))
        return any("audio" in p.lower() for p in parts)

    def _resolve_target_shape(self, target_key, is_clip, model, clip):
        """Resolve the actual target tensor shape for a target key."""
        offset = None
        if isinstance(target_key, tuple):
            actual_key = target_key[0]
            if len(target_key) > 1:
                offset = target_key[1]
        else:
            actual_key = target_key

        if is_clip:
            target_weight = comfy.utils.get_attr(clip.cond_stage_model, actual_key)
        else:
            target_weight = comfy.utils.get_attr(model.model, actual_key)

        target_shape = list(target_weight.shape)
        if offset is not None:
            target_shape[offset[0]] = offset[2]
        return torch.Size(target_shape)

    def _resolve_base_norm(self, target_key, is_clip, model, clip):
        """Frobenius norm of the base model weight at target_key — the reference
        for magnitude taming's delta/base ratio. Returns None if unreadable. The
        optional slice offset is ignored (full-weight norm is a close enough
        reference for the rare sliced-weight case)."""
        try:
            actual_key = target_key[0] if isinstance(target_key, tuple) else target_key
            if is_clip:
                w = comfy.utils.get_attr(clip.cond_stage_model, actual_key)
            else:
                w = comfy.utils.get_attr(model.model, actual_key)
            return float(w.float().norm().item())
        except (AttributeError, RuntimeError, IndexError, TypeError):
            return None

    @staticmethod
    def _resolve_branch_strength(item, is_clip):
        """Base per-LoRA strength for the target branch before auto-scaling.
        The global clip_strength_multiplier is deliberately NOT applied here —
        it scales the merged clip patches once at add_patches time."""
        if is_clip:
            if item["clip_strength"] is not None:
                return item["clip_strength"]
            return item["strength"]
        return item["strength"]

    def _apply_conflict_modes(self, diffs, eff_strengths, active_loras, merge_refinement="none"):
        """Apply per-LoRA conflict_mode masking to already-aggregated diffs."""
        if len(diffs) <= 1:
            return diffs
        if not any(active_loras[idx].get("conflict_mode", "all") != "all" for idx in diffs):
            return diffs

        indices = sorted(diffs.keys())
        ref = diffs[indices[0]]
        if merge_refinement != "none" and ref.dim() >= 2:
            out_dim = ref.shape[0]
            sign_sum = torch.zeros(out_dim, device=ref.device, dtype=torch.float32)
            for idx in indices:
                effective = diffs[idx] if eff_strengths[idx] >= 0 else -diffs[idx]
                sign_sum += effective.reshape(out_dim, -1).to(dtype=torch.float32).sum(dim=1).sign()
            majority_sign = torch.where(sign_sum >= 0, 1.0, -1.0)
            majority_sign = majority_sign.reshape(-1, *([1] * (ref.dim() - 1))).expand_as(ref)
        else:
            sign_sum = torch.zeros_like(ref, dtype=torch.float32)
            for idx in indices:
                effective = diffs[idx] if eff_strengths[idx] >= 0 else -diffs[idx]
                sign_sum += effective.sign()
            majority_sign = torch.where(sign_sum >= 0, 1.0, -1.0)

        masked = {}
        for idx in indices:
            diff = diffs[idx]
            effective = diff if eff_strengths[idx] >= 0 else -diff
            cm = active_loras[idx].get("conflict_mode", "all")
            if cm == "low_conflict":
                diff = diff * ((effective * majority_sign) > 0).float()
            elif cm == "high_conflict":
                diff = diff * ((effective * majority_sign) < 0).float()
            masked[idx] = diff
        return masked

    def _prepare_group_diffs(self, target_group, active_loras, model, clip, device,
                             clip_strength_multiplier=1.0, merge_refinement="none",
                             auto_scale=1.0, force_cpu=False):
        """
        Aggregate all alias contributions that resolve to the same target weight.
        Returns metadata plus one diff per contributing LoRA after key_filter and
        conflict_mode are applied.
        """
        target_key = self._group_target_key(target_group)
        is_clip = target_group["is_clip"]

        try:
            target_shape = self._resolve_target_shape(target_key, is_clip, model, clip)
        except (AttributeError, RuntimeError, IndexError):
            return None

        device = (torch.device("cpu") if force_cpu else
                  self._select_group_compute_device(
                      device, target_shape, len(active_loras)))
        use_gpu = device is not None and device.type != "cpu"
        if force_cpu and not getattr(self, "_cpu_fallback_reported", False):
            logging.info(
                "[LoRA Optimizer] Unknown third-party patch payload detected; "
                "using the capability-safe CPU path for that target.")
            self._cpu_fallback_reported = True
        if self._execution_stats is not None:
            key = "full_gpu" if use_gpu else "cpu"
            target_id = target_group.get("canonical", target_group.get("label_prefix", repr(target_shape)))
            self._execution_stats[key].add(target_id)
        aggregated = {}
        ranks = {}
        rank_bounds_known = {}
        raw_contributors = set()
        storage_dtype = None
        skip_count = 0
        # Per-LoRA cleaning (opt-in; no-op at defaults). STAR spectral truncate+
        # rescale then base-norm-anchored magnitude taming, applied to each raw
        # per-LoRA diff before it feeds analysis (auto-strength) and merge — so
        # order is clean -> auto-strength -> merge. Preserved LoRAs are exempt.
        _star_eta = getattr(self, '_star_eta', 100.0)
        _tame = getattr(self, '_tame_layers', 0.0)
        _clean_on = _star_eta < 100.0 or _tame > 0.0
        _base_norm = None
        _base_norm_tried = False

        for i, item in enumerate(active_loras):
            self._interrupt_check()
            diff_accum = None
            rank_sum = 0
            rank_bound_known = True

            # Virtual LoRAs from sub-merges store pre-computed diffs keyed by target key
            if item.get("_precomputed_diffs"):
                tkey = target_key
                raw = item["lora"].get(tkey)
                if raw is None and isinstance(tkey, tuple):
                    raw = item["lora"].get(tkey[0])
                if raw is not None:
                    if isinstance(raw, torch.Tensor):
                        diff = raw.float()
                    else:
                        # Move the small low-rank factors to the compute device
                        # BEFORE the up@down expand, so the matmul runs on-device
                        # and only the factors cross the bus — not the big dense
                        # [out x in] result (the previous code expanded on the
                        # factors' native CPU device, then shipped the dense diff
                        # over). Mirrors _compute_lora_diff's factor-first move.
                        # A ("diff", (tensor,)) payload is already dense — moving
                        # it before/after is equivalent; _move_patch_to_device
                        # handles that shape too.
                        expand_src = (self._move_patch_to_device(raw, device)
                                      if use_gpu and device is not None else raw)
                        diff = self._expand_patch_to_diff(expand_src)
                    if device is not None and diff.device != device:
                        diff = diff.to(device)
                    try:
                        diff = diff.reshape(target_shape)
                    except RuntimeError:
                        diff = None
                    if diff is not None:
                        raw_contributors.add(i)
                        # Preloaded items carry real weight adapters, not opaque
                        # dense diffs — read their true rank (mat_down.shape[0])
                        # so analysis does not treat every adapter as rank 1 and
                        # floor compress_rank to 64. Genuine dense diffs (bare tensor / ("diff",
                        # …)) and LoKr return None -> += 1 (rank unknown).
                        _pr = self._payload_rank(raw)
                        _weights = getattr(raw, "weights", ())
                        if (isinstance(_pr, int) and _pr > 0
                                and isinstance(raw, LoRAAdapter)
                                and not isinstance(raw, (LoKrAdapter, LoHaAdapter))
                                and self._is_plain_additive_payload(raw)
                                and len(target_shape) == 2
                                and len(_weights) >= 4
                                and _weights[3] is None
                                and isinstance(_weights[0], torch.Tensor)
                                and isinstance(_weights[1], torch.Tensor)
                                and _weights[0].dim() == 2
                                and _weights[1].dim() == 2):
                            rank_sum += _pr
                        else:
                            # Dense, LoKr and LoHa payloads do not expose a
                            # reliable matrix-rank upper bound. Keep rank=1 for
                            # analysis compatibility, but mark the compression
                            # budget unknown so "smart" never truncates them.
                            rank_sum += 1
                            rank_bound_known = False
                        if storage_dtype is None:
                            storage_dtype = raw.dtype if isinstance(raw, torch.Tensor) else diff.dtype
                        diff_accum = diff
                if diff_accum is not None:
                    aggregated[i] = diff_accum
                    ranks[i] = rank_sum
                    rank_bounds_known[i] = rank_bound_known
                continue

            for alias in target_group["aliases"]:
                lora_info = self._get_lora_key_info(item["lora"], alias)
                linear_rank_bound = False
                if lora_info is not None:
                    _up, _down, _alpha, _mid = lora_info
                    linear_rank_bound = (
                        _mid is None and len(target_shape) == 2
                        and isinstance(_up, torch.Tensor)
                        and isinstance(_down, torch.Tensor)
                        and _up.dim() == 2 and _down.dim() == 2)

                if lora_info is not None:
                    mat_up, mat_down, alpha, mid = lora_info
                    rank_sum += mat_down.shape[0]
                    if not linear_rank_bound:
                        rank_bound_known = False
                    raw_contributors.add(i)
                    if storage_dtype is None:
                        storage_dtype = mat_up.dtype
                    diff = self._compute_lora_diff(
                        mat_up, mat_down, alpha, mid, target_shape,
                        device=device if use_gpu else None,
                        to_cpu=not use_gpu,
                    )
                    if diff is None:
                        # LoRA has this key but its output dim doesn't match the
                        # target model weight — this layer is dropped from the
                        # merge. Surface it in the report instead of dropping it
                        # silently (the LoRA was likely trained on a different
                        # model variant/layout).
                        self._note_shape_mismatch(
                            item, target_key, int(mat_up.shape[0]),
                            int(target_shape[0]) if len(target_shape) > 0 else None)
                else:
                    # Try LoKr / LoHa formats
                    alt = self._get_lokr_diff(
                        item["lora"], alias,
                        device=device if use_gpu else None, to_cpu=not use_gpu,
                    )
                    if alt is None:
                        alt = self._get_loha_diff(
                            item["lora"], alias,
                            device=device if use_gpu else None, to_cpu=not use_gpu,
                        )
                    if alt is not None:
                        diff, alt_rank, alt_dtype = alt
                        try:
                            reshaped = diff.reshape(target_shape)
                        except RuntimeError:
                            # LoKr/LoHa reconstructs to a shape that doesn't match
                            # the target model weight (e.g. a narrow block trained
                            # on a different model variant) — this layer is dropped.
                            # Record it for the report instead of dropping silently.
                            self._note_shape_mismatch(
                                item, target_key,
                                int(diff.shape[0]) if diff.dim() > 0 else None,
                                int(target_shape[0]) if len(target_shape) > 0 else None)
                            reshaped = None
                        diff = reshaped
                        if diff is not None:
                            rank_sum += alt_rank
                            # The reported LoKr/LoHa rank is useful for
                            # analysis, but is not an upper bound on the rank of
                            # the reconstructed Kronecker/Hadamard product.
                            rank_bound_known = False
                            raw_contributors.add(i)
                            if storage_dtype is None:
                                storage_dtype = alt_dtype
                    else:
                        diff = None

                if diff is not None:
                    diff = diff.float()
                    diff_accum = diff if diff_accum is None else diff_accum + diff

            if diff_accum is not None:
                if _clean_on and not active_loras[i].get("preserve", False):
                    if _star_eta < 100.0:
                        diff_accum = self._star_truncate_rescale(diff_accum, _star_eta)
                    if _tame > 0.0:
                        if not _base_norm_tried:
                            _base_norm = self._resolve_base_norm(target_key, is_clip, model, clip)
                            _base_norm_tried = True
                        if _base_norm:
                            sc = self._tame_scale(
                                diff_accum.float().norm().item(), _base_norm,
                                getattr(self, '_tame_threshold', 0.3), _tame)
                            if sc != 1.0:
                                diff_accum = diff_accum * sc
                aggregated[i] = diff_accum
                ranks[i] = rank_sum
                rank_bounds_known[i] = rank_bound_known
            elif i in raw_contributors:
                pass  # contributed via cache but diff_accum ended up None (shouldn't happen)
            else:
                skip_count += 1

        raw_n = len(aggregated)
        if raw_n == 0:
            if skip_count > 0:
                return {
                    "label_prefix": target_group["label_prefix"],
                    "target_key": target_key,
                    "is_clip": is_clip,
                    "raw_n_loras": 0,
                    "diffs": {},
                    "eff_strengths": {},
                    "rank_sums": {},
                    "rank_bound": None,
                    "target_shape": target_shape,
                    "storage_dtype": storage_dtype,
                    "skip_count": skip_count,
                    "compute_device": device if device is not None else torch.device("cpu"),
                }
            return None

        filtered = {}
        eff_strengths = {}
        rank_sums = {}
        filtered_rank_bounds_known = {}
        is_audio_group = self._target_is_audio(target_group)
        for i, diff in aggregated.items():
            kf = active_loras[i].get("key_filter", "all")
            if kf == "shared_only" and raw_n < 2:
                continue
            if kf == "unique_only" and raw_n != 1:
                continue
            if kf == "audio_only" and not is_audio_group:
                continue
            if kf == "no_audio" and is_audio_group:
                continue
            filtered[i] = diff
            eff_strengths[i] = self._resolve_branch_strength(
                active_loras[i], is_clip
            ) * auto_scale
            rank_sums[i] = ranks.get(i, 0)
            filtered_rank_bounds_known[i] = rank_bounds_known.get(i, False)

        if not filtered:
            return {
                "label_prefix": target_group["label_prefix"],
                "target_key": target_key,
                "is_clip": is_clip,
                "raw_n_loras": raw_n,
                "diffs": {},
                "eff_strengths": {},
                "rank_sums": {},
                "rank_bound": None,
                "target_shape": target_shape,
                "storage_dtype": storage_dtype,
                "skip_count": skip_count,
                "compute_device": device if device is not None else torch.device("cpu"),
            }

        filtered = self._apply_conflict_modes(
            filtered, eff_strengths, active_loras, merge_refinement=merge_refinement
        )

        rank_bound = None
        if all(filtered_rank_bounds_known.values()):
            rank_bound = sum(rank_sums.values())

        return {
            "label_prefix": target_group["label_prefix"],
            "target_key": target_key,
            "is_clip": is_clip,
            "raw_n_loras": raw_n,
            "diffs": filtered,
            "eff_strengths": eff_strengths,
            "rank_sums": rank_sums,
            "rank_bound": rank_bound,
            "target_shape": target_shape,
            "storage_dtype": storage_dtype,
            "skip_count": skip_count,
            "compute_device": device if device is not None else torch.device("cpu"),
        }

    def _get_lora_key_info(self, lora_dict, key_prefix):
        """
        Extracts LoRA information for the given key.
        Returns (mat_up, mat_down, alpha, mid) or None for standard LoRA.
        """
        # LoRA key formats
        formats = [
            ("{}.lora_up.weight", "{}.lora_down.weight"),           # regular
            ("{}_lora.up.weight", "{}_lora.down.weight"),           # diffusers
            ("{}.lora_B.weight", "{}.lora_A.weight"),               # diffusers2
            ("{}.lora.up.weight", "{}.lora.down.weight"),           # diffusers3
        ]

        def _extract(up_key, down_key):
            mat_up = lora_dict[up_key]
            mat_down = lora_dict[down_key]
            alpha_key = "{}.alpha".format(key_prefix)
            alpha = lora_dict.get(alpha_key, None)
            if alpha is not None:
                alpha = alpha.item()
            else:
                alpha = mat_down.shape[0]
            mid_key = "{}.lora_mid.weight".format(key_prefix)
            mid = lora_dict.get(mid_key, None)
            return (mat_up, mat_down, alpha, mid)

        # Try cached format first
        dict_id = id(lora_dict)
        cached_fmt = self._lora_format_cache.get(dict_id)
        if cached_fmt is not None:
            up_key = formats[cached_fmt][0].format(key_prefix)
            down_key = formats[cached_fmt][1].format(key_prefix)
            if up_key in lora_dict and down_key in lora_dict:
                return _extract(up_key, down_key)

        for fmt_idx, (up_fmt, down_fmt) in enumerate(formats):
            up_key = up_fmt.format(key_prefix)
            down_key = down_fmt.format(key_prefix)
            if up_key in lora_dict and down_key in lora_dict:
                self._lora_format_cache[dict_id] = fmt_idx
                return _extract(up_key, down_key)

        return None

    @staticmethod
    def _has_lokr_keys(lora_dict, key_prefix):
        """Check if lora_dict has LoKr keys for the given prefix (no tensor loading)."""
        p = key_prefix
        has_w1 = f"{p}.lokr_w1" in lora_dict or (f"{p}.lokr_w1_a" in lora_dict and f"{p}.lokr_w1_b" in lora_dict)
        has_w2 = f"{p}.lokr_w2" in lora_dict or (f"{p}.lokr_w2_a" in lora_dict and f"{p}.lokr_w2_b" in lora_dict)
        return has_w1 and has_w2

    @staticmethod
    def _has_loha_keys(lora_dict, key_prefix):
        """Check if lora_dict has LoHa keys for the given prefix (no tensor loading)."""
        p = key_prefix
        return (f"{p}.hada_w1_a" in lora_dict and f"{p}.hada_w1_b" in lora_dict
                and f"{p}.hada_w2_a" in lora_dict and f"{p}.hada_w2_b" in lora_dict)

    def _get_lokr_diff(self, lora_dict, key_prefix, device=None, to_cpu=True):
        """
        Extract LoKr (Kronecker) factors and compute full diff.
        Returns (diff, rank, dtype) or None.
        """
        p = key_prefix
        w1 = lora_dict.get(f"{p}.lokr_w1")
        w2 = lora_dict.get(f"{p}.lokr_w2")
        w1_a = lora_dict.get(f"{p}.lokr_w1_a")
        w1_b = lora_dict.get(f"{p}.lokr_w1_b")
        w2_a = lora_dict.get(f"{p}.lokr_w2_a")
        w2_b = lora_dict.get(f"{p}.lokr_w2_b")
        t2 = lora_dict.get(f"{p}.lokr_t2")

        has_w1 = w1 is not None or (w1_a is not None and w1_b is not None)
        has_w2 = w2 is not None or (w2_a is not None and w2_b is not None)
        if not (has_w1 and has_w2):
            return None

        alpha = lora_dict.get(f"{p}.alpha")
        if alpha is not None:
            alpha = alpha.item()

        dim = None
        ref_tensor = w1 if w1 is not None else (w1_a if w1_a is not None else w2_a)
        dtype = ref_tensor.dtype

        if device is not None:
            w1 = w1.to(device) if w1 is not None else None
            w2 = w2.to(device) if w2 is not None else None
            w1_a = w1_a.to(device) if w1_a is not None else None
            w1_b = w1_b.to(device) if w1_b is not None else None
            w2_a = w2_a.to(device) if w2_a is not None else None
            w2_b = w2_b.to(device) if w2_b is not None else None
            t2 = t2.to(device) if t2 is not None else None

        if w1 is None:
            dim = w1_b.shape[0]
            w1 = torch.mm(w1_a.float(), w1_b.float())
        else:
            w1 = w1.float()
        if w2 is None:
            dim = w2_b.shape[0]
            if t2 is None:
                w2 = torch.mm(w2_a.float(), w2_b.float())
            else:
                w2 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    t2.float(), w2_b.float(), w2_a.float(),
                )
        else:
            w2 = w2.float()

        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)

        scale = alpha / dim if (alpha is not None and dim is not None) else 1.0
        diff = torch.kron(w1, w2) * scale
        rank = dim if dim is not None else min(w1.shape)

        if to_cpu and diff.device.type != "cpu":
            diff = diff.cpu()
        return (diff, rank, dtype)

    def _get_loha_diff(self, lora_dict, key_prefix, device=None, to_cpu=True):
        """
        Extract LoHa (Hadamard) factors and compute full diff.
        Returns (diff, rank, dtype) or None.
        """
        p = key_prefix
        w1a = lora_dict.get(f"{p}.hada_w1_a")
        w1b = lora_dict.get(f"{p}.hada_w1_b")
        w2a = lora_dict.get(f"{p}.hada_w2_a")
        w2b = lora_dict.get(f"{p}.hada_w2_b")
        if w1a is None or w1b is None or w2a is None or w2b is None:
            return None

        t1 = lora_dict.get(f"{p}.hada_t1")
        t2 = lora_dict.get(f"{p}.hada_t2")
        alpha = lora_dict.get(f"{p}.alpha")
        if alpha is not None:
            alpha = alpha.item()

        dtype = w1a.dtype
        rank = w1b.shape[0]

        if device is not None:
            w1a = w1a.to(device)
            w1b = w1b.to(device)
            w2a = w2a.to(device)
            w2b = w2b.to(device)
            t1 = t1.to(device) if t1 is not None else None
            t2 = t2.to(device) if t2 is not None else None

        if t1 is not None:
            m1 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                t1.float(), w1b.float(), w1a.float(),
            )
            m2 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                t2.float(), w2b.float(), w2a.float(),
            )
        else:
            m1 = torch.mm(w1a.float(), w1b.float())
            m2 = torch.mm(w2a.float(), w2b.float())

        scale = alpha / rank if alpha is not None else 1.0
        diff = (m1 * m2) * scale

        if to_cpu and diff.device.type != "cpu":
            diff = diff.cpu()
        return (diff, rank, dtype)
    
    def _compute_lora_diff(self, mat_up, mat_down, alpha, mid, target_shape, device=None, to_cpu=True):
        """
        Computes full diff for a single LoRA.
        diff = mat_up @ mat_down × (alpha / rank)
        When device is given, matrices are moved there for faster matmul,
        then the result is returned on CPU to avoid VRAM accumulation.
        """
        rank = mat_down.shape[0]
        scale = alpha / rank

        if device is not None:
            mat_up = mat_up.to(device)
            mat_down = mat_down.to(device)
            if mid is not None:
                mid = mid.to(device)

        if mid is not None:
            # LoCon with mid matrix (rare)
            final_shape = [mat_down.shape[1], mat_down.shape[0], mid.shape[2], mid.shape[3]]
            mat_down = (
                torch.mm(
                    mat_down.transpose(0, 1).flatten(start_dim=1).float(),
                    mid.transpose(0, 1).flatten(start_dim=1).float(),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )

        # Compute diff
        diff = torch.mm(
            mat_up.flatten(start_dim=1).float(),
            mat_down.flatten(start_dim=1).float()
        )

        # Try to reshape to target shape
        try:
            diff = diff.reshape(target_shape)
        except RuntimeError:
            # If shape doesn't match, skip
            return None

        # A large DiT projection can be hundreds of MB in fp32. Scaling out of
        # place briefly duplicates it and can OOM after the matmul already fit.
        diff.mul_(scale)
        if to_cpu and device is not None and device.type != "cpu":
            return diff.cpu()
        return diff

    def _note_shape_mismatch(self, item, target_key, lora_dim, model_dim):
        """Record a LoRA→model shape incompatibility for the merge report.

        Called when a LoRA has a tensor at a target layer but its output
        dimension doesn't match the model weight, so that layer is dropped from
        the merge. Best-effort (runs in analysis worker threads), capped, and
        deduped by target key so each dropped layer counts once per LoRA.
        """
        try:
            store = self._shape_mismatches
        except AttributeError:
            store = self._shape_mismatches = {}
        name = item.get("name", "?") if isinstance(item, dict) else str(item)
        key = target_key[0] if isinstance(target_key, tuple) else target_key
        per = store.setdefault(name, {})
        skey = str(key)
        if skey not in per and len(per) < 256:
            per[skey] = (lora_dim, model_dim)

    def _shape_mismatch_report_lines(self):
        """Render the shape-incompatibility warning for the merge report from the
        mismatches recorded in _note_shape_mismatch. Empty when there are none."""
        mismatches = getattr(self, '_shape_mismatches', None)
        if not mismatches:
            return []
        total = sum(len(v) for v in mismatches.values())
        lines = [
            "",
            f"  !! SHAPE INCOMPATIBILITY — {total} layer(s) DROPPED from the merge",
            "     A LoRA's tensor shape does not match the target model at these",
            "     layers, so they could not be merged and were skipped. The LoRA's",
            "     effect at these layers is MISSING from the result.",
            "     Likely cause: this LoRA was trained on a DIFFERENT model variant",
            "     or block layout than the model you are merging into.",
        ]
        for name, per in mismatches.items():
            short = name.split('/')[-1].split('\\')[-1]
            lines.append(f"     - {short}: {len(per)} layer(s) dropped")
            for k, (ld, md) in list(per.items())[:3]:
                lines.append(f"         {k}: LoRA dim={ld} vs model dim={md}")
            if len(per) > 3:
                lines.append(f"         ... and {len(per) - 3} more")
        lines.append("     Fix: merge LoRAs trained on the SAME base model, or accept that")
        lines.append("     these layers come only from the other LoRA(s)/base model.")
        return lines

    # --- Per-LoRA cleaning (opt-in Pass-1 preprocessing) ---
    # Feature inspired by CoreyCorza's comfyui-lora-loader
    # (https://github.com/CoreyCorza/comfyui-lora-loader): per-LoRA noise-tail
    # removal + layer taming. Reworked here to research-validated methods — STAR
    # spectral truncate+rescale (arXiv:2502.10339) and base-norm-anchored
    # magnitude taming / Norm-Anchor Scaling (arXiv:2602.02543).
    @staticmethod
    def _star_truncate_rescale(diff, eta):
        """STAR spectral truncate + nuclear-norm rescale (arXiv:2502.10339, NAACL 2025).

        SVD the per-layer delta, keep the top singular components whose cumulative
        singular-value sum first reaches eta% of the NUCLEAR norm (Σσ, not Σσ²),
        then rescale the kept singular values by (Σσ / Σ_kept σ) so the nuclear
        norm is restored. Truncation lowers the inter-task conflict bound for
        merging; the rescale keeps the delta's overall magnitude (the ablation
        that makes STAR work). eta >= 100 is a no-op. Data-free, per matrix.
        """
        if eta is None or eta >= 100.0:
            return diff
        if diff.ndim < 2:
            return diff
        orig_shape = diff.shape
        mat = diff.reshape(diff.shape[0], -1).float() if diff.ndim > 2 else diff.float()
        svd = _full_svd_robust(mat)
        if svd is None:
            return diff
        U, S, Vh = svd
        total = S.sum()
        if total.item() <= 0:
            return diff
        # smallest r whose cumulative singular-value sum reaches eta% of the total
        r = int((torch.cumsum(S, 0) < (eta / 100.0) * total).sum().item()) + 1
        r = max(1, min(r, S.shape[0]))
        kept = S[:r]
        kept_sum = kept.sum()
        if kept_sum.item() <= 0:
            return diff
        s_new = kept * (total / kept_sum)   # restore nuclear norm
        out = (U[:, :r] * s_new) @ Vh[:r, :]
        return out.reshape(orig_shape).to(diff.dtype)

    @staticmethod
    def _tame_scale(delta_norm, base_norm, threshold, strength, eps=1e-8):
        """Base-norm-anchored magnitude taming (Norm-Anchor Scaling, arXiv:2602.02543).

        A layer whose delta Frobenius norm is a large fraction of its BASE weight's
        norm is "hot" — it overwrites the base and pushes activations off-manifold.
        r = ||ΔW|| / max(||W_base||, eps); if r <= threshold the layer is within
        budget (scale 1.0), else scale by (threshold/r)^strength. strength=0 is a
        no-op; strength=1 scales the delta down to exactly threshold x base norm.
        Denominator floored to avoid the divide-by-small-norm singularity.
        """
        if strength <= 0.0 or threshold <= 0.0:
            return 1.0
        r = float(delta_norm) / max(float(base_norm), eps)
        if r <= threshold:
            return 1.0
        return (threshold / r) ** float(strength)

    @staticmethod
    def _stable_data_hash(value):
        """Create a compact stable hash for nested JSON-like data and tensors."""
        def _normalize(obj):
            if isinstance(obj, torch.Tensor):
                t = obj.detach().float().cpu()
                sample = t.flatten()[:16].tolist()
                return {
                    "__tensor__": True,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "sample": sample,
                }
            if isinstance(obj, dict):
                return {str(k): _normalize(obj[k]) for k in sorted(obj.keys(), key=str)}
            if isinstance(obj, (list, tuple)):
                return [_normalize(v) for v in obj]
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return repr(obj)

        payload = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _safe_unit_clamp(value):
        return max(-1.0, min(1.0, float(value)))


    @staticmethod
    def _compute_subspace_basis(diff, rank_hint=None, max_rank=8):
        if diff is None:
            return None

        if diff.dim() >= 2:
            mat = diff.reshape(diff.shape[0], -1).float()
        else:
            mat = diff.float().reshape(1, -1)

        if min(mat.shape) <= 0:
            return None

        rank_cap = min(max_rank, min(mat.shape))
        if rank_hint is not None:
            rank_cap = min(rank_cap, max(1, int(rank_hint)))
        rank_cap = max(rank_cap, 1)

        try:
            q = min(rank_cap, min(mat.shape))
            U, _S, V = torch.svd_lowrank(mat, q=q)
            return {"left": U[:, :q], "right": V[:, :q]}
        except Exception:
            try:
                left, _ = torch.linalg.qr(mat, mode="reduced")
                right, _ = torch.linalg.qr(mat.T, mode="reduced")
                return {
                    "left": left[:, :rank_cap],
                    "right": right[:, :rank_cap],
                }
            except Exception:
                return None

    @classmethod
    def _compute_subspace_overlap(cls, basis_a, basis_b):
        if not basis_a or not basis_b:
            return 0.0

        def _project_overlap(q_a, q_b):
            if q_a is None or q_b is None or q_a.numel() == 0 or q_b.numel() == 0:
                return None
            cross = torch.mm(q_a.transpose(0, 1), q_b)
            denom = max(1, min(q_a.shape[1], q_b.shape[1]))
            value = (cross.square().sum() / denom).item()
            return min(max(value, 0.0), 1.0)

        left = _project_overlap(basis_a.get("left"), basis_b.get("left"))
        right = _project_overlap(basis_a.get("right"), basis_b.get("right"))
        values = [v for v in (left, right) if v is not None]
        return sum(values) / len(values) if values else 0.0

    def _sample_pair_metrics(self, diff_a, diff_b, basis_a=None, basis_b=None, device=None):
        """
        Pairwise overlap/conflict metrics with a magnitude-aware noise floor,
        excess-conflict baseline, and optional subspace overlap.
        """
        flat_a = diff_a.flatten()
        flat_b = diff_b.flatten()
        if device is not None:
            flat_a = flat_a.to(device=device, dtype=torch.float32)
            flat_b = flat_b.to(device=device, dtype=torch.float32)
        elif flat_a.dtype != torch.float32:
            flat_a = flat_a.float()
            flat_b = flat_b.float()

        if flat_a.numel() != flat_b.numel():
            return {
                "overlap": 0,
                "conflict": 0,
                "dot": 0.0,
                "norm_a_sq": 0.0,
                "norm_b_sq": 0.0,
                "weighted_total": 0.0,
                "weighted_conflict": 0.0,
                "expected_conflict": 0.0,
                "excess_conflict": 0.0,
                "subspace_overlap": 0.0,
                "subspace_weight": 0.0,
            }

        n = flat_a.numel()
        sample_scale = 1.0
        if n > 100000:
            target_device = flat_a.device
            g = torch.Generator(device=target_device).manual_seed(42)
            indices = torch.randint(0, n, (100000,), device=target_device, generator=g)
            flat_a = flat_a[indices]
            flat_b = flat_b[indices]
            # Sampled sums estimate (k/n) of the full-tensor sums; rescale
            # dot/norm_* back to full scale so they are comparable with the
            # exact per-LoRA norms accumulated in branch_energy (energy
            # cross-terms) and so cross-prefix weighting is element-count
            # proportional instead of sample-count proportional.
            sample_scale = n / 100000.0

        mask = (flat_a != 0) & (flat_b != 0)
        n_overlap = mask.sum().item()
        if n_overlap == 0:
            return {
                "overlap": 0,
                "conflict": 0,
                "dot": 0.0,
                "norm_a_sq": 0.0,
                "norm_b_sq": 0.0,
                "weighted_total": 0.0,
                "weighted_conflict": 0.0,
                "expected_conflict": 0.0,
                "excess_conflict": 0.0,
                "subspace_overlap": 0.0,
                "subspace_weight": 0.0,
            }

        a_overlap = flat_a[mask]
        b_overlap = flat_b[mask]
        dot = (a_overlap * b_overlap).sum().item() * sample_scale
        norm_a_sq = (a_overlap * a_overlap).sum().item() * sample_scale
        norm_b_sq = (b_overlap * b_overlap).sum().item() * sample_scale
        n_conflict = (a_overlap.sign() != b_overlap.sign()).sum().item()

        a_rms = a_overlap.square().mean().sqrt()
        b_rms = b_overlap.square().mean().sqrt()
        noise_floor = max(a_rms.item(), b_rms.item()) * 0.05
        strong_mask = (a_overlap.abs() > noise_floor) & (b_overlap.abs() > noise_floor)
        if strong_mask.any():
            a_strong = a_overlap[strong_mask]
            b_strong = b_overlap[strong_mask]
        else:
            a_strong = a_overlap
            b_strong = b_overlap

        weights = torch.minimum(a_strong.abs(), b_strong.abs())
        weighted_total = weights.sum().item()
        mismatch = a_strong.sign() != b_strong.sign()
        weighted_conflict = weights[mismatch].sum().item() if weighted_total > 0 else 0.0
        weighted_ratio = (weighted_conflict / weighted_total) if weighted_total > 0 else 0.0

        # Excess conflict: measured sign-mismatch beyond the rate implied by the
        # correlation. Sheppard's theorem / the degree-0 arc-cosine kernel give
        # the UNWEIGHTED expectation P(mismatch) = arccos(rho)/pi — so both the
        # measured fraction and the correlation must be the plain (unweighted)
        # statistics of the SAME position set. The previous magnitude-weighted
        # ratio sat systematically below this baseline for correlated LoRAs
        # (mismatches concentrate on small-|min| positions; cf. Cho & Saul's
        # J1 != J0 angular dependence), under-detecting real conflict.
        n_strong = a_strong.numel()
        strong_mismatch_frac = (mismatch.sum().item() / n_strong) if n_strong > 0 else 0.0
        dot_strong = (a_strong * b_strong).sum().item()
        na_strong = (a_strong * a_strong).sum().item()
        nb_strong = (b_strong * b_strong).sum().item()
        denom_strong = (math.sqrt(na_strong * nb_strong)
                        if na_strong > 0 and nb_strong > 0 else 0.0)
        cos_strong = (self._safe_unit_clamp(dot_strong / denom_strong)
                      if denom_strong > 0 else 0.0)
        expected_conflict = math.acos(cos_strong) / math.pi if denom_strong > 0 else 0.0
        excess_conflict = max(strong_mismatch_frac - expected_conflict, 0.0)

        subspace_overlap = self._compute_subspace_overlap(basis_a, basis_b)
        subspace_weight = math.sqrt(norm_a_sq * norm_b_sq) if norm_a_sq > 0 and norm_b_sq > 0 else 0.0

        return {
            "overlap": n_overlap,
            "conflict": n_conflict,
            "dot": dot,
            "norm_a_sq": norm_a_sq,
            "norm_b_sq": norm_b_sq,
            "weighted_total": weighted_total,
            "weighted_conflict": weighted_conflict,
            "expected_conflict": expected_conflict,
            "excess_conflict": excess_conflict,
            "subspace_overlap": subspace_overlap,
            "subspace_weight": subspace_weight,
        }

    @staticmethod
    def _ties_trim(tensor, density):
        """
        TIES Step 1: Trim — keep only the top-k% values by absolute magnitude.
        Everything else is zeroed out (noise removal).
        """
        flat = tensor.flatten()
        n = flat.numel()
        k = max(1, int(n * density))
        if k >= n:
            return tensor
        _, indices = torch.topk(flat.abs(), k)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[indices] = True
        return (flat * mask).reshape(tensor.shape)

    @staticmethod
    def _dare_sparsify(tensor, density, generator=None, dampening=0.0):
        """
        DARE sparsification: randomly drop parameters and rescale survivors.
        Each element is kept with probability `density`, then rescaled by 1/q.
        With dampening=0: q=density (standard DARE, exact 1/(1-p) rescale).
        With dampening>0: q is interpolated toward 1.0, a data-free heuristic
        INSPIRED BY DAREx-q (ICLR 2025) — the paper itself selects q>1-p
        empirically (validation grid search / output-difference minimization),
        not via this closed form. Dampening>0 intentionally biases the
        estimator low (E[out] = density/q < 1) to tame variance blow-up.
        """
        if density >= 1.0:
            return tensor
        mask = torch.bernoulli(
            torch.full(tensor.shape, density, dtype=tensor.dtype, device=tensor.device),
            generator=generator
        )
        q = density + dampening * (1.0 - density)
        return tensor * mask * (1.0 / q)

    @staticmethod
    def _della_sparsify(tensor, density, epsilon=0.3, generator=None):
        """
        DELLA sparsification: magnitude-aware dropout.
        Low-magnitude elements are dropped with higher probability.
        Survivors are rescaled by 1/(1-p_i) to preserve expected value.
        """
        if density >= 1.0:
            return tensor
        original_shape = tensor.shape
        mat = tensor.unsqueeze(0) if tensor.dim() < 2 else tensor.reshape(tensor.shape[0], -1)
        nrows, ncols = mat.shape
        p_min = max((1.0 - density) - epsilon / 2.0, 0.0)
        # Double-argsort gives ascending magnitude ranks; invert so low-magnitude
        # elements get HIGH drop probability (rank 0 = highest magnitude → p_min)
        asc_ranks = mat.abs().argsort(dim=1).argsort(dim=1).float()
        ranks = (ncols - 1) - asc_ranks
        drop_probs = (p_min + (epsilon / ncols) * ranks).clamp(0.0, 1.0)
        keep_probs = 1.0 - drop_probs
        mask = torch.bernoulli(keep_probs, generator=generator)
        rescale = torch.where(mask > 0, 1.0 / keep_probs.clamp(min=1e-6), torch.zeros_like(keep_probs))
        return (mat * mask * rescale).reshape(original_shape)

    @staticmethod
    def _compute_conflict_mask(diffs_with_weights):
        """
        Boolean mask: True where 2+ diffs have opposing signs (actual interference).
        Uses sign-corrected diffs (weight sign applied).
        """
        has_positive = torch.zeros_like(diffs_with_weights[0][0], dtype=torch.bool)
        has_negative = torch.zeros_like(has_positive)
        for diff, weight in diffs_with_weights:
            effective = diff if weight >= 0 else -diff
            nonzero = effective != 0
            has_positive |= (nonzero & (effective > 0))
            has_negative |= (nonzero & (effective < 0))
        return has_positive & has_negative

    @staticmethod
    def _dare_sparsify_conflict(tensor, conflict_mask, density, generator=None, dampening=0.0):
        if density >= 1.0:
            return tensor
        rand_mask = torch.bernoulli(
            torch.full(tensor.shape, density, dtype=tensor.dtype, device=tensor.device),
            generator=generator
        )
        q = density + dampening * (1.0 - density)
        return torch.where(conflict_mask, tensor * rand_mask * (1.0 / q), tensor)

    @staticmethod
    def _della_sparsify_conflict(tensor, conflict_mask, density, epsilon=0.3, generator=None):
        if density >= 1.0:
            return tensor
        della_result = _LoRAMergeBase._della_sparsify(tensor, density, epsilon, generator)
        return torch.where(conflict_mask, della_result, tensor)

    @staticmethod
    def _estimate_patch_memory(patches_dict):
        """Estimate total bytes used by patch tensors in an add_patches-style dict."""
        total = 0
        for v in patches_dict.values():
            # LoRAAdapter: .weights = (mat_up, mat_down, alpha, ...)
            # diff patch:  ("diff", (tensor,))
            data = v.weights if hasattr(v, 'weights') else v
            if isinstance(data, (tuple, list)):
                for item in data:
                    if isinstance(item, torch.Tensor):
                        total += item.nelement() * item.element_size()
                    elif isinstance(item, (tuple, list)):
                        for sub in item:
                            if isinstance(sub, torch.Tensor):
                                total += sub.nelement() * sub.element_size()
        return total

    @staticmethod
    def _estimate_single_patch_bytes(patch):
        """Estimate byte size of a single patch entry (diff tuple or LoRAAdapter)."""
        total = 0
        data = patch.weights if hasattr(patch, 'weights') else patch
        if isinstance(data, (tuple, list)):
            for item in data:
                if isinstance(item, torch.Tensor):
                    total += item.nelement() * item.element_size()
                elif isinstance(item, (tuple, list)):
                    for sub in item:
                        if isinstance(sub, torch.Tensor):
                            total += sub.nelement() * sub.element_size()
        return total

    @staticmethod
    def _expand_patch_to_diff(patch):
        """Expand a patch (diff tuple, LoRAAdapter, LoKrAdapter, LoHaAdapter) to a float32 diff tensor."""
        if isinstance(patch, tuple):
            if _LoRAMergeBase._is_plain_additive_payload(patch):
                return patch[1][0].float()
            raise ValueError("Cannot expand an order-dependent or malformed tuple patch as a plain diff")
        if (isinstance(patch, (LoRAAdapter, LoKrAdapter, LoHaAdapter))
                and not _LoRAMergeBase._is_plain_additive_payload(patch)):
            raise ValueError("Cannot expand a base-dependent DoRA/reshape adapter as a plain diff")
        if isinstance(patch, LoKrAdapter):
            weights = patch.weights
            w1, w2, alpha = weights[0], weights[1], weights[2]
            w1_a, w1_b = weights[3], weights[4]
            w2_a, w2_b, t2 = weights[5], weights[6], weights[7]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            dim = None
            if w1 is None:
                dim = w1_b.shape[0]
                w1 = torch.mm(w1_a.float(), w1_b.float())
            else:
                w1 = w1.float()
            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    w2 = torch.mm(w2_a.float(), w2_b.float())
                else:
                    w2 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        t2.float(), w2_b.float(), w2_a.float(),
                    )
            else:
                w2 = w2.float()
            if len(w2.shape) == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)
            scale = alpha / dim if (alpha is not None and dim is not None) else 1.0
            return torch.kron(w1, w2) * scale
        elif isinstance(patch, LoHaAdapter):
            weights = patch.weights
            w1a, w1b, alpha = weights[0], weights[1], weights[2]
            w2a, w2b = weights[3], weights[4]
            t1, t2 = weights[5], weights[6]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            rank = w1b.shape[0]
            if t1 is not None:
                m1 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    t1.float(), w1b.float(), w1a.float(),
                )
                m2 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    t2.float(), w2b.float(), w2a.float(),
                )
            else:
                m1 = torch.mm(w1a.float(), w1b.float())
                m2 = torch.mm(w2a.float(), w2b.float())
            scale = alpha / rank if alpha is not None else 1.0
            return (m1 * m2) * scale
        elif isinstance(patch, LoRAAdapter):
            w = patch.weights
            mat_up, mat_down, alpha = w[0], w[1], w[2]
            mid = w[3]
            rank = mat_down.shape[0]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            if mid is not None:
                final_shape = [mat_down.shape[1], mat_down.shape[0], mid.shape[2], mid.shape[3]]
                mat_down = torch.mm(
                    mat_down.transpose(0, 1).flatten(start_dim=1).float(),
                    mid.transpose(0, 1).flatten(start_dim=1).float(),
                ).reshape(final_shape).transpose(0, 1)
            diff = torch.mm(
                mat_up.flatten(start_dim=1).float(),
                mat_down.flatten(start_dim=1).float(),
            )
            # alpha=None (PEFT/diffusers/AI-Toolkit LoRAs with no .alpha key)
            # means no alpha/rank rescale — use up@down as-is, matching comfy
            # (weight_adapter/lora.py) and the sibling LoKr/LoHa branches above.
            return diff * (alpha / rank if alpha is not None else 1.0)
        raise ValueError(f"Unknown patch format: {type(patch)}")

    @staticmethod
    def _move_patch_to_device(patch, device):
        """Move all tensors in a patch to the given device. Returns new patch."""
        if hasattr(patch, 'weights'):
            inner = patch.weights
            moved = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t
                for t in inner
            )
            if isinstance(patch, LoKrAdapter):
                return LoKrAdapter(patch.loaded_keys, moved)
            if isinstance(patch, LoHaAdapter):
                return LoHaAdapter(patch.loaded_keys, moved)
            return LoRAAdapter(patch.loaded_keys, moved)
        elif isinstance(patch, tuple) and len(patch) == 2 and isinstance(patch[0], str):
            # ("diff", (tensor,))
            moved_inner = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t
                for t in patch[1]
            )
            return (patch[0], moved_inner)
        return patch

    @staticmethod
    def _update_model_size(patcher, patches_dict):
        """Update a ModelPatcher's reported size to include patch memory.

        ComfyUI's model_size() only counts base model weights, making patches
        invisible to the memory manager. This causes large patched models to
        never be evicted when other models need RAM. By updating the cached
        size, ComfyUI's eviction logic sees the true memory footprint.
        """
        patch_bytes = _LoRAMergeBase._estimate_patch_memory(patches_dict)
        if patch_bytes > 0 and hasattr(patcher, 'model_size'):
            patcher.size = patcher.model_size() + patch_bytes
            logging.debug(f"[LoRA Optimizer] Updated model size: +{patch_bytes / (1024**2):.0f}MB patches")

    @staticmethod
    def _ties_elect_sign(trimmed_diffs, method="total"):
        """
        TIES Step 2: Elect Sign — determine majority sign direction per weight position.

        Args:
            trimmed_diffs: list of trimmed diff tensors (same shape)
            method: "frequency" (count votes) or "total" (sum magnitudes)

        Returns:
            majority_sign: tensor of +1/-1 per position
        """
        # Iterate instead of torch.stack to avoid allocating [N, *shape] tensor
        ref = trimmed_diffs[0]
        total = torch.zeros_like(ref, dtype=torch.float32)
        if method == "total":
            for d in trimmed_diffs:
                total.add_(d.to(dtype=torch.float32))
        else:
            # sign() returns -1/0/+1: zeros don't vote, matching original behavior
            for d in trimmed_diffs:
                total.add_(d.sign())
        # +1 where majority is positive or tied, -1 where majority is negative
        majority_sign = torch.where(total >= 0,
                                    torch.tensor(1.0, device=total.device, dtype=total.dtype),
                                    torch.tensor(-1.0, device=total.device, dtype=total.dtype))
        return majority_sign

    @staticmethod
    def _columnwise_elect_sign(trimmed_diffs, method="total"):
        """
        Column-wise sign election: each output neuron (row) votes as a unit
        instead of each element voting independently. For Conv2d tensors,
        [c_out, c_in, kH, kW] is reshaped to [c_out, -1] so each output
        channel votes together.

        Falls back to element-wise for 1D tensors (biases, norms).
        """
        ref = trimmed_diffs[0]
        if ref.dim() < 2:
            return _LoRAMergeBase._ties_elect_sign(trimmed_diffs, method)

        out_dim = ref.shape[0]
        total = torch.zeros(out_dim, device=ref.device, dtype=torch.float32)
        for d in trimmed_diffs:
            row_vals = d.reshape(out_dim, -1).to(dtype=torch.float32)
            if method == "total":
                total.add_(row_vals.sum(dim=1))
            else:
                total.add_(row_vals.sum(dim=1).sign())
        majority = torch.where(total >= 0, 1.0, -1.0)
        return majority.reshape(-1, *([1] * (ref.dim() - 1))).expand_as(ref)

    @staticmethod
    def _ties_disjoint_merge(trimmed_diffs, weights, majority_sign):
        """
        TIES Step 3: Disjoint Merge — average only contributors that agree
        with the elected majority sign at each position.

        Args:
            trimmed_diffs: list of trimmed diff tensors
            weights: list of scalar weights corresponding to each diff
            majority_sign: tensor of +1/-1 per position

        Returns:
            merged tensor
        """
        result = torch.zeros_like(trimmed_diffs[0], dtype=torch.float32)
        contributor_count = torch.zeros_like(result)

        for diff, weight in zip(trimmed_diffs, weights):
            diff_f = diff.to(dtype=torch.float32)
            # Positions where diff agrees with elected majority sign (and is non-zero)
            sign_match = (diff_f * majority_sign) > 0
            result.add_(torch.where(sign_match, diff_f * weight,
                                    torch.tensor(0.0, device=result.device, dtype=result.dtype)))
            contributor_count.add_(sign_match.float())

        # Average by number of contributors (avoid div-by-zero)
        contributor_count.clamp_(min=1.0)
        return result.div_(contributor_count)

    @staticmethod
    def _columnwise_disjoint_merge(trimmed_diffs, weights, majority_sign):
        """
        Column-wise disjoint merge: a row contributes entirely if its dominant
        sign matches the majority. Falls back to element-wise for 1D tensors.
        """
        ref = trimmed_diffs[0]
        if ref.dim() < 2:
            return _LoRAMergeBase._ties_disjoint_merge(trimmed_diffs, weights, majority_sign)

        out_dim = ref.shape[0]
        row_majority = majority_sign.reshape(out_dim, -1)[:, 0]

        result = torch.zeros_like(ref, dtype=torch.float32)
        contributor_count = torch.zeros(out_dim, device=ref.device, dtype=torch.float32)

        for diff, weight in zip(trimmed_diffs, weights):
            diff_f = diff.to(dtype=torch.float32)
            row_sign = diff_f.reshape(out_dim, -1).sum(dim=1).sign()
            sign_match = (row_sign * row_majority) > 0
            mask = sign_match.reshape(-1, *([1] * (ref.dim() - 1))).expand_as(ref)
            result.add_(torch.where(mask, diff_f * weight, torch.zeros_like(diff_f)))
            contributor_count.add_(sign_match.float())

        contributor_count.clamp_(min=1.0)
        return result.div_(contributor_count.reshape(-1, *([1] * (ref.dim() - 1))))

    @staticmethod
    @torch.no_grad()
    def _tall_masks(diffs_with_weights, lambda_threshold=1.0):
        """
        TALL-masks: identify per-parameter importance. "Selfish" weights
        (important to only one LoRA) are protected from merge averaging.
        Returns (consensus_diffs, selfish_additions) where selfish_additions
        is a tensor to add back after merging, or None if no selfish weights.
        """
        if len(diffs_with_weights) < 2:
            return diffs_with_weights, None

        ref = diffs_with_weights[0][0]

        # Tentative weighted sum
        d_merged = torch.zeros_like(ref, dtype=torch.float32)
        for d, w in diffs_with_weights:
            d_merged.add_(d.to(dtype=torch.float32) * w)

        # Per-LoRA importance masks
        masks = []
        for d, w in diffs_with_weights:
            contribution = d.to(dtype=torch.float32) * w
            others = d_merged - contribution
            mask = contribution.abs() >= others.abs() * lambda_threshold
            masks.append(mask)

        # Agreement count per position
        agreement = torch.zeros_like(ref, dtype=torch.float32)
        for m in masks:
            agreement.add_(m.float())

        # Separate selfish (agreement==1) from consensus
        selfish_additions = torch.zeros_like(ref, dtype=torch.float32)
        consensus_diffs = []
        has_selfish = False
        for i, (d, w) in enumerate(diffs_with_weights):
            selfish_mask = masks[i] & (agreement == 1)
            if selfish_mask.any():
                selfish_additions.add_(d.to(dtype=torch.float32) * w * selfish_mask.float())
                has_selfish = True
            consensus_d = torch.where(selfish_mask, torch.zeros(1, device=d.device, dtype=d.dtype), d)
            consensus_diffs.append((consensus_d, w))

        return consensus_diffs, selfish_additions if has_selfish else None

    @staticmethod
    @torch.no_grad()
    def _do_orthogonalize(diffs_with_weights):
        """
        DO-Merging: Decouple & Orthogonalize direction vectors while preserving magnitudes.
        Reduces directional interference between LoRA diffs by applying Modified Gram-Schmidt
        orthogonalization on unit direction vectors, then recombining with original magnitudes.

        Paper: arxiv 2505.15875 (May 2025)
        """
        if len(diffs_with_weights) < 2:
            return diffs_with_weights

        first = diffs_with_weights[0][0]
        if first.dim() < 2:
            return diffs_with_weights

        # Flatten, decompose into magnitude and unit direction
        dtype = first.dtype
        shapes = [d.shape for d, _ in diffs_with_weights]
        flat = [d.to(dtype=torch.float32).flatten() for d, _ in diffs_with_weights]
        weights = [w for _, w in diffs_with_weights]

        magnitudes = []
        directions = []
        for v in flat:
            mag = v.norm()
            magnitudes.append(mag)
            if mag > 1e-8:
                directions.append(v / mag)
            else:
                directions.append(torch.zeros_like(v))
        del flat

        # Modified Gram-Schmidt orthogonalization (numerically stable)
        # Operates in-place on directions (not used after this loop)
        ortho = []
        for i in range(len(directions)):
            q = directions[i]
            for j in range(len(ortho)):
                proj = torch.dot(q, ortho[j])
                q = q - proj * ortho[j]
            q_norm = q.norm()
            if q_norm > 1e-8:
                q = q / q_norm
            else:
                q = torch.zeros_like(q)
            ortho.append(q)
        del directions

        # Recombine: orthogonalized direction * original magnitude
        result = []
        for i in range(len(diffs_with_weights)):
            recombined = (ortho[i] * magnitudes[i]).to(dtype=dtype).reshape(shapes[i])
            result.append((recombined, weights[i]))

        return result

    @staticmethod
    @torch.no_grad()
    def _knots_align(diffs_with_weights, compute_device=None, svd_device=None):
        """
        KnOTS SVD alignment: project LoRA diffs into a shared SVD basis
        for better comparability before merging. Concatenates all diffs
        column-wise, computes truncated SVD, then reconstructs each diff
        in the shared basis.

        For [4096, 4096] with 5 LoRAs → M is [4096, 20480] ≈ 320MB.
        SVD rank capped at 256. Falls back to CPU on OOM.
        """
        if len(diffs_with_weights) < 2:
            return diffs_with_weights

        ref = diffs_with_weights[0][0]
        if ref.dim() < 2 or min(ref.shape) < 2:
            return diffs_with_weights

        n = len(diffs_with_weights)
        out_dim = ref.shape[0]
        in_dim = ref.reshape(out_dim, -1).shape[1]
        original_shape = ref.shape
        dev = svd_device if svd_device is not None else (compute_device or ref.device)

        # Concatenate column-wise: [out, N*in]
        cols = [d.reshape(out_dim, -1).to(device=dev, dtype=torch.float32)
                for d, _ in diffs_with_weights]
        M = torch.cat(cols, dim=1)
        del cols

        rank = min(out_dim, n * in_dim, 256)
        try:
            U, S, V = torch.svd_lowrank(M, q=rank)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            if M.is_cuda:
                logging.warning("[LoRA Optimizer] KnOTS SVD OOM on GPU, falling back to CPU")
                torch.cuda.empty_cache()
                M = M.cpu()
                try:
                    U, S, V = torch.svd_lowrank(M, q=rank)
                except RuntimeError:
                    logging.warning("[LoRA Optimizer] KnOTS SVD also failed on CPU, skipping alignment")
                    del M
                    return diffs_with_weights
            else:
                return diffs_with_weights
        del M

        # After CPU fallback, SVD results are on CPU — ensure aligned diffs
        # return on the same device as the input diffs
        output_device = compute_device if compute_device is not None else ref.device

        # Reconstruct each diff in shared basis
        aligned = []
        US = U * S.unsqueeze(0)  # [out, rank]
        for i, (_, w) in enumerate(diffs_with_weights):
            Vi = V[i * in_dim:(i + 1) * in_dim, :]  # [in, rank]
            aligned_diff = (US @ Vi.T).reshape(original_shape)
            if aligned_diff.device != output_device:
                aligned_diff = aligned_diff.to(output_device)
            aligned.append((aligned_diff, w))
        del U, S, V, US

        return aligned

    @staticmethod
    @torch.no_grad()
    def _procrustes_align(diffs_with_weights, compute_device=None, svd_device=None):
        """
        Procrustes alignment: rotate each LoRA diff toward the weighted-mean
        reference via optimal rotation. Uses batched_procrustes from kernel.py
        with whiten=False so Frobenius norms are preserved (pure rotation).

        Treats each diff as (n_samples=out_dim, N=in_dim): aligns input-space
        directions. Batches all diffs into a single (B, out_dim, in_dim) call.
        """
        if _batched_procrustes is None:
            return diffs_with_weights
        if len(diffs_with_weights) < 2:
            return diffs_with_weights

        ref = diffs_with_weights[0][0]
        if ref.dim() < 2 or min(ref.shape) < 2:
            return diffs_with_weights

        n = len(diffs_with_weights)
        out_dim = ref.shape[0]
        in_dim = ref.reshape(out_dim, -1).shape[1]
        original_shape = ref.shape
        original_dtype = ref.dtype

        dev = svd_device if svd_device is not None else (compute_device or ref.device)
        output_device = compute_device if compute_device is not None else ref.device

        # Weighted mean reference
        total_w = sum(abs(w) for _, w in diffs_with_weights)
        if total_w < 1e-12:
            return diffs_with_weights
        ref_mat = sum(
            d.reshape(out_dim, in_dim).to(device=dev, dtype=torch.float32) * (abs(w) / total_w)
            for d, w in diffs_with_weights
        )

        source_batch = torch.stack([
            d.reshape(out_dim, in_dim).to(device=dev, dtype=torch.float32)
            for d, _ in diffs_with_weights
        ], dim=0)
        target_batch = ref_mat.unsqueeze(0).expand(n, -1, -1)

        try:
            # Get rotation from batched_procrustes, then apply to uncentered source.
            # batched_procrustes centers internally, which corrupts LoRA diffs.
            # We extract R and compute src @ R ourselves.
            _, info = _batched_procrustes(
                source_batch, target_batch, whiten=False)
            R = info.get('rotation') if 'rotation' in info else info.get('rotation_k')
            if R is None:
                del source_batch, target_batch, ref_mat
                return diffs_with_weights
            if 'projection' in info:
                # Subspace path: R_k is in projected space, need to lift back
                P = info['projection']  # (B, N, k)
                P_T = P.transpose(1, 2)
                src_in = torch.bmm(source_batch, P)  # (B, out, k)
                src_perp = source_batch - torch.bmm(src_in, P_T)
                aligned_batch = torch.bmm(torch.bmm(src_in, R), P_T) + src_perp
            else:
                aligned_batch = torch.bmm(source_batch, R)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logging.warning(f"[LoRA Optimizer] Procrustes align failed ({e}), skipping")
            del source_batch, target_batch, ref_mat
            return diffs_with_weights
        del source_batch, target_batch, ref_mat

        result = []
        for i, (_, w) in enumerate(diffs_with_weights):
            aligned_diff = aligned_batch[i].reshape(original_shape)
            if aligned_diff.device != output_device:
                aligned_diff = aligned_diff.to(output_device)
            result.append((aligned_diff.to(dtype=original_dtype), w))
        del aligned_batch

        return result

    @staticmethod
    @torch.no_grad()
    def _truncated_svd_robust(mat, rank):
        """Truncated SVD -> (U[:, :rank], S[:rank], V[:, :rank]).

        Robust to SVD non-convergence, which is common on ROCm/MAGMA for
        ill-conditioned or repeated-singular-value matrices ("failed to converge
        ... too many repeated singular values"). Tries the input device, then
        retries on CPU (LAPACK is far more stable), then on CPU with a tiny jitter
        that separates repeated singular values. Returns None if all attempts fail.
        """
        min_dim = min(mat.shape)
        rank = max(1, min(rank, min_dim))

        def _compute(m):
            if rank > min_dim // 2:
                U, S, Vh = torch.linalg.svd(m, full_matrices=False)
                return U[:, :rank], S[:rank], Vh[:rank, :].T
            q = min(rank + max(20, rank // 5), min_dim)
            U, S, V = torch.svd_lowrank(m, q=q, niter=4)
            return U[:, :rank], S[:rank], V[:, :rank]

        try:
            return _compute(mat)
        except Exception:
            pass
        mat_cpu = mat.detach().to("cpu", torch.float32)
        try:
            return _compute(mat_cpu)
        except Exception:
            pass
        try:
            scale = mat_cpu.abs().mean()
            if scale > 0:
                g = torch.Generator().manual_seed(0)
                mat_cpu = mat_cpu + (scale * 1e-6) * torch.randn(
                    mat_cpu.shape, generator=g, dtype=mat_cpu.dtype)
            U, S, Vh = torch.linalg.svd(mat_cpu, full_matrices=False)
            return U[:, :rank], S[:rank], Vh[:rank, :].T
        except Exception:
            logging.warning("[LoRA Optimizer] SVD failed to converge on all paths "
                            "(GPU/CPU/jitter); keeping the dense diff for this layer.")
            return None

    @staticmethod
    @torch.no_grad()
    def _compress_to_lowrank(diff, rank, svd_device=None, output_dtype=None):
        """
        Re-compress a full-rank diff tensor to low-rank via truncated SVD.
        Returns ("lora", (mat_up, mat_down, alpha=rank, None)) so ComfyUI
        computes up @ down * (rank/rank) = up @ down (no extra scaling).

        svd_device: where to run SVD. GPU is ~10-50x faster. CPU if None.
        output_dtype: cast output to this dtype. None = same as input.
        For a [4096, 4096] diff at rank 128: 64MB → 2MB (~32x reduction).
        """
        original_shape = diff.shape
        if output_dtype is None:
            output_dtype = diff.dtype
        # Reshape to 2D for SVD: [out_features, in_features]
        mat = diff.reshape(original_shape[0], -1).float()
        rank = min(rank, min(mat.shape))

        # Move to requested device for SVD (GPU is much faster for matmul-heavy randomized SVD)
        if svd_device is not None and mat.device != svd_device:
            mat = mat.to(svd_device)
        svd = _LoRAOptimizerEngine._truncated_svd_robust(mat, rank)
        del mat
        if svd is None:
            # SVD failed on every device/path — signal the caller to keep the
            # dense diff rather than crash (seen on ROCm for repeated singular values).
            return None
        U, S, V = svd
        # U: [out, rank], S: [rank], V: [in, rank]
        # Reconstruct as: mat_up = U * sqrt(S), mat_down = sqrt(S) * V^T
        # Return on CPU for storage (ComfyUI moves to device when applying)
        sqrt_S = S.sqrt()
        mat_up = (U * sqrt_S.unsqueeze(0)).to(output_dtype).cpu()
        mat_down = ((V * sqrt_S.unsqueeze(0)).T).to(output_dtype).cpu()
        del U, S, V, sqrt_S
        # alpha=rank so ComfyUI computes: up @ down * (rank/rank) = up @ down
        return LoRAAdapter(set(), (mat_up, mat_down, float(rank), None, None, None))

    @staticmethod
    @torch.no_grad()
    def _estimate_save_rank(initial_rank, model_patches, clip_patches,
                            max_error=0.05, n_samples=3):
        """
        Estimate the minimum SVD rank needed to reconstruct sample diff patches
        within `max_error` relative Frobenius error.
        """
        samples = []
        for patch in list(model_patches.values()) + list(clip_patches.values()):
            if isinstance(patch, tuple) and patch[0] == "diff":
                samples.append(patch[1][0])
                if len(samples) >= n_samples:
                    break
        if not samples:
            return max(initial_rank, 64)

        rank = max(initial_rank, 64)
        for sample in samples:
            mat = sample.reshape(sample.shape[0], -1).float()
            singular_values = _triton_svdvals(mat, n_sv=min(mat.shape))
            total_sq = (singular_values ** 2).sum().item()
            if total_sq == 0:
                continue
            threshold_sq = (max_error * max_error) * total_sq
            cumulative_sq = 0.0
            needed = len(singular_values)
            for idx in range(len(singular_values)):
                cumulative_sq += singular_values[idx].item() ** 2
                if total_sq - cumulative_sq <= threshold_sq:
                    needed = idx + 1
                    break
            if needed > rank:
                rank = needed
            del singular_values, mat
        return rank

    @torch.no_grad()
    def _merge_diffs(self, diffs_with_weights, mode, density=0.5, majority_sign_method="total",
                     compute_device=None, sparsification="disabled",
                     sparsification_density=0.7, sparsification_generator=None,
                     merge_refinement="none", dare_dampening=0.0,
                     keep_on_gpu=False, preserve_flags=None):
        """
        Merges a list of diffs with their weights.
        When compute_device is given, tensors are moved there for faster ops,
        then the result is returned on CPU (unless keep_on_gpu=True).

        preserve_flags: optional list of bools aligned with diffs_with_weights.
        A LoRA marked preserve=True (a "style" LoRA the user tagged) is held OUT of
        the normal merge: the remaining (non-preserved) LoRAs are merged with `mode`
        as usual, then each preserved LoRA's full-strength delta is ADDED ON TOP.
        So it keeps its full emphasis instead of being averaged into a convex
        fraction, trimmed by sparsification, or deleted by TIES sign-election — but
        only that tagged LoRA, leaving ordinary multi-LoRA blends untouched.
        """
        self._interrupt_check()
        if len(diffs_with_weights) == 0:
            return None
        if preserve_flags is None:
            preserve_flags = [False] * len(diffs_with_weights)
        any_preserve = any(preserve_flags)

        if len(diffs_with_weights) == 1:
            diff, weight = diffs_with_weights[0]
            result = diff * weight
            if compute_device is not None and compute_device.type != "cpu" and result.is_cuda and not keep_on_gpu:
                return result.cpu()
            return result

        # All diffs should have the same shape (verified during computation)
        ref_diff = diffs_with_weights[0][0]
        dtype = ref_diff.dtype
        dev = compute_device if compute_device is not None else ref_diff.device
        to_cpu = (compute_device is not None and compute_device.type != "cpu"
                  and not keep_on_gpu)

        # Preserve overlay: tagged style LoRAs bypass the merge entirely. Blend the
        # rest with the requested mode, then add each preserved LoRA at full strength
        # on top. This is the only place sum-of-deltas behaviour is applied, and ONLY
        # to user-flagged LoRAs — the analyzer cannot tell "preserve this style" from
        # "blend these characters", so additive preservation must be opt-in.
        if any_preserve:
            rest = [dw for dw, p in zip(diffs_with_weights, preserve_flags) if not p]
            preserved_sum = None
            for (d, w), p in zip(diffs_with_weights, preserve_flags):
                if not p:
                    continue
                contrib = d.to(device=dev, dtype=torch.float32) * w
                preserved_sum = contrib if preserved_sum is None else preserved_sum + contrib
            if rest:
                blended = self._merge_diffs(
                    rest, mode, density=density, majority_sign_method=majority_sign_method,
                    compute_device=compute_device, sparsification=sparsification,
                    sparsification_density=sparsification_density,
                    sparsification_generator=sparsification_generator,
                    merge_refinement=merge_refinement, dare_dampening=dare_dampening,
                    keep_on_gpu=True, preserve_flags=None)
                result = blended.to(device=dev, dtype=torch.float32) + preserved_sum
            else:
                result = preserved_sum
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        # DARE/DELLA preprocessing for non-TIES modes
        # (TIES replaces its trim step instead — handled in the ties branch)
        if sparsification != "disabled" and mode != "ties":
            is_conflict = sparsification in ("dare_conflict", "della_conflict")

            if is_conflict:
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    diffs_with_weights[idx] = (diff.to(device=dev, dtype=torch.float32), weight)
                conflict_mask = self._compute_conflict_mask(diffs_with_weights)

                # Guard: if conflict mask covers >40% of positions, the "conflicts"
                # are likely base-rate noise from orthogonal LoRAs (expected ~50%
                # sign disagreement for uncorrelated vectors).  Skip sparsification
                # entirely — there are no real conflicts to resolve.
                conflict_frac = conflict_mask.float().mean().item()
                if conflict_frac > 0.40:
                    del conflict_mask
                    is_conflict = False
                    self._sparsification_skipped = getattr(self, '_sparsification_skipped', 0) + 1

            if is_conflict:
                is_dare = sparsification == "dare_conflict"
                sparsify_fn = (self._dare_sparsify_conflict if is_dare
                               else self._della_sparsify_conflict)
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    if preserve_flags[idx]:
                        continue  # tagged style LoRA: keep its diff dense
                    kwargs = dict(generator=sparsification_generator)
                    if is_dare:
                        kwargs["dampening"] = dare_dampening
                    diff = sparsify_fn(diff, conflict_mask, sparsification_density,
                                       **kwargs)
                    diffs_with_weights[idx] = (diff.to(dtype), weight)
                del conflict_mask
            else:
                is_dare = sparsification == "dare"
                sparsify_fn = (self._dare_sparsify if is_dare
                               else self._della_sparsify)
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    if preserve_flags[idx]:
                        # tagged style LoRA: keep its diff dense (still moved to dev)
                        diffs_with_weights[idx] = (diff.to(device=dev, dtype=torch.float32).to(dtype), weight)
                        continue
                    diff = diff.to(device=dev, dtype=torch.float32)
                    kwargs = dict(generator=sparsification_generator)
                    if is_dare:
                        kwargs["dampening"] = dare_dampening
                    diff = sparsify_fn(diff, sparsification_density,
                                       **kwargs)
                    diffs_with_weights[idx] = (diff.to(dtype), weight)

        # Refine/full merge refinement pipeline (non-TIES modes)
        # TIES has its own enhancement path below (after trim)
        # Order matters: TALL-masks must run BEFORE orthogonalization.
        # Orthogonalized diffs have uncorrelated element-wise distributions,
        # which causes TALL-masks to classify every position as "selfish"
        # (agreement=1 everywhere), zeroing out all consensus diffs.
        selfish_additions = None
        if merge_refinement != "none" and len(diffs_with_weights) >= 2 and mode != "ties":
            diffs_with_weights, selfish_additions = self._tall_masks(diffs_with_weights)
            first = diffs_with_weights[0][0]
            if first.dim() >= 2:
                diffs_with_weights = self._do_orthogonalize(diffs_with_weights)
            if merge_refinement == "full":
                first = diffs_with_weights[0][0]
                if first.dim() >= 2 and min(first.shape) >= 2:
                    if _batched_procrustes is not None:
                        diffs_with_weights = self._procrustes_align(
                            diffs_with_weights, compute_device=dev, svd_device=dev)
                    else:
                        diffs_with_weights = self._knots_align(
                            diffs_with_weights, compute_device=dev, svd_device=dev)

        if mode == "weighted_average":
            result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            total_weight = sum(abs(w) for _, w in diffs_with_weights)
            if total_weight == 0:
                return result.to(dtype).cpu() if to_cpu else result.to(dtype)
            for idx in range(len(diffs_with_weights)):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                result.add_(diff.to(device=dev, dtype=torch.float32), alpha=weight / total_weight)
            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "weighted_sum":
            result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            for idx in range(len(diffs_with_weights)):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                result.add_(diff.to(device=dev, dtype=torch.float32), alpha=weight)
            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "normalize":
            # Normalization by "energy" (sum of squared weights)
            weights = [w for _, w in diffs_with_weights]
            sum_sq = sum(w*w for w in weights)
            if sum_sq == 0:
                z = torch.zeros(ref_diff.shape, device=dev)
                return z.cpu() if to_cpu else z
            scale = 1.0 / math.sqrt(sum_sq)

            result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            for idx in range(len(diffs_with_weights)):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                result.add_(diff.to(device=dev, dtype=torch.float32), alpha=weight * scale)
            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "slerp":
            # Magnitude-preserving spherical blend for N diffs.
            # For 2 diffs: standard SLERP (exact spherical geodesic — the
            # Karcher mean reduces to it for N=2).
            # For 3+ diffs: weighted Karcher (Fréchet) mean on the unit
            # hypersphere. Iterative pairwise SLERP is order-dependent and
            # collapses empirically for N>=3 (multi-SLERP scores below plain
            # linear averaging at m=5 models; the Karcher mean is the
            # order-independent spherical mean it tried to approximate).
            # Final norm corrected to match weighted average of input norms.
            n_diffs = len(diffs_with_weights)

            # Handle negative weights by negating diff direction
            items = []
            for idx in range(n_diffs):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                v = diff.to(device=dev, dtype=torch.float32).flatten()
                del diff
                if weight < 0:
                    v = -v
                items.append((v, abs(weight)))

            # Sort by descending weight (strongest LoRA anchors direction)
            items.sort(key=lambda x: x[1], reverse=True)

            # All-zero weights: return zero (consistent with other modes)
            total_w = sum(w for _, w in items)
            if total_w == 0:
                z = torch.zeros(ref_diff.shape, device=dev, dtype=dtype)
                return z.cpu() if to_cpu else z

            if n_diffs == 2:
                # Pre-compute norm-correction target before vectors are consumed
                input_norms = [(v.norm().item(), w) for v, w in items]
                target_norm = sum(n * w for n, w in input_norms) / total_w
                # Standard pairwise SLERP
                acc_v, acc_w = items[0]
                items[0] = None  # Free tensor reference
                next_v, next_w = items[1]
                items[1] = None  # Free tensor reference
                frac = next_w / (acc_w + next_w) if (acc_w + next_w) > 0 else 0.5

                # Compute angle between the two vectors
                norm_acc = acc_v.norm()
                norm_next = next_v.norm()
                denom = norm_acc * norm_next
                if denom > 0:
                    cos_theta = (torch.dot(acc_v, next_v) / denom).clamp(-1.0, 1.0)
                else:
                    cos_theta = torch.tensor(1.0, device=dev)
                theta = torch.acos(cos_theta)

                # Nearly-parallel fallback to linear interpolation
                if theta.item() < 1e-6:
                    acc_v = (1.0 - frac) * acc_v + frac * next_v
                else:
                    sin_theta = torch.sin(theta)
                    a = torch.sin((1.0 - frac) * theta) / sin_theta
                    b = torch.sin(frac * theta) / sin_theta
                    acc_v = a * acc_v + b * next_v
                del next_v, items
            else:
                # Weighted Karcher mean, BATCHED: all unit vectors live in one
                # [N, D] matrix so each iteration is two matvecs instead of
                # per-unit kernels with .item() syncs (measured 1.7-2.9x
                # faster at 4-16M elements). Iterates tangent-space (log-map)
                # weighted average -> exp-map back, from the chordal-mean
                # init. Converges in a few iterations for vectors in a
                # half-space (the common case after negative-weight folding).
                numel = items[0][0].numel()
                U = torch.empty((n_diffs, numel), device=dev, dtype=torch.float32)
                w_raw = []
                for idx in range(n_diffs):
                    v, w = items[idx]
                    items[idx] = None  # Free tensor reference
                    U[idx].copy_(v)
                    w_raw.append(w)
                    del v
                del items
                row_norms = U.norm(dim=1)  # [N]
                w_raw = torch.tensor(w_raw, device=dev, dtype=torch.float32)
                keep = row_norms > 1e-12
                if not bool(keep.any()):
                    del U
                    z = torch.zeros(ref_diff.shape, device=dev, dtype=dtype)
                    return z.cpu() if to_cpu else z
                # Norm-correction target from the same row norms (zero-norm
                # rows contribute ~0, matching the old per-vector reads)
                target_norm = ((row_norms * w_raw).sum() / total_w).item()
                if not bool(keep.all()):
                    U = U[keep]
                    row_norms = row_norms[keep]
                    w_raw = w_raw[keep]
                U.div_(row_norms.unsqueeze(1))  # unit rows
                w_t = w_raw / total_w

                # Init: normalized weighted chordal (Euclidean) mean
                m = U.t().mv(w_t)
                m_norm = m.norm()
                m = U[0].clone() if m_norm.item() < 1e-8 else m / m_norm

                for _ in range(8):
                    self._interrupt_check()
                    # log_m(u_i) = theta_i/sin(theta_i) * (u_i - cos_i*m)
                    # cos clamped away from ±1: theta -> 0 contributes ~0,
                    # antipodal (cut locus) kept finite
                    cos = (U @ m).clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # [N]
                    theta = torch.acos(cos)
                    coef = torch.where(theta < 1e-7,
                                       torch.zeros_like(theta),
                                       w_t * theta / torch.sin(theta))
                    tangent = U.t().mv(coef) - (coef * cos).sum() * m
                    t_norm = tangent.norm()
                    if t_norm.item() < 1e-7:
                        break
                    # exp_m(t) = cos(|t|)*m + sin(|t|)*t/|t|
                    m = torch.cos(t_norm) * m + (torch.sin(t_norm) / t_norm) * tangent
                    m = m / m.norm().clamp(min=1e-12)
                del U
                acc_v = m

            # Norm correction: rescale to match weighted average of input norms
            current_norm = acc_v.norm().item()
            if current_norm > 1e-8:
                acc_v = acc_v * (target_norm / current_norm)

            result = acc_v.reshape(ref_diff.shape)
            del acc_v
            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "consensus":
            # Consensus merge: Fisher-proxy + MAGIC calibration + MonoSoup spectral cleanup
            # Optimized for similar LoRAs (high cosine similarity, low conflict)

            # Step 1: Fisher-Proxy weighted merge
            # Weight each parameter by |diff|^2 as importance proxy
            numerator = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            denominator = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            input_norms = []
            abs_weight_list = []

            for idx in range(len(diffs_with_weights)):
                d, w = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free early
                d_f = d.to(device=dev, dtype=torch.float32)
                del d
                importance = d_f.square()
                aw = abs(w)
                numerator.add_(d_f * w * importance)
                denominator.add_(aw * importance)
                input_norms.append(d_f.norm().item() * aw)
                abs_weight_list.append(aw)
                del d_f, importance

            # Safe division (zero importance → zero result)
            result = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
            del numerator, denominator

            # Step 2: MAGIC magnitude calibration
            # Rescale merged result so L2 norm matches weighted average of input norms
            merged_norm = result.norm().item()
            total_aw = sum(abs_weight_list)
            if total_aw > 0 and merged_norm > 1e-8:
                target_norm = sum(input_norms) / total_aw
                result.mul_(target_norm / merged_norm)
            del input_norms, abs_weight_list

            # Step 3: MonoSoup spectral cleanup (2D+ weights only)
            if result.dim() >= 2 and min(result.shape) >= 4:
                mat = result.reshape(result.shape[0], -1)
                try:
                    rank_budget = min(min(mat.shape), 128)
                    U, S, V = torch.svd_lowrank(mat, q=rank_budget)

                    # Entropy-based effective rank
                    s_sum = S.sum()
                    if s_sum < 1e-10:
                        del U, S, V, mat
                        raise RuntimeError("zero singular values")
                    p = S / s_sum
                    p = p.clamp(min=1e-10)  # avoid log(0)
                    entropy = -(p * p.log()).sum().item()
                    eff_rank = min(int(math.exp(entropy) + 0.5), rank_budget)
                    eff_rank = max(eff_rank, 1)

                    # Soft spectral gate: smooth transition instead of hard cutoff
                    gate = torch.sigmoid(4.0 * (torch.arange(rank_budget, device=dev, dtype=torch.float32) - eff_rank) * (-1.0 / max(eff_rank, 1)))
                    S_gated = S * gate

                    # Preserve original norm (spectral cleanup shouldn't change magnitude)
                    pre_norm = result.norm()
                    result = (U * S_gated.unsqueeze(0)) @ V.T
                    result = result.reshape(ref_diff.shape)
                    post_norm = result.norm()
                    if post_norm > 1e-8:
                        result.mul_(pre_norm / post_norm)
                    del U, S, V, S_gated, gate, mat
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    pass  # SVD failed, skip spectral cleanup

            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "ties":
            # TIES-Merging: Trim, Elect Sign, Disjoint Merge
            # Pre-multiply diffs by sign(weight) so negative strengths vote correctly,
            # then use abs(weight) for magnitude in disjoint merge.
            # Memory-optimized: free input diffs after trimming to reduce peak VRAM.
            trimmed = []
            abs_weights = []
            # Tagged style LoRAs bypass TIES: they are NOT trimmed and NOT subject
            # to sign election (which would delete the minority-sign direction a
            # style often holds). Their full-strength contribution is summed here
            # and added on top of the TIES-merged content below.
            preserved_sum = None
            is_conflict = sparsification in ("dare_conflict", "della_conflict")

            if is_conflict:
                signed_diffs = []
                for idx in range(len(diffs_with_weights)):
                    d, w = diffs_with_weights[idx]
                    diffs_with_weights[idx] = None
                    d_f = d.to(device=dev, dtype=torch.float32)
                    del d
                    if w < 0:
                        d_f = -d_f
                    if preserve_flags[idx]:
                        contrib = d_f * abs(w)
                        preserved_sum = contrib if preserved_sum is None else preserved_sum + contrib
                        continue
                    signed_diffs.append(d_f)
                    abs_weights.append(abs(w))

                if signed_diffs:
                    conflict_mask = self._compute_conflict_mask(
                        [(d, 1.0) for d in signed_diffs])
                    is_dare = sparsification == "dare_conflict"
                    sparsify_fn = (self._dare_sparsify_conflict if is_dare
                                   else self._della_sparsify_conflict)
                    for d_f in signed_diffs:
                        kwargs = dict(generator=sparsification_generator)
                        if is_dare:
                            kwargs["dampening"] = dare_dampening
                        trimmed.append(sparsify_fn(d_f, conflict_mask, sparsification_density,
                                                   **kwargs))
                    del conflict_mask
                del signed_diffs
            else:
                for idx in range(len(diffs_with_weights)):
                    d, w = diffs_with_weights[idx]
                    diffs_with_weights[idx] = None  # Free input diff early
                    d_f = d.to(device=dev, dtype=torch.float32)
                    del d
                    if w < 0:
                        d_f = -d_f
                    if preserve_flags[idx]:
                        contrib = d_f * abs(w)
                        preserved_sum = contrib if preserved_sum is None else preserved_sum + contrib
                        continue
                    # DARE/DELLA replaces TIES trim step when enabled
                    if sparsification == "dare":
                        trimmed.append(self._dare_sparsify(d_f, sparsification_density, generator=sparsification_generator, dampening=dare_dampening))
                    elif sparsification == "della":
                        trimmed.append(self._della_sparsify(d_f, sparsification_density, generator=sparsification_generator))
                    else:
                        trimmed.append(self._ties_trim(d_f, density))
                    abs_weights.append(abs(w))

            # Every contributor was a tagged style LoRA — nothing to TIES-merge.
            if not trimmed:
                result = (preserved_sum if preserved_sum is not None
                          else torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev))
                result = result.to(dtype)
                return result.cpu() if to_cpu else result

            # Refine/full merge refinement pipeline for TIES
            # TALL-masks before orthogonalization (see non-TIES comment above)
            ties_selfish = None
            if merge_refinement != "none" and len(trimmed) >= 2:
                # Re-pair trimmed diffs with abs_weights for enhancement pipeline
                trimmed_pairs = list(zip(trimmed, abs_weights))
                trimmed_pairs, ties_selfish = self._tall_masks(trimmed_pairs)
                first = trimmed_pairs[0][0]
                if first.dim() >= 2:
                    trimmed_pairs = self._do_orthogonalize(trimmed_pairs)
                if merge_refinement == "full":
                    first = trimmed_pairs[0][0]
                    if first.dim() >= 2 and min(first.shape) >= 2:
                        if _batched_procrustes is not None:
                            trimmed_pairs = self._procrustes_align(
                                trimmed_pairs, compute_device=dev, svd_device=dev)
                        else:
                            trimmed_pairs = self._knots_align(
                                trimmed_pairs, compute_device=dev, svd_device=dev)
                trimmed = [d for d, _ in trimmed_pairs]
                abs_weights = [w for _, w in trimmed_pairs]
                del trimmed_pairs

            # Step 2: Elect majority sign
            if merge_refinement != "none":
                majority_sign = self._columnwise_elect_sign(trimmed, majority_sign_method)
            else:
                majority_sign = self._ties_elect_sign(trimmed, majority_sign_method)

            # Step 3: Disjoint merge
            if merge_refinement != "none":
                result = self._columnwise_disjoint_merge(trimmed, abs_weights, majority_sign)
            else:
                result = self._ties_disjoint_merge(trimmed, abs_weights, majority_sign)
            del trimmed, majority_sign
            if ties_selfish is not None:
                result = result.to(dtype=torch.float32) + ties_selfish.to(device=result.device, dtype=torch.float32)
            if preserved_sum is not None:
                # Tagged style added at full strength on top of the TIES-merged content.
                result = result.to(dtype=torch.float32) + preserved_sum.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        return None

    @staticmethod
    def _stringify_lora_keys(lora_dict):
        """Return a string-keyed VIEW of a state dict for architecture
        detection. Captured (virtual) items may key by TUPLES (str_key,
        offset) from fused-QKV splits, and _detect_architecture indexes
        k.lower() / 'substr in k', which crash/misbehave on tuple keys. The
        values are shared (no copy) and the original dict is never mutated."""
        return {(k[0] if isinstance(k, tuple) else k): v
                for k, v in lora_dict.items()}

    def _normalize_stack(self, lora_stack, normalize_keys="disabled",
                         _arch_hint=None):
        """
        Normalize a LoRA stack into a consistent list of dicts.

        Accepts two formats:
        - Standard tuples: [(lora_name, model_strength, clip_strength), ...]
          Used by Efficiency Nodes, Comfyroll, and other popular node packs.
          LoRAs are loaded from disk (cached in self.loaded_loras).
        - Preloaded dicts: [{"name": str, "lora": dict, "strength": float}, ...]
          Already loaded, clip_strength defaults to None (use global multiplier).

        Returns list of dicts with keys: name, lora, strength, clip_strength.
        clip_strength is None when the global multiplier should be used.
        """
        if not lora_stack:
            return []

        normalized = []

        # Dispatch per item, not on the first element: a stack can mix tuple
        # entries (file references from LoRA Manager) with dict entries carrying
        # preloaded in-memory weights. The
        # old first-element branch silently dropped the minority type.
        for entry in lora_stack:
            if isinstance(entry, (tuple, list)):
                # Standard format: (lora_name, model_strength, clip_strength[, conflict_mode[, key_filter[, preserve]]])
                if len(entry) < 3:
                    logging.warning("[LoRA Optimizer] Skipping malformed tuple entry (expected 3 elements)")
                    continue
                lora_name, model_str, clip_str = entry[0], entry[1], entry[2]
                conflict_mode = entry[3] if len(entry) > 3 else "all"
                key_filter = entry[4] if len(entry) > 4 else "all"
                preserve = bool(entry[5]) if len(entry) > 5 else False

                # Load LoRA with caching
                if lora_name in self.loaded_loras:
                    lora_dict = self.loaded_loras[lora_name]
                    lora_path = None
                else:
                    try:
                        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                        lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        self.loaded_loras[lora_name] = lora_dict
                    except Exception as e:
                        logging.warning(f"[LoRA Optimizer] Failed to load LoRA '{lora_name}': {e}")
                        continue

                metadata = {}
                if lora_path is not None:
                    metadata = _read_safetensors_metadata(lora_path)

                normalized.append({
                    "name": lora_name,
                    "lora": lora_dict,
                    "strength": model_str,
                    "clip_strength": clip_str,
                    "conflict_mode": conflict_mode,
                    "key_filter": key_filter,
                    "preserve": preserve,
                    "metadata": metadata,
                })

            elif isinstance(entry, dict):
                # preloaded adapter format: already loaded dicts
                if "lora" not in entry or "strength" not in entry or "name" not in entry:
                    logging.warning("[LoRA Optimizer] Skipping malformed dict entry (expected keys: name, lora, strength)")
                    continue
                normalized_item = {
                    "name": entry["name"],
                    "lora": entry["lora"],
                    "strength": entry["strength"],
                    "clip_strength": entry.get("clip_strength", None),
                    "conflict_mode": entry.get("conflict_mode", "all"),
                    "key_filter": entry.get("key_filter", "all"),
                    "preserve": bool(entry.get("preserve", False)),
                    "metadata": entry.get("metadata", {}),
                    "_precomputed_diffs": entry.get("_precomputed_diffs", False),
                }
                # Virtual payload identity must survive normalization. A
                # resolved loader filename reconciles captured runs with the
                # file-based cache; the memoized hash avoids re-hashing large
                # captured factor dictionaries for every tuner candidate.
                for identity_key in ("_resolved_file_name", "_content_hash"):
                    if identity_key in entry:
                        normalized_item[identity_key] = entry[identity_key]
                normalized.append(normalized_item)

            else:
                logging.warning("[LoRA Optimizer] Skipping unrecognized stack entry")
                continue

        # Always detect architecture (used for preset selection even without key normalization)
        if len(normalized) > 0:
            arch = "unknown"
            # File-based items carry trainer-format LoRA keys the detector /
            # normalizer understands — try them first, exactly as before.
            for item in normalized:
                if item.get("_precomputed_diffs"):
                    continue  # virtual LoRAs have model-space keys, not LoRA keys
                detected = self._detect_architecture(item["lora"])
                if detected != "unknown":
                    arch = detected
                    break
            # Fallback for all-virtual or mixed stacks whose
            # file items didn't resolve: virtual items are keyed by MODEL-SPACE
            # target keys, which STILL carry structural architecture markers
            # (diffusion_model.transformer_blocks... for LTX,
            # diffusion_model.layers.N.attention... for Z-Image, etc.).
            # _detect_architecture's structural heuristics work on those. Keys
            # may be TUPLES (str_key, offset) from fused-QKV captures, so detect
            # on a stringified VIEW (never mutate the real lora dict). This only
            # fires when file detection failed — a strict improvement over the
            # old unknown->sd_unet fallthrough for virtual payloads.
            if arch == "unknown":
                has_virtual = any(item.get("_precomputed_diffs")
                                  for item in normalized)
                if has_virtual and _arch_hint and _arch_hint != "unknown":
                    # Model-class detection (from the real MODEL object) is
                    # authoritative for virtual adapter payloads — it resolves
                    # architectures that are indistinguishable from model-space
                    # keys alone (e.g. attention-only Qwen-Image vs ACE-Step
                    # v1.0, which share transformer_blocks.N.attn.to_q). Guarded
                    # by has_virtual so FILE-based stacks stay pure key-based.
                    arch = _arch_hint
                else:
                    for item in normalized:
                        if not item.get("_precomputed_diffs"):
                            continue
                        detected = self._detect_architecture(
                            self._stringify_lora_keys(item["lora"]))
                        if detected != "unknown":
                            arch = detected
                            break
            self._detected_arch = arch if arch != "unknown" else None

            # Architecture-aware key normalization (only when enabled)
            if normalize_keys == "enabled":
                if arch != "unknown":
                    logging.info(f"[LoRA Optimizer] Architecture detected: {arch}")
                    logging.info(f"[LoRA Optimizer] Normalizing keys for {len(normalized)} LoRAs...")
                    for item in normalized:
                        if not item.get("_precomputed_diffs"):
                            item["lora"] = self._normalize_keys(item["lora"], arch)
                else:
                    logging.info("[LoRA Optimizer] Architecture: unknown (no key normalization applied)")

            # Update loaded_loras cache to point at normalized dicts so the
            # pre-normalization copies can be garbage-collected.  This avoids
            # keeping both raw and normalized state dicts in memory (saves
            # 500MB-3GB for large models like Qwen).
            for item in normalized:
                name = item["name"]
                if name in self.loaded_loras:
                    self.loaded_loras[name] = item["lora"]
        else:
            self._detected_arch = None

        return normalized

def _full_svd_robust(mat: torch.Tensor):
    """Compute a complete SVD with stable CPU fallbacks."""
    try:
        return torch.linalg.svd(mat, full_matrices=False)
    except Exception:
        pass
    mat_cpu = mat.detach().to("cpu", torch.float32)
    try:
        return torch.linalg.svd(mat_cpu, full_matrices=False)
    except Exception:
        pass
    try:
        scale = mat_cpu.abs().mean()
        if scale > 0:
            generator = torch.Generator().manual_seed(0)
            mat_cpu = mat_cpu + (scale * 1e-6) * torch.randn(
                mat_cpu.shape, generator=generator, dtype=mat_cpu.dtype)
        return torch.linalg.svd(mat_cpu, full_matrices=False)
    except Exception:
        logging.warning("[LoRA Optimizer] SVD failed on GPU and CPU; skipping this layer.")
        return None


class _LoRAOptimizerEngine(_LoRAMergeBase):
    """
    Auto-optimizer that analyzes a LoRA stack (sign conflicts, magnitude
    distributions, overlap) and automatically selects merge modes
    and parameters, then performs the merge.

    Outputs the merged model/clip plus an analysis report explaining
    what was chosen and why.

    Two-pass streaming architecture:
      Pass 1 — Analysis: resolves aliases to target groups, computes diffs
        per group, samples conflict and magnitude statistics, then discards
        diffs immediately. Only lightweight scalars and small sample tensors
        are kept in memory.
      Pass 2 — Merge: recomputes diffs per group and merges with the
        auto-selected strategy. Each group's diffs are freed after merging.
    Peak memory is roughly one target group at a time, but the exact peak
    still depends on layer size, overlap, and enabled quality/compression
    options.

    Limitation: the optimizer only analyzes LoRAs in its own stack. It has
    no visibility into LoRA patches already applied to the model by upstream
    nodes (via Load LoRA, etc.). Those patches stack additively on top of
    the optimizer's output, which could cause overexposure. Fully baked
    merges (safetensors checkpoints) are indistinguishable from base weights
    and cannot be detected at all.
    """

    def __init__(self):
        super().__init__()
        self._merge_cache = {}  # single-entry: {cache_key: (model_patches, clip_patches, report, clip_strength_out, lora_data)}
        self._persistent_cache = PersistentPatchCache()
        self._detected_arch = None

    @staticmethod
    def _compute_cache_key(lora_stack, output_strength, clip_strength_multiplier, auto_strength, optimization_mode="per_prefix", patch_compression="smart", svd_device="gpu", normalize_keys="disabled", sparsification="disabled", sparsification_density=0.7, dare_dampening=0.0, merge_refinement="none", strategy_set="full", architecture_preset="auto", auto_strength_floor=-1.0, decision_smoothing=0.25, smooth_slerp_gate=False, star_eta=100.0, tame_layers=0.0, tame_threshold=0.3):
        """
        Build a deterministic SHA-256 hash (16 hex chars) from the stack
        configuration. Used by IS_CHANGED to let ComfyUI skip re-execution
        when nothing changed.
        """
        h = hashlib.sha256()
        if lora_stack:
            entries = []
            # Dispatch per item, not on the first element: a stack can mix tuple
            # entries (LoRA Manager and compatible providers) with dict entries
            # carrying preloaded weights. Both produce the same
            # 6-field shape so a mixed stack still hashes consistently.
            for entry in lora_stack:
                if isinstance(entry, (tuple, list)):
                    cm = entry[3] if len(entry) > 3 else "all"
                    kf = entry[4] if len(entry) > 4 else "all"
                    pres = bool(entry[5]) if len(entry) > 5 else False
                    cs = float(entry[2]) if entry[2] is not None else -1.0
                    entries.append((str(entry[0]), float(entry[1]), cs, cm, kf, pres))
                elif isinstance(entry, dict):
                    cm = entry.get("conflict_mode", "all")
                    kf = entry.get("key_filter", "all")
                    pres = bool(entry.get("preserve", False))
                    cs_raw = entry.get("clip_strength", None)
                    cs = float(cs_raw) if cs_raw is not None else -1.0
                    entries.append((str(entry.get("name", "")), float(entry.get("strength", 0)), cs, cm, kf, pres))
            entries.sort()
            h.update(json.dumps(entries).encode())
        h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}|om={optimization_mode}|cp={patch_compression}|sd={svd_device}|nk={normalize_keys}|sp={sparsification}|spd={sparsification_density}|dd={dare_dampening}|mq={merge_refinement}|bp={strategy_set}|ap={architecture_preset}|asf={auto_strength_floor}|ds={decision_smoothing}|ssg={smooth_slerp_gate}".encode())
        # Fold per-LoRA cleaning ONLY when active, so default-off keys stay
        # byte-identical to pre-feature keys (existing caches remain valid).
        if star_eta < 100.0 or tame_layers > 0.0:
            h.update(f"|clean={star_eta},{tame_layers},{tame_threshold}".encode())
        return h.hexdigest()[:16]

    def _persistent_cache_key(self, lora_stack, model, clip, config_key,
                              controller, tame_layers=0.0):
        if tame_layers > 0.0:
            logging.info(
                "[LoRA Optimizer Cache] Persistent cache disabled for TAME because "
                "its result depends on the base checkpoint weights")
            return None
        try:
            source_digest = self._persistent_cache.source_digest(
                lora_stack, controller)
            model_signature = self._persistent_cache.model_signature(
                model, clip, controller)
        except (OSError, PersistentCacheUnsupported) as error:
            logging.info(
                "[LoRA Optimizer Cache] Persistent cache unavailable: %s", error)
            return None
        return self._persistent_cache.build_key(
            config_key, source_digest, model_signature,
            ANALYSIS_CACHE_VERSION)

    def _apply_cached_merge(self, model, clip, cached, cache_kind):
        self._interrupt_check()
        model_patches = cached["model_patches"]
        clip_patches = cached["clip_patches"]
        lora_data = cached["lora_data"]
        output_strength = lora_data.get("output_strength", 1.0)
        clip_strength = lora_data.get("clip_strength", output_strength)
        new_model = model
        new_clip = clip
        if model is not None and model_patches:
            new_model = model.clone()
            new_model.add_patches(model_patches, output_strength)
            self._update_model_size(new_model, model_patches)
        if clip is not None and clip_patches:
            new_clip = clip.clone()
            new_clip.add_patches(clip_patches, clip_strength)
            self._update_model_size(new_clip, clip_patches)
        report = cached.get("report", "")
        report += (
            "\n\nPersistent Cache\n" + "-" * 40 + "\n"
            f"  Cache hit: {cache_kind}\n"
            "  Analysis and merge were skipped; cached patches were applied directly.\n"
        )
        logging.info(
            "[LoRA Optimizer Cache] HIT (%s) — applied %d MODEL + %d CLIP patches",
            cache_kind, len(model_patches), len(clip_patches))
        return (new_model, new_clip, report, None, lora_data)

    @staticmethod
    def _advanced_merge_kwargs(settings):
        """Map the public settings object to the merge engine arguments."""
        return dict(
            auto_strength=settings["auto_strength"],
            auto_strength_floor=settings["auto_strength_floor"],
            optimization_mode=settings["optimization_mode"],
            sparsification=settings["sparsification"],
            sparsification_density=settings["sparsification_density"],
            dare_dampening=settings["dare_dampening"],
            merge_refinement=settings["merge_refinement"],
            strategy_set=settings["strategy_set"],
            normalize_keys=settings["normalize_keys"],
            architecture_preset=settings["architecture_preset"],
            decision_smoothing=settings["decision_smoothing"],
            smooth_slerp_gate=settings["smooth_slerp_gate"],
            star_eta=settings.get("star_eta", 100.0),
            tame_layers=settings.get("tame_layers", 0.0),
            tame_threshold=settings.get("tame_threshold", 0.3),
            cache_patches=settings["cache_patches"],
            persistent_cache=settings.get("persistent_cache", "enabled"),
            patch_compression=settings["patch_compression"],
            svd_device=settings["svd_device"],
            free_vram_between_passes=settings["free_vram_between_passes"],
            vram_budget=settings["vram_budget"],
        )

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, clip=None,
                   clip_strength_multiplier=1.0, auto_strength="enabled",
                   auto_strength_floor=-1.0,
                   free_vram_between_passes="disabled", vram_budget=0.0,
                   optimization_mode="per_prefix",
                   cache_patches="enabled", persistent_cache="enabled",
                   patch_compression="smart",
                   svd_device="gpu", normalize_keys="enabled",
                   sparsification="disabled", sparsification_density=0.7,
                   dare_dampening=0.0, merge_refinement="none",
                   strategy_set="full", architecture_preset="auto",
                   decision_smoothing=0.25, smooth_slerp_gate=False,
                   star_eta=100.0, tame_layers=0.0, tame_threshold=0.3):
        base_key = cls._compute_cache_key(
            lora_stack, output_strength, clip_strength_multiplier,
            auto_strength, optimization_mode, patch_compression, svd_device,
            normalize_keys, sparsification, sparsification_density,
            dare_dampening, merge_refinement,
            strategy_set, architecture_preset, auto_strength_floor,
            decision_smoothing, smooth_slerp_gate, star_eta, tame_layers,
            tame_threshold)
        return (
            f"{base_key}|mid={id(model)}|memory={cache_patches}"
            f"|persistent={persistent_cache}")

    @torch.no_grad()
    def _analyze_target_group(self, target_group, active_loras, model, clip, device,
                              clip_strength_multiplier=1.0,
                              merge_refinement="none", n_magnitude_samples=1000):
        """
        Pass 1 analysis for one resolved target group. All aliases that hit the
        same underlying weight are aggregated per LoRA before statistics are
        computed, so mixed-trainer overlaps are analyzed as one merge unit.
        """
        self._interrupt_check()
        force_cpu_source = False
        if device is not None:
            source_group = self._prepare_group_sources(
                target_group, active_loras, model, clip)
            force_cpu_source = bool(source_group and source_group.get("unsupported"))
            if (source_group is not None and not force_cpu_source and source_group["sources"]
                    and (getattr(self, "_star_eta", 100.0) >= 100.0 or all(
                        source.rank > 0 for source in source_group["sources"].values()))):
                source_count = len(source_group["sources"])
                factor_bytes = sum(source.factor_bytes
                                   for source in source_group["sources"].values())
                plan = self._execution_planner.plan(
                    device, source_group["target_shape"], source_count,
                    max(source_count + 4, source_count * 2 + 3),
                    factor_bytes=factor_bytes, chunkable=True)
                if plan.mode in ("tiled_gpu", "cpu"):
                    if plan.mode == "tiled_gpu" and not self._tiled_gpu_reported:
                        logging.info(
                            f"[LoRA Optimizer] Large targets will use tiled GPU "
                            f"({plan.rows_per_tile} rows for first target); CPU is only a fallback.")
                        self._tiled_gpu_reported = True
                    try:
                        return self._analyze_target_group_tiled(
                            source_group, active_loras, model, clip, plan,
                            merge_refinement, n_magnitude_samples)
                    finally:
                        for source in source_group["sources"].values():
                            source.release()

        prepared = self._prepare_group_diffs(
            target_group, active_loras, model, clip, device,
            clip_strength_multiplier=clip_strength_multiplier,
            merge_refinement=merge_refinement,
            force_cpu=force_cpu_source,
        )
        if prepared is None:
            return None

        diffs = prepared["diffs"]
        eff_strengths = prepared["eff_strengths"]
        rank_sums = prepared["rank_sums"]
        skip_count = prepared["skip_count"]
        target_key = prepared["target_key"]
        is_clip = prepared["is_clip"]
        raw_n = prepared["raw_n_loras"]
        group_device = prepared["compute_device"]

        if len(diffs) == 0:
            if skip_count > 0 or raw_n > 0:
                return (
                    target_group["label_prefix"], [], {}, [], (target_key, is_clip),
                    skip_count, raw_n, {}
                )
            return None

        partial_stats = []
        per_lora_norm_sq = {}
        bases = {}
        for i, diff in diffs.items():
            norm = torch.linalg.vector_norm(diff.float()).item()
            norm_sq = norm * norm
            per_lora_norm_sq[i] = norm_sq
            display_l2 = norm * abs(active_loras[i]["strength"])
            partial_stats.append((i, rank_sums.get(i, 0), display_l2, norm_sq))
            bases[i] = self._compute_subspace_basis(diff, rank_hint=rank_sums.get(i, 1))

        pair_conflicts = {}
        lora_indices = sorted(diffs.keys())
        for ai in range(len(lora_indices)):
            for bi in range(ai + 1, len(lora_indices)):
                i, j = lora_indices[ai], lora_indices[bi]
                diff_i = diffs[i] if eff_strengths[i] >= 0 else -diffs[i]
                diff_j = diffs[j] if eff_strengths[j] >= 0 else -diffs[j]
                pair_conflicts[(i, j)] = self._sample_pair_metrics(
                    diff_i, diff_j, basis_a=bases.get(i), basis_b=bases.get(j),
                    device=group_device if group_device.type != "cpu" else None
                )

        magnitude_samples = []
        # zlib.crc32: stable across processes (hash() varies with PYTHONHASHSEED)
        seed = zlib.crc32(target_group["label_prefix"].encode("utf-8")) & 0xFFFFFFFF
        sample_dev = diffs[lora_indices[0]].device
        mag_g = torch.Generator(device=sample_dev).manual_seed(seed)
        for i in lora_indices:
            flat = diffs.pop(i).flatten()
            n = flat.numel()
            if n > n_magnitude_samples:
                indices = torch.randint(0, n, (n_magnitude_samples,),
                                        device=sample_dev, generator=mag_g)
                flat = flat[indices]
            flat = flat.abs().float() * abs(eff_strengths[i])
            magnitude_samples.append(flat.cpu())

        return (
            target_group["label_prefix"],
            partial_stats,
            pair_conflicts,
            magnitude_samples,
            (target_key, is_clip),
            skip_count,
            raw_n,
            per_lora_norm_sq,
        )

    def _run_group_analysis(self, target_groups, active_loras, model, clip,
                            compute_device, clip_strength_multiplier=1.0,
                            merge_refinement="none",
                            decision_smoothing=0.0, progress_cb=None):
        """Run Pass 1 once and retain only scalar statistics and small samples."""
        self._interrupt_check()
        use_gpu = compute_device.type != "cpu"
        per_lora_stats = [{
            "name": item["name"],
            "strength": item["strength"],
            "ranks": [],
            "key_count": 0,
            "l2_norms": [],
        } for item in active_loras]
        pairs = [(i, j) for i in range(len(active_loras))
                         for j in range(i + 1, len(active_loras))]
        branch_energy = {
            branch: {
                "norm_sq": [0.0] * len(active_loras),
                "dot": {(i, j): 0.0 for i, j in pairs},
            } for branch in ("model", "clip")
        }
        pair_accum = {
            pair: {
                "overlap": 0, "conflict": 0, "dot": 0.0,
                "norm_a_sq": 0.0, "norm_b_sq": 0.0,
                "weighted_total": 0.0, "weighted_conflict": 0.0,
                "expected_conflict_weighted": 0.0,
                "excess_conflict_weighted": 0.0,
                "subspace_num": 0.0, "subspace_den": 0.0,
            } for pair in pairs
        }
        all_magnitude_samples = []
        all_key_targets = {}
        prefix_stats = {}
        skipped_keys = 0
        prefix_count = 0

        def collect(result):
            nonlocal skipped_keys, prefix_count
            if result is None:
                return
            (prefix, partial_stats, pair_conflicts, mag_samples, target_info,
             skips, raw_n, per_lora_norm_sq) = result
            branch_name = "clip" if target_info[1] else "model"
            skipped_keys += skips
            if partial_stats:
                all_key_targets[prefix] = target_info
                prefix_count += 1
            for idx, rank, l2, norm_sq in partial_stats:
                per_lora_stats[idx]["ranks"].append(rank)
                per_lora_stats[idx]["key_count"] += 1
                per_lora_stats[idx]["l2_norms"].append(l2)
                branch_energy[branch_name]["norm_sq"][idx] += norm_sq
            for pair, metrics in pair_conflicts.items():
                acc = pair_accum[pair]
                acc["overlap"] += metrics["overlap"]
                acc["conflict"] += metrics["conflict"]
                acc["dot"] += metrics["dot"]
                acc["norm_a_sq"] += metrics["norm_a_sq"]
                acc["norm_b_sq"] += metrics["norm_b_sq"]
                acc["weighted_total"] += metrics["weighted_total"]
                acc["weighted_conflict"] += metrics["weighted_conflict"]
                acc["expected_conflict_weighted"] += metrics["expected_conflict"] * metrics["weighted_total"]
                acc["excess_conflict_weighted"] += metrics["excess_conflict"] * metrics["weighted_total"]
                acc["subspace_num"] += metrics["subspace_overlap"] * metrics["subspace_weight"]
                acc["subspace_den"] += metrics["subspace_weight"]
                branch_energy[branch_name]["dot"][pair] += metrics["dot"]
            all_magnitude_samples.extend(mag_samples)
            if not partial_stats:
                return
            overlap = sum(item["overlap"] for item in pair_conflicts.values())
            conflict = sum(item["conflict"] for item in pair_conflicts.values())
            conflict_ratio = conflict / overlap if overlap > 0 else 0.0
            weighted_total = sum(item["weighted_total"] for item in pair_conflicts.values())
            weighted_conflict = sum(item["weighted_conflict"] for item in pair_conflicts.values())
            weighted_ratio = weighted_conflict / weighted_total if weighted_total > 0 else conflict_ratio
            expected_conflict = (
                sum(item["expected_conflict"] * item["weighted_total"] for item in pair_conflicts.values()) / weighted_total
                if weighted_total > 0 else 0.0)
            excess_conflict = (
                sum(item["excess_conflict"] * item["weighted_total"] for item in pair_conflicts.values()) / weighted_total
                if weighted_total > 0 else 0.0)
            l2_values = [math.sqrt(value) for value in per_lora_norm_sq.values() if value > 0]
            magnitude_ratio = max(l2_values) / min(l2_values) if len(l2_values) >= 2 else 1.0
            cosine_values = []
            for metrics in pair_conflicts.values():
                denominator = math.sqrt(metrics["norm_a_sq"] * metrics["norm_b_sq"])
                if denominator > 0:
                    cosine_values.append(metrics["dot"] / denominator)
            subspace_den = sum(item["subspace_weight"] for item in pair_conflicts.values())
            prefix_stats[prefix] = {
                "n_loras": len(partial_stats),
                "raw_n_loras": raw_n,
                "conflict_ratio": conflict_ratio,
                "weighted_conflict_ratio": weighted_ratio,
                "expected_conflict": expected_conflict,
                "excess_conflict": excess_conflict,
                "magnitude_ratio": magnitude_ratio,
                "activation_ratio": magnitude_ratio,
                "magnitude_samples": list(mag_samples),
                "avg_cos_sim": sum(cosine_values) / len(cosine_values) if cosine_values else 0.0,
                "avg_subspace_overlap": (
                    sum(item["subspace_overlap"] * item["subspace_weight"] for item in pair_conflicts.values()) / subspace_den
                    if subspace_den > 0 else 0.0),
                "per_lora_norm_sq": dict(per_lora_norm_sq),
                "pairwise_dots": {pair: values["dot"] for pair, values in pair_conflicts.items()},
            }

        group_items = list(target_groups.values())
        if use_gpu:
            for target_group in group_items:
                self._interrupt_check()
                result = self._analyze_target_group(
                    target_group, active_loras, model, clip, compute_device,
                    clip_strength_multiplier=clip_strength_multiplier,
                    merge_refinement=merge_refinement)
                collect(result)
                if progress_cb is not None:
                    progress_cb(target_group["label_prefix"])
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(4, max(1, len(group_items))))
            futures = {}
            try:
                for target_group in group_items:
                    self._interrupt_check()
                    future = executor.submit(
                        self._analyze_target_group, target_group, active_loras,
                        model, clip, compute_device, clip_strength_multiplier,
                        merge_refinement)
                    futures[future] = target_group["label_prefix"]
                pending = set(futures)
                while pending:
                    self._interrupt_check()
                    done, pending = concurrent.futures.wait(
                        pending, timeout=0.05,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        collect(future.result())
                        if progress_cb is not None:
                            progress_cb(futures[future])
            except BaseException:
                self._interrupt_controller.cancel()
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(
                    wait=not self._interrupt_controller.event.is_set(),
                    cancel_futures=True)

        return {
            "all_key_targets": all_key_targets,
            "target_groups": dict(target_groups),
            "prefix_stats": self._apply_block_smoothing(
                prefix_stats, strength=decision_smoothing),
            "per_lora_stats": per_lora_stats,
            "pair_accum": pair_accum,
            "branch_energy": branch_energy,
            "all_magnitude_samples": all_magnitude_samples,
            "prefix_count": prefix_count,
            "skipped_keys": skipped_keys,
            "pairs": pairs,
        }

    def _estimate_density(self, all_key_diffs, arch_preset=None):
        """
        Estimate TIES density parameter from magnitude distribution.
        Uses fraction of values above noise floor of the max magnitude as a
        sparsity proxy. Thresholds come from arch_preset.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
        samples = []
        max_samples_per_key = 1000
        g = torch.Generator().manual_seed(42)

        for key, diffs_list in all_key_diffs.items():
            for entry in diffs_list:
                diff, strength = entry[0], entry[1]
                flat = diff.flatten().abs().float().cpu() * abs(strength)
                n = flat.numel()
                if n > max_samples_per_key:
                    indices = torch.randperm(n, generator=g)[:max_samples_per_key]
                    flat = flat[indices]
                samples.append(flat)

        if len(samples) == 0:
            return 0.5

        all_samples = torch.cat(samples)
        if all_samples.numel() == 0:
            return 0.5

        max_val = all_samples.max().item()
        if max_val <= 0:
            return 0.5

        noise_floor = max_val * arch_preset["density_noise_floor_ratio"]
        above_noise = (all_samples > noise_floor).float().mean().item()

        return max(arch_preset["density_clamp_min"], min(arch_preset["density_clamp_max"], above_noise))

    def _estimate_density_from_samples(self, magnitude_samples, arch_preset=None):
        """
        Estimate TIES density from pre-sampled magnitude tensors.
        Takes a list of 1D CPU float tensors (from _analyze_prefix).
        Thresholds come from arch_preset.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
        if len(magnitude_samples) == 0:
            return 0.5

        all_samples = torch.cat(magnitude_samples)
        if all_samples.numel() == 0:
            return 0.5

        max_val = all_samples.max().item()
        if max_val <= 0:
            return 0.5

        noise_floor = max_val * arch_preset["density_noise_floor_ratio"]
        above_noise = (all_samples > noise_floor).float().mean().item()

        return max(arch_preset["density_clamp_min"], min(arch_preset["density_clamp_max"], above_noise))

    def _compute_branch_auto_scale(self, branch_name, strengths, norm_sq, dot_accum,
                                   arch_preset=None, detected_arch=None,
                                   auto_strength_floor=-1.0, is_full_rank=False):
        """Exact streamed auto-strength scale for one branch."""
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
        n = len(strengths)
        effective = [abs(strengths[i]) * math.sqrt(max(norm_sq[i], 0.0)) for i in range(n)]
        nonzero = [e for e in effective if e > 0]
        reasoning = []
        if len(nonzero) <= 1:
            reasoning.append(f"{branch_name}: single contributing branch or none — no adjustment needed")
            return {
                "scale": 1.0,
                "new_strengths": list(strengths),
                "reasoning": reasoning,
            }

        energy_sq = 0.0
        orthogonal_energy_sq = 0.0
        pairwise_cos = []
        for i in range(n):
            energy_sq += (strengths[i] ** 2) * norm_sq[i]
            orthogonal_energy_sq += (strengths[i] ** 2) * norm_sq[i]
        for (i, j), dot in dot_accum.items():
            if norm_sq[i] <= 0 or norm_sq[j] <= 0:
                continue
            energy_sq += 2.0 * strengths[i] * strengths[j] * dot
            denom = math.sqrt(norm_sq[i]) * math.sqrt(norm_sq[j])
            if denom > 0:
                pairwise_cos.append(dot / denom)

        energy_sq = max(energy_sq, 0.0)
        current_energy = math.sqrt(energy_sq)
        orthogonal_energy = math.sqrt(max(orthogonal_energy_sq, 0.0))
        reference_energy = max(effective)
        scale = min(reference_energy / current_energy, 1.0) if current_energy > 0 else 1.0

        floor_applied = False
        floor = None
        explicit_floor = auto_strength_floor >= 0
        if explicit_floor:
            # An explicitly set floor bounds the reduction REGARDLESS of stack
            # alignment — the widget promises "how much the weakest LoRA is
            # allowed to be scaled down", full stop. Previously the
            # orthogonality gate below silently ignored user floors on
            # aligned/opposing stacks.
            floor = auto_strength_floor
        elif pairwise_cos:
            # The -1 defaults are orthogonality-noise heuristics: aligned
            # stacks compound energy and genuinely need the reduction, so
            # the default floors only apply to mostly-orthogonal stacks.
            avg_cos = sum(pairwise_cos) / len(pairwise_cos)
            alignment_thresh = arch_preset["alignment_threshold"]
            if abs(avg_cos) <= alignment_thresh:
                if is_full_rank:
                    floor = arch_preset.get("full_rank", {}).get("auto_strength_floor", 1.0)
                else:
                    floor = _VIDEO_ARCH_ORTHOGONAL_FLOOR.get(
                        detected_arch,
                        arch_preset.get("auto_strength_orthogonal_floor", 0.85),
                    )
        if floor is not None and scale < floor:
            scale = floor
            floor_applied = True

        new_strengths = [s * scale if effective[i] > 0 else s for i, s in enumerate(strengths)]

        reasoning.append(f"{branch_name}: scale factor {scale:.4f}")
        if pairwise_cos:
            avg_cos = sum(pairwise_cos) / len(pairwise_cos)
            alignment_thresh = arch_preset["alignment_threshold"]
            if avg_cos > alignment_thresh:
                alignment_desc = "mostly aligned (reinforcing)"
            elif avg_cos < -alignment_thresh:
                alignment_desc = "mostly opposing (cancelling)"
            else:
                alignment_desc = "mostly orthogonal (independent)"
            if floor_applied:
                arch_label = detected_arch or "unknown"
                if explicit_floor:
                    reasoning.append(
                        f"{branch_name}: user floor {floor:.2f} applied "
                        f"(bounds auto-strength reduction regardless of alignment)"
                    )
                elif is_full_rank:
                    reasoning.append(
                        f"{branch_name}: full-rank orthogonal floor {floor:.2f} applied "
                        f"to preserve complete weight deltas"
                    )
                else:
                    reasoning.append(
                        f"{branch_name}: orthogonal floor {floor:.2f} applied for {arch_label} "
                        f"to preserve independent contributions"
                    )
            reasoning.append(
                f"{branch_name}: exact streamed energy {current_energy:.4f} "
                f"(orthogonal baseline {orthogonal_energy:.4f}, avg cos {avg_cos:.3f} — {alignment_desc})"
            )
        return {
            "scale": scale,
            "new_strengths": new_strengths,
            "reasoning": reasoning,
        }

    def _compute_auto_strengths(self, active_loras, branch_energy,
                                clip_strength_multiplier=1.0, arch_preset=None,
                                detected_arch=None, auto_strength_floor=-1.0,
                                is_full_rank=False):
        """
        Compute exact streamed auto-strength scaling separately for model and
        CLIP branches, using accumulated Frobenius norms and pairwise dots.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]

        model_strengths = [item["strength"] for item in active_loras]
        clip_strengths = [
            item["clip_strength"] if item["clip_strength"] is not None else item["strength"]
            for item in active_loras
        ]

        model_info = self._compute_branch_auto_scale(
            "Model",
            model_strengths,
            branch_energy["model"]["norm_sq"],
            branch_energy["model"]["dot"],
            arch_preset=arch_preset,
            detected_arch=detected_arch,
            auto_strength_floor=auto_strength_floor,
            is_full_rank=is_full_rank,
        )
        clip_info = self._compute_branch_auto_scale(
            "CLIP",
            clip_strengths,
            branch_energy["clip"]["norm_sq"],
            branch_energy["clip"]["dot"],
            arch_preset=arch_preset,
            detected_arch=detected_arch,
            auto_strength_floor=auto_strength_floor,
            is_full_rank=is_full_rank,
        )

        reasoning = []
        reasoning.extend(model_info["reasoning"])
        if any(v > 0 for v in branch_energy["clip"]["norm_sq"]):
            reasoning.extend(clip_info["reasoning"])

        return {
            "model_scale": model_info["scale"],
            "clip_scale": clip_info["scale"],
            "model_strengths": model_info["new_strengths"],
            "clip_strengths": clip_info["new_strengths"],
            "original_model_strengths": model_strengths,
            "original_clip_strengths": clip_strengths,
            "names": [item["name"] for item in active_loras],
            "clip_uses_global_multiplier": [
                item["clip_strength"] is None for item in active_loras
            ],
            "clip_strength_multiplier": clip_strength_multiplier,
            "reasoning": reasoning,
        }

    @staticmethod
    def _virtual_payload_is_linear_ok(payload):
        """True if a captured payload is a plain 2D LoRAAdapter (no LoCon mid) —
        the only shape the low-rank concat fast path can emit bit-equivalently
        (within float tol) of the dense expand. LoKr/LoHa/dense-tensor/
        ("diff",…)/mid!=None/non-2D payloads return False and keep their group
        on the dense _prepare_group_diffs path."""
        if (not isinstance(payload, LoRAAdapter)
                or not _LoRAMergeBase._is_plain_additive_payload(payload)):
            return False
        # LoKr/LoHa are sibling classes, not LoRAAdapter subclasses; be explicit
        # in case a future comfy makes them subclass a shared base.
        if isinstance(payload, (LoKrAdapter, LoHaAdapter)):
            return False
        w = getattr(payload, "weights", None)
        if not isinstance(w, (tuple, list)) or len(w) < 4:
            return False
        up, down, mid = w[0], w[1], w[3]
        if mid is not None:
            return False
        if not (isinstance(up, torch.Tensor) and isinstance(down, torch.Tensor)):
            return False
        return up.dim() == 2 and down.dim() == 2

    def _virtual_group_is_linear_ok(self, target_group, active_loras, model, clip):
        """True iff every captured (virtual) contributor to this group is a
        plain 2D LoRAAdapter targeting a 2D linear weight — the exact case the
        low-rank concat fast path reproduces (within float tol) of the dense
        _prepare_group_diffs + _merge_diffs path. Any LoKr/LoHa/dense/mid!=None
        contributor, an offset-sliced (tuple) target key, or a non-2D target
        keeps the whole group on the dense path (returns False). A group with no
        virtual contributor returns True — the file path is left untouched."""
        target_key = target_group["target_key"]
        # Offset/sliced targets (e.g. Z-Image QKV refusion) reshape the dense
        # diff to a slice; the low-rank concat can't reproduce that. Keep dense.
        if isinstance(target_key, tuple):
            return False
        has_virtual = False
        for item in active_loras:
            if not item.get("_precomputed_diffs"):
                continue
            payload = item["lora"].get(target_key)
            if payload is None:
                continue  # this virtual item doesn't touch this group
            has_virtual = True
            if not self._virtual_payload_is_linear_ok(payload):
                return False
        if not has_virtual:
            return True
        # Confirm the resolved target weight is a 2D linear so up@down maps 1:1
        # (a 4D conv target would reshape in the dense path but not here).
        try:
            target_shape = self._resolve_target_shape(
                target_key, target_group["is_clip"], model, clip)
        except (AttributeError, RuntimeError, IndexError):
            return False
        return len(target_shape) == 2

    def _build_exact_linear_patch(self, target_group, active_loras, raw_n_loras,
                                  mode, is_clip_key=False, model_scale=1.0):
        """
        Build an exact low-rank patch for linear merges by concatenating factors
        instead of materializing a dense diff. Falls back to None when the group
        contains unsupported parameterizations (for example LoCon mid matrices).

        Handles both file items (trainer-format lora_up/down keys, read by
        alias) and captured/virtual items (_precomputed_diffs: a LoRAAdapter
        payload keyed by the model target key). Virtual items must first pass
        _virtual_group_is_linear_ok at the call site; the per-payload guard here
        is a defensive re-check that falls back to dense on anything unexpected.
        """
        if mode not in ("weighted_sum", "weighted_average", "normalize"):
            return None

        pieces = []
        lora_weights = {}
        has_conflict_modes = False
        is_audio_group = self._target_is_audio(target_group)

        for i, item in enumerate(active_loras):
            kf = item.get("key_filter", "all")
            if kf == "shared_only" and raw_n_loras < 2:
                continue
            if kf == "unique_only" and raw_n_loras != 1:
                continue
            if kf == "audio_only" and not is_audio_group:
                continue
            if kf == "no_audio" and is_audio_group:
                continue
            if item.get("conflict_mode", "all") != "all":
                has_conflict_modes = True
                break

            if is_clip_key:
                base_weight = item["clip_strength"] if item["clip_strength"] is not None else item["strength"]
            else:
                base_weight = item["strength"] * model_scale

            contributed = False
            if item.get("_precomputed_diffs"):
                # Captured chain item: its lora dict maps the MODEL TARGET KEY ->
                # adapter payload (not trainer-format {prefix}.lora_up keys), so
                # read the factors by target key (mirrors _prepare_group_diffs'
                # virtual branch, including the tuple fallback). Only plain 2D
                # LoRAAdapters are eligible; anything else falls back to dense.
                tk = target_group["target_key"]
                payload = item["lora"].get(tk)
                if payload is None and isinstance(tk, tuple):
                    payload = item["lora"].get(tk[0])
                if payload is not None:
                    if not self._virtual_payload_is_linear_ok(payload):
                        return None
                    mat_up, mat_down = payload.weights[0], payload.weights[1]
                    alpha = payload.weights[2]
                    if isinstance(alpha, torch.Tensor):
                        alpha = alpha.item()
                    # alpha=None -> scale 1.0. Fold it as alpha==rank so the
                    # shared piece_scale = weight * (alpha/rank) below reproduces
                    # _expand_patch_to_diff's (alpha/rank if alpha is not None
                    # else 1.0) exactly — the dense path's per-LoRA scale.
                    if alpha is None:
                        alpha = mat_down.shape[0]
                    pieces.append((i, mat_up, mat_down, alpha))
                    contributed = True
            else:
                for alias in target_group["aliases"]:
                    lora_info = self._get_lora_key_info(item["lora"], alias)
                    if lora_info is None:
                        # Check if this alias has LoKr/LoHa keys — can't represent
                        # as low-rank factors, fall through to dense diff path
                        if self._has_lokr_keys(item["lora"], alias) or self._has_loha_keys(item["lora"], alias):
                            return None
                        continue
                    mat_up, mat_down, alpha, mid = lora_info
                    if mid is not None:
                        return None
                    pieces.append((i, mat_up, mat_down, alpha))
                    contributed = True

            if contributed:
                lora_weights[i] = base_weight

        if has_conflict_modes or not pieces:
            return None

        if mode == "weighted_average":
            total_weight = sum(abs(w) for w in lora_weights.values())
            if total_weight == 0:
                return None
            per_lora_scales = {idx: w / total_weight for idx, w in lora_weights.items()}
        elif mode == "normalize":
            denom = math.sqrt(sum(w * w for w in lora_weights.values()))
            if denom == 0:
                return None
            per_lora_scales = {idx: w / denom for idx, w in lora_weights.items()}
        else:
            per_lora_scales = dict(lora_weights)

        up_parts = []
        down_parts = []
        total_rank = 0
        factor_device = pieces[0][1].device
        for lora_idx, mat_up, mat_down, alpha in pieces:
            weight = per_lora_scales[lora_idx]
            rank = mat_down.shape[0]
            total_rank += rank
            piece_scale = weight * (alpha / rank)
            # Comfy expands LoRA factors in float32 and only then applies the
            # patch strength. Scaling fp16/bf16 factors in their storage dtype
            # first introduces visible rounding (and float8 can fail outright).
            # Fuse in float32 to mirror the ordinary loader path.
            up_parts.append(mat_up.to(device=factor_device, dtype=torch.float32)
                            * piece_scale)
            down_parts.append(mat_down.to(device=factor_device, dtype=torch.float32))

        if total_rank <= 0:
            return None

        fused_up = torch.cat(up_parts, dim=1)
        fused_down = torch.cat(down_parts, dim=0)
        return {
            "patch": LoRAAdapter(set(), (fused_up, fused_down, float(total_rank), None, None, None)),
            "weights": per_lora_scales,
        }

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _extract_block_name(prefix):
        """
        Extract a human-readable block name from a LoRA key prefix.
        Handles common architectures: SD1.5, SDXL, Flux, Wan, etc.

        Examples:
          lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_q -> input_blocks.4
          lora_unet_double_blocks_12_img_attn_proj -> double_blocks.12
          lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k -> down_blocks.2
          diffusion_model.joint_blocks.5.x_block.attn.qkv -> joint_blocks.5
          transformer.blocks.8.attn1.to_q -> blocks.8

        Falls back to the first two meaningful segments if no pattern matches.
        Pure function of the prefix — cached (called per prefix per smoothing
        pass, with two regex evaluations each).
        """
        # Normalize separators: lora_unet_input_blocks_4 -> input_blocks.4
        # Strip common prefixes
        p = prefix
        for strip in ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_",
                       "diffusion_model.", "transformer.", "model."]:
            if p.startswith(strip):
                p = p[len(strip):]
                break

        # Replace underscores with dots for pattern matching
        p_dots = re.sub(r'_', '.', p)

        # Match: word.number (e.g., input.blocks.4, double.blocks.12, down.blocks.2)
        m = re.match(r'([a-z]+(?:\.[a-z]+)*?)\.(\d+)', p_dots)
        if m:
            block_type = m.group(1).replace('.', '_')
            block_num = m.group(2)
            return f"{block_type}.{block_num}"

        # Fallback: first segment
        parts = re.split(r'[._]', prefix)
        meaningful = [p for p in parts if p not in ("lora", "unet", "te", "te1", "te2",
                                                     "diffusion", "model", "transformer")]
        if len(meaningful) >= 2:
            return f"{meaningful[0]}.{meaningful[1]}"
        elif meaningful:
            return meaningful[0]
        return prefix[:30]

    def _apply_block_smoothing(self, prefix_stats, strength=0.0):
        """
        Smooth noisy per-group decision metrics toward the average of their
        surrounding logical block. Raw metrics are preserved; decision_* fields
        are what Pass 2 should consume.
        """
        if not prefix_stats:
            return prefix_stats

        strength = max(0.0, min(1.0, float(strength or 0.0)))
        block_groups = {}
        for prefix, stat in prefix_stats.items():
            block_name = self._extract_block_name(prefix)
            stat["block_name"] = block_name
            block_groups.setdefault(block_name, []).append((prefix, stat))

        metric_keys = (
            "conflict_ratio",
            "weighted_conflict_ratio",
            "expected_conflict",
            "excess_conflict",
            "avg_cos_sim",
            "avg_subspace_overlap",
            "magnitude_ratio",
            "activation_ratio",
        )

        for entries in block_groups.values():
            if not entries:
                continue

            weights = []
            for _prefix, stat in entries:
                weight = sum(stat.get("per_lora_norm_sq", {}).values())
                weights.append(weight if weight > 0 else 1.0)

            total_weight = sum(weights) if weights else 0.0
            block_means = {}
            for key in metric_keys:
                values = [stat.get(key) for _prefix, stat in entries if stat.get(key) is not None]
                if not values:
                    continue
                weighted_sum = 0.0
                for weight, (_prefix, stat) in zip(weights, entries):
                    weighted_sum += stat.get(key, 0.0) * weight
                block_means[key] = weighted_sum / total_weight if total_weight > 0 else sum(values) / len(values)

            for prefix, stat in entries:
                stat["block_size"] = len(entries)
                stat["block_smoothing_strength"] = strength
                for key, mean_value in block_means.items():
                    raw_value = stat.get(key, mean_value)
                    smoothed = raw_value if strength <= 0 or len(entries) == 1 else ((1.0 - strength) * raw_value + strength * mean_value)
                    stat[f"smoothed_{key}"] = smoothed
                stat["decision_conflict"] = stat.get("smoothed_excess_conflict", stat.get("excess_conflict", stat.get("conflict_ratio", 0.0)))
                stat["decision_cosine"] = stat.get("smoothed_avg_cos_sim", stat.get("avg_cos_sim", 0.0))
                stat["decision_subspace_overlap"] = stat.get("smoothed_avg_subspace_overlap", stat.get("avg_subspace_overlap", 0.0))
                stat["decision_magnitude_ratio"] = stat.get("smoothed_activation_ratio", stat.get("activation_ratio", stat.get("magnitude_ratio", 1.0)))
        return prefix_stats

    def _auto_select_params(self, avg_conflict_ratio, magnitude_ratio, all_key_diffs=None,
                            magnitude_samples=None, avg_cos_sim=0.0,
                            avg_excess_conflict=None, avg_subspace_overlap=0.0,
                            strategy_set="full", arch_preset=None,
                            precomputed_density=None):
        """
        Decision logic for auto-selecting merge parameters.
        Returns (mode, density, sign_method, reasoning_lines).

        Density can be estimated from either all_key_diffs (legacy bulk path)
        or magnitude_samples (streaming path). Thresholds come from arch_preset.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
        reasoning = []

        effective_conflict = avg_conflict_ratio
        if avg_excess_conflict is not None:
            effective_conflict = max(avg_excess_conflict, 0.0)
            if avg_subspace_overlap > 0:
                effective_conflict *= (0.5 + 0.5 * avg_subspace_overlap)

        # High similarity + low conflict → consensus mode (Fisher-proxy + magnitude calibration)
        if (strategy_set == "full"
                and avg_cos_sim > arch_preset["consensus_cos_sim_min"]
                and effective_conflict < arch_preset["consensus_conflict_max"]
                and avg_subspace_overlap >= 0.35):
            mode = "consensus"
            reasoning.append(f"Cosine similarity {avg_cos_sim:.2f} > {arch_preset['consensus_cos_sim_min']} "
                             f"and excess conflict {effective_conflict:.1%} < {arch_preset['consensus_conflict_max']:.0%} -> consensus mode")
            reasoning.append("  Fisher-proxy importance weighting + magnitude calibration + spectral cleanup")
            density = 0.5  # unused
            sign_method = "frequency"  # unused
            return (mode, density, sign_method, reasoning)

        # Near-orthogonal LoRAs: ~50% sign conflict is the base rate for
        # independent vectors, not actual semantic conflict. TIES trimming
        # destroys both signals. Use weighted_average (balanced blend), upgraded
        # to SLERP per-prefix in the full strategy set to preserve magnitude.
        if (strategy_set in ("full", "no_slerp")
                and abs(avg_cos_sim) < arch_preset["orthogonal_cos_sim_max"]
                and effective_conflict < arch_preset["orthogonal_conflict_max"]
                and avg_subspace_overlap < 0.35):
            mode = "weighted_average"
            reasoning.append(f"Cosine similarity {avg_cos_sim:.2f} near zero (orthogonal LoRAs) — "
                             f"sign conflict {avg_conflict_ratio:.1%} is base-rate noise, not real conflict")
            if strategy_set == "full":
                reasoning.append("  Using weighted_average (upgraded to SLERP per-prefix to preserve magnitude)")
            else:
                reasoning.append("  Using weighted_average to preserve both signals (SLERP upgrade disabled by profile)")
            density = 0.5
            sign_method = "frequency"
            return (mode, density, sign_method, reasoning)

        # Select mode based on sign conflict
        if effective_conflict > arch_preset["ties_conflict_threshold"]:
            mode = "ties"
            reasoning.append(f"Excess conflict {effective_conflict:.1%} > {arch_preset['ties_conflict_threshold']:.0%} threshold -> TIES mode selected")
            if avg_subspace_overlap > 0:
                reasoning.append(f"  Subspace overlap {avg_subspace_overlap:.2f} suggests the conflicting LoRAs target similar directions")
            reasoning.append("  TIES resolves sign conflicts via trim + elect sign + disjoint merge")
        else:
            mode = "weighted_average"
            reasoning.append(f"Excess conflict {effective_conflict:.1%} <= {arch_preset['ties_conflict_threshold']:.0%} threshold -> weighted_average mode selected")
            reasoning.append("  Low conflict means LoRAs are mostly compatible, simple averaging works well")

        # Auto-density (TIES only)
        if mode == "ties":
            if magnitude_samples is not None:
                density = self._estimate_density_from_samples(magnitude_samples, arch_preset=arch_preset)
            elif all_key_diffs is not None:
                density = self._estimate_density(all_key_diffs, arch_preset=arch_preset)
            elif precomputed_density is not None:
                density = precomputed_density
            else:
                density = 0.5
            reasoning.append(f"Auto-density estimated at {density:.2f} from magnitude distribution")
        else:
            density = 0.5  # unused but set for completeness

        # Sign method (only relevant for TIES mode).
        # Always magnitude-weighted "total" voting: the TIES paper (NeurIPS
        # 2023) defines sign election ONLY as gamma = sgn(sum of trimmed task
        # vectors) — the sign with greater total magnitude wins. There is no
        # frequency vote in the paper; it remains available as an explicit
        # merge_strategy override only.
        if mode == "ties":
            sign_method = "total"
            reasoning.append("Sign election: 'total' (magnitude-weighted voting, TIES-canonical)")
        else:
            sign_method = "total"  # unused, default for completeness

        return (mode, density, sign_method, reasoning)

    def _decide_prefix_mode(self, pf, strategy_set, arch_preset, smooth_slerp_gate,
                            is_full_rank, fr_preset):
        """
        Per-prefix merge mode decision for per_prefix mode (n_loras >= 2).
        Single source of truth for Pass 2; keep all mode gating here.
        Returns (mode, density, sign_method, orthogonal, opposing).
        """
        pf_mode, pf_density, pf_sign, _ = self._auto_select_params(
            pf["conflict_ratio"], pf.get("decision_magnitude_ratio", pf["magnitude_ratio"]),
            magnitude_samples=pf.get("magnitude_samples"),
            avg_cos_sim=pf.get("decision_cosine", pf.get("avg_cos_sim", 0.0)),
            avg_excess_conflict=pf.get("decision_conflict", pf.get("excess_conflict", pf.get("conflict_ratio", 0.0))),
            avg_subspace_overlap=pf.get("decision_subspace_overlap", pf.get("avg_subspace_overlap", 0.0)),
            strategy_set=strategy_set,
            arch_preset=arch_preset,
            precomputed_density=pf.get("precomputed_density"),
        )
        # Upgrade weighted_average → slerp for 2+ non-opposing LoRAs.
        # SLERP preserves magnitude better than weighted_average's /N reduction,
        # which is critical for video LoRAs where motion energy matters.
        # Skip for opposing LoRAs (cos < 0): SLERP interpolates between opposing
        # directions while preserving magnitude, amplifying artifacts.
        pf_raw_cos = pf.get("decision_cosine", pf.get("avg_cos_sim", 0.0)) if smooth_slerp_gate else pf.get("avg_cos_sim", 0.0)
        pf_orthogonal = abs(pf_raw_cos) < arch_preset["orthogonal_cos_sim_max"]
        pf_opposing = pf_raw_cos < 0
        # NOTE: orthogonal groups are NOT auto-routed to sum_preserve. The
        # analyzer cannot distinguish "preserve this style LoRA's full emphasis"
        # (wants additive) from "blend these character LoRAs evenly" (wants the
        # average) — both look orthogonal. Auto-selecting sum_preserve oversaturates
        # ordinary multi-LoRA blends (Σ wᵢdᵢ grows with N). Style preservation is
        # therefore driven by the explicit per-LoRA `preserve` flag instead (handled
        # in _merge_diffs: blend the rest, add the tagged LoRA at full on top).
        # Per-prefix magnitude imbalance gate. SLERP averages DIRECTIONS and
        # discards magnitude: for an orthogonal pair where one LoRA dominates,
        # the Karcher mean rotates the strong LoRA ~halfway toward the weak one
        # and rescales to the AVERAGE of the two norms — washing out the
        # dominant LoRA (≈40% retained at a 6x imbalance) while over-amplifying
        # the weak one (≈220%). So only upgrade to SLERP when the contributors
        # are reasonably balanced.
        pf_mag_ratio = pf.get("magnitude_ratio", 1.0)
        slerp_imbalance_cap = arch_preset.get("slerp_imbalance_ratio", 2.0)
        pf_imbalanced = pf_mag_ratio >= slerp_imbalance_cap
        if (pf_mode == "weighted_average" and pf["n_loras"] >= 2
                and strategy_set == "full"
                and not pf_opposing
                and not pf_imbalanced
                and not (is_full_rank and fr_preset.get("disable_slerp_upgrade", False))):
            pf_mode = "slerp"
        # A strongly-imbalanced ORTHOGONAL pair: additive (weighted_sum)
        # preserves the dominant LoRA at full while SLERP/weighted_average both
        # wash it out. Safe from oversaturation because the dominant LoRA
        # defines the auto-strength reference energy (scale ≈ 1.0); the
        # floor-clamped oversaturation risk is a balanced-multi-LoRA effect, so
        # this is gated to n_loras == 2. The existing full-rank rule is kept.
        if (pf_mode == "weighted_average" and pf_orthogonal and not pf_opposing
                and ((pf_imbalanced and pf["n_loras"] == 2)
                     or (is_full_rank and fr_preset.get("prefer_sum_orthogonal", False)))):
            pf_mode = "weighted_sum"
        return pf_mode, pf_density, pf_sign, pf_orthogonal, pf_opposing

    def _build_report(self, lora_stats, pairwise_conflicts, collection_stats,
                      mode, density, sign_method, reasoning, merge_summary,
                      auto_strength_info=None, strategy_counts=None, optimization_mode="global",
                      prefix_decisions=None, detected_arch=None, normalize_keys="disabled",
                      sparsification="disabled", sparsification_density=0.7,
                      dare_dampening=0.0,
                      merge_refinement="none",
                      compatibility_warnings=None,
                      strategy_set="full",
                      architecture_preset=None,
                      is_full_rank=False):
        """Format analysis as a multi-line report string."""
        lines = []
        lines.append("=" * 50)
        lines.append("LORA OPTIMIZER - ANALYSIS REPORT")
        lines.append("=" * 50)

        # Architecture preset info
        if architecture_preset and architecture_preset in _ARCH_PRESETS:
            preset_info = _ARCH_PRESETS[architecture_preset]
            lines.append(f"Architecture preset: {architecture_preset} ({preset_info['display_name']})")
            if detected_arch:
                arch_names = {
                    'zimage': 'Z-Image Turbo (Lumina2)',
                    'flux': 'FLUX',
                    'wan': 'Wan 2.1/2.2',
                    'acestep': 'ACE-Step',
                    'sdxl': 'SDXL',
                    'sd15': 'SD 1.5',
                    'ltx': 'LTX Video',
                    'qwen_image': 'Qwen-Image',
                    'anima': 'Anima (Cosmos-Predict2 DiT)',
                    'krea2': 'Krea 2',
                }
                lines.append(f"Detected architecture: {arch_names.get(detected_arch, detected_arch)}")
            if normalize_keys == "enabled":
                lines.append(f"Key normalization: enabled")
            lines.append("")
        elif normalize_keys == "enabled" and detected_arch:
            arch_names = {
                'zimage': 'Z-Image Turbo (Lumina2)',
                'flux': 'FLUX',
                'wan': 'Wan 2.1/2.2',
                'acestep': 'ACE-Step',
                'sdxl': 'SDXL',
                'ltx': 'LTX Video',
                'qwen_image': 'Qwen-Image',
                'anima': 'Anima (Cosmos-Predict2 DiT)',
            }
            lines.append(f"Architecture: {arch_names.get(detected_arch, detected_arch)} (auto-detected)")
            lines.append(f"Key normalization: enabled")
            lines.append("")

        if compatibility_warnings:
            lines.append("")
            lines.append("!!! COMPATIBILITY WARNING !!!")
            for warn in compatibility_warnings:
                lines.append(f"  {warn['name_i']} vs {warn['name_j']}: cosine similarity = {warn['cosine_sim']:.3f}")
                lines.append(f"    These LoRAs work against each other and may cancel out.")
                lines.append(f"    Consider removing one or using conflict_mode='high_conflict'")
            lines.append("")

        # Per-LoRA Analysis
        lines.append("")
        lines.append("--- Per-LoRA Analysis ---")
        for stat in lora_stats:
            lines.append(f"  {stat['name']}:")
            lines.append(f"    Strength: {stat.get('original_strength', stat['strength'])}")
            lines.append(f"    Keys: {stat['key_count']}")
            if stat['key_count'] > 0:
                lines.append(f"    Avg rank: {stat['avg_rank']:.0f}")
                lines.append(f"    L2 norm (mean): {stat['l2_mean']:.4f}")
            else:
                lines.append(f"    Avg rank: N/A (no compatible keys)")
                lines.append(f"    L2 norm (mean): N/A")
            if stat.get("conflict_mode", "all") != "all":
                lines.append(f"    Conflict mode: {stat['conflict_mode']}")
            if stat.get("key_filter", "all") != "all":
                lines.append(f"    Key filter: {stat['key_filter']}")

        # Rank variation hint — suggest SVD scoring when ranks differ significantly
        ranks_with_keys = [s['avg_rank'] for s in lora_stats if s['key_count'] > 0 and s['avg_rank'] > 0]
        if len(ranks_with_keys) >= 2:
            rank_ratio = max(ranks_with_keys) / min(ranks_with_keys) if min(ranks_with_keys) > 0 else 1.0
            if rank_ratio >= 2.0:
                lines.append("")
                lines.append(f"  ** Rank variation detected (ratio {rank_ratio:.1f}x) — try scoring_svd='lora_rank' or 'full'")
                lines.append(f"     SVD scoring may produce better rankings when LoRA ranks differ significantly.")

        # Auto-Strength Adjustment (between Per-LoRA and Pairwise)
        if auto_strength_info is not None:
            lines.append("")
            lines.append("--- Auto-Strength Adjustment ---")
            for i, name in enumerate(auto_strength_info["names"]):
                orig = auto_strength_info["original_model_strengths"][i]
                new = auto_strength_info["model_strengths"][i]
                line = f"  {name}: model {orig} -> {new:.4f}"
                if i < len(auto_strength_info.get("original_clip_strengths", [])):
                    clip_orig = auto_strength_info["original_clip_strengths"][i]
                    clip_new = auto_strength_info["clip_strengths"][i]
                    line += f", clip {clip_orig} -> {clip_new:.4f}"
                    if auto_strength_info.get("clip_uses_global_multiplier", [False] * len(auto_strength_info["names"]))[i]:
                        line += " (pre-global multiplier)"
                lines.append(line)
            for r in auto_strength_info["reasoning"]:
                lines.append(f"  {r}")

        # Pairwise Analysis
        if pairwise_conflicts:
            lines.append("")
            lines.append("--- Pairwise Analysis ---")
            for pc in pairwise_conflicts:
                lines.append(f"  {pc['pair']}:")
                lines.append(f"    Overlapping positions: {pc['overlap']}")
                lines.append(f"    Sign conflicts: {pc['conflicts']} ({pc['ratio']:.1%})")
                if 'weighted_ratio' in pc:
                    lines.append(f"    Magnitude-weighted conflict: {pc['weighted_ratio']:.1%}")
                if 'expected_conflict' in pc:
                    lines.append(f"    Excess conflict over cosine baseline: {pc.get('excess_conflict', 0.0):.1%} (expected {pc['expected_conflict']:.1%})")
                if 'cosine_sim' in pc:
                    lines.append(f"    Cosine similarity: {pc['cosine_sim']:.3f}")
                if 'subspace_overlap' in pc:
                    lines.append(f"    Subspace overlap: {pc['subspace_overlap']:.2f}")

        # Collection Statistics
        lines.append("")
        lines.append("--- Collection Statistics ---")
        lines.append(f"  Total LoRAs: {collection_stats['n_loras']}")
        lines.append(f"  Total target groups: {collection_stats['total_keys']}")
        lines.append(f"  Avg sign conflict ratio: {collection_stats['avg_conflict']:.1%}")
        if "avg_weighted_conflict" in collection_stats:
            lines.append(f"  Avg weighted conflict ratio: {collection_stats['avg_weighted_conflict']:.1%}")
        if "avg_excess_conflict" in collection_stats:
            lines.append(f"  Avg excess conflict: {collection_stats['avg_excess_conflict']:.1%}")
        if "avg_subspace_overlap" in collection_stats:
            lines.append(f"  Avg subspace overlap: {collection_stats['avg_subspace_overlap']:.2f}")
        lines.append(f"  Importance ratio (max/min frobenius): {collection_stats['magnitude_ratio']:.2f}x")
        if collection_stats.get("decision_smoothing", 0.0) > 0:
            lines.append(f"  Decision smoothing: {collection_stats['decision_smoothing']:.2f}")

        # Auto-Selected Parameters
        lines.append("")
        lines.append("--- Auto-Selected Parameters ---")
        if optimization_mode == "additive":
            lines.append(f"  Merge mode: weighted_sum (forced by additive)")
            lines.append(f"  Auto-detected mode: {mode} (overridden)")
        else:
            lines.append(f"  Merge mode: {mode}")
        if mode == "ties":
            lines.append(f"  Density: {density:.2f}")
            lines.append(f"  Sign method: {sign_method}")
        if optimization_mode == "per_prefix":
            lines.append("  (global fallback — each target group uses its own parameters)")
        if sparsification != "disabled":
            display_name = {
                "dare": "DARE", "della": "DELLA",
                "dare_conflict": "DARE (conflict-aware)",
                "della_conflict": "DELLA (conflict-aware)",
            }.get(sparsification, sparsification.upper())
            lines.append(f"  Sparsification: {display_name}")
            lines.append(f"  Sparsification density: {sparsification_density:.2f} (keep rate)")
            if dare_dampening > 0 and sparsification in ("dare", "dare_conflict"):
                q = sparsification_density + dare_dampening * (1.0 - sparsification_density)
                lines.append(f"  DAREx dampening: {dare_dampening:.2f} (rescale factor: 1/{q:.2f} = {1.0/q:.2f}x vs standard 1/{sparsification_density:.2f} = {1.0/sparsification_density:.2f}x)")
            if optimization_mode == "per_prefix":
                lines.append(f"  For TIES prefixes: replaces trim step; others: preprocessing")
            elif optimization_mode == "additive":
                lines.append(f"  Applied as preprocessing before weighted_sum")
            elif mode == "ties":
                lines.append(f"  Note: {display_name} replaces TIES trim step")
            else:
                lines.append(f"  Applied as preprocessing before {mode}")

        if merge_refinement != "none":
            quality_desc = {
                "refine": "Refine (orthogonalize + TALL-masks)",
                "full": "Full (orthogonalize + KnOTS SVD alignment + TALL-masks)",
            }
            lines.append(f"  Merge refinement: {quality_desc.get(merge_refinement, merge_refinement)}")

        if strategy_set != "full":
            profile_desc = {
                "no_slerp": "no_slerp (full detection, no SLERP upgrade)",
                "basic": "basic (TIES vs weighted_average only)",
            }
            lines.append(f"  Strategy set: {profile_desc.get(strategy_set, strategy_set)}")

        # Per-Prefix Strategy breakdown (only in per_prefix mode)
        if optimization_mode == "per_prefix" and strategy_counts:
            lines.append("")
            lines.append("--- Per-Group Strategy ---")
            total_pf = sum(strategy_counts.values())
            if total_pf > 0:
                if strategy_counts.get("weighted_sum", 0) > 0:
                    n = strategy_counts["weighted_sum"]
                    lines.append(f"  weighted_sum (single LoRA):      {n:>4} groups ({n/total_pf:.0%})")
                if strategy_counts.get("slerp", 0) > 0:
                    n = strategy_counts["slerp"]
                    lines.append(f"  slerp (low conflict):            {n:>4} groups ({n/total_pf:.0%})")
                if strategy_counts.get("weighted_average", 0) > 0:
                    n = strategy_counts["weighted_average"]
                    lines.append(f"  weighted_average (orthogonal):   {n:>4} groups ({n/total_pf:.0%})")
                if strategy_counts.get("consensus", 0) > 0:
                    n = strategy_counts["consensus"]
                    lines.append(f"  consensus (high similarity):     {n:>4} groups ({n/total_pf:.0%})")
                if strategy_counts.get("ties", 0) > 0:
                    n = strategy_counts["ties"]
                    lines.append(f"  ties (high conflict):            {n:>4} groups ({n/total_pf:.0%})")
                lines.append(f"  Total:                           {total_pf:>4} groups")

        # Block Strategy Map (per_prefix mode only)
        if optimization_mode == "per_prefix" and prefix_decisions:
            # Group prefixes by block name
            block_data = {}  # block_name -> list of (mode, conflict, n_loras)
            for prefix, pf_mode, conflict, n_loras in prefix_decisions:
                block_name = self._extract_block_name(prefix)
                if block_name not in block_data:
                    block_data[block_name] = []
                block_data[block_name].append((pf_mode, conflict, n_loras))

            # Aggregate per block: dominant strategy, avg conflict, max n_loras
            # Priority-based dominant: ties > slerp > avg > sum (show most interesting)
            mode_priority = {"ties": 4, "consensus": 3, "slerp": 2, "weighted_average": 1, "weighted_sum": 0}
            block_summary = []
            for block_name, entries in block_data.items():
                modes = [e[0] for e in entries]
                conflicts = [e[1] for e in entries]
                n_loras_max = max(e[2] for e in entries)
                mode_counts = {}
                for m in modes:
                    mode_counts[m] = mode_counts.get(m, 0) + 1
                # Pick highest-priority mode present (not most frequent)
                dominant = max(mode_counts, key=lambda m: mode_priority.get(m, -1))
                # Avg conflict only over multi-LoRA prefixes (sum prefixes have 0 conflict)
                multi_conflicts = [c for m, c in zip(modes, conflicts) if m != "weighted_sum"]
                avg_conflict = sum(multi_conflicts) / len(multi_conflicts) if multi_conflicts else 0
                n_prefixes = len(entries)
                block_summary.append((block_name, dominant, avg_conflict, n_loras_max, n_prefixes, mode_counts))

            # Sort by block name for consistent ordering
            block_summary.sort(key=lambda x: x[0])

            lines.append("")
            lines.append("--- Block Strategy Map ---")
            symbols = {"weighted_sum": "====", "slerp": "~~~~", "weighted_average": "----", "ties": "####", "consensus": "++++"}
            labels = {"weighted_sum": "sum", "slerp": "slrp", "weighted_average": "avg", "ties": "TIES", "consensus": "cons"}
            # Find max block name length for alignment
            max_name = max(len(b[0]) for b in block_summary) if block_summary else 10
            for block_name, dominant, avg_conflict, n_loras_max, n_prefixes, mode_counts in block_summary:
                sym = symbols.get(dominant, "????")
                lbl = labels.get(dominant, dominant)
                if len(mode_counts) == 1 and dominant == "weighted_sum":
                    detail = "1 LoRA"
                else:
                    # Show breakdown when block has mixed strategies
                    parts = []
                    for m in ("weighted_sum", "weighted_average", "slerp", "consensus", "ties"):
                        if mode_counts.get(m, 0) > 0:
                            parts.append(f"{mode_counts[m]} {labels.get(m, m)}")
                    detail = f"{avg_conflict:.0%} conflict ({', '.join(parts)})"
                count_str = f"({n_prefixes}x)" if n_prefixes > 1 else ""
                lines.append(f"  {block_name:<{max_name}}  {sym}  {lbl:<5} {detail} {count_str}")
            lines.append(f"  Legend: ==== sum  ~~~~ slerp  ---- avg  ++++ cons  #### TIES")

        # Reasoning
        lines.append("")
        lines.append("--- Reasoning ---")
        for r in reasoning:
            lines.append(f"  {r}")

        # Merge Summary
        lines.append("")
        lines.append("--- Merge Summary ---")
        lines.append(f"  Keys processed: {merge_summary['keys_processed']}")
        lines.append(f"  Model patches: {merge_summary['model_patches']}")
        lines.append(f"  CLIP patches: {merge_summary['clip_patches']}")
        if merge_summary.get('skipped_keys', 0) > 0:
            lines.append(f"  Skipped keys: {merge_summary['skipped_keys']} (no matching model weight)")
        lines.extend(self._shape_mismatch_report_lines())
        os_val = merge_summary['output_strength']
        auto_os = merge_summary.get('auto_output_strength', False)
        if auto_os:
            lines.append(f"  Output strength: {os_val:.2f} (auto — suggested max)")
        else:
            lines.append(f"  Output strength: {os_val}")
        lines.append(f"  CLIP strength: {merge_summary['clip_strength']}")
        if not auto_os and merge_summary.get('suggested_max_strength') is not None:
            sms = merge_summary['suggested_max_strength']
            lines.append(f"  Suggested max output_strength: {sms:.2f}")
            if sms >= 3.0:
                lines.append(f"    (capped at 3.0 — actual headroom may be higher)")
            elif sms == 1.0:
                lines.append(f"    (energy preserved — no compensation needed)")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def optimize_merge(self, model, lora_stack, output_strength, clip=None,
                       clip_strength_multiplier=1.0, auto_strength="enabled",
                       auto_strength_floor=-1.0,
                       free_vram_between_passes="disabled", vram_budget=0.0,
                       optimization_mode="per_prefix", cache_patches="enabled",
                       persistent_cache="enabled", patch_compression="smart",
                       svd_device="gpu", normalize_keys="enabled",
                       sparsification="disabled", sparsification_density=0.7,
                       dare_dampening=0.0, merge_refinement="none",
                       strategy_set="full",
                       architecture_preset="auto", decision_smoothing=0.25,
                       smooth_slerp_gate=False, star_eta=100.0,
                       tame_layers=0.0, tame_threshold=0.3):
        """
        Main entry point. Two-pass streaming architecture:
        Pass 1: Resolve aliases to target groups, compute diffs, sample metrics, discard diffs
        Decision: Finalize stats, auto-select params from lightweight accumulators
        Pass 2: Recompute diffs per target group, merge immediately, discard
        Peak memory tracks the largest active target group, not the whole stack.
        """
        self._interrupt_controller = InterruptController()
        self._interrupt_check()
        self._progress_state = None
        self._execution_stats = {"full_gpu": set(), "tiled_gpu": set(), "cpu": set(), "tile_rows": []}
        self._tiled_gpu_reported = False
        self._cpu_fallback_reported = False
        self._shape_mismatches = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._star_eta = star_eta
        self._tame_layers = tame_layers
        self._tame_threshold = tame_threshold

        # Free stale cached models when the input model/clip changes — prevents
        # the old patched clones from staying in RAM after switching models.
        if self._track_model_identity(model, clip):
            self._merge_cache.clear()
            gc.collect()

        # Normalize stack format (standard tuples or preloaded dicts)
        if not lora_stack or len(lora_stack) == 0:
            return (model, clip, "No LoRAs in stack.", None, None)

        config_cache_key = self._compute_cache_key(
            lora_stack, output_strength, clip_strength_multiplier,
            auto_strength, optimization_mode, patch_compression,
            svd_device, normalize_keys, sparsification,
            sparsification_density, dare_dampening,
            merge_refinement, strategy_set,
            architecture_preset, auto_strength_floor,
            decision_smoothing, smooth_slerp_gate,
            star_eta, tame_layers, tame_threshold)
        persistent_cache_key = None
        if persistent_cache == "enabled" and len(lora_stack) > 1:
            persistent_cache_key = self._persistent_cache_key(
                lora_stack, model, clip, config_cache_key,
                self._interrupt_controller, tame_layers=tame_layers)
        runtime_cache_key = (
            f"{persistent_cache_key or config_cache_key}|mid={id(model)}")

        if cache_patches == "enabled" and runtime_cache_key in self._merge_cache:
            model_patches, clip_patches, report, _clip_strength, lora_data = (
                self._merge_cache[runtime_cache_key])
            return self._apply_cached_merge(model, clip, {
                "model_patches": model_patches,
                "clip_patches": clip_patches,
                "report": report,
                "lora_data": lora_data,
            }, "memory")

        if persistent_cache_key is not None:
            cached = self._persistent_cache.load(
                persistent_cache_key, self._interrupt_controller)
            if cached is not None:
                lora_data = cached["lora_data"]
                if cache_patches == "enabled":
                    self._merge_cache = {runtime_cache_key: (
                        cached["model_patches"], cached["clip_patches"],
                        cached["report"], lora_data.get("clip_strength", 1.0),
                        lora_data)}
                return self._apply_cached_merge(
                    model, clip, cached, "disk")

        # For virtual adapter payloads, the comfy model class is an authoritative
        # architecture signal that model-space keys cannot always provide. Pass
        # it as a hint; _normalize_stack consults it only for virtual items and
        # only when key-based file detection failed (file stacks unchanged).
        arch_hint = self._model_class_arch(model)
        normalized_stack = self._normalize_stack(
            lora_stack, normalize_keys=normalize_keys, _arch_hint=arch_hint)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]

        if len(active_loras) == 0:
            return (model, clip, "No LoRAs in stack (all zero strength or malformed).", None, None)

        # Resolve architecture preset from override or auto-detection
        preset_key, arch_preset = _resolve_arch_preset(
            architecture_preset, getattr(self, '_detected_arch', None) or 'unknown')
        logging.info(f"[LoRA Optimizer] Architecture preset: {preset_key} ({arch_preset['display_name']})")

        # Single LoRA: skip analysis, apply directly via ComfyUI's standard
        # additive LoRA application (faster than diff-based pipeline).
        # auto_strength is a no-op with a single LoRA (scale would be 1.0).
        # Skip fast path for Z-Image: normalized keys (to_q/to_k/to_v) won't
        # match the model's fused qkv keys — need full pipeline + re-fusion.
        # Skip for virtual items (_precomputed_diffs): their lora dict maps
        # MODEL-TARGET keys to adapters/diffs, not trainer-format keys, so
        # comfy.sd.load_lora_for_models would resolve nothing and silently
        # apply NO LoRA at all (fatal for a provider that has already
        # stripped the originals off the model).
        if (len(active_loras) == 1
                and getattr(self, '_detected_arch', None) != 'zimage'
                and not active_loras[0].get("_precomputed_diffs")):
            item = active_loras[0]
            lora_dict = item["lora"]
            strength = item["strength"]
            resolved_output_strength = 1.0 if output_strength < 0 else output_strength

            if item["clip_strength"] is not None:
                clip_str = item["clip_strength"]
            else:
                clip_str = strength * clip_strength_multiplier
            new_model, new_clip = comfy.sd.load_lora_for_models(
                model, clip, lora_dict, resolved_output_strength * strength, resolved_output_strength * clip_str
            )

            report = (
                "=" * 50 + "\n"
                "LORA OPTIMIZER - ANALYSIS REPORT\n"
                "=" * 50 + "\n\n"
                "Single LoRA detected — bypassing analysis.\n"
                f"  Name: {item['name']}\n"
                f"  Strength: {strength}\n"
                f"  Applied directly with output_strength={resolved_output_strength}\n"
                "\n" + "=" * 50
            )
            return (new_model, new_clip, report, None, None)

        logging.info(f"[LoRA Optimizer] Starting analysis of {len(active_loras)} LoRAs")
        t_start = time.time()

        compute_device = self._get_compute_device()
        use_gpu = compute_device.type != "cpu"
        _standard_pbar = None

        # Key maps + target groups are only built when Pass 1 actually runs —
        # with cached analysis they come from the cache,
        # so building them here would be discarded work on every candidate.
        model_keys = self._get_model_keys(model)
        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})

        # Build target groups so all aliases for the same underlying weight
        # are analyzed and merged together.
        all_lora_prefixes = self._collect_lora_prefixes(active_loras)
        target_groups = self._build_target_groups(all_lora_prefixes, model_keys, clip_keys)
        if not target_groups:
            return (model, clip, "No compatible LoRA keys found. "
                    "LoRAs may be incompatible with this model architecture.", None, None)
        progress_total = (len(target_groups) * 2 + 1) * 1000
        _standard_pbar = comfy.utils.ProgressBar(progress_total)
        self._progress_state = {
            "bar": _standard_pbar,
            "total": progress_total,
            "value": 0,
            "targets": {},
            "decision": False,
            "lock": threading.Lock(),
        }

        # =====================================================================
        # Pass 1 — Analysis (streaming: diffs computed, sampled, and discarded)
        # =====================================================================
        logging.info("[LoRA Optimizer] Pass 1: Analyzing weight diffs (streaming)...")
        logging.info(f"[LoRA Optimizer]   {len(target_groups)} target groups from {len(all_lora_prefixes)} aliases across {len(active_loras)} LoRAs")
        logging.info(f"[LoRA Optimizer]   Compute device: {compute_device}"
                     f" ({'sequential' if use_gpu else 'threaded'})")
        t_pass1 = time.time()
        # Periodic console progress so a long Pass 1 (many groups / LoRAs) isn't silent.
        _p1_total = len(target_groups)
        _p1 = {"n": 0, "logged": 0.0}
        _p1_every = max(1, _p1_total // 10)

        def _p1_progress(prefix=None):
            _p1["n"] += 1
            self._progress_update("pass1", prefix, 1.0)
            c = _p1["n"]
            now = time.monotonic()
            if c == _p1_total or (c % _p1_every == 0 and now - _p1["logged"] >= 1.0):
                _p1["logged"] = now
                logging.info(f"[LoRA Optimizer]   Analyzed {c}/{_p1_total} target "
                             f"groups ({time.time() - t_pass1:.0f}s)")

        analysis_data = self._run_group_analysis(
            target_groups, active_loras, model, clip, compute_device,
            clip_strength_multiplier=clip_strength_multiplier,
            merge_refinement=merge_refinement,
            decision_smoothing=decision_smoothing,
            progress_cb=_p1_progress,
        )
        all_key_targets = analysis_data["all_key_targets"]
        target_groups = analysis_data["target_groups"]
        prefix_stats = analysis_data["prefix_stats"]
        per_lora_stats = analysis_data["per_lora_stats"]
        pair_accum = analysis_data["pair_accum"]
        branch_energy = analysis_data["branch_energy"]
        all_magnitude_samples = analysis_data["all_magnitude_samples"]
        prefix_count = analysis_data["prefix_count"]
        skipped_keys = analysis_data["skipped_keys"]
        pairs = analysis_data["pairs"]

        if prefix_count == 0:
            return (model, clip, "No compatible LoRA keys found. "
                    "LoRAs may be incompatible with this model architecture.", None, None)

        # Log per-LoRA summaries
        for i, stat in enumerate(per_lora_stats):
            avg_r = sum(stat["ranks"]) / len(stat["ranks"]) if stat["ranks"] else 0
            logging.info(f"[LoRA Optimizer]   {stat['name']} ({i+1}/{len(active_loras)}): "
                         f"{stat['key_count']} groups, avg rank {avg_r:.0f}")
        logging.info(f"[LoRA Optimizer]   Total: {prefix_count} target groups ({time.time() - t_pass1:.1f}s)")

        # =====================================================================
        # Decision — finalize stats, auto-select params (scalars only)
        # =====================================================================

        # Finalize per-LoRA stats
        lora_stats = []
        l2_means = []
        for i, stat in enumerate(per_lora_stats):
            avg_rank = sum(stat["ranks"]) / len(stat["ranks"]) if stat["ranks"] else 0
            l2_mean = sum(stat["l2_norms"]) / len(stat["l2_norms"]) if stat["l2_norms"] else 0
            l2_means.append(l2_mean)
            lora_stats.append({
                "name": stat["name"],
                "strength": stat["strength"],
                "key_count": stat["key_count"],
                "avg_rank": avg_rank,
                "l2_mean": l2_mean,
                "conflict_mode": active_loras[i].get("conflict_mode", "all"),
                "key_filter": active_loras[i].get("key_filter", "all"),
            })

        fr_preset = arch_preset.get("full_rank", {})
        fr_rank_threshold = fr_preset.get("rank_threshold", 512)
        global_avg_rank = (sum(s["avg_rank"] for s in lora_stats) / len(lora_stats)) if lora_stats else 0
        is_full_rank = global_avg_rank >= fr_rank_threshold
        logging.info(f"[LoRA Optimizer] Global avg rank: {global_avg_rank:.0f} (full-rank threshold: {fr_rank_threshold})")
        if is_full_rank:
            logging.info(
                f"[LoRA Optimizer] Full-rank LoRAs detected "
                f"(avg rank {global_avg_rank:.0f} >= {fr_rank_threshold})"
            )
            if fr_preset.get("disable_slerp_upgrade", False):
                logging.info("[LoRA Optimizer]   SLERP upgrade disabled for full-rank patches")
            if fr_preset.get("prefer_sum_orthogonal", False):
                logging.info("[LoRA Optimizer]   Orthogonal full-rank patches will use weighted_sum (additive)")

        # Pairwise conflict stats and cosine similarity from accumulated counts
        total_overlap = 0
        total_conflict = 0
        total_weighted_total = 0.0
        total_weighted_conflict = 0.0
        total_expected_conflict_weighted = 0.0
        total_excess_conflict_weighted = 0.0
        total_subspace_num = 0.0
        total_subspace_den = 0.0
        pairwise_conflicts = []
        pairwise_similarities = {}
        for i, j in pairs:
            pair_metrics = pair_accum[(i, j)]
            pair_overlap = pair_metrics["overlap"]
            pair_conflict = pair_metrics["conflict"]
            pair_dot = pair_metrics["dot"]
            pair_na_sq = pair_metrics["norm_a_sq"]
            pair_nb_sq = pair_metrics["norm_b_sq"]
            total_overlap += pair_overlap
            total_conflict += pair_conflict
            total_weighted_total += pair_metrics["weighted_total"]
            total_weighted_conflict += pair_metrics["weighted_conflict"]
            total_expected_conflict_weighted += pair_metrics["expected_conflict_weighted"]
            total_excess_conflict_weighted += pair_metrics["excess_conflict_weighted"]
            total_subspace_num += pair_metrics["subspace_num"]
            total_subspace_den += pair_metrics["subspace_den"]
            ratio = pair_conflict / pair_overlap if pair_overlap > 0 else 0
            weighted_ratio = (pair_metrics["weighted_conflict"] / pair_metrics["weighted_total"]) if pair_metrics["weighted_total"] > 0 else ratio
            expected_conflict = (pair_metrics["expected_conflict_weighted"] / pair_metrics["weighted_total"]) if pair_metrics["weighted_total"] > 0 else 0.0
            excess_conflict = (pair_metrics["excess_conflict_weighted"] / pair_metrics["weighted_total"]) if pair_metrics["weighted_total"] > 0 else 0.0
            subspace_overlap = (pair_metrics["subspace_num"] / pair_metrics["subspace_den"]) if pair_metrics["subspace_den"] > 0 else 0.0
            denom = math.sqrt(pair_na_sq) * math.sqrt(pair_nb_sq)
            cos_sim = pair_dot / denom if denom > 0 else 0.0
            pairwise_similarities[(i, j)] = cos_sim
            name_i = active_loras[i]['name']
            name_j = active_loras[j]['name']
            if name_i == name_j:
                pair_label = f"{name_i} [#{i+1}, str={active_loras[i]['strength']}] vs {name_j} [#{j+1}, str={active_loras[j]['strength']}]"
            else:
                pair_label = f"{name_i} vs {name_j}"
            pairwise_conflicts.append({
                "pair": pair_label,
                "overlap": pair_overlap,
                "conflicts": pair_conflict,
                "ratio": ratio,
                "weighted_ratio": weighted_ratio,
                "expected_conflict": expected_conflict,
                "excess_conflict": excess_conflict,
                "cosine_sim": cos_sim,
                "subspace_overlap": subspace_overlap,
            })
            logging.info(f"[LoRA Optimizer]   {pair_label} -> raw={ratio:.1%}, excess={excess_conflict:.1%}, cos_sim={cos_sim:.3f}, subspace={subspace_overlap:.2f}")

        # Compatibility warnings for opposing LoRAs
        compatibility_warnings = []
        for (i, j), cos_sim in pairwise_similarities.items():
            if cos_sim < -0.1:
                compatibility_warnings.append({
                    "name_i": active_loras[i]['name'],
                    "name_j": active_loras[j]['name'],
                    "cosine_sim": cos_sim,
                })
                logging.warning(
                    f"[LoRA Optimizer] WARNING: {active_loras[i]['name']} vs {active_loras[j]['name']} "
                    f"have negative cosine similarity ({cos_sim:.3f}) — they work against each other"
                )

        avg_conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0
        avg_weighted_conflict_ratio = total_weighted_conflict / total_weighted_total if total_weighted_total > 0 else avg_conflict_ratio
        avg_expected_conflict = total_expected_conflict_weighted / total_weighted_total if total_weighted_total > 0 else 0.0
        avg_excess_conflict = total_excess_conflict_weighted / total_weighted_total if total_weighted_total > 0 else 0.0
        avg_subspace_overlap = total_subspace_num / total_subspace_den if total_subspace_den > 0 else 0.0
        if len(active_loras) > 1:
            logging.info(f"[LoRA Optimizer]   Average conflict ratio: {avg_conflict_ratio:.1%}")
            logging.info(f"[LoRA Optimizer]   Excess conflict: {avg_excess_conflict:.1%} | subspace overlap: {avg_subspace_overlap:.2f}")

        # Magnitude ratio
        branch_measure = branch_energy["model"]["norm_sq"]
        model_effective = [
            abs(active_loras[i]["strength"]) * math.sqrt(max(branch_measure[i], 0.0))
            for i in range(len(active_loras))
        ]
        valid_l2 = [m for m in model_effective if m > 0]
        if len(valid_l2) >= 2:
            magnitude_ratio = max(valid_l2) / min(valid_l2)
        else:
            magnitude_ratio = 1.0

        collection_stats = {
            "n_loras": len(active_loras),
            "total_keys": prefix_count,
            "avg_conflict": avg_conflict_ratio,
            "avg_weighted_conflict": avg_weighted_conflict_ratio,
            "avg_expected_conflict": avg_expected_conflict,
            "avg_excess_conflict": avg_excess_conflict,
            "avg_subspace_overlap": avg_subspace_overlap,
            "magnitude_ratio": magnitude_ratio,
            "decision_smoothing": decision_smoothing,
        }

        # Auto-select parameters (density estimated from pre-sampled magnitudes)
        if len(active_loras) == 1:
            mode = "weighted_average"
            density = 0.5
            sign_method = "frequency"
            reasoning = ["Single LoRA — applied directly"]
        else:
            global_avg_cos_sim = (sum(ps.get("cosine_sim", 0.0) for ps in pairwise_conflicts)
                                  / len(pairwise_conflicts)) if pairwise_conflicts else 0.0
            mode, density, sign_method, reasoning = self._auto_select_params(
                avg_conflict_ratio, magnitude_ratio, magnitude_samples=all_magnitude_samples,
                avg_cos_sim=global_avg_cos_sim, strategy_set=strategy_set,
                avg_excess_conflict=avg_excess_conflict,
                avg_subspace_overlap=avg_subspace_overlap,
                arch_preset=arch_preset
            )
        del all_magnitude_samples

        logging.info(f"[LoRA Optimizer] Decision: {mode} ({reasoning[0] if reasoning else 'no reasoning'})")
        if mode == "ties":
            logging.info(f"[LoRA Optimizer]   density={density:.2f}, sign_method={sign_method}")

        # Auto-strength adjustment
        auto_strength_info = None
        model_auto_scale = 1.0
        clip_auto_scale = 1.0
        if auto_strength == "enabled":
            auto_strength_info = self._compute_auto_strengths(
                active_loras, branch_energy,
                clip_strength_multiplier=clip_strength_multiplier,
                arch_preset=arch_preset,
                detected_arch=getattr(self, '_detected_arch', None),
                auto_strength_floor=auto_strength_floor,
                is_full_rank=is_full_rank,
            )
            model_auto_scale = auto_strength_info["model_scale"]
            clip_auto_scale = auto_strength_info["clip_scale"]
            for i, stat in enumerate(lora_stats):
                stat["original_strength"] = stat["strength"]
                stat["strength"] = auto_strength_info["model_strengths"][i]

            if auto_strength_info["reasoning"]:
                logging.info(f"[LoRA Optimizer] Auto-strength: {auto_strength_info['reasoning'][0]}")
            for i in range(len(active_loras)):
                logging.info(
                    f"[LoRA Optimizer]   {active_loras[i]['name']}: "
                    f"model {active_loras[i]['strength']} -> {auto_strength_info['model_strengths'][i]:.4f}"
                )

        # Free GPU cache between passes if requested
        if free_vram_between_passes == "enabled" and use_gpu:
            torch.cuda.empty_cache()

        # Resolve patch_compression rank (sum of input LoRA ranks)
        sum_rank = sum(int(stat["avg_rank"]) for stat in lora_stats if stat["avg_rank"] > 0)
        compress_rank = 0  # 0 = disabled
        if patch_compression in ("smart", "aggressive"):
            compress_rank = max(sum_rank, 64)  # floor at 64
            logging.info(f"[LoRA Optimizer] Patch compression: {patch_compression} (rank {compress_rank} from sum of input LoRA ranks)")

        # Resolve SVD device for compression
        resolved_svd_device = None
        if compress_rank > 0 and svd_device == "gpu" and torch.cuda.is_available():
            resolved_svd_device = torch.device("cuda")
        elif compress_rank > 0 and svd_device == "cpu":
            resolved_svd_device = None  # CPU is the default in _compress_to_lowrank

        # =====================================================================
        # Pass 2 — Merge (recompute diffs per target group, merge, discard)
        # =====================================================================
        self._interrupt_check()
        self._progress_decision()
        logging.info(f"[LoRA Optimizer] Pass 2: Merging {len(target_groups)} target groups "
                     f"({optimization_mode} strategy, "
                     f"{'sequential' if use_gpu else 'threaded'})...")
        t_pass2 = time.time()
        model_patches = {}
        clip_patches = {}
        processed_keys = 0
        compressed_count = 0

        # Opt-in per-phase merge profiling (LORA_OPTIMIZER_PROFILE_MERGE=1).
        # _merge_prof maps phase -> [count, seconds]; updates are locked because
        # the non-GPU path merges groups across threads.
        _profiling_on = _merge_profiling_enabled()
        _merge_prof = {} if _profiling_on else None
        _merge_prof_lock = threading.Lock() if _profiling_on else None
        _prof_cuda = (use_gpu and compute_device is not None
                      and getattr(compute_device, "type", None) == "cuda")
        # A GPU compression SVD (torch.svd_lowrank) can race asynchronously on
        # some stacks (Blackwell/cu130), corrupting state and aborting a later
        # kernel — a C++ abort the Python try/except can't catch. When the
        # compression SVD runs on GPU, serialize each group so the SVD finishes
        # before the next group launches. Gated tightly (compression ON + GPU
        # SVD) so fast no-compression merges keep their pipelined speed.
        _compress_sync = (_prof_cuda and compress_rank > 0
                          and resolved_svd_device is not None)

        def _prof_t():
            """Synced timestamp — without the sync, async CUDA kernels would
            attribute their cost to whatever line later forces a sync."""
            if _prof_cuda:
                torch.cuda.synchronize(compute_device)
            return time.perf_counter()

        def _prof_add(phase, dt):
            with _merge_prof_lock:
                e = _merge_prof.get(phase)
                if e is None:
                    _merge_prof[phase] = [1, dt]
                else:
                    e[0] += 1
                    e[1] += dt
        strategy_counts = {"weighted_sum": 0, "weighted_average": 0, "slerp": 0, "ties": 0, "consensus": 0}
        prefix_decisions = []  # list of (prefix, mode, conflict_ratio, n_loras) for block map
        has_virtual_loras = any(item.get("_precomputed_diffs") for item in active_loras)
        # conflict_mode masking depends on merge_refinement, so Pass-1 norms
        # (computed with refinement "none") can't be reused when any LoRA masks
        stack_has_conflict_modes = any(
            item.get("conflict_mode", "all") != "all" for item in active_loras)
        # A tagged preserve LoRA needs the dense _merge_diffs overlay (blend the
        # rest, add the preserved one at full on top) — the exact-linear fast path
        # would average it into the blend, so disable it for the whole stack.
        stack_has_preserve = any(
            item.get("preserve", False) for item in active_loras)

        # VRAM budget for patch storage
        vram_budget_bytes = 0
        gpu_patch_bytes = 0
        if vram_budget > 0 and use_gpu:
            free_vram = comfy.model_management.get_free_memory(compute_device)
            safety_margin = 256 * 1024 * 1024  # 256MB headroom
            usable = max(0, free_vram - safety_margin)
            vram_budget_bytes = int(usable * vram_budget)
            logging.info(f"[LoRA Optimizer] VRAM patch budget: {vram_budget_bytes // (1024**2)}MB "
                         f"({vram_budget*100:.0f}% of {usable // (1024**2)}MB free)")

        def _can_place_patch_on_gpu(patch_bytes):
            if (vram_budget_bytes <= 0
                    or gpu_patch_bytes + patch_bytes > vram_budget_bytes
                    or compute_device is None or compute_device.type != "cuda"):
                return False
            live_free = comfy.model_management.get_free_memory(compute_device)
            total_memory = torch.cuda.get_device_properties(compute_device).total_memory
            safety = max(512 * 1024 * 1024, int(total_memory * 0.10))
            return patch_bytes + safety <= live_free

        def _merge_one_group(label_prefix, target_group):
            """Recompute diffs for one target group, merge, return patch or None."""
            self._interrupt_check()
            nonlocal gpu_patch_bytes
            should_keep = vram_budget_bytes > 0 and gpu_patch_bytes < vram_budget_bytes
            target_key = target_group["target_key"]
            is_clip_key = target_group["is_clip"]

            # Determine strategy BEFORE computing diffs (use Pass 1 stats)
            pf_conflict = 0.0
            pf_n_loras = 0
            pf_mode = mode
            pf_density = density
            pf_sign = sign_method
            pf_orthogonal = False
            pf_opposing = False
            if optimization_mode == "additive":
                pf_mode = "weighted_sum"
                pf_n_loras = prefix_stats.get(label_prefix, {}).get("n_loras", 0)
                pf_conflict = prefix_stats.get(label_prefix, {}).get("decision_conflict",
                                                                     prefix_stats.get(label_prefix, {}).get("conflict_ratio", 0.0))
            elif optimization_mode == "global":
                # Global mode: the selected/overridden merge mode applies to
                # multi-LoRA groups. pf_n_loras MUST be populated here — when
                # it stayed 0, the single-contributor fallback below silently
                # forced weighted_sum for EVERY group, so global-mode
                # candidates never actually merged with their declared mode.
                pf_n_loras = prefix_stats.get(label_prefix, {}).get("n_loras", 0)
                pf_conflict = prefix_stats.get(label_prefix, {}).get("decision_conflict",
                                                                     prefix_stats.get(label_prefix, {}).get("conflict_ratio", 0.0))
            elif optimization_mode == "per_prefix" and label_prefix in prefix_stats:
                pf = prefix_stats[label_prefix]
                pf_conflict = pf.get("decision_conflict", pf.get("conflict_ratio", 0.0))
                pf_n_loras = pf["n_loras"]
                if pf["n_loras"] <= 1:
                    pf_mode = "weighted_sum"
                    pf_density = 0.5
                    pf_sign = "frequency"
                else:
                    pf_mode, pf_density, pf_sign, pf_orthogonal, pf_opposing = (
                        self._decide_prefix_mode(
                            pf, strategy_set, arch_preset, smooth_slerp_gate,
                            is_full_rank, fr_preset))

            pf = prefix_stats.get(label_prefix, {})
            raw_n = pf.get("raw_n_loras", pf_n_loras)
            if pf_n_loras <= 1 and pf_mode != "weighted_sum":
                pf_mode = "weighted_sum"

            linear_stats = None
            # Single-LoRA groups (layers only one LoRA touches) have no conflict
            # to sparsify and no second diff to align, so TIES/DARE/refinement
            # are no-ops on them. Take the exact low-rank fast path for them even
            # when those global toggles are on — this skips the wasteful
            # dense-materialize + compression SVD and emits the LoRA's native
            # factors directly. Size-safe: a single LoRA's compress_rank is
            # max(rank, 64) >= its native rank, so compression never shrinks it
            # (only pads rank<64 ones to 64) — bypassing is always equal-or-smaller.
            _single_lora_group = pf_n_loras == 1  # exactly one contributor
            _linear_quality_ok = (sparsification == "disabled"
                                  and merge_refinement == "none")
            # Captured (virtual) chains were previously excluded wholesale by
            # has_virtual_loras, forcing every group through the dense
            # _prepare_group_diffs materialize (~10x slower on repeated virtual-adapter
            # sweeps). Allow the fast path for virtual groups that qualify —
            # every captured contributor is a plain 2D LoRAAdapter (mid None)
            # on a 2D linear target — which _build_exact_linear_patch emits
            # bit-equivalently. Non-qualifying virtual groups keep the dense
            # path (the check returns False, or the builder bails to None).
            if (pf_mode in ("weighted_sum", "weighted_average", "normalize")
                    and (not has_virtual_loras
                         or self._virtual_group_is_linear_ok(
                             target_group, active_loras, model, clip))
                    and not stack_has_preserve
                    and (_single_lora_group or _linear_quality_ok)):
                _t_lin = _prof_t() if _merge_prof is not None else 0.0
                linear_patch_info = self._build_exact_linear_patch(
                    target_group, active_loras, raw_n, pf_mode,
                    is_clip_key=is_clip_key, model_scale=model_auto_scale,
                )
                if linear_patch_info is not None:
                    if _merge_prof is not None:
                        _prof_add("linear_fast", _prof_t() - _t_lin)
                    patch = linear_patch_info["patch"]
                    weights = linear_patch_info["weights"]
                    input_norms_mean = (
                        sum(math.sqrt(max(pf.get("per_lora_norm_sq", {}).get(i, 0.0), 0.0)) * abs(w)
                            for i, w in weights.items()) / len(weights)
                    ) if weights else 0.0
                    energy_sq = 0.0
                    for i, weight in weights.items():
                        energy_sq += (weight ** 2) * pf.get("per_lora_norm_sq", {}).get(i, 0.0)
                    for (i, j), dot in pf.get("pairwise_dots", {}).items():
                        if i in weights and j in weights:
                            energy_sq += 2.0 * weights[i] * weights[j] * dot
                    merged_norm = math.sqrt(max(energy_sq, 0.0))
                    linear_stats = (input_norms_mean, merged_norm)
                    if should_keep:
                        p_bytes = self._estimate_single_patch_bytes(patch)
                        if _can_place_patch_on_gpu(p_bytes):
                            patch = self._move_patch_to_device(patch, compute_device)
                            gpu_patch_bytes += p_bytes
                    result = (
                        target_key, is_clip_key, patch, pf_mode, label_prefix,
                        pf_conflict, max(pf_n_loras, 1), False,
                        linear_stats[0], linear_stats[1],
                        None,  # LoRAAdapter patches are scored from factors
                    )
                    return result

            _t_prep = _prof_t() if _merge_prof is not None else 0.0
            tiled_plan = None
            tiled_source_group = self._prepare_group_sources(
                target_group, active_loras, model, clip,
                auto_scale=model_auto_scale if not is_clip_key else 1.0)
            force_cpu_source = bool(
                tiled_source_group and tiled_source_group.get("unsupported"))
            tiled_modes = {"weighted_sum", "weighted_average", "normalize", "slerp", "consensus", "ties"}
            if (pf_mode in tiled_modes
                    and merge_refinement in ("none", "refine", "full")):
                if (tiled_source_group is not None and not force_cpu_source
                        and tiled_source_group["sources"]
                        and (star_eta >= 100.0 or all(
                            source.rank > 0
                            for source in tiled_source_group["sources"].values()))):
                    source_count = len(tiled_source_group["sources"])
                    factor_bytes = sum(source.factor_bytes
                                       for source in tiled_source_group["sources"].values())
                    tiled_buffers = source_count + 4 if pf_mode == "ties" else 4
                    tiled_plan = self._execution_planner.plan(
                        compute_device, tiled_source_group["target_shape"], source_count,
                        tiled_buffers, factor_bytes=factor_bytes, chunkable=True)
            if tiled_plan is not None and tiled_plan.mode in ("tiled_gpu", "cpu"):
                self._interrupt_check()
                if factor_bytes <= tiled_plan.workset_bytes:
                    for source in tiled_source_group["sources"].values():
                        source.stage(tiled_plan.device)
                diff_indices = sorted(tiled_source_group["sources"])
                preserve_list = [bool(active_loras[i].get("preserve", False))
                                 for i in diff_indices]
                merged_diff = self._merge_group_sources_tiled(
                    tiled_source_group, active_loras, tiled_plan, pf_mode,
                    pf_density, pf_sign, preserve_list,
                    sparsification=sparsification,
                    sparsification_density=sparsification_density,
                    dare_dampening=dare_dampening,
                    merge_refinement=merge_refinement,
                    model=model,
                    clip=clip,
                    star_eta=star_eta,
                    tame_layers=tame_layers,
                    tame_threshold=tame_threshold)
                pf_norm_sq = pf.get("per_lora_norm_sq") or {}
                input_norms_mean = (
                    sum(math.sqrt(max(pf_norm_sq.get(i, 0.0), 0.0))
                        * abs(tiled_source_group["eff_strengths"][i]) for i in diff_indices)
                    / len(diff_indices)) if diff_indices else 0.0
                merged_norm = torch.linalg.vector_norm(merged_diff).item()
                rank_bound = tiled_source_group.get("rank_bound")
                smart_safe = (
                    isinstance(rank_bound, int) and rank_bound <= compress_rank
                    and pf_mode in ("weighted_sum", "weighted_average", "normalize", "slerp"))
                should_compress = (compress_rank > 0 and
                                   (patch_compression == "aggressive" or
                                    (patch_compression == "smart" and smart_safe)))
                storage_dtype = tiled_source_group["storage_dtype"] or torch.float32
                score_stats = None
                if should_compress:
                    chunk_svd_device = (tiled_plan.device if svd_device == "gpu"
                                        else torch.device("cpu"))
                    u, singular, vh = chunked_randomized_svd(
                        lambda start, end, device: merged_diff.reshape(
                            merged_diff.shape[0], -1)[start:end].to(device),
                        merged_diff.shape, compress_rank, chunk_svd_device,
                        tiled_plan.rows_per_tile, self._interrupt_controller,
                        niter=2, seed=42)
                    sqrt_s = singular.sqrt()
                    mat_up = (u * sqrt_s.unsqueeze(0)).to(storage_dtype)
                    mat_down = (vh * sqrt_s.unsqueeze(1)).to(storage_dtype)
                    patch = LoRAAdapter(
                        set(), (mat_up, mat_down, float(mat_up.shape[1]), None, None, None))
                    is_compressed = True
                else:
                    if storage_dtype != torch.float32:
                        merged_diff = merged_diff.to(storage_dtype)
                    patch = ("diff", (merged_diff,))
                    is_compressed = False
                if should_keep:
                    p_bytes = self._estimate_single_patch_bytes(patch)
                    if _can_place_patch_on_gpu(p_bytes):
                        patch = self._move_patch_to_device(patch, compute_device)
                        gpu_patch_bytes += p_bytes
                if self._execution_stats is not None:
                    if tiled_plan.mode == "tiled_gpu":
                        self._execution_stats["tiled_gpu"].add(target_key)
                        self._execution_stats["tile_rows"].append(tiled_plan.rows_per_tile)
                    else:
                        self._execution_stats["cpu"].add(target_key)
                result = (target_key, is_clip_key, patch, pf_mode, label_prefix,
                          pf_conflict, max(pf_n_loras, 1), is_compressed,
                          input_norms_mean, merged_norm, score_stats)
                return result

            prepared = self._prepare_group_diffs(
                target_group, active_loras, model, clip, compute_device,
                clip_strength_multiplier=clip_strength_multiplier,
                merge_refinement=merge_refinement,
                auto_scale=model_auto_scale if not is_clip_key else 1.0,
                force_cpu=force_cpu_source,
            )
            if _merge_prof is not None:
                _prof_add("diff_prep", _prof_t() - _t_prep)
            if prepared is None or len(prepared["diffs"]) == 0:
                return None

            diffs_list = []
            preserve_list = []
            storage_dtype = prepared["storage_dtype"]
            group_device = prepared["compute_device"]
            group_svd_device = (resolved_svd_device
                                if group_device.type != "cpu" else None)
            prepared_diffs = prepared["diffs"]
            diff_indices = sorted(prepared_diffs.keys())
            for i in diff_indices:
                diffs_list.append((prepared_diffs[i], prepared["eff_strengths"][i]))
                preserve_list.append(bool(active_loras[i].get("preserve", False)))
            # _merge_diffs drops consumed inputs as it goes. Remove this second
            # owner so those large dense tensors can actually be released.
            prepared_diffs.clear()

            if len(diffs_list) <= 1 and pf_mode != "weighted_sum":
                pf_mode = "weighted_sum"
                _gc_key = None  # decision diverged from the planned key — don't cache

            # Create deterministic per-group RNG for reproducible sparsification
            sp_gen = None
            if sparsification != "disabled":
                seed = int(hashlib.sha256(label_prefix.encode()).hexdigest(), 16) % (2**63)
                sp_gen = torch.Generator(device=group_device)
                sp_gen.manual_seed(seed)

            # Reuse the exact per-LoRA norms from Pass 1 instead of re-reading
            # every diff tensor here (a full pass + sync per diff per group per
            # candidate). Fallback covers conflict-mode masking (differs by
            # refinement) and prefixes missing from the analysis stats.
            pf_norm_sq = pf.get("per_lora_norm_sq") or {}
            if (diff_indices and not stack_has_conflict_modes
                    and all(i in pf_norm_sq for i in diff_indices)):
                input_norms_mean = (
                    sum(math.sqrt(max(pf_norm_sq[i], 0.0)) * abs(prepared["eff_strengths"][i])
                        for i in diff_indices) / len(diff_indices)
                )
            else:
                input_norms_mean = (sum(d.float().norm().item() * abs(w) for d, w in diffs_list)
                                    / len(diffs_list)) if diffs_list else 0.0

            # For orthogonal/opposing weighted_average, force standard quality.
            # TALL-masks classifies most positions as "selfish" for orthogonal
            # LoRAs (each dominates independent positions) and adds them back at
            # full strength AFTER the merge, bypassing weighted_average's /N
            # normalization.  This converts weighted_average → weighted_sum (~Nx energy).
            # Opposing LoRAs have a similar problem: the magnitude calibration
            # in enhanced quality doesn't account for the directional cancellation
            # that weighted_average provides.
            pf_quality = merge_refinement
            if (pf_mode == "weighted_average" and pf_opposing):
                pf_quality = "none"

            _t_merge = _prof_t() if _merge_prof is not None else 0.0
            merged_diff = self._merge_diffs(
                diffs_list, pf_mode,
                density=pf_density, majority_sign_method=pf_sign,
                compute_device=group_device,
                sparsification=sparsification,
                sparsification_density=sparsification_density,
                sparsification_generator=sp_gen,
                merge_refinement=pf_quality,
                dare_dampening=dare_dampening,
                # Keep the result on the selected group device: the norm and
                # dtype downcast below run there before any final transfer.
                keep_on_gpu=group_device.type != "cpu",
                preserve_flags=preserve_list,
            )
            if _merge_prof is not None:
                _prof_add(f"merge:{pf_mode}", _prof_t() - _t_merge)
            merged_norm = merged_diff.float().norm().item() if merged_diff is not None else 0.0
            diffs_list.clear()  # Free input diffs from GPU
            if merged_diff is None:
                return None
            # "smart" compression is only allowed when the result has a known
            # rank upper bound that fits the budget and no rank-increasing mask
            # or refinement was applied. Dense captured diffs and LoKr/LoHa do
            # not provide such a bound; truncating them at the global rank-64
            # floor was a major additive-merge quality regression. "aggressive"
            # remains the explicit opt-in to lossy truncation.
            rank_bound = prepared.get("rank_bound")
            smart_compression_safe = (
                isinstance(rank_bound, int)
                and rank_bound <= compress_rank
                and pf_mode in ("weighted_sum", "weighted_average", "normalize", "slerp")
                and sparsification == "disabled"
                and merge_refinement == "none"
                and not stack_has_conflict_modes
                and star_eta >= 100.0
            )
            should_compress = (compress_rank > 0 and
                               (patch_compression == "aggressive" or
                                (patch_compression == "smart" and smart_compression_safe)))
            # Downcast from float32 to native weight dtype (e.g. fp16/bf16)
            # to halve memory — ComfyUI handles dtype conversion when applying
            if storage_dtype is not None and merged_diff.dtype != storage_dtype:
                merged_diff = merged_diff.to(storage_dtype)
            score_stats = None
            if (merged_diff.is_cuda and not should_keep
                    and not (should_compress and group_svd_device is not None)):
                merged_diff = merged_diff.cpu()
            if should_compress:
                _t_comp = _prof_t() if _merge_prof is not None else 0.0
                patch = self._compress_to_lowrank(merged_diff, compress_rank, svd_device=group_svd_device)
                if _merge_prof is not None:
                    _prof_add("compress", _prof_t() - _t_comp)
                if patch is None:
                    # SVD failed to converge (e.g. ROCm) — keep the dense diff.
                    patch = ("diff", (merged_diff.cpu() if merged_diff.is_cuda else merged_diff,))
                    is_compressed = False
                else:
                    del merged_diff
                    is_compressed = True
            else:
                patch = ("diff", (merged_diff,))
                is_compressed = False
            if should_keep:
                p_bytes = self._estimate_single_patch_bytes(patch)
                if _can_place_patch_on_gpu(p_bytes):
                    patch = self._move_patch_to_device(patch, compute_device)
                    gpu_patch_bytes += p_bytes
                else:
                    patch = self._move_patch_to_device(patch, torch.device("cpu"))
            result = (target_key, is_clip_key, patch, pf_mode, label_prefix, pf_conflict, max(pf_n_loras, 1), is_compressed, input_norms_mean, merged_norm, score_stats)
            return result

        lowrank_count = 0
        total_input_energy = 0.0
        total_merged_energy = 0.0

        _overwrite_count = 0
        _overwrite_examples = []

        def _collect_merge_result(result):
            nonlocal processed_keys, lowrank_count, compressed_count
            nonlocal total_input_energy, total_merged_energy
            nonlocal _overwrite_count
            if result is None:
                return
            (target_key, is_clip_key, patch, used_mode, prefix, conflict,
             n_loras, is_compressed, inp_norm, mrg_norm, score_stats) = result
            self._progress_update("pass2", prefix, 1.0)
            total_input_energy += inp_norm
            total_merged_energy += mrg_norm

            target_dict = clip_patches if is_clip_key else model_patches
            if target_key in target_dict:
                # Target-key collision: accumulate diffs instead of overwriting
                _overwrite_count += 1
                existing = target_dict[target_key]
                existing_diff = self._expand_patch_to_diff(existing)
                new_diff = self._expand_patch_to_diff(patch)
                accumulated = existing_diff + new_diff
                # Use the smaller dtype to save memory
                store_dt = existing_diff.dtype if existing_diff.dtype != torch.float32 else new_diff.dtype
                if store_dt != torch.float32:
                    accumulated = accumulated.to(store_dt)
                target_dict[target_key] = ("diff", (accumulated,))
                # Fix lowrank_count if existing was a low-rank adapter (now replaced by diff)
                if isinstance(existing, (LoRAAdapter, LoKrAdapter, LoHaAdapter)):
                    lowrank_count -= 1
                if len(_overwrite_examples) < 3:
                    _overwrite_examples.append(f"{'CLIP' if is_clip_key else 'MODEL'} {prefix} -> {target_key}")
            else:
                target_dict[target_key] = patch
                if isinstance(patch, (LoRAAdapter, LoKrAdapter, LoHaAdapter)):
                    lowrank_count += 1
            processed_keys += 1
            if is_compressed:
                compressed_count += 1
            strategy_counts[used_mode] = strategy_counts.get(used_mode, 0) + 1
            prefix_decisions.append((prefix, used_mode, conflict, n_loras))

        if use_gpu:
            group_items = list(target_groups.items())
            for idx, (label_prefix, target_group) in enumerate(group_items):
                self._interrupt_check()
                if _merge_prof is not None:
                    debug_key = target_group.get("target_key")
                    debug_key = debug_key[0] if isinstance(debug_key, tuple) else debug_key
                    logging.info(
                        f"[merge-debug] processing {idx + 1}/{len(group_items)}: "
                        f"{debug_key}  (n_loras={prefix_stats.get(label_prefix, {}).get('n_loras', '?')})")
                result = _merge_one_group(label_prefix, target_group)
                if _compress_sync or (_merge_prof is not None and _prof_cuda):
                    torch.cuda.synchronize(compute_device)
                _collect_merge_result(result)
                self._interrupt_check()
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(4, max(1, len(target_groups))))
            futures = {}
            try:
                for label_prefix, target_group in target_groups.items():
                    self._interrupt_check()
                    futures[executor.submit(
                        _merge_one_group, label_prefix, target_group)] = label_prefix
                pending = set(futures)
                while pending:
                    self._interrupt_check()
                    done, pending = concurrent.futures.wait(
                        pending, timeout=0.05,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        _collect_merge_result(future.result())
            except BaseException:
                self._interrupt_controller.cancel()
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(
                    wait=not self._interrupt_controller.event.is_set(),
                    cancel_futures=True)

        fullrank_count = processed_keys - lowrank_count
        if _overwrite_count > 0:
            logging.info(f"[LoRA Optimizer] {_overwrite_count} target-key collisions resolved "
                         f"(different LoRA key formats targeting the same model weight — diffs accumulated)")
        _pass2_secs = time.time() - t_pass2
        logging.info(f"[LoRA Optimizer]   Model patches: {len(model_patches)}, "
                     f"CLIP patches: {len(clip_patches)} ({_pass2_secs:.1f}s)")
        if _merge_prof:
            profiled = sum(v[1] for v in _merge_prof.values())
            # % is against wall-clock Pass-2 time, and an "other" row accounts
            # for whatever the timed phases don't cover (cache lookups, patch
            # device moves, score-during-merge, result collection) — so the
            # breakdown sums to ~100% and untimed overhead can't hide.
            denom = max(_pass2_secs, profiled, 1e-9)
            logging.info("[LoRA Optimizer]   Merge profile (phase: total_s, count, avg_ms, %):")
            for phase, (cnt, secs) in sorted(_merge_prof.items(), key=lambda kv: -kv[1][1]):
                logging.info(f"[LoRA Optimizer]     {phase:<22} {secs:6.2f}s  "
                             f"n={cnt:<5} {secs / cnt * 1000:6.1f}ms  {secs / denom * 100:4.1f}%")
            _other = max(0.0, _pass2_secs - profiled)
            if _other > 0.05:
                logging.info(f"[LoRA Optimizer]     {'other (cache/move/score)':<22} "
                             f"{_other:6.2f}s  {'':<7} {'':>6}  {_other / denom * 100:4.1f}%")
            # Auto-flag: merge *math* (slerp/consensus/ties/weighted_*) is allowed
            # to dominate — that's the real work. OVERHEAD phases dominating is a
            # pathology (this is how the diff-cache disk spill would have announced
            # itself). Warn with a targeted hint so issues surface without a manual
            # read of the numbers.
            _OVERHEAD_HINTS = {
                "diff_prep": "diff reconstruction — low-rank diffs should recompute in ~ms; "
                             "check diff_cache_mode (disk spill?) and that the temp dir is on a fast local disk",
                "compress": "compression SVD — patch_compression rebuilds low-rank every candidate; "
                            "try patch_compression='disabled' during the sweep",
                "linear_fast": "fast linear path — should be ~ms/item; unexpectedly slow, check input LoRA sizes",
                "other (cache/move/score)": "untimed overhead (cache I/O, patch device moves, score-during-merge) "
                                            "— check vram_budget / cache_patches / diff_cache_mode",
            }
            if _pass2_secs >= 1.0:
                _rows = list(_merge_prof.items()) + (
                    [("other (cache/move/score)", (1, _other))] if _other > 0.05 else [])
                for phase, (cnt, secs) in _rows:
                    pct = secs / denom * 100
                    if phase in _OVERHEAD_HINTS and pct >= 40.0:
                        logging.warning(
                            f"[LoRA Optimizer]   ⚠ merge overhead: '{phase}' = {pct:.0f}% of Pass 2 "
                            f"({secs:.1f}s) — {_OVERHEAD_HINTS[phase]}")
        if lowrank_count > 0:
            logging.info(f"[LoRA Optimizer]   Low-rank patches: {lowrank_count} "
                         f"(full-rank: {fullrank_count}) — "
                         f"~{lowrank_count}/{processed_keys} keys use minimal RAM")
        if optimization_mode == "per_prefix":
            logging.info(f"[LoRA Optimizer]   Per-prefix strategies: "
                         f"{strategy_counts.get('weighted_sum', 0)} sum, "
                         f"{strategy_counts.get('slerp', 0)} slerp, "
                         f"{strategy_counts.get('weighted_average', 0)} avg, "
                         f"{strategy_counts.get('consensus', 0)} cons, "
                         f"{strategy_counts.get('ties', 0)} ties")
        spars_skipped = getattr(self, '_sparsification_skipped', 0)
        if spars_skipped > 0:
            logging.info(f"[LoRA Optimizer]   Conflict-aware sparsification skipped for "
                         f"{spars_skipped} groups (base-rate noise from orthogonal LoRAs)")
            self._sparsification_skipped = 0
        if compressed_count > 0:
            passthrough_count = lowrank_count - compressed_count
            logging.info(f"[LoRA Optimizer]   SVD-compressed: {compressed_count} patches "
                         f"(rank {compress_rank}), passthrough: {passthrough_count}, "
                         f"full-rank: {fullrank_count}")
        if vram_budget_bytes > 0:
            gpu_count = 0
            cpu_count = 0
            for p in list(model_patches.values()) + list(clip_patches.values()):
                data = p.weights if hasattr(p, 'weights') else p
                if isinstance(data, (tuple, list)):
                    for item in data:
                        if isinstance(item, torch.Tensor):
                            gpu_count += 1 if item.is_cuda else 0
                            cpu_count += 1 if not item.is_cuda else 0
                            break
                        elif isinstance(item, (tuple, list)):
                            for sub in item:
                                if isinstance(sub, torch.Tensor):
                                    gpu_count += 1 if sub.is_cuda else 0
                                    cpu_count += 1 if not sub.is_cuda else 0
                                    break
                            break
            logging.info(f"[LoRA Optimizer]   VRAM budget: {gpu_count} patches on GPU "
                         f"({gpu_patch_bytes // (1024**2)}MB), {cpu_count} on CPU")

        suggested_max_strength = None
        if total_merged_energy > 0 and total_input_energy > 0:
            norm_ratio = total_merged_energy / total_input_energy
            suggested_max_strength = max(1.0, min(1.0 / norm_ratio, arch_preset["suggested_max_strength_cap"]))

        prefix_stats.clear()
        self.loaded_loras.clear()
        if use_gpu:
            torch.cuda.empty_cache()

        if getattr(self, '_detected_arch', None) == 'zimage':
            if len(model_patches) > 0:
                model_patches = self._refuse_zimage_patches(model_patches)
                logging.info(f"[LoRA Optimizer] Re-fused Z-Image QKV patches ({len(model_patches)} model patches)")

        # Build reverse key map: target_key → canonical prefix metadata
        # (used by SaveMergedLoRA to reconstruct standard LoRA key names)
        reverse_key_map = {}
        for label_prefix, target_group in target_groups.items():
            target_key = target_group["target_key"]
            tkey = target_key[0] if isinstance(target_key, tuple) else target_key
            entry = {
                "canonical_prefix": label_prefix,
                "aliases": list(target_group["aliases"]),
            }
            reverse_key_map[target_key] = entry
            reverse_key_map[tkey] = entry

        # Apply patches
        new_model = model
        new_clip = clip

        auto_output_strength = False
        if output_strength < 0 and suggested_max_strength is not None:
            output_strength = suggested_max_strength
            auto_output_strength = True
            logging.info(f"[LoRA Optimizer] Auto output_strength: {output_strength:.2f} (suggested max)")
        elif output_strength < 0:
            output_strength = 1.0
            logging.info("[LoRA Optimizer] Auto output_strength: no suggestion available, using 1.0")

        all_explicit_clip = all(item["clip_strength"] is not None for item in active_loras)
        force_clip_multiplier = bool(
            getattr(self, "_force_explicit_clip_multiplier", False))
        if all_explicit_clip and not force_clip_multiplier:
            clip_strength_out = output_strength * clip_auto_scale
        else:
            clip_strength_out = output_strength * clip_strength_multiplier * clip_auto_scale

        self._interrupt_check()
        if model is not None and len(model_patches) > 0:
            new_model = model.clone()
            new_model.add_patches(model_patches, output_strength)
            self._update_model_size(new_model, model_patches)

        if clip is not None and len(clip_patches) > 0:
            new_clip = clip.clone()
            new_clip.add_patches(clip_patches, clip_strength_out)
            self._update_model_size(new_clip, clip_patches)

        merge_summary = {
            "keys_processed": processed_keys,
            "model_patches": len(model_patches),
            "clip_patches": len(clip_patches),
            "skipped_keys": skipped_keys,
            "output_strength": output_strength,
            "clip_strength": clip_strength_out,
            "suggested_max_strength": suggested_max_strength,
            "auto_output_strength": auto_output_strength,
        }

        report = self._build_report(
            lora_stats, pairwise_conflicts, collection_stats,
            mode, density, sign_method, reasoning, merge_summary,
            auto_strength_info=auto_strength_info,
            strategy_counts=strategy_counts if optimization_mode == "per_prefix" else None,
            optimization_mode=optimization_mode,
            prefix_decisions=prefix_decisions if optimization_mode == "per_prefix" else None,
            detected_arch=getattr(self, '_detected_arch', None),
            normalize_keys=normalize_keys,
            sparsification=sparsification,
            sparsification_density=sparsification_density,
            dare_dampening=dare_dampening,
            merge_refinement=merge_refinement,
            compatibility_warnings=compatibility_warnings,
            strategy_set=strategy_set,
            architecture_preset=preset_key,
        )
        tile_rows = self._execution_stats.get("tile_rows", [])
        tile_text = (f"{min(tile_rows)}-{max(tile_rows)} rows"
                     if tile_rows else "n/a")
        peak_vram = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                     if torch.cuda.is_available() else 0.0)
        report += (
            "\n\nExecution Plan\n" + "-" * 40 + "\n"
            f"  Full GPU targets: {len(self._execution_stats.get('full_gpu', ()))}\n"
            f"  Tiled GPU targets: {len(self._execution_stats.get('tiled_gpu', ()))}\n"
            f"  CPU targets: {len(self._execution_stats.get('cpu', ()))}\n"
            f"  Tiled row range: {tile_text}\n"
            f"  Peak GPU allocated: {peak_vram:.1f} MiB\n"
        )

        # Derive per-prefix decision map from the decision log (last-wins if a
        # prefix somehow appears twice — shouldn't, but defensive).
        per_prefix_decisions = {
            prefix: mode
            for prefix, mode, _conflict, _n in prefix_decisions
        }

        # Bundle LORA_DATA for optional downstream saving
        lora_data = {
            "model_patches": model_patches,
            "clip_patches": clip_patches,
            "key_map": reverse_key_map,
            "output_strength": output_strength,
            "clip_strength": clip_strength_out,
            "suggested_max_strength": suggested_max_strength,
            "sum_rank": compress_rank if compress_rank > 0 else 128,
            "per_prefix_decisions": per_prefix_decisions,
            "merge_metadata": {
                "source_loras": [{"name": item["name"], "strength": item["strength"]} for item in active_loras],
                "mode": mode,
                "optimization_mode": optimization_mode,
                "architecture": getattr(self, '_detected_arch', None) or 'unknown',
                "architecture_preset": preset_key,
                "auto_strength": auto_strength,
                "sparsification": sparsification,
                "sparsification_density": sparsification_density,
                "merge_refinement": merge_refinement,
                "strategy_set": strategy_set,
                "bake_strength_output": output_strength,
                "bake_strength_clip": clip_strength_out,
            },
        }

        # Cache only atomically completed patches. The disk entry is content-
        # addressed and survives ComfyUI restarts; the single RAM entry keeps
        # repeated executions in the current process instant.
        self._interrupt_check()
        if persistent_cache_key is not None:
            validated_cache_key = self._persistent_cache_key(
                lora_stack, model, clip, config_cache_key,
                self._interrupt_controller, tame_layers=tame_layers)
            if validated_cache_key != persistent_cache_key:
                logging.warning(
                    "[LoRA Optimizer Cache] Source files changed during the merge; "
                    "the persistent result was not cached")
                persistent_cache_key = None
            else:
                try:
                    self._persistent_cache.save(
                        persistent_cache_key, model_patches, clip_patches,
                        report, lora_data, self._interrupt_controller)
                except (OSError, PersistentCacheUnsupported) as error:
                    logging.warning(
                        "[LoRA Optimizer Cache] Persistent cache write skipped: %s",
                        error)
        self._interrupt_check()
        if cache_patches == "enabled":
            self._merge_cache = {runtime_cache_key: (
                model_patches, clip_patches, report,
                clip_strength_out, lora_data)}
        else:
            self._merge_cache = {}
            logging.info("[LoRA Optimizer] Memory patch cache disabled")

        self._progress_finish()
        logging.info(f"[LoRA Optimizer] Done! {processed_keys} keys processed ({time.time() - t_start:.1f}s total)")

        return (new_model, new_clip, report, None, lora_data)

class LoRAOptimizerSettings:
    """All advanced controls for the single public optimizer node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_strength": (["enabled", "disabled"], {"default": "enabled", "tooltip": "Automatically reduces excessive combined LoRA strength."}),
                "optimization_mode": (["per_prefix", "global", "additive"], {"default": "per_prefix", "tooltip": "per_prefix automatically chooses the best merge strategy for each model area."}),
                "merge_refinement": (["none", "refine", "full"], {"default": "none", "tooltip": "Optional interference cleanup after merging."}),
                "sparsification": (["disabled", "dare", "della", "dare_conflict", "della_conflict"], {"default": "disabled", "tooltip": "Optional DARE/DELLA sparsification before merging."}),
                "sparsification_density": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.05, "tooltip": "Fraction of weights retained when sparsification is enabled."}),
                "dare_dampening": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Reduces DARE/DELLA compensation after pruning."}),
                "patch_compression": (["smart", "aggressive", "disabled"], {"default": "smart", "tooltip": "smart preserves exact low-rank results; aggressive also compresses nonlinear results."}),
                "svd_device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device used for SVD compression."}),
                "free_vram_between_passes": (["disabled", "enabled"], {"default": "disabled", "tooltip": "Release unused CUDA cache between analysis and merge passes."}),
                "strategy_set": (["full", "no_slerp", "basic"], {"default": "full", "tooltip": "Limits which merge strategies automatic mode may select."}),
                "normalize_keys": (["disabled", "enabled"], {"default": "enabled", "tooltip": "Normalize LoRA keys produced by different training tools."}),
                "architecture_preset": (["auto", "sd_unet", "dit", "acestep_dit", "llm"], {"default": "auto", "tooltip": "Model-family hint used by automatic strategy selection."}),
                "auto_strength_floor_mode": (["auto", "manual"], {"default": "auto", "tooltip": "Use the architecture-aware floor or the manual value below."}),
                "auto_strength_floor": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Lowest multiplier auto-strength may apply in manual mode."}),
                "decision_smoothing": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Encourages similar layers to use consistent merge strategies."}),
                "smooth_slerp_gate": ("BOOLEAN", {"default": False, "tooltip": "Use a smooth decision boundary when selecting SLERP."}),
                "vram_budget": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Maximum fraction of VRAM used to retain completed patches; 0 keeps them in RAM."}),
                "cache_patches": (["enabled", "disabled"], {"default": "enabled", "tooltip": "Keep one completed merge in memory for instant reuse during this ComfyUI session."}),
                "persistent_cache": (["enabled", "disabled"], {"default": "enabled", "tooltip": "Save completed merge patches under the ComfyUI user directory so they can be reused after a restart. Disable to prevent local cache files from being read or written."}),
                "star_eta": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 100.0, "step": 5.0, "tooltip": "Per-LoRA STAR spectral cleaning; 100 disables it."}),
                "tame_layers": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Strength of base-weight-aware magnitude taming; 0 disables it."}),
                "tame_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 2.0, "step": 0.05, "tooltip": "Delta-to-base norm threshold used by magnitude taming."}),
            },
        }

    RETURN_TYPES = ("OPTIMIZER_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build_settings"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Complete merge, memory, persistent-cache, and performance settings for LoRA Optimizer."

    def build_settings(self, auto_strength, optimization_mode, merge_refinement,
                       sparsification, sparsification_density, dare_dampening,
                       patch_compression, svd_device, free_vram_between_passes,
                       strategy_set, normalize_keys, architecture_preset,
                       auto_strength_floor_mode, auto_strength_floor,
                       decision_smoothing, smooth_slerp_gate, vram_budget,
                       cache_patches, persistent_cache, star_eta, tame_layers,
                       tame_threshold):
        resolved_floor = (max(0.0, min(1.0, auto_strength_floor))
                          if auto_strength_floor_mode == "manual" else -1.0)
        return ({
            "mode": "advanced",
            "auto_strength": auto_strength,
            "optimization_mode": optimization_mode,
            "merge_refinement": merge_refinement,
            "sparsification": sparsification,
            "sparsification_density": sparsification_density,
            "dare_dampening": dare_dampening,
            "patch_compression": patch_compression,
            "svd_device": svd_device,
            "free_vram_between_passes": free_vram_between_passes,
            "strategy_set": strategy_set,
            "normalize_keys": normalize_keys,
            "architecture_preset": architecture_preset,
            "auto_strength_floor": resolved_floor,
            "decision_smoothing": decision_smoothing,
            "smooth_slerp_gate": smooth_slerp_gate,
            "vram_budget": vram_budget,
            "cache_patches": cache_patches,
            "persistent_cache": persistent_cache,
            "star_eta": star_eta,
            "tame_layers": tame_layers,
            "tame_threshold": tame_threshold,
        },)


class LoRAOptimizerSimple(_LoRAOptimizerEngine):
    """The single public LoRA merge node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base model that receives the optimized LoRA patches."}),
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRA stack supplied by LoRA Manager or another compatible stack provider."}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 10.0, "step": 0.05, "tooltip": "Final merged strength; -1 lets the optimizer choose a safe value."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional text encoder for LoRAs that contain CLIP patches."}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Strength multiplier applied to merged CLIP patches."}),
                "settings": ("OPTIMIZER_SETTINGS", {"tooltip": "Optional LoRA Optimizer Settings; recommended defaults are used when disconnected."}),
            },
        }

    FUNCTION = "execute_simple"
    DESCRIPTION = "Analyze, optimize, merge, cache, and apply a LoRA Manager stack."

    _SIMPLE_DEFAULTS = dict(
        auto_strength="enabled",
        auto_strength_floor=-1.0,
        free_vram_between_passes="disabled",
        vram_budget=0.0,
        optimization_mode="per_prefix",
        cache_patches="enabled",
        persistent_cache="enabled",
        patch_compression="smart",
        svd_device="gpu",
        normalize_keys="enabled",
        sparsification="disabled",
        sparsification_density=0.7,
        dare_dampening=0.0,
        merge_refinement="none",
        strategy_set="full",
        architecture_preset="auto",
        decision_smoothing=0.25,
        smooth_slerp_gate=False,
        star_eta=100.0,
        tame_layers=0.0,
        tame_threshold=0.3,
    )

    def execute_simple(self, model, lora_stack, output_strength,
                       clip=None, clip_strength_multiplier=1.0,
                       settings=None):
        kwargs = (self._advanced_merge_kwargs(settings)
                  if settings is not None else dict(self._SIMPLE_DEFAULTS))
        return super().optimize_merge(
            model, lora_stack, output_strength,
            clip=clip, clip_strength_multiplier=clip_strength_multiplier,
            **kwargs,
        )

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength,
                   clip=None, clip_strength_multiplier=1.0,
                   settings=None):
        base = _LoRAOptimizerEngine.IS_CHANGED(
            model, lora_stack, output_strength,
            clip=clip, clip_strength_multiplier=clip_strength_multiplier,
            **cls._SIMPLE_DEFAULTS,
        )
        if settings is not None:
            settings_hash = hashlib.md5(
                json.dumps(settings, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
            return f"{base}|settings={settings_hash}"
        return base


class SaveMergedLoRA:
    """
    Saves merged LoRA patches from LoRA Optimizer as a standalone .safetensors
    file that can be loaded by any standard LoRA loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_folders = folder_paths.get_folder_paths("loras")
        folder_choices = lora_folders if lora_folders else [os.path.join(folder_paths.models_dir, "loras")]
        return {
            "required": {
                "lora_data": ("LORA_DATA", {"tooltip": "Connect the lora_data output from LoRA Optimizer here."}),
                "save_folder": (folder_choices, {"tooltip": "Which loras folder to save into. Lists configured lora paths from ComfyUI and extra_model_paths.yaml."}),
                "filename": ("STRING", {"default": "merged_lora", "tooltip": "Name for the saved file. Subdirectories are allowed (e.g. 'merged/my_lora'). Extension .safetensors is added automatically."}),
                "save_rank": ("INT", {
                    "default": 0, "min": 0, "max": 2048, "step": 4,
                    "tooltip": "0 = auto: uses the sum of all input LoRA ranks (e.g. 3 rank-32 LoRAs → rank 96). Non-zero = compress all layers to exactly this rank via SVD. Use a non-zero value to reduce output file size — set it to match your largest input LoRA's rank or lower. Higher values = more accurate but larger file."
                }),
                "bake_strength": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, the saved LoRA reproduces your exact merge when loaded at strength 1.0. When disabled, strengths are not baked in — you'll need to set the strength manually when loading."
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Example prompt or trigger words to embed in the file metadata. Useful for sharing — some UIs display this automatically."
                }),
                "description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional description or notes about this merged LoRA. Stored in file metadata."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_lora"
    CATEGORY = "LoRA Optimizer"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves merged LoRA data as a standalone .safetensors file that can be loaded by any standard LoRA loader."

    def save_lora(self, lora_data, save_folder, filename, save_rank=0, bake_strength=True, prompt="", description=""):
        _throw_if_processing_interrupted()
        if lora_data is None:
            logging.warning("[Save Merged LoRA] No lora_data received (optimizer may have returned early). Nothing to save.")
            return ("",)

        save_path = _resolve_safe_output_path(save_folder, filename, ".safetensors", "Save Merged LoRA")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model_patches = lora_data["model_patches"]
        clip_patches = lora_data["clip_patches"]
        key_map = lora_data["key_map"]
        output_strength = lora_data["output_strength"]
        clip_strength = lora_data["clip_strength"]

        auto_rank = save_rank == 0

        # Auto mode: adapt rank for any full-rank diffs that need compression.
        if auto_rank:
            has_diffs = any(
                isinstance(patch, tuple) and patch[0] == "diff"
                for patch in list(model_patches.values()) + list(clip_patches.values())
            )
            if has_diffs:
                initial_rank = lora_data.get("sum_rank", 128)
                fallback_rank = _LoRAOptimizerEngine._estimate_save_rank(initial_rank, model_patches, clip_patches)
                logging.info(f"[Save Merged LoRA] Auto rank for diffs: {fallback_rank} "
                             f"(initial estimate {initial_rank}, adapted from sample diffs)")
            else:
                fallback_rank = lora_data.get("sum_rank", 128)
                n_loras = len(lora_data.get("merge_metadata", {}).get("source_loras", []))
                logging.info(
                    f"[Save Merged LoRA] Auto rank: {fallback_rank} (sum of {n_loras} input LoRA ranks). "
                    f"Set save_rank to a fixed value to compress below this."
                )

        save_dtype = None
        for patch in list(model_patches.values()) + list(clip_patches.values()):
            if isinstance(patch, tuple) and patch[0] == "diff":
                dtype = patch[1][0].dtype
            elif hasattr(patch, "weights") and patch.weights[0] is not None:
                dtype = patch.weights[0].dtype
            else:
                continue
            if dtype not in (torch.float32, torch.float64):
                save_dtype = dtype
                break
        if save_dtype is None:
            save_dtype = torch.float16
        logging.info(f"[Save Merged LoRA] Output dtype: {save_dtype}")

        # SVD decomposition is the bottleneck here — especially with a high auto
        # rank over many diffs. The diffs live on CPU, so run the SVD on the GPU
        # when one is available (_compress_to_lowrank moves each diff to the device).
        svd_device = _LoRAOptimizerEngine._get_compute_device()
        if svd_device.type != "cpu":
            logging.info(f"[Save Merged LoRA] SVD device: {svd_device}")

        state_dict = {}

        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            for target_key, patch in patches.items():
                _throw_if_processing_interrupted()
                tkey = target_key[0] if isinstance(target_key, tuple) else target_key
                key_info = key_map.get(target_key)
                if key_info is None:
                    key_info = key_map.get(tkey, tkey)
                if isinstance(key_info, dict):
                    lora_prefix = key_info.get("canonical_prefix", tkey)
                else:
                    lora_prefix = key_info

                if isinstance(patch, (LoKrAdapter, LoHaAdapter)):
                    diff_tensor = _LoRAMergeBase._expand_patch_to_diff(patch)
                    rank = fallback_rank if auto_rank else save_rank
                    compressed = _LoRAOptimizerEngine._compress_to_lowrank(diff_tensor, rank, svd_device=svd_device, output_dtype=save_dtype)
                    if compressed is None:
                        logging.warning(f"[Save Merged LoRA] SVD failed for {lora_prefix}; skipping this layer.")
                        continue
                    mat_up, mat_down, alpha, mid, _, _ = compressed.weights
                    alpha = float(alpha)
                elif isinstance(patch, LoRAAdapter):
                    mat_up, mat_down, alpha, mid, _, _ = patch.weights
                    alpha = float(alpha) if alpha is not None else float(mat_down.shape[0])
                    current_rank = int(mat_down.shape[0])
                    target_rank = fallback_rank if auto_rank else save_rank
                    # The fast linear path (_build_exact_linear_patch) produces LoRAAdapters
                    # with rank = sum of all input ranks, bypassing patch_compression entirely.
                    # Compress here when the patch rank exceeds the requested target rank.
                    # Skip LoCon (mid != None): mid tensor requires special reshape handling.
                    if target_rank > 0 and current_rank > target_rank and mid is None:
                        diff = _LoRAMergeBase._expand_patch_to_diff(patch)
                        compressed = _LoRAOptimizerEngine._compress_to_lowrank(diff, target_rank, svd_device=svd_device, output_dtype=save_dtype)
                        del diff
                        if compressed is None:
                            logging.warning(f"[Save Merged LoRA] SVD failed for {lora_prefix}; saving at full patch rank.")
                        else:
                            mat_up, mat_down, alpha, mid, _, _ = compressed.weights
                            alpha = float(alpha)
                elif isinstance(patch, tuple) and len(patch) == 2 and patch[0] == "diff":
                    diff_tensor = patch[1][0]
                    # All-zero diff (untrained layer) merges to nothing — skip it
                    # rather than waste an SVD and write a dead layer to the file.
                    if diff_tensor.abs().amax().item() == 0:
                        continue
                    rank = fallback_rank if auto_rank else save_rank
                    compressed = _LoRAOptimizerEngine._compress_to_lowrank(diff_tensor, rank, svd_device=svd_device, output_dtype=save_dtype)
                    if compressed is None:
                        logging.warning(f"[Save Merged LoRA] SVD failed for {lora_prefix}; skipping this layer.")
                        continue
                    mat_up, mat_down, alpha, mid, _, _ = compressed.weights
                    alpha = float(alpha)
                else:
                    logging.warning(f"[Save Merged LoRA] Skipping unknown patch type for {lora_prefix}: {type(patch)}")
                    continue

                if bake_strength:
                    strength = clip_strength if is_clip else output_strength
                    alpha *= strength

                state_dict[f"{lora_prefix}.lora_up.weight"] = mat_up.to(save_dtype).cpu().contiguous()
                state_dict[f"{lora_prefix}.lora_down.weight"] = mat_down.to(save_dtype).cpu().contiguous()
                state_dict[f"{lora_prefix}.alpha"] = torch.tensor(alpha)

        unmapped_keys = []
        nan_keys = []
        zero_keys = []
        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            for target_key, _patch in patches.items():
                direct = key_map.get(target_key)
                if direct is None:
                    tkey = target_key[0] if isinstance(target_key, tuple) else target_key
                    fallback = key_map.get(tkey)
                    label = f"{'CLIP' if is_clip else 'MODEL'} {tkey}"
                    if fallback is None:
                        unmapped_keys.append(f"{label} (using raw key as prefix)")
                    else:
                        unmapped_keys.append(f"{label} (fallback to base key)")

        for state_key, tensor in state_dict.items():
            if state_key.endswith(".alpha"):
                continue
            if torch.isnan(tensor).any():
                nan_keys.append(state_key)
            if tensor.abs().max().item() == 0:
                zero_keys.append(state_key)

        if unmapped_keys:
            logging.warning(f"[Save Merged LoRA] {len(unmapped_keys)} keys fell through to fallback mapping:")
            for item in unmapped_keys[:5]:
                logging.warning(f"  {item}")
        if nan_keys:
            logging.error(f"[Save Merged LoRA] {len(nan_keys)} tensors contain NaN")
            for item in nan_keys[:5]:
                logging.error(f"  {item}")
        if zero_keys:
            logging.warning(f"[Save Merged LoRA] {len(zero_keys)} tensors are all zeros")

        prefixes = sorted(set(key.rsplit(".lora_", 1)[0] for key in state_dict if ".lora_" in key))
        if prefixes:
            logging.info(f"[Save Merged LoRA] Sample prefixes: {prefixes[:3]} ... ({len(prefixes)} total)")

        # Reconstruction-error check is a diagnostic — sample a few diffs rather
        # than reconstructing all of them on CPU (that's minutes on a large model
        # like LTX-2, and drags vram_budget GPU patches back to CPU). 32 is plenty.
        svd_errors = []
        _MAX_SVD_CHECKS = 32
        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            if len(svd_errors) >= _MAX_SVD_CHECKS:
                break
            for target_key, patch in patches.items():
                if len(svd_errors) >= _MAX_SVD_CHECKS:
                    break
                if not (isinstance(patch, tuple) and patch[0] == "diff"):
                    continue
                key_info = key_map.get(target_key)
                lora_prefix = (key_info.get("canonical_prefix")
                               if isinstance(key_info, dict) else key_info)
                if not lora_prefix:
                    continue
                up_key = f"{lora_prefix}.lora_up.weight"
                if up_key not in state_dict:  # e.g. an all-zero layer we skipped
                    continue
                original_diff = patch[1][0].float()
                strength = clip_strength if is_clip else output_strength
                reference = original_diff * strength if bake_strength else original_diff
                original_norm = reference.norm().item()
                if original_norm <= 0:
                    continue
                # Diagnostic only — never let it abort the save (device/shape, etc.).
                try:
                    saved_up = state_dict[up_key].float()
                    saved_down = state_dict[f"{lora_prefix}.lora_down.weight"].float()
                    rank = saved_down.shape[0]
                    scale = state_dict[f"{lora_prefix}.alpha"].item() / rank
                    reconstructed = torch.mm(saved_up, saved_down) * scale
                    ref = reference.to(reconstructed.device)
                    svd_errors.append((reconstructed - ref).norm().item() / original_norm)
                except Exception:
                    pass
        if svd_errors:
            avg_error = sum(svd_errors) / len(svd_errors)
            max_error = max(svd_errors)
            logging.info(f"[Save Merged LoRA] SVD reconstruction error: "
                         f"avg={avg_error:.4f}, max={max_error:.4f} "
                         f"({len(svd_errors)} diffs checked)")
            # High error = the merged diffs are higher-rank than the save rank can
            # capture (common for TIES / refine merges). A larger save_rank retains
            # more detail. Only nudge in auto mode; a fixed save_rank is the user's call.
            eff_rank = fallback_rank if auto_rank else save_rank
            if avg_error > 0.05 or max_error > 0.20:
                suggested = min(eff_rank * 2, 1024) if eff_rank > 0 else 256
                logging.warning(
                    f"[Save Merged LoRA] Lossy compression at rank {eff_rank} "
                    f"(avg {avg_error*100:.1f}%, max {max_error*100:.1f}% reconstruction "
                    f"error) — the merge is higher-rank than this. For more fidelity set "
                    f"save_rank to ~{suggested} (larger file); rank 0 = auto.")

        # Build safetensors metadata header
        metadata = {"tool": "ComfyUI-ZImage-LoRA-Merger"}
        merge_meta = lora_data.get("merge_metadata", {})
        if merge_meta:
            source_loras = merge_meta.get("source_loras", [])
            if source_loras:
                metadata["source_loras"] = ", ".join(
                    f"{s['name']} @ {s['strength']}" for s in source_loras
                )
            for key in ("mode", "optimization_mode", "architecture", "architecture_preset",
                        "auto_strength", "sparsification", "merge_refinement", "strategy_set"):
                val = merge_meta.get(key)
                if val is not None:
                    metadata[f"merge_{key}"] = str(val)
            if merge_meta.get("sparsification_density") is not None:
                metadata["merge_sparsification_density"] = str(merge_meta["sparsification_density"])
            metadata["merge_output_strength"] = str(merge_meta.get("bake_strength_output", output_strength))
            metadata["merge_clip_strength"] = str(merge_meta.get("bake_strength_clip", clip_strength))
            metadata["merge_bake_strength"] = str(bake_strength)
        if prompt.strip():
            metadata["prompt"] = prompt.strip()
        if description.strip():
            metadata["description"] = description.strip()

        _throw_if_processing_interrupted()
        save_file(state_dict, save_path, metadata=metadata)
        logging.info(f"[Save Merged LoRA] Saved {len(state_dict) // 3} LoRA keys to {save_path}")

        return (save_path,)


NODE_CLASS_MAPPINGS = {
    "LoRAOptimizerSimple": LoRAOptimizerSimple,
    "LoRAOptimizerSettings": LoRAOptimizerSettings,
    "SaveMergedLoRA": SaveMergedLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAOptimizerSimple": "LoRA Optimizer",
    "LoRAOptimizerSettings": "LoRA Optimizer Settings",
    "SaveMergedLoRA": "Save Merged LoRA",
}
