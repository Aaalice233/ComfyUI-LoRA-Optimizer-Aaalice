import contextlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _install_stubs():
    tmpdir = tempfile.gettempdir()

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = tmpdir
    folder_paths.add_model_folder_path = lambda *args, **kwargs: None
    folder_paths.get_temp_directory = lambda: tmpdir
    folder_paths.get_user_directory = lambda: tmpdir
    folder_paths.get_folder_paths = lambda _kind: [tmpdir]
    folder_paths.get_filename_list = lambda _kind: []
    folder_paths.get_full_path_or_raise = lambda _kind, name: name
    folder_paths.get_full_path = lambda _kind, name: name
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")

    def get_attr(obj, path):
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.value = 0

        def update(self, amount):
            self.value += amount

    utils.get_attr = get_attr
    utils.load_torch_file = lambda _path, safe_load=True: {}
    utils.ProgressBar = ProgressBar

    sd = types.ModuleType("comfy.sd")
    sd.load_lora_for_models = lambda model, clip, lora_dict, model_strength, clip_strength: (model, clip)

    lora = types.ModuleType("comfy.lora")
    lora.model_lora_keys_unet = lambda model, mapping: {}
    lora.model_lora_keys_clip = lambda clip, mapping: {}

    model_management = types.ModuleType("comfy.model_management")
    model_management.get_free_memory = lambda _device: 1 << 60

    weight_adapter = types.ModuleType("comfy.weight_adapter")
    weight_adapter_lora = types.ModuleType("comfy.weight_adapter.lora")
    weight_adapter_lokr = types.ModuleType("comfy.weight_adapter.lokr")
    weight_adapter_loha = types.ModuleType("comfy.weight_adapter.loha")

    class LoRAAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights

    class LoKrAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights

    class LoHaAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights

    weight_adapter_lora.LoRAAdapter = LoRAAdapter
    weight_adapter_lokr.LoKrAdapter = LoKrAdapter
    weight_adapter_loha.LoHaAdapter = LoHaAdapter

    comfy.utils = utils
    comfy.sd = sd
    comfy.lora = lora
    comfy.model_management = model_management

    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils
    sys.modules["comfy.sd"] = sd
    sys.modules["comfy.lora"] = lora
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.weight_adapter"] = weight_adapter
    sys.modules["comfy.weight_adapter.lora"] = weight_adapter_lora
    sys.modules["comfy.weight_adapter.lokr"] = weight_adapter_lokr
    sys.modules["comfy.weight_adapter.loha"] = weight_adapter_loha

    try:
        import safetensors
        import safetensors.torch as safetensors_torch
    except ModuleNotFoundError:
        safetensors = types.ModuleType("safetensors")
        safetensors.safe_open = mock.MagicMock()
        safetensors_torch = types.ModuleType("safetensors.torch")
        safetensors_torch.save_file = lambda state_dict, path, metadata=None: None
        safetensors.torch = safetensors_torch
        sys.modules["safetensors"] = safetensors
        sys.modules["safetensors.torch"] = safetensors_torch


def _load_module():
    """Load lora_optimizer module with stubs in place."""
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "lora_optimizer",
        os.path.join(os.path.dirname(__file__), "..", "lora_optimizer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if torch is not None:
    _install_stubs()
    MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lora_optimizer.py")
    SPEC = importlib.util.spec_from_file_location("lora_optimizer_under_test", MODULE_PATH)
    lora_optimizer = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(lora_optimizer)
    lora_optimizer.LoRAOptimizer = lora_optimizer._LoRAOptimizerEngine
    # Register under the plain name too so mock.patch("lora_optimizer.X")
    # modifies the same module instance used by these tests.
    sys.modules["lora_optimizer"] = lora_optimizer
else:
    lora_optimizer = None


def _make_model():
    layer = types.SimpleNamespace(weight=torch.zeros(1, 1))
    return types.SimpleNamespace(model=types.SimpleNamespace(layer=layer))


def _make_lora_entry(prefix_to_value, strength=1.0, clip_strength=None, key_filter="all", conflict_mode="all", name="demo"):
    lora = {}
    for prefix, value in prefix_to_value.items():
        lora[f"{prefix}.lora_up.weight"] = torch.tensor([[float(value)]], dtype=torch.float32)
        lora[f"{prefix}.lora_down.weight"] = torch.tensor([[1.0]], dtype=torch.float32)
    return {
        "name": name,
        "lora": lora,
        "strength": strength,
        "clip_strength": clip_strength,
        "key_filter": key_filter,
        "conflict_mode": conflict_mode,
    }


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LoRAOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.optimizer = lora_optimizer.LoRAOptimizer()
        self.model = _make_model()

    def test_oversized_group_falls_back_to_cpu_before_dense_expansion(self):
        device = torch.device("cuda")
        shape = (1024, 1024)
        n_diffs = 9

        with mock.patch.object(
                lora_optimizer.comfy.model_management, "get_free_memory",
                return_value=16 * 1024 * 1024):
            selected = self.optimizer._select_group_compute_device(
                device, shape, n_diffs)

        self.assertEqual(selected.type, "cpu")

    def test_group_stays_on_gpu_when_estimated_peak_fits(self):
        device = torch.device("cuda")

        with mock.patch.object(
                lora_optimizer.comfy.model_management, "get_free_memory",
                return_value=1024 * 1024 * 1024):
            selected = self.optimizer._select_group_compute_device(
                device, (1024, 1024), 9)

        self.assertEqual(selected.type, "cuda")

    def test_lora_format_cache_avoids_repeated_detection(self):
        """After detecting a LoRA's format once, subsequent prefixes should reuse it."""
        optimizer = lora_optimizer.LoRAOptimizer()
        lora_dict = {
            "unet.a.lora_B.weight": torch.tensor([[1.0]], dtype=torch.float32),
            "unet.a.lora_A.weight": torch.tensor([[1.0]], dtype=torch.float32),
            "unet.b.lora_B.weight": torch.tensor([[2.0]], dtype=torch.float32),
            "unet.b.lora_A.weight": torch.tensor([[1.0]], dtype=torch.float32),
        }
        result1 = optimizer._get_lora_key_info(lora_dict, "unet.a")
        self.assertIsNotNone(result1)
        self.assertIn(id(lora_dict), optimizer._lora_format_cache)
        result2 = optimizer._get_lora_key_info(lora_dict, "unet.b")
        self.assertIsNotNone(result2)

    def test_target_groups_merge_aliases_for_same_target(self):
        groups = self.optimizer._build_target_groups(
            ["alias_a", "alias_b", "other"],
            {"alias_a": "layer.weight", "alias_b": "layer.weight", "other": "other.weight"},
            {},
        )

        self.assertEqual(set(groups.keys()), {"alias_a", "other"})
        self.assertEqual(groups["alias_a"]["aliases"], ["alias_a", "alias_b"])

    def test_group_analysis_detects_alias_overlap(self):
        active_loras = [
            _make_lora_entry({"alias_a": 1.0}, name="A"),
            _make_lora_entry({"alias_b": -1.0}, name="B"),
        ]
        target_groups = self.optimizer._build_target_groups(
            ["alias_a", "alias_b"],
            {"alias_a": "layer.weight", "alias_b": "layer.weight"},
            {},
        )

        analysis = self.optimizer._run_group_analysis(
            target_groups, active_loras, self.model, None, torch.device("cpu")
        )

        self.assertEqual(analysis["prefix_count"], 1)
        stats = analysis["prefix_stats"]["alias_a"]
        self.assertEqual(stats["n_loras"], 2)
        self.assertGreater(stats["conflict_ratio"], 0.99)

    def test_same_lora_aliases_are_aggregated_before_analysis(self):
        target_group = {
            "target_key": "layer.weight",
            "is_clip": False,
            "aliases": ["alias_a", "alias_b"],
            "label_prefix": "alias_a",
        }
        active_loras = [
            _make_lora_entry({"alias_a": 1.0, "alias_b": 2.0}, name="A"),
        ]

        prepared = self.optimizer._prepare_group_diffs(
            target_group, active_loras, self.model, None, torch.device("cpu")
        )

        self.assertAlmostEqual(prepared["diffs"][0].item(), 3.0)

    def test_exact_linear_patch_matches_dense_sum(self):
        target_group = {
            "target_key": "layer.weight",
            "is_clip": False,
            "aliases": ["alias_a", "alias_b"],
            "label_prefix": "alias_a",
        }
        active_loras = [
            _make_lora_entry({"alias_a": 1.0}, name="A"),
            _make_lora_entry({"alias_b": 2.0}, name="B"),
        ]

        patch_info = self.optimizer._build_exact_linear_patch(
            target_group, active_loras, raw_n_loras=2, mode="weighted_sum"
        )

        diff = self.optimizer._expand_patch_to_diff(patch_info["patch"])
        self.assertAlmostEqual(diff.item(), 3.0)

    def test_expand_patch_to_diff_supports_lokr_and_loha(self):
        lokr_patch = lora_optimizer.LoKrAdapter(
            set(),
            (
                torch.tensor([[2.0]], dtype=torch.float32),
                torch.tensor([[3.0]], dtype=torch.float32),
                1.0,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
        loha_patch = lora_optimizer.LoHaAdapter(
            set(),
            (
                torch.tensor([[2.0]], dtype=torch.float32),
                torch.tensor([[3.0]], dtype=torch.float32),
                1.0,
                torch.tensor([[4.0]], dtype=torch.float32),
                torch.tensor([[5.0]], dtype=torch.float32),
                None,
                None,
                None,
            ),
        )

        self.assertAlmostEqual(self.optimizer._expand_patch_to_diff(lokr_patch).item(), 6.0)
        self.assertAlmostEqual(self.optimizer._expand_patch_to_diff(loha_patch).item(), 120.0)

    def test_auto_strength_uses_exact_streamed_energy(self):
        active_loras = [
            {"name": "A", "strength": 1.0, "clip_strength": None},
            {"name": "B", "strength": 1.0, "clip_strength": None},
        ]
        branch_energy = {
            "model": {
                "norm_sq": [1.0, 1.0],
                "dot": {(0, 1): 1.0},
            },
            "clip": {
                "norm_sq": [0.0, 0.0],
                "dot": {(0, 1): 0.0},
            },
        }

        info = self.optimizer._compute_auto_strengths(active_loras, branch_energy)
        self.assertAlmostEqual(info["model_scale"], 0.5)
        self.assertAlmostEqual(info["model_strengths"][0], 0.5)
        self.assertAlmostEqual(info["model_strengths"][1], 0.5)

    def test_pair_metrics_capture_excess_conflict_and_subspace_overlap(self):
        diff_a = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        diff_b = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        basis_a = self.optimizer._compute_subspace_basis(diff_a, rank_hint=1)
        basis_b = self.optimizer._compute_subspace_basis(diff_b, rank_hint=1)

        metrics = self.optimizer._sample_pair_metrics(diff_a, diff_b, basis_a=basis_a, basis_b=basis_b)

        self.assertEqual(metrics["overlap"], 0)
        self.assertAlmostEqual(metrics["subspace_overlap"], 0.0, places=4)
        self.assertAlmostEqual(metrics["excess_conflict"], 0.0, places=4)

    def test_block_smoothing_populates_decision_metrics(self):
        prefix_stats = {
            "block_0.attn.q": {
                "n_loras": 2,
                "conflict_ratio": 0.10,
                "excess_conflict": 0.10,
                "avg_cos_sim": 0.20,
                "avg_subspace_overlap": 0.30,
                "magnitude_ratio": 1.0,
                "per_lora_norm_sq": {0: 1.0, 1: 1.0},
            },
            "block_0.attn.k": {
                "n_loras": 2,
                "conflict_ratio": 0.50,
                "excess_conflict": 0.50,
                "avg_cos_sim": 0.20,
                "avg_subspace_overlap": 0.30,
                "magnitude_ratio": 1.0,
                "per_lora_norm_sq": {0: 1.0, 1: 1.0},
            },
        }

        smoothed = self.optimizer._apply_block_smoothing(prefix_stats, strength=0.5)

        self.assertIn("decision_conflict", smoothed["block_0.attn.q"])
        self.assertGreater(smoothed["block_0.attn.q"]["decision_conflict"], 0.10)
        self.assertLess(smoothed["block_0.attn.q"]["decision_conflict"], 0.50)
        self.assertEqual(smoothed["block_0.attn.q"]["block_name"], smoothed["block_0.attn.k"]["block_name"])

    def test_auto_select_uses_excess_conflict_and_subspace(self):
        mode, _density, _sign, _reasoning = self.optimizer._auto_select_params(
            0.55, 1.0, avg_cos_sim=0.0,
            avg_excess_conflict=0.05, avg_subspace_overlap=0.10,
        )
        self.assertEqual(mode, "weighted_average")

        mode, _density, _sign, _reasoning = self.optimizer._auto_select_params(
            0.55, 1.0, avg_cos_sim=0.15,
            avg_excess_conflict=0.40, avg_subspace_overlap=0.85,
        )
        self.assertEqual(mode, "ties")

    def test_save_merged_lora_uses_canonical_prefix(self):
        saver = lora_optimizer.SaveMergedLoRA()
        patch = lora_optimizer.LoRAAdapter(
            set(),
            (torch.tensor([[1.0]]), torch.tensor([[1.0]]), 1.0, None, None, None),
        )
        captured = {}

        with mock.patch.object(lora_optimizer, "save_file", side_effect=lambda state_dict, path, metadata=None: captured.update({"state_dict": state_dict, "path": path, "metadata": metadata})):
            save_path, = saver.save_lora(
                {
                    "model_patches": {"layer.weight": patch},
                    "clip_patches": {},
                    "key_map": {
                        "layer.weight": {
                            "canonical_prefix": "canonical_alias",
                            "aliases": ["alias_a", "canonical_alias"],
                        }
                    },
                    "output_strength": 1.0,
                    "clip_strength": 1.0,
                },
                tempfile.gettempdir(),
                "merged_test",
                save_rank=0,
                bake_strength=False,
            )

        self.assertTrue(save_path.endswith(".safetensors"))
        self.assertIn("canonical_alias.lora_up.weight", captured["state_dict"])

    def test_save_node_blocks_directory_traversal(self):
        merged_saver = lora_optimizer.SaveMergedLoRA()
        patch = lora_optimizer.LoRAAdapter(
            set(),
            (torch.tensor([[1.0]]), torch.tensor([[1.0]]), 1.0, None, None, None),
        )

        with self.assertRaises(ValueError):
            merged_saver.save_lora(
                {
                    "model_patches": {"layer.weight": patch},
                    "clip_patches": {},
                    "key_map": {"layer.weight": "alias"},
                    "output_strength": 1.0,
                    "clip_strength": 1.0,
                },
                tempfile.gettempdir(),
                "../escape",
                save_rank=0,
                bake_strength=False,
            )


class PrefixModeRoutingTests(unittest.TestCase):
    """Orthogonal groups BLEND by default (weighted_average / slerp). Additive
    preservation is never auto-selected — the analyzer can't tell 'preserve this
    style' from 'blend these characters', and auto-additive oversaturates ordinary
    multi-LoRA merges. Style preservation is opt-in via the preserve flag only."""

    def setUp(self):
        self.optimizer = lora_optimizer.LoRAOptimizer()
        self.arch = lora_optimizer._ARCH_PRESETS["dit"]

    def _pf(self, cos, n_loras=2, mag_ratio=1.0):
        return {
            "conflict_ratio": 0.5,
            "magnitude_ratio": mag_ratio,
            "n_loras": n_loras,
            "avg_cos_sim": cos,
            "excess_conflict": 0.0,
            "avg_subspace_overlap": 0.0,
        }

    def _decide(self, cos, strategy_set, n_loras=2, mag_ratio=1.0):
        return self.optimizer._decide_prefix_mode(
            self._pf(cos, n_loras, mag_ratio), strategy_set, self.arch,
            smooth_slerp_gate=False, is_full_rank=False, fr_preset={})

    def test_orthogonal_no_slerp_weighted_average(self):
        mode, _d, _s, orth, opp = self._decide(0.0, "no_slerp")
        self.assertEqual(mode, "weighted_average")
        self.assertTrue(orth)

    def test_orthogonal_basic_weighted_average(self):
        mode, *_ = self._decide(0.0, "basic")
        self.assertEqual(mode, "weighted_average")

    def test_orthogonal_full_slerp(self):
        mode, *_ = self._decide(0.0, "full")
        self.assertEqual(mode, "slerp")

    def test_nonorthogonal_aligned_full_slerp(self):
        mode, *_ = self._decide(0.35, "full")
        self.assertEqual(mode, "slerp")

    def test_opposing_weighted_average(self):
        mode, _d, _s, orth, opp = self._decide(-0.1, "no_slerp")
        self.assertEqual(mode, "weighted_average")
        self.assertTrue(opp)

    def test_additive_not_auto_selected_for_balanced(self):
        # Regression for the multi-LoRA oversaturation: a BALANCED stack
        # (magnitude_ratio ~1) must never auto-route to a sum mode regardless of
        # orthogonality/strategy.
        for cos in (0.0, 0.1, -0.1, 0.35, 0.6):
            for ss in ("full", "no_slerp", "basic"):
                mode, *_ = self._decide(cos, ss, mag_ratio=1.0)
                self.assertNotIn(mode, ("sum_preserve", "weighted_sum"))

    def test_imbalanced_orthogonal_pair_routes_to_weighted_sum(self):
        # A strongly-imbalanced orthogonal PAIR routes to additive: SLERP/
        # weighted_average wash out the dominant LoRA, weighted_sum preserves it
        # (and can't oversaturate — the dominant defines the auto-strength ref).
        mode, _d, _s, orth, opp = self._decide(0.0, "full", mag_ratio=6.0)
        self.assertEqual(mode, "weighted_sum")
        self.assertTrue(orth)
        self.assertFalse(opp)

    def test_imbalanced_orthogonal_suppresses_slerp(self):
        # Below the cap → SLERP; at/above the cap → not SLERP.
        self.assertEqual(self._decide(0.0, "full", mag_ratio=1.5)[0], "slerp")
        self.assertNotEqual(self._decide(0.0, "full", mag_ratio=2.5)[0], "slerp")

    def test_imbalanced_aligned_not_weighted_sum(self):
        # Imbalanced but ALIGNED (non-orthogonal): SLERP is suppressed but it must
        # NOT go additive (aligned LoRAs reinforce → additive would oversaturate).
        mode, *_ = self._decide(0.35, "full", mag_ratio=6.0)
        self.assertEqual(mode, "weighted_average")

    def test_imbalanced_orthogonal_triple_stays_blended(self):
        # 3+ LoRAs: weighted_sum is gated to pairs (a balanced sub-pair among 3+
        # could compound past the auto-strength floor), so imbalanced triples fall
        # back to weighted_average, never additive.
        mode, *_ = self._decide(0.0, "full", n_loras=3, mag_ratio=6.0)
        self.assertEqual(mode, "weighted_average")


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class ShapeMismatchReportTests(unittest.TestCase):
    """A LoRA whose tensor shape doesn't match the target model at a layer is
    dropped from the merge; _note_shape_mismatch records it (deduped, capped) so
    the report can surface the incompatibility instead of dropping it silently."""

    def setUp(self):
        self.optimizer = lora_optimizer.LoRAOptimizer()
        self.optimizer._shape_mismatches = {}

    def test_records_and_dedups_by_key(self):
        item = {"name": "concept/KNP_V2.safetensors"}
        self.optimizer._note_shape_mismatch(item, "blocks.0.attn.gate", 2560, 6144)
        self.optimizer._note_shape_mismatch(item, "blocks.0.attn.gate", 2560, 6144)  # dup
        self.optimizer._note_shape_mismatch(item, "blocks.1.attn.wq", 2560, 6144)
        per = self.optimizer._shape_mismatches["concept/KNP_V2.safetensors"]
        self.assertEqual(len(per), 2)  # deduped by target key
        self.assertEqual(per["blocks.0.attn.gate"], (2560, 6144))

    def test_tuple_target_key_uses_first_element(self):
        self.optimizer._note_shape_mismatch({"name": "x"}, ("blocks.0.attn.gate", True), 2560, 6144)
        self.assertIn("blocks.0.attn.gate", self.optimizer._shape_mismatches["x"])

    def test_capped_at_256(self):
        for i in range(300):
            self.optimizer._note_shape_mismatch({"name": "x"}, f"blocks.{i}", 2560, 6144)
        self.assertLessEqual(len(self.optimizer._shape_mismatches["x"]), 256)

    def test_lazy_init_when_attribute_absent(self):
        opt = lora_optimizer.LoRAOptimizer()
        if hasattr(opt, "_shape_mismatches"):
            del opt._shape_mismatches
        opt._note_shape_mismatch({"name": "y"}, "blocks.0", 2560, 6144)
        self.assertIn("y", opt._shape_mismatches)

    def test_report_lines_empty_when_no_mismatch(self):
        self.assertEqual(self.optimizer._shape_mismatch_report_lines(), [])

    def test_report_lines_render_warning(self):
        item = {"name": "Krea 2/concept/KNP_V2.safetensors"}
        for i in range(2):
            for proj in ("gate", "wq", "wk", "wv"):
                self.optimizer._note_shape_mismatch(item, f"blocks.{i}.attn.{proj}", 2560, 6144)
        text = "\n".join(self.optimizer._shape_mismatch_report_lines())
        self.assertIn("SHAPE INCOMPATIBILITY", text)
        self.assertIn("8 layer(s) DROPPED", text)        # 2 blocks x 4 projections
        self.assertIn("KNP_V2.safetensors", text)        # basename only, no dir
        self.assertNotIn("concept/", text)
        self.assertIn("LoRA dim=2560 vs model dim=6144", text)
        self.assertIn("... and", text)                   # truncated past 3 examples
        self.assertIn("SAME base model", text)           # actionable fix hint

    def test_lokr_shape_mismatch_is_recorded(self):
        # A LoKr concept whose reconstructed delta doesn't match the model weight
        # must be recorded too — LoKr/LoHa go through a different branch than plain
        # LoRA, so the capture has to cover it (regression for snofs-style LoKrs).
        model = _make_model()  # model.layer.weight is (1, 1)
        lora = {
            "layer.lokr_w1": torch.eye(2, dtype=torch.float32),          # (2, 2)
            "layer.lokr_w2": torch.tensor([[1.0]], dtype=torch.float32),  # (1, 1)
        }  # kron(w1, w2) -> (2, 2), cannot reshape to the (1, 1) model weight
        active_loras = [{
            "name": "concept/snofs_krea_v1.safetensors", "lora": lora, "strength": 1.0,
            "clip_strength": None, "key_filter": "all", "conflict_mode": "all",
        }]
        target_group = {
            "target_key": "layer.weight", "is_clip": False,
            "aliases": ["layer"], "label_prefix": "layer",
        }
        self.optimizer._prepare_group_diffs(
            target_group, active_loras, model, None, torch.device("cpu"))
        per = self.optimizer._shape_mismatches.get("concept/snofs_krea_v1.safetensors")
        self.assertIsNotNone(per)
        self.assertEqual(per["layer.weight"], (2, 1))  # LoKr dim 2 vs model dim 1


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LoraCleaningTests(unittest.TestCase):
    """Per-LoRA cleaning primitives: STAR spectral truncate+rescale (nuclear-norm
    preserving) and base-norm-anchored magnitude taming (Norm-Anchor Scaling)."""

    def setUp(self):
        self.opt = lora_optimizer.LoRAOptimizer()

    def test_star_eta_100_is_identity(self):
        d = torch.randn(6, 4)
        torch.testing.assert_close(self.opt._star_truncate_rescale(d.clone(), 100.0), d)

    def test_star_preserves_nuclear_norm_and_reduces_rank(self):
        d = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))  # nuclear norm 10, rank 4
        out = self.opt._star_truncate_rescale(d.clone(), 70.0)
        sv = torch.linalg.svdvals(out)
        self.assertAlmostEqual(sv.sum().item(), 10.0, places=4)   # nuclear norm restored
        self.assertEqual(int((sv > 1e-4).sum().item()), 2)         # eta=70 keeps 2 of 4

    def test_star_preserves_nuclear_norm_random(self):
        torch.manual_seed(0)
        d = torch.randn(8, 5)
        nuc0 = torch.linalg.svdvals(d).sum().item()
        out = self.opt._star_truncate_rescale(d.clone(), 50.0)
        self.assertAlmostEqual(torch.linalg.svdvals(out).sum().item(), nuc0, places=3)

    def test_star_1d_passthrough(self):
        d = torch.randn(5)
        torch.testing.assert_close(self.opt._star_truncate_rescale(d.clone(), 50.0), d)

    def test_tame_within_budget_no_scale(self):
        self.assertEqual(self.opt._tame_scale(2.0, 10.0, 0.3, 1.0), 1.0)  # r=0.2 <= 0.3

    def test_tame_hot_layer_scaled_to_threshold_at_strength_1(self):
        s = self.opt._tame_scale(5.0, 10.0, 0.3, 1.0)  # r=0.5 > 0.3
        self.assertAlmostEqual(s, 0.6, places=6)
        self.assertAlmostEqual(5.0 * s, 0.3 * 10.0, places=6)  # tamed to exactly threshold*base

    def test_tame_strength_0_is_off(self):
        self.assertEqual(self.opt._tame_scale(5.0, 10.0, 0.3, 0.0), 1.0)

    def test_tame_partial_strength(self):
        s = self.opt._tame_scale(5.0, 10.0, 0.3, 0.5)
        self.assertAlmostEqual(s, math.sqrt(0.6), places=6)

    def test_tame_zero_base_norm_floored(self):
        s = self.opt._tame_scale(1.0, 0.0, 0.3, 1.0)  # denom floored, no div-by-zero
        self.assertGreater(s, 0.0)
        self.assertLess(s, 1e-6)

    def _lora_and_model(self, base_scale):
        # base weight = base_scale*I(4) (norm 2*base_scale); LoRA delta = diag(4,3,2,1)
        layer = types.SimpleNamespace(weight=torch.eye(4) * base_scale)
        model = types.SimpleNamespace(model=types.SimpleNamespace(layer=layer))
        lora = {"layer.lora_up.weight": torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])),
                "layer.lora_down.weight": torch.eye(4)}
        active = [{"name": "A", "lora": lora, "strength": 1.0, "clip_strength": None,
                   "key_filter": "all", "conflict_mode": "all"}]
        tg = {"target_key": "layer.weight", "is_clip": False,
              "aliases": ["layer"], "label_prefix": "layer"}
        return active, tg, model

    def _diff(self, active, tg, model):
        return self.opt._prepare_group_diffs(tg, active, model, None, torch.device("cpu"))["diffs"][0]

    def test_prepare_applies_star(self):
        active, tg, model = self._lora_and_model(3.0)
        self.opt._star_eta = 100.0; self.opt._tame_layers = 0.0
        self.assertEqual(int((torch.linalg.svdvals(self._diff(active, tg, model)) > 1e-4).sum()), 4)
        self.opt._star_eta = 70.0
        sv = torch.linalg.svdvals(self._diff(active, tg, model))
        self.assertEqual(int((sv > 1e-4).sum()), 2)                 # rank reduced
        self.assertAlmostEqual(sv.sum().item(), 10.0, places=3)     # nuclear norm preserved

    def test_prepare_applies_tame(self):
        active, tg, model = self._lora_and_model(5.0)  # base norm 10; delta norm sqrt(30)
        self.opt._star_eta = 100.0; self.opt._tame_layers = 1.0; self.opt._tame_threshold = 0.3
        self.assertAlmostEqual(self._diff(active, tg, model).norm().item(), 3.0, places=3)  # -> 0.3*base

    def test_prepare_preserve_is_exempt_from_cleaning(self):
        active, tg, model = self._lora_and_model(5.0)
        active[0]["preserve"] = True
        self.opt._star_eta = 50.0; self.opt._tame_layers = 1.0; self.opt._tame_threshold = 0.3
        out = self._diff(active, tg, model)
        self.assertEqual(int((torch.linalg.svdvals(out) > 1e-4).sum()), 4)   # untouched
        self.assertAlmostEqual(out.norm().item(), math.sqrt(30.0), places=3)

    def test_cache_key_folds_cleaning_only_when_active(self):
        stack = [{"name": "x", "strength": 1.0}]
        k_off = self.opt._compute_cache_key(stack, 1.0, 1.0, "disabled")
        k_off2 = self.opt._compute_cache_key(stack, 1.0, 1.0, "disabled", star_eta=100.0, tame_layers=0.0)
        self.assertEqual(k_off, k_off2)  # default-off must not change the key
        k_on = self.opt._compute_cache_key(stack, 1.0, 1.0, "disabled", star_eta=40.0)
        self.assertNotEqual(k_off, k_on)  # active cleaning gets a distinct key


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class PreserveFlagTests(unittest.TestCase):
    """A per-LoRA preserve flag protects a tagged style LoRA from TIES sign-election
    deletion and from sparsification trimming in conflict merges."""

    def setUp(self):
        self.optimizer = lora_optimizer.LoRAOptimizer()

    def test_preserve_overlay_weighted_average(self):
        # The style-LoRA use case: a flagged style is added at FULL strength on top
        # of the weighted_average blend of the rest, instead of being averaged down.
        content = torch.tensor([2.0, 0.0, 0.0, 0.0])   # not preserved
        style = torch.tensor([0.0, 3.0, 0.0, 0.0])     # preserved (orthogonal)
        blended = self.optimizer._merge_diffs(
            [(content.clone(), 1.0), (style.clone(), 1.0)], "weighted_average")
        kept = self.optimizer._merge_diffs(
            [(content.clone(), 1.0), (style.clone(), 1.0)], "weighted_average",
            preserve_flags=[False, True])
        # plain blend halves both; preserve keeps content blended (single -> full) and
        # the style at full on top
        torch.testing.assert_close(blended, torch.tensor([1.0, 1.5, 0.0, 0.0]))
        torch.testing.assert_close(kept, torch.tensor([2.0, 3.0, 0.0, 0.0]))

    def test_preserve_does_not_affect_unflagged_blend(self):
        # No flags -> ordinary balanced blend is untouched (the multi-LoRA case).
        a = torch.tensor([2.0, 0.0])
        b = torch.tensor([0.0, 4.0])
        res = self.optimizer._merge_diffs(
            [(a.clone(), 1.0), (b.clone(), 1.0)], "weighted_average")
        torch.testing.assert_close(res, torch.tensor([1.0, 2.0]))

    def test_ties_deletes_minority_sign_without_preserve(self):
        # Content (+10) out-votes a style (-1) in TIES sign election; the style's
        # minority-sign direction is dropped entirely.
        content = torch.tensor([10.0, 10.0, 10.0, 10.0])
        style = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        base = self.optimizer._merge_diffs(
            [(content.clone(), 1.0), (style.clone(), 1.0)], "ties", density=1.0)
        self.assertAlmostEqual(base[0].item(), 10.0, places=5)

    def test_ties_preserve_keeps_minority_sign_style(self):
        # With preserve on the style, its full contribution is added on top of the
        # TIES-merged content: 10 + (-1) = 9 (the style survives the conflict).
        content = torch.tensor([10.0, 10.0, 10.0, 10.0])
        style = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        kept = self.optimizer._merge_diffs(
            [(content.clone(), 1.0), (style.clone(), 1.0)], "ties", density=1.0,
            preserve_flags=[False, True])
        self.assertAlmostEqual(kept[0].item(), 9.0, places=5)

    def test_ties_all_preserved_is_full_sum(self):
        # Every contributor tagged -> nothing to TIES-merge -> plain full sum.
        a = torch.tensor([3.0, 3.0])
        b = torch.tensor([-2.0, -2.0])
        res = self.optimizer._merge_diffs(
            [(a.clone(), 1.0), (b.clone(), 1.0)], "ties", density=0.5,
            preserve_flags=[True, True])
        torch.testing.assert_close(res, torch.tensor([1.0, 1.0]))

    def test_sparsification_all_preserved_equals_disabled(self):
        # All preserved -> sparsification is skipped -> identical to disabled.
        d1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        d2 = torch.tensor([0.5, 1.5, 2.5, 3.5])
        disabled = self.optimizer._merge_diffs(
            [(d1.clone(), 1.0), (d2.clone(), 1.0)], "weighted_sum",
            sparsification="disabled")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        all_pres = self.optimizer._merge_diffs(
            [(d1.clone(), 1.0), (d2.clone(), 1.0)], "weighted_sum",
            sparsification="dare", sparsification_density=0.5,
            sparsification_generator=gen, preserve_flags=[True, True])
        torch.testing.assert_close(disabled, all_pres)

    def test_normalize_stack_carries_preserve_tuple_and_dict(self):
        opt = lora_optimizer.LoRAOptimizer()
        opt.loaded_loras = {"loraA": {}, "loraB": {}}
        tup = opt._normalize_stack([
            ("loraA", 1.0, 1.0, "all", "all", True),
            ("loraB", 1.0, 1.0, "all", "all"),  # legacy 5-tuple -> preserve False
        ])
        self.assertTrue(tup[0]["preserve"])
        self.assertFalse(tup[1]["preserve"])

        dct = opt._normalize_stack([
            {"name": "x", "lora": {}, "strength": 1.0, "preserve": True},
        ])
        self.assertTrue(dct[0]["preserve"])

    def test_normalize_stack_mixed_tuple_and_dict(self):
        """A stack mixing file references with preloaded adapter entries keeps
        both; the first element must not determine the format of the rest."""
        opt = lora_optimizer.LoRAOptimizer()
        opt.loaded_loras = {"loraA": {"k": 1}}
        extracted = {"name": "<extracted>", "lora": {"w": 2}, "strength": 1.5}

        # tuple first, dict last
        out = opt._normalize_stack([("loraA", 1.0, 1.0), extracted])
        self.assertEqual([e["name"] for e in out], ["loraA", "<extracted>"])
        self.assertEqual(out[1]["lora"], {"w": 2})

        # dict first, tuple last
        out2 = opt._normalize_stack([extracted, ("loraA", 1.0, 1.0)])
        self.assertEqual([e["name"] for e in out2], ["<extracted>", "loraA"])

    def test_compute_cache_key_mixed_tuple_and_dict(self):
        """_compute_cache_key must not crash on a mixed stack (tuple entry
        first, extracted dict entry second) — it used to do entry[3] on the
        dict and raise KeyError: 3."""
        key = lora_optimizer.LoRAOptimizer._compute_cache_key(
            [("loraA", 1.0, 1.0), {"name": "<extracted>", "lora": {}, "strength": 1.5}],
            output_strength=1.0, auto_strength="disabled",
        )
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 16)
        # order-independent: same entries reversed hash identically
        key2 = lora_optimizer.LoRAOptimizer._compute_cache_key(
            [{"name": "<extracted>", "lora": {}, "strength": 1.5}, ("loraA", 1.0, 1.0)],
            output_strength=1.0, auto_strength="disabled",
        )
        self.assertEqual(key, key2)

@unittest.skipIf(torch is None, "torch is not installed in this environment")
class UnifiedOutputStrengthTests(unittest.TestCase):
    def test_single_lora_scales_model_and_clip_with_output_strength(self):
        class Model:
            pass

        class Clip:
            pass

        model = Model()
        model.model = object()
        clip = Clip()
        optimizer = lora_optimizer.LoRAOptimizer()
        optimizer.loaded_loras = {"one.safetensors": {}}
        captured = {}

        def apply(model_arg, clip_arg, _lora, model_strength, clip_strength):
            captured["model"] = model_strength
            captured["clip"] = clip_strength
            return model_arg, clip_arg

        with mock.patch.object(
                lora_optimizer.comfy.sd, "load_lora_for_models", side_effect=apply):
            optimizer.optimize_merge(
                model, [("one.safetensors", 0.8, 0.4)], 1.25, clip=clip,
                cache_patches="disabled", persistent_cache="disabled")

        self.assertAlmostEqual(captured["model"], 1.0)
        self.assertAlmostEqual(captured["clip"], 0.5)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LoRASettingsNodeTests(unittest.TestCase):
    def test_only_three_nodes_are_registered(self):
        self.assertEqual(set(lora_optimizer.NODE_CLASS_MAPPINGS), {
            "LoRAOptimizerSimple", "LoRAOptimizerSettings", "SaveMergedLoRA"})

    def test_main_optimizer_has_complete_node_metadata(self):
        node = lora_optimizer.LoRAOptimizerSimple
        self.assertEqual(node.CATEGORY, "LoRA Optimizer")
        self.assertEqual(node.RETURN_TYPES, ("MODEL", "CLIP", "STRING", "LORA_DATA"))
        self.assertEqual(node.RETURN_NAMES, ("model", "clip", "analysis_report", "lora_data"))

    def test_removed_nodes_leave_no_obsolete_input_sockets(self):
        optimizer_inputs = lora_optimizer.LoRAOptimizerSimple.INPUT_TYPES()
        settings_inputs = lora_optimizer.LoRAOptimizerSettings.INPUT_TYPES()
        self.assertNotIn("clip_strength_multiplier", optimizer_inputs.get("optional", {}))
        self.assertNotIn("tuner_data", optimizer_inputs.get("optional", {}))
        self.assertNotIn("merge_settings", settings_inputs.get("optional", {}))
        self.assertNotIn("merge_strategy_override", settings_inputs.get("optional", {}))

    def test_settings_include_separate_memory_and_persistent_cache_switches(self):
        inputs = lora_optimizer.LoRAOptimizerSettings.INPUT_TYPES()["required"]
        self.assertEqual(inputs["cache_patches"][1]["default"], "enabled")
        self.assertEqual(inputs["persistent_cache"][1]["default"], "enabled")

    def test_settings_build_matches_simple_defaults(self):
        node = lora_optimizer.LoRAOptimizerSettings()
        values = {
            key: schema[1].get("default", schema[0][0])
            for key, schema in node.INPUT_TYPES()["required"].items()
        }
        settings = node.build_settings(**values)[0]
        for key, value in lora_optimizer.LoRAOptimizerSimple._SIMPLE_DEFAULTS.items():
            self.assertEqual(settings[key], value, key)

    def test_persistent_cache_switch_changes_execution_hash(self):
        node = lora_optimizer.LoRAOptimizerSimple
        base = node.IS_CHANGED(None, [], 1.0, settings={**node._SIMPLE_DEFAULTS, "persistent_cache": "enabled"})
        disabled = node.IS_CHANGED(None, [], 1.0, settings={**node._SIMPLE_DEFAULTS, "persistent_cache": "disabled"})
        self.assertNotEqual(base, disabled)


class TestIdeogram4Support(unittest.TestCase):
    """Ideogram 4 detection (must beat the Z-Image check — both are NextDiT
    with layers.N.attention.qkv), key normalization, and preset routing."""

    def _detect(self, sd):
        return lora_optimizer._LoRAMergeBase._detect_architecture(sd)

    @staticmethod
    def _zeros_sd(keys):
        return {k: torch.zeros(1) for k in keys}

    def test_ai_toolkit_native_format_detected(self):
        sd = self._zeros_sd([
            "diffusion_model.layers.0.attention.qkv.lora_A.weight",
            "diffusion_model.layers.0.attention.qkv.lora_B.weight",
            "diffusion_model.layers.0.attention.o.lora_A.weight",
            "diffusion_model.layers.0.attention.o.lora_B.weight",
            "diffusion_model.layers.0.feed_forward.w1.lora_A.weight",
            "diffusion_model.layers.0.feed_forward.w1.lora_B.weight",
        ])
        self.assertEqual(self._detect(sd), "ideogram4")

    def test_fal_conditional_transformer_prefix_detected(self):
        sd = self._zeros_sd([
            "conditional_transformer.layers.3.attention.qkv.lora_A.weight",
            "conditional_transformer.layers.3.attention.qkv.lora_B.weight",
            "conditional_transformer.layers.3.attention.qkv.alpha",
        ])
        self.assertEqual(self._detect(sd), "ideogram4")

    def test_lokr_output_proj_detected(self):
        sd = self._zeros_sd([
            "diffusion_model.layers.7.attention.o.lokr_w1",
            "diffusion_model.layers.7.attention.o.lokr_w2",
            "diffusion_model.layers.7.attention.o.alpha",
        ])
        self.assertEqual(self._detect(sd), "ideogram4")

    def test_lowercase_adaln_plus_ffn_detected(self):
        sd = self._zeros_sd([
            "diffusion_model.layers.2.adaln_modulation.lora_A.weight",
            "diffusion_model.layers.2.feed_forward.w2.lora_A.weight",
        ])
        self.assertEqual(self._detect(sd), "ideogram4")

    def test_qkv_only_disambiguated_by_fused_width(self):
        # qkv-only LoRA: no o/adaln markers — the 13824-row (3x4608) fused
        # up matrix is the Ideogram tell; 6912 (3x2304) is Z-Image Turbo
        ideo = {
            "diffusion_model.layers.0.attention.qkv.lora_A.weight": torch.zeros(8, 4608),
            "diffusion_model.layers.0.attention.qkv.lora_B.weight": torch.zeros(13824, 8),
        }
        self.assertEqual(self._detect(ideo), "ideogram4")
        zim = {
            "diffusion_model.layers.0.attention.qkv.lora_A.weight": torch.zeros(8, 2304),
            "diffusion_model.layers.0.attention.qkv.lora_B.weight": torch.zeros(6912, 8),
        }
        self.assertEqual(self._detect(zim), "zimage")

    def test_zimage_keys_still_detect_zimage(self):
        """Regression: Z-Image markers (attention.out, adaLN_modulation) must
        not be claimed by the Ideogram 4 checks."""
        sd = self._zeros_sd([
            "diffusion_model.layers.0.attention.qkv.lora_up.weight",
            "diffusion_model.layers.0.attention.qkv.lora_down.weight",
            "diffusion_model.layers.0.attention.out.lora_up.weight",
            "diffusion_model.layers.0.attention.out.lora_down.weight",
            "diffusion_model.layers.0.adaLN_modulation.1.lora_up.weight",
        ])
        self.assertEqual(self._detect(sd), "zimage")

    def test_normalize_fal_and_peft_prefixes(self):
        norm = lora_optimizer._LoRAMergeBase._normalize_keys_ideogram4
        sd = self._zeros_sd([
            "conditional_transformer.layers.3.attention.qkv.lora_A.weight",
            "transformer.layers.4.feed_forward.w3.lora_B.weight",
            "base_model.model.layers.5.attention.o.lora_A.weight",
            "layers.6.adaln_modulation.lora_B.weight",
            "diffusion_model.layers.7.attention.qkv.lora_A.weight",  # passthrough
        ])
        out = norm(sd)
        self.assertIn("diffusion_model.layers.3.attention.qkv.lora_A.weight", out)
        self.assertIn("diffusion_model.layers.4.feed_forward.w3.lora_B.weight", out)
        self.assertIn("diffusion_model.layers.5.attention.o.lora_A.weight", out)
        self.assertIn("diffusion_model.layers.6.adaln_modulation.lora_B.weight", out)
        self.assertIn("diffusion_model.layers.7.attention.qkv.lora_A.weight", out)
        self.assertEqual(len(out), len(sd))

    def test_normalize_kohya_underscores(self):
        norm = lora_optimizer._LoRAMergeBase._normalize_keys_ideogram4
        sd = self._zeros_sd([
            "lora_unet_layers_0_attention_qkv.lora_down.weight",
            "lora_unet_layers_12_feed_forward_w2.lora_up.weight",
            "lora_unet_layers_3_adaln_modulation.alpha",
        ])
        out = norm(sd)
        self.assertIn("diffusion_model.layers.0.attention.qkv.lora_down.weight", out)
        self.assertIn("diffusion_model.layers.12.feed_forward.w2.lora_up.weight", out)
        self.assertIn("diffusion_model.layers.3.adaln_modulation.alpha", out)

    def test_preset_routes_to_dit(self):
        key, preset = lora_optimizer._resolve_arch_preset("auto", "ideogram4")
        self.assertEqual(key, "dit")

    def test_normalize_dispatch(self):
        sd = {"conditional_transformer.layers.0.attention.o.lora_A.weight": torch.zeros(1)}
        out = lora_optimizer._LoRAMergeBase._normalize_keys(sd, "ideogram4")
        self.assertIn("diffusion_model.layers.0.attention.o.lora_A.weight", out)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestAceStepDetection(unittest.TestCase):
    """Test _detect_architecture for ACE-Step v1.0 and v1.5 key patterns."""

    def _detect(self, keys):
        sd = {k: torch.zeros(1) for k in keys}
        return lora_optimizer._LoRAMergeBase._detect_architecture(sd)

    # --- v1.5 PEFT format ---
    def test_v15_peft_self_attn(self):
        keys = [
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    def test_v15_peft_cross_attn(self):
        keys = [
            "base_model.model.layers.12.cross_attn.k_proj.lora_A.weight",
            "base_model.model.layers.12.cross_attn.k_proj.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    def test_v15_peft_mlp_only_not_detected(self):
        """MLP-only LoRA without attn keys should not detect as acestep."""
        keys = [
            "base_model.model.layers.0.mlp.gate_proj.lora_A.weight",
            "base_model.model.layers.0.mlp.gate_proj.lora_B.weight",
        ]
        # No self_attn/cross_attn keys, so won't match acestep pattern
        self.assertNotEqual(self._detect(keys), "acestep")

    def test_v15_bare_layers(self):
        """v1.5 keys without base_model.model. prefix."""
        keys = [
            "layers.5.self_attn.v_proj.lora_up.weight",
            "layers.5.self_attn.v_proj.lora_down.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    # --- v1.0 diffusers format ---
    def test_v10_transformer_blocks(self):
        keys = [
            "transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer_blocks.0.attn.to_q.lora_B.weight",
            "transformer_blocks.0.cross_attn.to_k.lora_A.weight",
            "transformer_blocks.0.cross_attn.to_k.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    def test_v10_speaker_embedder(self):
        keys = [
            "speaker_embedder.lora_A.weight",
            "speaker_embedder.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    def test_v10_lyric_encoder(self):
        keys = [
            "lyric_encoder.encoders.0.self_attn.linear_q.lora_A.weight",
            "lyric_encoder.encoders.0.self_attn.linear_q.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestAceStepNormalization(unittest.TestCase):
    """Test _normalize_keys_acestep for v1.0 and v1.5 key formats."""

    def _norm(self, keys):
        sd = {k: torch.zeros(1) for k in keys}
        return lora_optimizer._LoRAMergeBase._normalize_keys_acestep(sd)

    # --- v1.5 PEFT format ---
    def test_v15_peft_strips_prefix(self):
        result = self._norm([
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        ])
        self.assertIn("diffusion_model.layers.0.self_attn.q_proj.lora_A.weight", result)

    def test_v15_bare_layers_adds_prefix(self):
        result = self._norm(["layers.5.cross_attn.k_proj.lora_up.weight"])
        self.assertIn("diffusion_model.layers.5.cross_attn.k_proj.lora_up.weight", result)

    def test_v15_kohya_underscore(self):
        result = self._norm(["lora_unet_layers_3_self_attn_q_proj.lora_down.weight"])
        self.assertIn("diffusion_model.layers.3.self_attn.q_proj.lora_down.weight", result)

    def test_v15_mlp_keys(self):
        result = self._norm([
            "base_model.model.layers.10.mlp.gate_proj.lora_A.weight",
        ])
        self.assertIn("diffusion_model.layers.10.mlp.gate_proj.lora_A.weight", result)

    # --- v1.0 → v1.5 mapping ---
    def test_v10_transformer_blocks_to_layers(self):
        result = self._norm([
            "transformer_blocks.7.attn.to_q.lora_A.weight",
        ])
        self.assertIn("diffusion_model.layers.7.self_attn.q_proj.lora_A.weight", result)

    def test_v10_cross_attn_preserved(self):
        result = self._norm([
            "transformer_blocks.3.cross_attn.to_v.lora_B.weight",
        ])
        self.assertIn("diffusion_model.layers.3.cross_attn.v_proj.lora_B.weight", result)

    def test_v10_to_out_0_to_o_proj(self):
        result = self._norm([
            "transformer_blocks.0.attn.to_out.0.lora_A.weight",
        ])
        self.assertIn("diffusion_model.layers.0.self_attn.o_proj.lora_A.weight", result)

    def test_v10_cross_attn_to_out_0(self):
        result = self._norm([
            "transformer_blocks.5.cross_attn.to_out.0.lora_B.weight",
        ])
        self.assertIn("diffusion_model.layers.5.cross_attn.o_proj.lora_B.weight", result)

    def test_v10_speaker_embedder(self):
        result = self._norm(["speaker_embedder.lora_A.weight"])
        self.assertIn("diffusion_model.speaker_embedder.lora_A.weight", result)

    def test_v10_lyric_encoder(self):
        result = self._norm([
            "lyric_encoder.encoders.2.self_attn.linear_q.lora_A.weight",
        ])
        self.assertIn(
            "diffusion_model.lyric_encoder.encoders.2.self_attn.q_proj.lora_A.weight",
            result,
        )

    def test_v10_lyric_encoder_linear_v(self):
        result = self._norm([
            "lyric_encoder.encoders.0.self_attn.linear_v.lora_B.weight",
        ])
        self.assertIn(
            "diffusion_model.lyric_encoder.encoders.0.self_attn.v_proj.lora_B.weight",
            result,
        )

    # --- Mixed format: ensure no cross-contamination ---
    def test_self_attn_not_double_prefixed(self):
        """self_attn should not become self_self_attn."""
        result = self._norm([
            "transformer_blocks.0.cross_attn.to_q.lora_A.weight",
        ])
        key = list(result.keys())[0]
        self.assertNotIn("self_self_attn", key)
        self.assertNotIn("self_cross_attn", key)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestAceStepPreset(unittest.TestCase):
    """Test ACE-Step architecture preset and auto-detection integration."""

    def test_acestep_maps_to_dedicated_preset(self):
        self.assertEqual(lora_optimizer._ARCH_TO_PRESET["acestep"], "acestep_dit")
        self.assertIn("acestep_dit", lora_optimizer._ARCH_PRESETS)

    def test_acestep_preset_has_wider_orthogonal_band(self):
        dit = lora_optimizer._ARCH_PRESETS["dit"]
        ace = lora_optimizer._ARCH_PRESETS["acestep_dit"]
        self.assertGreater(ace["orthogonal_cos_sim_max"], dit["orthogonal_cos_sim_max"])

    def test_acestep_preset_has_higher_ties_threshold(self):
        dit = lora_optimizer._ARCH_PRESETS["dit"]
        ace = lora_optimizer._ARCH_PRESETS["acestep_dit"]
        self.assertGreater(ace["ties_conflict_threshold"], dit["ties_conflict_threshold"])

    def test_acestep_preset_full_magnitude_preservation(self):
        ace = lora_optimizer._ARCH_PRESETS["acestep_dit"]
        self.assertEqual(ace["auto_strength_orthogonal_floor"], 1.0)

    def test_resolve_arch_preset_acestep(self):
        key, preset = lora_optimizer._resolve_arch_preset("auto", "acestep")
        self.assertEqual(key, "acestep_dit")
        self.assertEqual(preset["display_name"], "ACE-Step (Music DiT)")

    def test_resolve_arch_preset_manual_override(self):
        key, preset = lora_optimizer._resolve_arch_preset("acestep_dit", "unknown")
        self.assertEqual(key, "acestep_dit")


class TestExcessConflictBaseline(unittest.TestCase):
    """excess_conflict must compare the UNWEIGHTED sign-mismatch fraction
    against the unweighted arccos(rho)/pi baseline on the same position set
    (Sheppard / degree-0 arc-cosine kernel)."""

    def setUp(self):
        self.opt = lora_optimizer.LoRAOptimizer()

    def test_identical_vectors_no_excess(self):
        g = torch.Generator().manual_seed(7)
        a = torch.randn(20000, generator=g)
        m = self.opt._sample_pair_metrics(a, a.clone())
        self.assertAlmostEqual(m["excess_conflict"], 0.0, places=5)
        self.assertAlmostEqual(m["expected_conflict"], 0.0, places=3)

    def test_negated_vectors_no_excess(self):
        g = torch.Generator().manual_seed(7)
        a = torch.randn(20000, generator=g)
        m = self.opt._sample_pair_metrics(a, -a)
        # Mismatch fraction 1.0 is exactly what rho=-1 predicts
        self.assertAlmostEqual(m["expected_conflict"], 1.0, places=3)
        self.assertAlmostEqual(m["excess_conflict"], 0.0, places=3)

    def test_independent_gaussians_no_excess(self):
        g = torch.Generator().manual_seed(11)
        a = torch.randn(50000, generator=g)
        b = torch.randn(50000, generator=g)
        m = self.opt._sample_pair_metrics(a, b)
        # ~50% mismatch is the rho~0 base rate, not real conflict
        self.assertLess(m["excess_conflict"], 0.03)

    def test_count_conflict_detected_beyond_weighted_baseline(self):
        """Mismatches concentrated on smaller (but above-noise-floor)
        magnitudes: the old magnitude-weighted ratio sat BELOW the baseline
        (excess clamped to 0); the unweighted fraction detects it."""
        g = torch.Generator().manual_seed(3)
        n = 20000
        signs = torch.where(torch.rand(n, generator=g) < 0.5, 1.0, -1.0)
        mag = torch.where(torch.arange(n) < n // 2,
                          torch.full((n,), 2.0), torch.full((n,), 0.3))
        a = signs * mag
        b = a.clone()
        # Flip half of the small-magnitude positions (25% of total count)
        flip = torch.arange(n) >= (3 * n) // 4
        b[flip] = -b[flip]
        m = self.opt._sample_pair_metrics(a, b)
        # Old weighted ratio: 0.25*0.3/(0.5*2+0.5*0.3) ~ 0.065 < baseline
        # arccos(0.978)/pi ~ 0.067 -> old excess clamped to ~0.
        # New unweighted: 0.25 - 0.067 ~ 0.18.
        self.assertGreater(m["excess_conflict"], 0.10)


class TestTiesSignElection(unittest.TestCase):
    """Sign election is always the magnitude-weighted 'total' vote — the only
    mechanism defined in the TIES paper (frequency is override-only)."""

    def test_ties_mode_elects_total_regardless_of_magnitude_ratio(self):
        opt = lora_optimizer.LoRAOptimizer()
        for mag_ratio in (1.0, 2.0, 10.0):
            mode, _density, sign, _r = opt._auto_select_params(
                0.6, mag_ratio, avg_cos_sim=0.5,
                avg_excess_conflict=0.6, avg_subspace_overlap=0.8,
                strategy_set="basic", precomputed_density=0.7)
            self.assertEqual(mode, "ties")
            self.assertEqual(sign, "total")


class TestKarcherSlerp(unittest.TestCase):
    """N>=3 slerp mode is a weighted Karcher mean: order-independent,
    symmetric, magnitude-corrected. N=2 remains standard SLERP."""

    def setUp(self):
        self.opt = lora_optimizer.LoRAOptimizer()

    def _merge(self, pairs):
        return self.opt._merge_diffs([(t.clone(), w) for t, w in pairs], "slerp")

    def test_two_vector_slerp_unchanged(self):
        e1 = torch.zeros(8); e1[0] = 1.0
        e2 = torch.zeros(8); e2[1] = 1.0
        out = self._merge([(e1, 1.0), (e2, 1.0)])
        expected = (e1 + e2) / math.sqrt(2.0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_three_orthogonal_symmetric_mean(self):
        scale = 2.0
        vecs = []
        for i in range(3):
            e = torch.zeros(8); e[i] = scale
            vecs.append(e)
        out = self._merge([(v, 1.0) for v in vecs])
        # Direction: equal cosine to all three inputs; norm: weighted avg = 2.0
        for v in vecs:
            cos = torch.dot(out.flatten(), v.flatten()) / (out.norm() * v.norm())
            self.assertAlmostEqual(cos.item(), 1.0 / math.sqrt(3.0), places=3)
        self.assertAlmostEqual(out.norm().item(), scale, places=3)

    def test_order_independence(self):
        g = torch.Generator().manual_seed(5)
        vs = [torch.randn(64, generator=g) for _ in range(4)]
        ws = [1.0, 0.8, 0.6, 0.4]
        out1 = self._merge(list(zip(vs, ws)))
        perm = [2, 0, 3, 1]
        out2 = self._merge([(vs[i], ws[i]) for i in perm])
        self.assertTrue(torch.allclose(out1, out2, atol=1e-4),
                        f"max diff {(out1 - out2).abs().max().item()}")

    def test_weight_pulls_toward_heavier_vector(self):
        e1 = torch.zeros(8); e1[0] = 1.0
        e2 = torch.zeros(8); e2[1] = 1.0
        e3 = torch.zeros(8); e3[2] = 1.0
        out = self._merge([(e1, 10.0), (e2, 1.0), (e3, 1.0)])
        u = out / out.norm()
        self.assertGreater(torch.dot(u, e1).item(), torch.dot(u, e2).item())
        self.assertGreater(torch.dot(u, e1).item(), 0.8)


class TestModelIdentityTracking(unittest.TestCase):

    def test_same_object_not_changed(self):
        opt = lora_optimizer.LoRAOptimizer()
        a = types.SimpleNamespace()
        self.assertFalse(opt._track_model_identity(a))  # first sighting
        self.assertFalse(opt._track_model_identity(a))  # unchanged

    def test_new_object_changed(self):
        opt = lora_optimizer.LoRAOptimizer()
        a, b = types.SimpleNamespace(), types.SimpleNamespace()
        opt._track_model_identity(a)
        self.assertTrue(opt._track_model_identity(b))

    def test_clip_swap_detected(self):
        opt = lora_optimizer.LoRAOptimizer()
        m = types.SimpleNamespace()
        c1, c2 = types.SimpleNamespace(), types.SimpleNamespace()
        opt._track_model_identity(m, c1)
        self.assertTrue(opt._track_model_identity(m, c2))

    def test_id_reuse_detected_via_dead_weakref(self):
        import weakref, gc as _gc

        class _Obj:
            pass

        opt = lora_optimizer.LoRAOptimizer()
        b = _Obj()
        # Simulate address reuse: stored id matches b but the original
        # object the cache was built from is gone
        dead = weakref.ref(_Obj())
        _gc.collect()
        self.assertIsNone(dead())
        opt._cached_model_id = id(b)
        opt._cached_clip_id = None
        opt._cached_model_ref = dead
        opt._cached_clip_ref = None
        self.assertTrue(opt._track_model_identity(b))


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestAutoStrengthFloor(unittest.TestCase):
    """Explicit auto_strength_floor bounds the reduction on ANY stack;
    only the -1 defaults stay gated on orthogonality."""

    def _scale(self, dot, floor):
        opt = lora_optimizer.LoRAOptimizer()
        preset = lora_optimizer._ARCH_PRESETS["dit"]
        info = opt._compute_branch_auto_scale(
            "Model", [1.0, 1.0], [1.0, 1.0], {(0, 1): dot},
            arch_preset=preset, detected_arch="wan",
            auto_strength_floor=floor, is_full_rank=False)
        return info["scale"]

    def test_explicit_floor_applies_to_aligned_stacks(self):
        # aligned (cos=0.5): unfloored auto scale is 1/sqrt(2+2*0.5) ~ 0.577
        self.assertAlmostEqual(self._scale(0.5, -1.0), 1.0 / math.sqrt(3.0), places=4)
        self.assertAlmostEqual(self._scale(0.5, 0.85), 0.85, places=6)
        self.assertAlmostEqual(self._scale(0.5, 1.0), 1.0, places=6)

    def test_explicit_floor_applies_to_orthogonal_stacks(self):
        self.assertAlmostEqual(self._scale(0.0, 0.85), 0.85, places=6)
        self.assertAlmostEqual(self._scale(0.0, 1.0), 1.0, places=6)

    def test_default_floor_still_gated_on_orthogonality(self):
        # orthogonal + default on wan -> video floor 1.0
        self.assertAlmostEqual(self._scale(0.0, -1.0), 1.0, places=6)
        # aligned + default -> no floor, raw auto scale
        self.assertLess(self._scale(0.5, -1.0), 0.85)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestSingleLoraSkipsCompression(unittest.TestCase):
    """Single-LoRA groups (layers only one LoRA touches) take the exact
    low-rank fast path even when sparsification/refinement is on — those are
    conflict/multi-diff ops that are no-ops on a lone LoRA. This avoids the
    wasteful dense-materialize + compression SVD and emits native factors
    (a LoRAAdapter), while genuine multi-LoRA groups still go dense+compress."""

    def _run(self, **kwargs):
        class FakePatcher:
            def __init__(self, model):
                self.model = model

            def clone(self):
                return FakePatcher(self.model)

            def add_patches(self, patches, strength=1.0, strength_clip=None):
                return list(patches.keys())

        inner = types.SimpleNamespace(
            layer=types.SimpleNamespace(weight=torch.zeros(32, 32)),
            layer2=types.SimpleNamespace(weight=torch.zeros(32, 32)))
        model = FakePatcher(inner)

        def make_lora(seed, scale, prefixes):
            g = torch.Generator().manual_seed(seed)
            d = {}
            for prefix in prefixes:
                d[f"{prefix}.lora_up.weight"] = torch.randn(32, 4, generator=g) * scale
                d[f"{prefix}.lora_down.weight"] = torch.randn(4, 32, generator=g)
                d[f"{prefix}.alpha"] = torch.tensor(4.0)
            return d

        stack = [
            # alias_a is shared (2 LoRAs -> conflict); alias_b is single-LoRA
            {"name": "A", "lora": make_lora(1, 0.1, ("alias_a", "alias_b")), "strength": 1.0},
            {"name": "B", "lora": make_lora(2, 0.1, ("alias_a",)), "strength": 0.8},
        ]
        opt = lora_optimizer.LoRAOptimizer()
        opt._get_model_keys = lambda m: {"alias_a": "layer.weight", "alias_b": "layer2.weight"}
        _, _, _, lora_data = opt.optimize_merge(
            model, stack, 1.0, cache_patches="disabled", **kwargs)
        patches = {}
        for k, v in lora_data["model_patches"].items():
            patches[k[0] if isinstance(k, tuple) else k] = v
        return patches

    def _is_native_rank4(self, patch):
        # The fast path emits the LoRA's native rank-4 factors untouched.
        # The dense+compress path (DARE breaks the low-rank structure, then SVD
        # re-fits) lands at a higher rank or stays a dense ("diff",) tuple.
        return (isinstance(patch, lora_optimizer.LoRAAdapter)
                and patch.weights[1].shape[0] == 4)

    def test_single_lora_skips_compression_under_sparsification(self):
        # sparsification on + compression on: the single-LoRA group must STILL
        # take the low-rank fast path (no dense diff, no sparsification, no
        # compression SVD) — proven by its native rank 4 surviving intact; the
        # multi-LoRA group must NOT (it genuinely needs deconfliction).
        patches = self._run(sparsification="dare", sparsification_density=0.9,
                            patch_compression="smart")
        self.assertTrue(self._is_native_rank4(patches["layer2.weight"]),
                        "single-LoRA group should emit native rank-4 factors")
        self.assertFalse(self._is_native_rank4(patches["layer.weight"]),
                         "multi-LoRA group should still go dense+compress")

    def test_single_lora_lowrank_path_is_size_safe(self):
        # The bypassed single-LoRA patch is stored at the LoRA's native rank (4),
        # never padded up to the rank-64 compression floor.
        patches = self._run(sparsification="dare", sparsification_density=0.9,
                            patch_compression="smart")
        adapter = patches["layer2.weight"]
        mat_up, mat_down = adapter.weights[0], adapter.weights[1]
        self.assertEqual(mat_down.shape[0], 4)  # native rank, not 64
        self.assertEqual(mat_up.shape[1], 4)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestBatchedKarcher(unittest.TestCase):
    """The batched Karcher implementation must match the per-unit reference
    loop it replaced (same math, different reduction order)."""

    @staticmethod
    def _reference_karcher(vecs, weights):
        """The pre-1.9.3 per-unit implementation, kept as the oracle."""
        total_w = sum(weights)
        units = []
        for v, w in zip(vecs, weights):
            vn = v.norm()
            if vn.item() > 1e-12:
                units.append((v / vn, w / total_w))
        m = None
        for u, wn in units:
            m = u * wn if m is None else m.add_(u, alpha=wn)
        m_norm = m.norm()
        m = units[0][0].clone() if m_norm.item() < 1e-8 else m / m_norm
        for _ in range(8):
            tangent = torch.zeros_like(m)
            for u, wn in units:
                cos_i = torch.dot(u, m).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                theta_i = torch.acos(cos_i)
                if theta_i.item() < 1e-7:
                    continue
                coef = wn * (theta_i / torch.sin(theta_i))
                tangent.add_(u - cos_i * m, alpha=coef.item())
            t_norm = tangent.norm()
            if t_norm.item() < 1e-7:
                break
            m = torch.cos(t_norm) * m + (torch.sin(t_norm) / t_norm) * tangent
            m = m / m.norm().clamp(min=1e-12)
        # norm correction as in _merge_diffs
        input_norms = [(v.norm().item(), w) for v, w in zip(vecs, weights)]
        target_norm = sum(n * w for n, w in input_norms) / total_w
        cur = m.norm().item()
        if cur > 1e-8:
            m = m * (target_norm / cur)
        return m

    def test_batched_matches_reference(self):
        opt = lora_optimizer.LoRAOptimizer()
        g = torch.Generator().manual_seed(11)
        for n in (3, 4, 5):
            vecs = [torch.randn(256, generator=g) for _ in range(n)]
            weights = [1.0, 0.8, 0.6, 0.4, 0.9][:n]
            ref = self._reference_karcher([v.clone() for v in vecs], weights)
            out = opt._merge_diffs(list(zip([v.clone() for v in vecs], weights)), "slerp")
            self.assertTrue(torch.allclose(out, ref, atol=1e-5),
                            f"n={n} max diff {(out - ref).abs().max().item()}")

    def test_zero_norm_vector_is_filtered(self):
        opt = lora_optimizer.LoRAOptimizer()
        g = torch.Generator().manual_seed(12)
        vecs = [torch.randn(64, generator=g), torch.zeros(64), torch.randn(64, generator=g)]
        weights = [1.0, 0.7, 0.5]
        out = opt._merge_diffs(list(zip([v.clone() for v in vecs], weights)), "slerp")
        ref = self._reference_karcher([v.clone() for v in vecs], weights)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_all_zero_vectors_return_zero(self):
        opt = lora_optimizer.LoRAOptimizer()
        out = opt._merge_diffs(
            [(torch.zeros(16), 1.0), (torch.zeros(16), 0.5), (torch.zeros(16), 0.3)],
            "slerp")
        self.assertTrue(torch.equal(out, torch.zeros(16)))


@unittest.skipIf(torch is None, "torch is not installed")
def _sd(keys):
    return {key: (torch.zeros(2, 2) if torch is not None else None) for key in keys}


class AnimaDetectionTests(unittest.TestCase):
    """Anima (CircleStone Labs / Cosmos-Predict2 DiT) detection — real key forms."""

    det = staticmethod(lambda s: lora_optimizer.LoRAOptimizer._detect_architecture(s))

    def test_diffusion_pipe_comfyui_form(self):
        s = _sd(["diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
                 "diffusion_model.blocks.0.cross_attn.output_proj.lora_up.weight",
                 "diffusion_model.blocks.0.mlp.layer1.lora_down.weight",
                 "diffusion_model.llm_adapter.blocks.0.cross_attn.k_proj.lora_down.weight"])
        self.assertEqual(self.det(s), "anima")

    def test_kohya_form(self):
        s = _sd(["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight",
                 "lora_unet_blocks_0_cross_attn_output_proj.lora_up.weight",
                 "lora_unet_blocks_0_mlp_layer1.lora_down.weight",
                 "lora_te_layers_0_self_attn_q_proj.lora_down.weight"])
        self.assertEqual(self.det(s), "anima")

    def test_no_collision_acestep_wan_ltx(self):
        # These previously-supported archs must still win, not get mistaken for Anima.
        ace = _sd(["diffusion_model.layers.0.self_attn.q_proj.lora_down.weight",
                   "diffusion_model.layers.0.cross_attn.k_proj.lora_up.weight"])
        wan = _sd(["diffusion_model.blocks.0.self_attn.q.a",
                   "diffusion_model.blocks.0.cross_attn.k.b",
                   "diffusion_model.blocks.0.ffn.0.c"])
        ltx = _sd(["transformer_blocks.0.attn1.to_q.a", "adaln_single.linear.b"])
        self.assertEqual(self.det(ace), "acestep")
        self.assertEqual(self.det(wan), "wan")
        self.assertEqual(self.det(ltx), "ltx")


class Krea2DetectionTests(unittest.TestCase):
    """Krea 2 (krea/Krea-2, from-scratch single-stream image DiT) detection."""

    det = staticmethod(lambda s: lora_optimizer.LoRAOptimizer._detect_architecture(s))

    def test_comfy_native_form(self):
        # Mirrors the official Comfy-Org/Krea-2 rank-64 LoRA: GQA attn.wq/wk/wv/wo
        # + sigmoid attn.gate + SwiGLU mlp.gate/up/down under diffusion_model.blocks.N
        s = _sd(["diffusion_model.blocks.0.attn.wq.lora_up.weight",
                 "diffusion_model.blocks.0.attn.wk.lora_up.weight",
                 "diffusion_model.blocks.0.attn.wv.lora_up.weight",
                 "diffusion_model.blocks.0.attn.wo.lora_up.weight",
                 "diffusion_model.blocks.0.attn.gate.lora_up.weight",
                 "diffusion_model.blocks.0.mlp.gate.lora_up.weight",
                 "diffusion_model.blocks.0.mlp.up.lora_up.weight",
                 "diffusion_model.blocks.0.mlp.down.lora_up.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_kohya_underscore_form(self):
        s = _sd(["lora_unet_blocks_0_attn_wq.lora_down.weight",
                 "lora_unet_blocks_0_mlp_gate.lora_up.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_trainer_diffusion_model_form(self):
        # The "krea_2" trainer: diffusion_model.transformer_blocks.N.attn.to_*
        # with a sigmoid attn.to_gate. Must NOT be mistaken for ACE-Step (which
        # also matches transformer_blocks.N.attn.to_q).
        s = _sd(["diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.attn.to_v.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.attn.to_gate.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.ff.gate.lora_A.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_diffusers_transformer_form(self):
        # diffusers form: transformer.transformer_blocks.N.attn.to_* + to_gate
        # + text_fusion. Must NOT be mistaken for Qwen-Image (transformer.transformer_blocks).
        s = _sd(["transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
                 "transformer.transformer_blocks.0.attn.to_gate.lora_A.weight",
                 "transformer.transformer_blocks.0.ff.gate.lora_A.weight",
                 "transformer.text_fusion.refiner_blocks.0.attn.to_gate.lora_A.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_gate_plus_mlp_gate_fallback(self):
        # Backup discriminator when attn.w{q,k,v,o} isn't in a partial LoRA.
        s = _sd(["diffusion_model.blocks.3.attn.gate.lora_up.weight",
                 "diffusion_model.blocks.3.mlp.gate.lora_up.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_no_collision_with_wan_flux_qwen(self):
        # Krea is checked before WAN; must not steal these, and they must not steal Krea.
        wan = _sd(["diffusion_model.blocks.0.self_attn.q.a",
                   "diffusion_model.blocks.0.cross_attn.k.b",
                   "diffusion_model.blocks.0.ffn.0.c"])
        flux = _sd(["diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight"])
        qwen = _sd(["transformer.transformer_blocks.0.attn.to_q.a",
                    "transformer.transformer_blocks.0.img_mlp.net.a"])
        self.assertEqual(self.det(wan), "wan")
        self.assertEqual(self.det(flux), "flux")
        self.assertEqual(self.det(qwen), "qwen_image")

    def test_ltx2_gate_logits_not_stolen_by_krea2(self):
        # Regression: LTX-2 (LTXV 2.3) has dual video/audio + cross-modal attention
        # with '*_attn.to_gate_logits'. The substring '..._attn.to_gate' must NOT
        # trip krea2's gate detection (krea2's gate is a leaf followed by '.', LTX-2's
        # is 'to_gate_logits' with '_' after). These LTX LoRAs were mis-detected as
        # krea2 -> wrong normalization (transformer_blocks->blocks) -> nothing merged.
        s = _sd([
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_gate_logits.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn2.to_k.lora_A.weight",
            "diffusion_model.transformer_blocks.0.audio_to_video_attn.to_gate_logits.lora_A.weight",
            "diffusion_model.transformer_blocks.0.video_to_audio_attn.to_v.lora_A.weight",
        ])
        self.assertNotEqual(self.det(s), "krea2")
        self.assertEqual(self.det(s), "ltx")


class Krea2NormalizationTests(unittest.TestCase):
    """Both Krea 2 trainer forms normalize to the model-native diffusion_model.* keys.
    Mappings are shape-verified against krea2_turbo_bf16 (224/224, 264/264)."""

    norm = staticmethod(lambda s: lora_optimizer.LoRAOptimizer._normalize_keys_krea2(s))

    def test_trainer_form_attn_and_ff(self):
        n = self.norm(_sd([
            "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_B.weight",
            "diffusion_model.transformer_blocks.0.attn.to_gate.lora_A.weight",
            "diffusion_model.transformer_blocks.0.ff.gate.lora_B.weight",
            "diffusion_model.transformer_blocks.0.ff.down.lora_A.weight",
        ]))
        self.assertIn("diffusion_model.blocks.0.attn.wq.lora_A.weight", n)
        self.assertIn("diffusion_model.blocks.0.attn.wo.lora_B.weight", n)
        self.assertIn("diffusion_model.blocks.0.attn.gate.lora_A.weight", n)
        self.assertIn("diffusion_model.blocks.0.mlp.gate.lora_B.weight", n)
        self.assertIn("diffusion_model.blocks.0.mlp.down.lora_A.weight", n)

    def test_diffusers_form_transformer_prefix_and_txtfusion(self):
        n = self.norm(_sd([
            "transformer.transformer_blocks.5.attn.to_k.lora_A.weight",
            "transformer.text_fusion.layerwise_blocks.2.attn.to_v.lora_B.weight",
            "transformer.text_fusion.refiner_blocks.0.mlp.gate.lora_A.weight",
            "transformer.text_fusion.projector.lora_A.weight",
        ]))
        self.assertIn("diffusion_model.blocks.5.attn.wk.lora_A.weight", n)
        self.assertIn("diffusion_model.txtfusion.layerwise_blocks.2.attn.wv.lora_B.weight", n)
        self.assertIn("diffusion_model.txtfusion.refiner_blocks.0.mlp.gate.lora_A.weight", n)
        self.assertIn("diffusion_model.txtfusion.projector.lora_A.weight", n)

    def test_named_non_block_projections(self):
        n = self.norm(_sd([
            "transformer.img_in.lora_A.weight",
            "transformer.final_layer.linear.lora_B.weight",
            "transformer.time_mod_proj.lora_A.weight",
            "transformer.time_embed.linear_1.lora_A.weight",
            "transformer.time_embed.linear_2.lora_B.weight",
            "transformer.txt_in.linear_1.lora_A.weight",
            "transformer.txt_in.linear_2.lora_B.weight",
        ]))
        for expect in [
            "diffusion_model.first.lora_A.weight",
            "diffusion_model.last.linear.lora_B.weight",
            "diffusion_model.tproj.1.lora_A.weight",
            "diffusion_model.tmlp.0.lora_A.weight",
            "diffusion_model.tmlp.2.lora_B.weight",
            "diffusion_model.txtmlp.1.lora_A.weight",
            "diffusion_model.txtmlp.3.lora_B.weight",
        ]:
            self.assertIn(expect, n)

    def test_alpha_and_idempotent(self):
        # alpha keys are remapped too; already-canonical keys are unchanged.
        n = self.norm(_sd([
            "diffusion_model.transformer_blocks.0.attn.to_q.alpha",
            "diffusion_model.blocks.0.attn.wq.lora_A.weight",
        ]))
        self.assertIn("diffusion_model.blocks.0.attn.wq.alpha", n)
        self.assertIn("diffusion_model.blocks.0.attn.wq.lora_A.weight", n)
        # idempotent: re-normalizing canonical keys is a no-op
        self.assertEqual(set(self.norm(n).keys()), set(n.keys()))


@unittest.skipIf(torch is None, "torch is not installed")
class AnimaNormalizationTests(unittest.TestCase):
    """All trainer forms normalize to canonical diffusion_model.blocks.N.* keys."""

    norm = staticmethod(lambda s: lora_optimizer.LoRAOptimizer._normalize_keys_anima(s))

    def test_diffusion_pipe_passthrough(self):
        s = _sd(["diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight"])
        self.assertIn("diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight", self.norm(s))

    def test_kohya_restores_dots(self):
        n = self.norm(_sd([
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight",
            "lora_unet_blocks_0_cross_attn_output_proj.lora_up.weight",
            "lora_unet_blocks_0_mlp_layer1.lora_down.weight",
            "lora_unet_blocks_0_adaln_modulation_self_attn_1.lora_up.weight",
            "lora_unet_llm_adapter_blocks_0_cross_attn_q_proj.lora_down.weight",
        ]))
        for expect in [
            "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
            "diffusion_model.blocks.0.cross_attn.output_proj.lora_up.weight",
            "diffusion_model.blocks.0.mlp.layer1.lora_down.weight",
            "diffusion_model.blocks.0.adaln_modulation_self_attn.1.lora_up.weight",
            "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_down.weight",
        ]:
            self.assertIn(expect, n)

    def test_diffusers_to_canonical(self):
        n = self.norm(_sd([
            "transformer_blocks.0.attn1.to_q.lora_down.weight",
            "transformer_blocks.0.attn2.to_out.0.lora_up.weight",
            "transformer_blocks.0.ff.net.0.proj.lora_down.weight",
        ]))
        for expect in [
            "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
            "diffusion_model.blocks.0.cross_attn.output_proj.lora_up.weight",
            "diffusion_model.blocks.0.mlp.layer1.lora_down.weight",
        ]:
            self.assertIn(expect, n)

    def test_three_forms_converge(self):
        a = self.norm(_sd(["diffusion_model.blocks.0.self_attn.q_proj.w"]))
        b = self.norm(_sd(["lora_unet_blocks_0_self_attn_q_proj.w"]))
        c = self.norm(_sd(["transformer_blocks.0.attn1.to_q.w"]))
        key = "diffusion_model.blocks.0.self_attn.q_proj.w"
        self.assertIn(key, a)
        self.assertIn(key, b)
        self.assertIn(key, c)
