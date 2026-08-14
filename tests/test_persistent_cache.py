import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

_PLUGIN_DIR = Path(__file__).resolve().parents[1]
_COMFYUI_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PLUGIN_DIR))
sys.path.insert(0, str(_COMFYUI_DIR))

import persistent_cache


class _Interrupted(Exception):
    pass


class _Controller:
    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.checks = 0

    def check(self):
        self.checks += 1
        if self.fail_after is not None and self.checks >= self.fail_after:
            raise _Interrupted()


class PersistentPatchCacheTests(unittest.TestCase):
    def _payload(self):
        model_patches = {
            "model.block.weight": persistent_cache.LoRAAdapter(
                set(), (torch.randn(4, 2), torch.randn(2, 3), 2.0,
                        None, None, None)),
            "model.loha.weight": persistent_cache.LoHaAdapter(
                set(), (torch.randn(4, 2), torch.randn(2, 3),
                        torch.randn(4, 2), torch.randn(2, 3),
                        2.0, None, None)),
            "model.lokr.weight": persistent_cache.LoKrAdapter(
                set(), (torch.randn(2, 2), torch.randn(2, 2),
                        None, None, None, None, 2.0, None, None)),
            ("model.qkv.weight", (4, 2)): (
                "diff", (torch.randn(2, 3),)),
        }
        clip_patches = {
            "clip.weight": ("diff", (torch.randn(3, 2),)),
        }
        lora_data = {
            "model_patches": model_patches,
            "clip_patches": clip_patches,
            "key_map": {
                "model.block.weight": ["alias.one", "alias.two"],
                ("model.qkv.weight", (4, 2)): ["alias.q"],
            },
            "output_strength": 0.8,
            "clip_strength": 0.6,
            "suggested_max_strength": 0.8,
            "sum_rank": 2,
            "per_prefix_decisions": {"model.block": {"mode": "slerp"}},
            "merge_metadata": {"architecture": "krea2"},
        }
        return model_patches, clip_patches, lora_data

    def test_round_trip_preserves_patch_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = persistent_cache.PersistentPatchCache(
                cache_dir=directory, max_cache_gb=1)
            model_patches, clip_patches, lora_data = self._payload()
            path = cache.save(
                "abc", model_patches, clip_patches, "report", lora_data,
                _Controller())
            self.assertTrue(os.path.isfile(path))

            loaded = persistent_cache.PersistentPatchCache(
                cache_dir=directory, max_cache_gb=1).load("abc", _Controller())
            self.assertEqual(loaded["report"], "report")
            self.assertEqual(set(loaded["model_patches"]), set(model_patches))
            loaded_adapter = loaded["model_patches"]["model.block.weight"]
            self.assertIsInstance(loaded_adapter, persistent_cache.LoRAAdapter)
            self.assertIsInstance(
                loaded["model_patches"]["model.loha.weight"], persistent_cache.LoHaAdapter)
            self.assertIsInstance(
                loaded["model_patches"]["model.lokr.weight"], persistent_cache.LoKrAdapter)
            for actual, expected in zip(loaded_adapter.weights[:2],
                                        model_patches["model.block.weight"].weights[:2]):
                self.assertTrue(torch.equal(actual, expected))
            offset_patch = loaded["model_patches"][("model.qkv.weight", (4, 2))]
            self.assertEqual(offset_patch[0], "diff")
            self.assertTrue(torch.equal(
                offset_patch[1][0],
                model_patches[("model.qkv.weight", (4, 2))][1][0]))
            self.assertEqual(loaded["lora_data"]["key_map"], lora_data["key_map"])
            self.assertEqual(loaded["lora_data"]["output_strength"], 0.8)

    def test_corrupt_cache_is_removed_and_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = persistent_cache.PersistentPatchCache(
                cache_dir=directory, max_cache_gb=1)
            path = cache.cache_path("corrupt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a safetensors file")
            self.assertIsNone(cache.load("corrupt", _Controller()))
            self.assertFalse(path.exists())

    def test_source_digest_reuses_hash_and_invalidates_modified_file(self):
        with tempfile.TemporaryDirectory() as directory:
            lora_path = os.path.join(directory, "source.safetensors")
            with open(lora_path, "wb") as handle:
                handle.write(b"first")
            cache = persistent_cache.PersistentPatchCache(
                cache_dir=directory, max_cache_gb=1)
            original_resolver = persistent_cache.folder_paths.get_full_path
            persistent_cache.folder_paths.get_full_path = (
                lambda _kind, _name: lora_path)
            try:
                first = cache.source_digest(
                    [("source.safetensors", 1.0, 1.0)], _Controller())
                second = cache.source_digest(
                    [("source.safetensors", 1.0, 1.0)], _Controller())
                self.assertEqual(first, second)
                with open(lora_path, "wb") as handle:
                    handle.write(b"second-version")
                os.utime(lora_path, None)
                third = cache.source_digest(
                    [("source.safetensors", 1.0, 1.0)], _Controller())
                self.assertNotEqual(first, third)
            finally:
                persistent_cache.folder_paths.get_full_path = original_resolver

    def test_cancelled_write_is_not_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = persistent_cache.PersistentPatchCache(
                cache_dir=directory, max_cache_gb=1)
            model_patches, clip_patches, lora_data = self._payload()
            with self.assertRaises(_Interrupted):
                cache.save(
                    "cancelled", model_patches, clip_patches, "report",
                    lora_data, _Controller(fail_after=3))
            self.assertFalse(cache.cache_path("cancelled").exists())
            self.assertEqual(list(cache.cache_dir.glob(".*.tmp.safetensors")), [])

    def test_interrupt_after_atomic_replace_removes_committed_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = persistent_cache.PersistentPatchCache(
                cache_dir=directory, max_cache_gb=1)
            model_patches, clip_patches, lora_data = self._payload()
            final_path = cache.cache_path("cancel-after-replace")

            class FailAfterCommit:
                def check(self):
                    if final_path.exists():
                        raise _Interrupted()

            with self.assertRaises(_Interrupted):
                cache.save(
                    "cancel-after-replace", model_patches, clip_patches,
                    "report", lora_data, FailAfterCommit())
            self.assertFalse(final_path.exists())

    def test_large_cache_write_checks_cancellation_between_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = persistent_cache.PersistentPatchCache(
                cache_dir=directory, max_cache_gb=1)
            dense = torch.ones(5 * 1024 * 1024, dtype=torch.float32)
            lora_data = {
                "model_patches": {"large.weight": ("diff", (dense,))},
                "clip_patches": {},
                "key_map": {},
                "output_strength": 1.0,
                "clip_strength": 1.0,
            }
            with self.assertRaises(_Interrupted):
                cache.save(
                    "large-cancelled", lora_data["model_patches"], {},
                    "report", lora_data, _Controller(fail_after=6))
            self.assertFalse(cache.cache_path("large-cancelled").exists())
            self.assertEqual(list(cache.cache_dir.glob(".*.tmp.safetensors")), [])

    def test_virtual_payload_has_no_persistent_identity(self):
        cache = persistent_cache.PersistentPatchCache(
            cache_dir=tempfile.gettempdir(), max_cache_gb=1)
        with self.assertRaises(persistent_cache.PersistentCacheUnsupported):
            cache.source_digest([{
                "name": "virtual",
                "strength": 1.0,
                "_precomputed_diffs": {"key": torch.ones(1)},
            }], _Controller())


class OptimizerPersistentCacheIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tests import test_lora_optimizer as helper
        except ImportError:
            import test_lora_optimizer as helper
        cls.optimizer_module = helper.lora_optimizer

    @staticmethod
    def _model():
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, 1))

        class Patcher:
            def __init__(self):
                self.model = Inner()
                self.size = 4
                self.applied = []

            def model_size(self):
                return 4

            def clone(self):
                clone = Patcher()
                clone.applied = self.applied
                return clone

            def add_patches(self, patches, strength):
                self.applied.append((patches, strength))

        return Patcher()

    def test_restart_cache_hit_skips_lora_loading_analysis_and_merge(self):
        module = self.optimizer_module
        with tempfile.TemporaryDirectory() as directory:
            cache = persistent_cache.PersistentPatchCache(directory)
            patch = ("diff", (torch.tensor([[2.0]]),))
            lora_data = {
                "model_patches": {"layer.weight": patch},
                "clip_patches": {},
                "key_map": {"layer.weight": "layer"},
                "output_strength": 0.8,
                "clip_strength": 0.8,
            }
            cache.save("restart-key", lora_data["model_patches"], {}, "cached", lora_data, _Controller())

            optimizer = module._LoRAOptimizerEngine()
            optimizer._persistent_cache = cache
            model = self._model()
            with mock.patch.object(optimizer, "_persistent_cache_key", return_value="restart-key"), \
                    mock.patch.object(optimizer, "_normalize_stack", side_effect=AssertionError("cache miss")):
                output = optimizer.optimize_merge(
                    model, [("a.safetensors", 1.0, 1.0), ("b.safetensors", 1.0, 1.0)], 1.0,
                    cache_patches="disabled", persistent_cache="enabled")

            self.assertIn("Persistent Cache", output[2])
            self.assertEqual(len(model.applied), 1)
            self.assertEqual(model.applied[0][1], 0.8)

    def test_disabled_switch_never_reads_or_writes_disk_cache(self):
        module = self.optimizer_module
        optimizer = module._LoRAOptimizerEngine()
        with mock.patch.object(optimizer, "_persistent_cache_key") as identity, \
                mock.patch.object(optimizer, "_normalize_stack", return_value=[]):
            optimizer.optimize_merge(
                self._model(), [("a.safetensors", 1.0, 1.0), ("b.safetensors", 1.0, 1.0)], 1.0,
                cache_patches="disabled", persistent_cache="disabled")
        identity.assert_not_called()


if __name__ == "__main__":
    unittest.main()
