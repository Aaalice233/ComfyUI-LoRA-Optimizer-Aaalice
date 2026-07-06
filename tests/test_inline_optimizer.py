"""Tests for the inline chain-filter optimizer node."""
import struct
import types
import unittest

import torch

# Reuse the stub installer / module instance from the main test module.
from tests.test_lora_optimizer import lora_optimizer


def _adapter(rank=4, out_dim=8, in_dim=8):
    """Minimal LoRAAdapter-like payload the engine can expand."""
    return lora_optimizer.LoRAAdapter(
        loaded_keys=set(),
        weights=(torch.randn(out_dim, rank), torch.randn(rank, in_dim),
                 float(rank), None, None, None),
    )


def _entry(strength, payload, strength_model=1.0, offset=None, function=None):
    """A ModelPatcher patch-list entry, shaped like model_patcher.py:807."""
    return (strength, payload, strength_model, offset, function)


class TestPatchClassification(unittest.TestCase):
    def test_adapter_is_capturable(self):
        e = _entry(0.8, _adapter())
        self.assertTrue(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_diff_tuple_is_capturable(self):
        e = _entry(1.0, ("diff", (torch.randn(4, 4),)))
        self.assertTrue(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_set_tuple_passes_through(self):
        e = _entry(1.0, ("set", (torch.randn(4, 4),)))
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_nonunit_strength_model_passes_through(self):
        e = _entry(1.0, _adapter(), strength_model=0.5)
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_function_entry_passes_through(self):
        e = _entry(1.0, _adapter(), function=lambda w: w)
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_unknown_object_passes_through(self):
        e = _entry(1.0, object())
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))


def _chain_patches(*loras):
    """Simulate a loader chain: each lora is {key: payload} applied at a
    strength. Distinct float objects per call, entries appended in order —
    exactly what ModelPatcher.add_patches does."""
    patches = {}
    seen_ids = set()
    for strength_value, lora in loras:
        # struct round-trip mints a FRESH float object even for values whose
        # literals CPython folds/interns (e.g. 1.0 used twice in one test).
        # float(x) would return the SAME object for an exact float input.
        s = struct.unpack("d", struct.pack("d", strength_value))[0]
        assert id(s) not in seen_ids, "float object reused across simulated calls"
        seen_ids.add(id(s))
        for key, payload in lora.items():
            patches.setdefault(key, []).append(_entry(s, payload))
    return patches


class TestChainGroupReconstruction(unittest.TestCase):
    def _reconstruct(self, patches):
        return lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(patches)

    def test_single_lora(self):
        patches = _chain_patches((0.8, {"a": _adapter(), "b": _adapter()}))
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(groups[0]["strength"], 0.8)
        self.assertEqual(set(groups[0]["entries"]), {"a", "b"})

    def test_two_loras_chain_order(self):
        patches = _chain_patches(
            (0.8, {"a": _adapter(), "b": _adapter()}),
            (0.5, {"a": _adapter(), "c": _adapter()}),
        )
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(groups[0]["strength"], 0.8)
        self.assertAlmostEqual(groups[1]["strength"], 0.5)
        self.assertEqual(set(groups[1]["entries"]), {"a", "c"})

    def test_subset_alignment(self):
        # A patches attn only; B patches attn+mlp. Naive index-as-identity
        # would misattribute B's mlp entry (position 0 there) to A.
        patches = _chain_patches(
            (0.8, {"attn": _adapter()}),
            (0.5, {"attn": _adapter(), "mlp": _adapter()}),
        )
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 2)
        self.assertEqual(set(groups[0]["entries"]), {"attn"})
        self.assertEqual(set(groups[1]["entries"]), {"attn", "mlp"})

    def test_same_strength_value_distinct_objects(self):
        patches = _chain_patches(
            (1.0, {"a": _adapter()}),
            (1.0, {"a": _adapter(), "b": _adapter()}),
        )
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 2)

    def test_shared_float_object_fallback(self):
        # Pathological: two calls share ONE float object (interning). The
        # id-group then holds two entries on key "a" — must split by per-key
        # order instead of returning a corrupt group.
        s = 1.0
        patches = {}
        for lora in ({"a": _adapter()}, {"a": _adapter(), "b": _adapter()}):
            for key, payload in lora.items():
                patches.setdefault(key, []).append(_entry(s, payload))
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 2)
        for g in groups:
            keys = [k for k in g["entries"]]
            self.assertEqual(len(keys), len(set(keys)))

    def test_offset_entries_get_tuple_keys(self):
        off = (0, 0, 4)
        patches = {"qkv": [_entry(0.7, _adapter(), offset=off)]}
        groups = self._reconstruct(patches)
        self.assertEqual(list(groups[0]["entries"]), [("qkv", off)])

    def test_noncapturable_entries_ignored(self):
        patches = _chain_patches((0.8, {"a": _adapter()}))
        patches["a"].append(_entry(1.0, ("set", (torch.zeros(2, 2),))))
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]["entries"]), {"a"})


class _FakePatcher:
    """Minimal ModelPatcher stand-in: ordered patch lists + clone()."""
    def __init__(self, patches=None):
        self.patches = patches if patches is not None else {}
        self.patches_uuid = object()

    def clone(self):
        return _FakePatcher({k: v[:] for k, v in self.patches.items()})


class TestStripCaptured(unittest.TestCase):
    def test_strips_only_captured_entries(self):
        keep = _entry(1.0, ("set", (torch.zeros(2, 2),)))
        patches = _chain_patches((0.8, {"a": _adapter(), "b": _adapter()}))
        patches["a"].append(keep)
        patcher = _FakePatcher(patches)
        groups = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(patcher.patches)
        clone = patcher.clone()
        lora_optimizer._LoRAMergeBase._strip_captured_entries(clone, groups)
        self.assertEqual(clone.patches, {"a": [keep]})
        self.assertEqual(len(patcher.patches["a"]), 2)  # original untouched

    def test_uuid_regenerated(self):
        patcher = _FakePatcher(_chain_patches((0.8, {"a": _adapter()})))
        groups = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(patcher.patches)
        clone = patcher.clone()
        before = clone.patches_uuid
        lora_optimizer._LoRAMergeBase._strip_captured_entries(clone, groups)
        self.assertNotEqual(clone.patches_uuid, before)


if __name__ == "__main__":
    unittest.main()
