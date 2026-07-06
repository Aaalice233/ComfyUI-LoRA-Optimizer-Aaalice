"""Tests for the inline chain-filter optimizer node."""
import struct
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

    def test_non_5_tuple_entry_passes_through(self):
        # older/nonstandard third-party nodes may store short entries — the
        # classifier must not crash on unpack, just pass them through
        e = (1.0, _adapter(), 1.0)
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_padded_diff_passes_through(self):
        # comfy pads the BASE weight at apply time for this shape — we cannot
        # faithfully expand it to a bare diff tensor
        e = _entry(1.0, ("diff", (torch.randn(4, 4), {"pad_weight": True})))
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_malformed_diff_passes_through(self):
        e = _entry(1.0, ("diff", None))
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

    def test_empty_patches(self):
        self.assertEqual(self._reconstruct({}), [])

    def test_non_5_tuple_entry_ignored_without_raising(self):
        patches = _chain_patches((0.8, {"a": _adapter()}))
        patches["a"].append((1.0, _adapter(), 1.0))  # nonstandard 3-tuple
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]["entries"]), {"a"})

    def test_interleaved_collision_no_fragmentation(self):
        # Distinct-strength loader X on {a} first, then two interned-strength
        # loaders B and C both patching {a, b}. The collision sub-gid must be
        # the per-target-key collision ORDINAL — splitting on the absolute
        # per-key position fragments this into 4 groups [a],[a,b],[a],[b]
        # because X shifts the positions on "a" but not on "b".
        d = struct.unpack("d", struct.pack("d", 0.6))[0]
        s = 1.0
        pX = _adapter()
        pB_a, pB_b = _adapter(), _adapter()
        pC_a, pC_b = _adapter(), _adapter()
        patches = {
            "a": [_entry(d, pX), _entry(s, pB_a), _entry(s, pC_a)],
            "b": [_entry(s, pB_b), _entry(s, pC_b)],
        }
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 3)
        self.assertEqual(set(groups[0]["entries"]), {"a"})
        self.assertEqual(set(groups[1]["entries"]), {"a", "b"})
        self.assertEqual(set(groups[2]["entries"]), {"a", "b"})
        # ordinal-k entries belong to the k-th colliding call on EVERY key
        self.assertIs(groups[1]["entries"]["a"], pB_a)
        self.assertIs(groups[1]["entries"]["b"], pB_b)
        self.assertIs(groups[2]["entries"]["a"], pC_a)
        self.assertIs(groups[2]["entries"]["b"], pC_b)

    def test_collision_subgroup_reordered_by_precedence(self):
        # Chain: two interned-strength calls A, B on "a", then distinct call D
        # on {z, a}. Dict iteration starts at "z", so D's group is CREATED
        # first; both collision groups (base + ordinal sub-group) must be
        # sorted forward past D via the shared-key "a" positions — this
        # exercises the insertion sort on a collision-created sub-group.
        s = 1.0
        d = struct.unpack("d", struct.pack("d", 0.5))[0]
        pA, pB, pD_a, pD_z = _adapter(), _adapter(), _adapter(), _adapter()
        patches = {
            "z": [_entry(d, pD_z)],
            "a": [_entry(s, pA), _entry(s, pB), _entry(d, pD_a)],
        }
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 3)
        self.assertIs(groups[0]["entries"]["a"], pA)
        self.assertIs(groups[1]["entries"]["a"], pB)
        self.assertEqual(set(groups[2]["entries"]), {"z", "a"})


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


def _slot(enabled=True, strength=1.0, model_strength=1.0, clip_strength=1.0,
          conflict_mode="all", key_filter="all", preserve=False):
    return dict(enabled=enabled, strength=strength, model_strength=model_strength,
                clip_strength=clip_strength, conflict_mode=conflict_mode,
                key_filter=key_filter, preserve=preserve)


class TestChainStackBuild(unittest.TestCase):
    def _build(self, model_groups, clip_groups, slots, visibility="simple"):
        return lora_optimizer.LoRAOptimizerInline._chain_groups_to_stack(
            model_groups, clip_groups, slots, visibility)

    def test_basic_item_schema(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        stack = self._build(mg, [], [_slot()])
        item = stack[0]
        self.assertTrue(item["_precomputed_diffs"])
        self.assertAlmostEqual(item["strength"], 0.8)     # loader strength kept
        self.assertIsNone(item["clip_strength"])
        self.assertEqual(item["conflict_mode"], "all")
        self.assertIn("a", item["lora"])

    def test_simple_mode_multiplier(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        stack = self._build(mg, [], [_slot(strength=0.5)])
        self.assertAlmostEqual(stack[0]["strength"], 0.4)  # 0.8 loader × 0.5 slot

    def test_advanced_mode_split_multipliers(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot(model_strength=0.5, clip_strength=2.0)],
                            visibility="advanced")
        self.assertAlmostEqual(stack[0]["strength"], 0.4)        # 0.8 × 0.5
        self.assertAlmostEqual(stack[0]["clip_strength"], 1.2)   # 0.6 × 2.0
        self.assertIn("te.a", stack[0]["lora"])                  # clip keys merged in

    def test_disabled_slot_excluded(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        stack = self._build(mg, [], [_slot(enabled=False), _slot()])
        self.assertEqual(len(stack), 1)
        self.assertAlmostEqual(stack[0]["strength"], 0.5)

    def test_missing_slots_get_defaults(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        stack = self._build(mg, [], [_slot(preserve=True)])   # only 1 slot for 2 loras
        self.assertEqual(len(stack), 2)
        self.assertTrue(stack[0]["preserve"])
        self.assertFalse(stack[1]["preserve"])

    def test_leftover_clip_groups_become_clip_only_items(self):
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build([], cg, [])
        self.assertEqual(len(stack), 1)
        self.assertIn("te.a", stack[0]["lora"])


if __name__ == "__main__":
    unittest.main()
