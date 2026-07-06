"""Tests for the inline chain-filter optimizer node."""
import struct
import types
import unittest
import uuid
from unittest import mock

import torch

# Reuse the stub installer / module instance from the main test module.
from tests.test_lora_optimizer import lora_optimizer


def _adapter(rank=4, out_dim=8, in_dim=8, up=None, down=None):
    """Minimal LoRAAdapter-like payload the engine can expand. Explicit
    up/down matrices make the expanded diff deterministic (alpha == rank, so
    the diff is exactly up @ down)."""
    if up is None:
        up = torch.randn(out_dim, rank)
    if down is None:
        down = torch.randn(rank, in_dim)
    return lora_optimizer.LoRAAdapter(
        loaded_keys=set(),
        weights=(up, down, float(rank), None, None, None),
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

    def test_leftover_strength_is_clip_product_nonzero(self):
        # the engine drops strength == 0 items (active_loras filter in
        # optimize_merge) — clip-only items must ride on the clip product
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build([], cg, [])
        self.assertNotEqual(stack[0]["strength"], 0.0)
        self.assertAlmostEqual(stack[0]["strength"], 0.6)
        self.assertAlmostEqual(stack[0]["clip_strength"], 0.6)

    def test_zeroed_model_branch_keeps_live_clip_branch(self):
        # stock-UI case: LoraLoader with strength_model=0, strength_clip=1.0
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot(model_strength=0.0, clip_strength=1.0)],
                            visibility="advanced")
        self.assertEqual(len(stack), 1)
        self.assertNotIn("a", stack[0]["lora"])       # unet keys dropped
        self.assertIn("te.a", stack[0]["lora"])
        self.assertAlmostEqual(stack[0]["strength"], 0.6)      # clip product
        self.assertAlmostEqual(stack[0]["clip_strength"], 0.6)

    def test_both_branches_zeroed_item_excluded(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot(model_strength=0.0, clip_strength=0.0)],
                            visibility="advanced")
        self.assertEqual(stack, [])

    def test_leftover_after_model_groups_uses_own_slot_index(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()}), (0.4, {"te.b": _adapter()})))
        stack = self._build(mg, cg, [_slot(), _slot()])
        self.assertEqual(len(stack), 2)
        self.assertEqual(stack[1]["name"], "chain lora #2 (clip-only)")
        self.assertIn("te.b", stack[1]["lora"])
        self.assertAlmostEqual(stack[1]["strength"], 0.4)
        # slot index j (not 0) gates the leftover — disabling slot 2 drops it
        stack = self._build(mg, cg, [_slot(), _slot(enabled=False)])
        self.assertEqual(len(stack), 1)
        self.assertNotIn("(clip-only)", stack[0]["name"])

    def test_simple_visibility_scales_paired_clip(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot(strength=0.5)])   # simple visibility
        self.assertAlmostEqual(stack[0]["strength"], 0.4)          # 0.8 × 0.5
        self.assertAlmostEqual(stack[0]["clip_strength"], 0.3)     # 0.6 × 0.5

    def test_extra_slots_ignored(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        stack = self._build(mg, [], [_slot(), _slot(strength=9.0), _slot(enabled=False)])
        self.assertEqual(len(stack), 1)
        self.assertAlmostEqual(stack[0]["strength"], 0.8)


class _FakeCLIP:
    def __init__(self, patcher):
        self.patcher = patcher
        self.cond_stage_model = types.SimpleNamespace()

    def clone(self):
        return _FakeCLIP(self.patcher.clone())


class TestInlineExecute(unittest.TestCase):
    def _node(self):
        return lora_optimizer.LoRAOptimizerInline()

    def _model(self, patches):
        m = _FakePatcher(patches)
        m.model = types.SimpleNamespace()
        return m

    @staticmethod
    def _out(result):
        return result["result"] if isinstance(result, dict) else result

    def _capture_merge(self, node, seen):
        """Mock optimize_merge, recording every kwarg + the stack/model."""
        def fake_merge(m, stack, output_strength, **kw):
            seen["model"] = m
            seen["stack"] = stack
            seen["output_strength"] = output_strength
            seen.update(kw)
            return (m, kw.get("clip"), "engine report", None, None)
        node.optimize_merge = fake_merge

    def test_no_lora_patches_passthrough(self):
        model = self._model({})
        node = self._node()
        result = node.execute_inline(model, output_strength=1.0)
        out = self._out(result)
        self.assertIs(out[0], model)          # unchanged object back
        self.assertIn("No LoRA patches", out[2])
        self.assertIn("AFTER your Load LoRA nodes", out[2])
        self.assertIsNone(out[3])
        self.assertIsNone(out[4])

    def test_non_mergeable_only_patches_reported_honestly(self):
        # Patches exist but none are capturable (OFT/"set"/function entries):
        # "place this node AFTER your loaders" would be WRONG advice.
        patches = {"a": [_entry(1.0, ("set", (torch.zeros(2, 2),))),
                         _entry(1.0, _adapter(), function=lambda w: w)]}
        model = self._model(patches)
        node = self._node()
        result = node.execute_inline(model, output_strength=1.0)
        out = self._out(result)
        self.assertIs(out[0], model)
        self.assertIn("2 non-mergeable patch entries", out[2])
        self.assertIn("passed through", out[2])
        self.assertNotIn("AFTER your Load LoRA nodes", out[2])

    def test_merge_called_with_virtual_stack_and_stripped_model(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        node.execute_inline(model, output_strength=1.0)
        self.assertEqual(len(seen["stack"]), 2)
        self.assertTrue(all(i["_precomputed_diffs"] for i in seen["stack"]))
        self.assertEqual(seen["model"].patches, {})   # stripped clone
        self.assertIsNot(seen["model"], model)
        self.assertEqual(seen["normalize_keys"], "disabled")

    def test_report_prepends_fingerprints(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter(), "b": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "engine report", None, None)
        result = node.execute_inline(model, output_strength=1.0)
        report = self._out(result)[2]
        self.assertIn("#1", report)
        self.assertIn("2 keys", report)
        self.assertIn("0.80", report)
        self.assertIn("rank 4", report)
        self.assertIn("engine report", report)

    def test_disabled_slot_stripped_but_not_merged(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        node.execute_inline(model, output_strength=1.0, enabled_1=False)
        self.assertEqual(len(seen["stack"]), 1)
        self.assertAlmostEqual(seen["stack"][0]["strength"], 0.5)
        self.assertEqual(seen["model"].patches, {})   # disabled LoRA stripped too

    def test_fingerprint_warns_on_group_count_mismatch(self):
        # 2 model groups vs 1 clip group (both non-empty) -> order-attribution
        # warning naming the usual suspects
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        clip = _FakeCLIP(_FakePatcher(_chain_patches((0.6, {"te.a": _adapter()}))))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        result = node.execute_inline(model, output_strength=1.0, clip=clip)
        report = self._out(result)[2]
        self.assertIn("2 model-side vs 1 clip-side", report)
        self.assertIn("LoraLoaderModelOnly", report)
        # the clip clone passed to the engine is stripped, original untouched
        self.assertIsNot(seen["clip"], clip)
        self.assertEqual(seen["clip"].patcher.patches, {})
        self.assertEqual(len(clip.patcher.patches["te.a"]), 1)

    def test_fingerprint_counts_passthrough_entries(self):
        patches = _chain_patches((0.8, {"a": _adapter()}))
        patches["a"].append(_entry(1.0, ("set", (torch.zeros(2, 2),))))
        model = self._model(patches)
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        result = node.execute_inline(model, output_strength=1.0)
        report = self._out(result)[2]
        self.assertIn("1 non-LoRA", report)
        self.assertIn("passed through untouched", report)

    def test_no_passthrough_note_when_all_captured(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertNotIn("non-LoRA", report)

    def test_fingerprint_notes_fewer_slots_than_loras(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(
            model, output_strength=1.0, lora_count=1))[2]
        self.assertIn("2 LoRAs detected but only 1 option slot", report)
        self.assertIn("default options", report)

    def test_fingerprint_notes_more_slots_than_loras(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(
            model, output_strength=1.0, lora_count=3))[2]
        self.assertIn("3 option slots but only 1 LoRA", report)
        self.assertIn("extra slots ignored", report)

    def test_autotuner_settings_fall_back_to_defaults(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        result = node.execute_inline(model, output_strength=1.0,
                                     settings={"mode": "autotuner", "top_n": 3})
        report = self._out(result)[2]
        self.assertIn("AutoTuner settings are not supported inline", report)
        self.assertNotIn("top_n", seen)                      # defaults used
        self.assertEqual(seen["normalize_keys"], "disabled")

    def test_advanced_settings_delegated_normalize_keys_pinned(self):
        settings = {
            "mode": "advanced",
            "auto_strength": "disabled", "auto_strength_floor": 0.5,
            "optimization_mode": "global",
            "sparsification": "dare", "sparsification_density": 0.5,
            "dare_dampening": 0.1,
            "merge_refinement": "refine", "strategy_set": "basic",
            "normalize_keys": "enabled",   # must NOT reach the engine
            "architecture_preset": "dit", "decision_smoothing": 0.1,
            "smooth_slerp_gate": True,
            "star_eta": 0.9, "tame_layers": 0.1, "tame_threshold": 0.2,
            "cache_patches": "disabled", "patch_compression": "disabled",
            "svd_device": "cpu", "free_vram_between_passes": "enabled",
            "vram_budget": 0.25,
        }
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        result = node.execute_inline(model, output_strength=1.0, settings=settings)
        report = self._out(result)[2]
        self.assertNotIn("not supported inline", report)
        self.assertEqual(seen["normalize_keys"], "disabled")  # pinned
        self.assertEqual(seen["optimization_mode"], "global")
        self.assertEqual(seen["sparsification"], "dare")
        self.assertAlmostEqual(seen["sparsification_density"], 0.5)
        self.assertEqual(seen["cache_patches"], "disabled")
        self.assertEqual(seen["svd_device"], "cpu")
        self.assertAlmostEqual(seen["star_eta"], 0.9)
        # explicit preset from the settings node -> no arch-unknown warning
        self.assertNotIn("architecture: unknown", report)

    def test_fingerprint_warns_arch_unknown_without_preset(self):
        # Virtual items skip _normalize_stack's arch detection, so auto
        # detection can never resolve — the report must say so.
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertIn("architecture: unknown (inline capture)", report)
        self.assertIn("architecture_preset", report)

    def test_passthrough_count_includes_clip_side(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        clip_patches = _chain_patches((0.6, {"te.a": _adapter()}))
        clip_patches["te.a"].append(_entry(1.0, ("set", (torch.zeros(2, 2),))))
        clip = _FakeCLIP(_FakePatcher(clip_patches))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, kw.get("clip"), "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0,
                                               clip=clip))[2]
        self.assertIn("1 non-LoRA", report)          # the clip-side "set" entry
        self.assertIn("passed through untouched", report)

    def test_lokr_payload_reports_dense_rank(self):
        # LoKr weights[1] is the w2 Kronecker factor, NOT a rank — the
        # fingerprint must not misreport its shape as "rank N".
        lokr = lora_optimizer.LoKrAdapter(
            loaded_keys=set(),
            weights=(torch.randn(4, 4), torch.randn(2, 2), None, None,
                     None, None, None, None))
        model = self._model(_chain_patches((0.8, {"a": lokr})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertIn("dense", report)
        self.assertNotIn("rank 2", report)

    def test_advanced_visibility_widgets_reach_stack(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        clip = _FakeCLIP(_FakePatcher(_chain_patches(
            (0.6, {"te.a": _adapter()}), (0.4, {"te.b": _adapter()}))))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        node.execute_inline(model, output_strength=1.0, clip=clip,
                            settings_visibility="advanced",
                            model_strength_2=0.5, clip_strength_2=2.0)
        self.assertEqual(len(seen["stack"]), 2)
        self.assertAlmostEqual(seen["stack"][0]["strength"], 0.8)       # defaults
        self.assertAlmostEqual(seen["stack"][0]["clip_strength"], 0.6)
        self.assertAlmostEqual(seen["stack"][1]["strength"], 0.25)      # 0.5 x 0.5
        self.assertAlmostEqual(seen["stack"][1]["clip_strength"], 0.8)  # 0.4 x 2.0

    def test_salt_names_differ_across_source_patchers(self):
        # Two DIFFERENT loader chains at identical strengths must yield
        # different virtual item names, or optimize_merge's in-node cache
        # would false-hit when the user swaps a LoRA file.
        names_per_run = []
        for _ in range(2):
            model = self._model(_chain_patches(
                (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
            model.patches_uuid = uuid.uuid4()
            node = self._node()
            seen = {}
            self._capture_merge(node, seen)
            node.execute_inline(model, output_strength=1.0)
            names_per_run.append([i["name"] for i in seen["stack"]])
        self.assertNotEqual(names_per_run[0], names_per_run[1])
        for names in names_per_run:
            self.assertTrue(names[0].startswith("chain lora #1"))
            self.assertTrue(names[1].startswith("chain lora #2"))

    def test_salt_names_stable_for_same_patcher(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        model.patches_uuid = uuid.uuid4()
        node = self._node()
        names_per_run = []
        def fake_merge(m, stack, o, **kw):
            names_per_run.append([i["name"] for i in stack])
            return (m, None, "r", None, None)
        node.optimize_merge = fake_merge
        node.execute_inline(model, output_strength=1.0)
        node.execute_inline(model, output_strength=1.0)
        self.assertEqual(names_per_run[0], names_per_run[1])

    def test_input_types_surface(self):
        cls = lora_optimizer.LoRAOptimizerInline
        it = cls.INPUT_TYPES()
        req, opt = it["required"], it["optional"]
        # widgets the user types/toggles live in "required" (optional widget
        # inputs render as sockets in some frontends); JS control widgets
        # must all be present and required
        for w in ("model", "settings_visibility", "lora_count",
                  "output_strength", "clip_strength_multiplier"):
            self.assertIn(w, req)
        for i in (1, cls.MAX_LORAS):
            for base in ("enabled", "strength", "model_strength",
                         "clip_strength", "conflict_mode", "key_filter",
                         "preserve"):
                self.assertIn(f"{base}_{i}", req)
        # genuine node-to-node wires stay optional
        self.assertIn("clip", opt)
        self.assertIn("settings", opt)
        self.assertEqual(cls.FUNCTION, "execute_inline")
        # inherited from LoRAOptimizer — lora_data keeps SaveMergedLoRA working
        self.assertEqual(cls.RETURN_TYPES,
                         ("MODEL", "CLIP", "STRING", "TUNER_DATA", "LORA_DATA"))
        self.assertNotIn("RETURN_TYPES", vars(cls))   # not redeclared


class TestInlineIsChanged(unittest.TestCase):
    """LoRAOptimizerInline.IS_CHANGED must accept the node's ACTUAL inputs
    (ComfyUI passes every declared widget; the inherited LoRAOptimizer
    signature expects lora_stack and would raise -> node re-merges on every
    queue press) and key on the upstream patcher state + consulted widgets."""

    @staticmethod
    def _full_widget_kwargs():
        cls = lora_optimizer.LoRAOptimizerInline
        kw = {"settings_visibility": "simple", "lora_count": 3,
              "output_strength": 1.0, "clip_strength_multiplier": 1.0}
        for i in range(1, cls.MAX_LORAS + 1):
            kw[f"enabled_{i}"] = True
            kw[f"strength_{i}"] = 1.0
            kw[f"model_strength_{i}"] = 1.0
            kw[f"clip_strength_{i}"] = 1.0
            kw[f"conflict_mode_{i}"] = "all"
            kw[f"key_filter_{i}"] = "all"
            kw[f"preserve_{i}"] = False
        return kw

    def _model(self):
        m = _FakePatcher(_chain_patches((0.8, {"a": _adapter()})))
        m.patches_uuid = uuid.uuid4()
        return m

    def test_accepts_full_widget_set_and_is_stable(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        kw = self._full_widget_kwargs()
        first = cls.IS_CHANGED(model, **kw)     # must not raise
        self.assertEqual(first, cls.IS_CHANGED(model, **kw))

    def test_changes_when_consulted_widget_changes(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        kw = self._full_widget_kwargs()
        first = cls.IS_CHANGED(model, **kw)
        changed = dict(kw, enabled_2=False)     # slot 2 <= lora_count=3
        self.assertNotEqual(first, cls.IS_CHANGED(model, **changed))

    def test_ignores_widgets_beyond_lora_count(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        kw = self._full_widget_kwargs()
        first = cls.IS_CHANGED(model, **kw)
        beyond = dict(kw, strength_9=0.25)      # slot 9 > lora_count=3
        self.assertEqual(first, cls.IS_CHANGED(model, **beyond))

    def test_changes_when_upstream_chain_changes(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        kw = self._full_widget_kwargs()
        first = cls.IS_CHANGED(model, **kw)
        model.patches_uuid = uuid.uuid4()       # loader chain re-executed
        self.assertNotEqual(first, cls.IS_CHANGED(model, **kw))

    def test_changes_with_clip_and_settings(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        kw = self._full_widget_kwargs()
        first = cls.IS_CHANGED(model, **kw)
        clip = _FakeCLIP(_FakePatcher(_chain_patches((0.6, {"te.a": _adapter()}))))
        clip.patcher.patches_uuid = uuid.uuid4()
        with_clip = cls.IS_CHANGED(model, clip=clip, **kw)
        self.assertNotEqual(first, with_clip)
        clip.patcher.patches_uuid = uuid.uuid4()
        self.assertNotEqual(with_clip, cls.IS_CHANGED(model, clip=clip, **kw))
        self.assertNotEqual(first, cls.IS_CHANGED(
            model, settings={"mode": "advanced"}, **kw))


def _realistic_load_lora_for_models(calls):
    """A comfy.sd.load_lora_for_models stand-in that mimics the REAL failure
    mode instead of blindly passing the model through: it resolves only
    TRAINER-format keys ({x}.lora_up.weight / .lora_down.weight). Virtual
    chain items carry model-target keys, so nothing matches and the models
    come back unchanged — exactly what live comfy does, silently."""
    def stub(model, clip, lora_dict, strength_model, strength_clip):
        matched = [k for k in lora_dict
                   if isinstance(k, str) and (k.endswith(".lora_up.weight")
                                              or k.endswith(".lora_down.weight"))]
        calls.append({"matched": matched, "n_keys": len(lora_dict)})
        return (model, clip)   # nothing baked for unmatched keys
    return stub


def _pipeline_model(patches, applied, dim=16):
    """Fake ModelPatcher threading the REAL optimize_merge pipeline: a .model
    with a dim x dim base weight so _resolve_target_shape/_resolve_base_norm
    work, faithful clone() (copies the CURRENT patch lists, like the real
    ModelPatcher.clone — so a clone of the stripped clone stays stripped),
    and an add_patches recorder capturing the merged patches the engine
    re-applies (plus which patcher received them)."""
    base = types.SimpleNamespace(
        layer=types.SimpleNamespace(weight=torch.zeros(dim, dim)))

    def _attach(p):
        p.model = base
        p.add_patches = lambda patches_, strength=1.0, strength_clip=None: (
            applied.update(patches=dict(patches_), strength=strength,
                           patcher=p),
            list(patches_.keys()))[1]
        p.clone = lambda: _attach(
            _FakePatcher({k: v[:] for k, v in p.patches.items()}))
        return p

    return _attach(_FakePatcher(patches))


class TestSingleLoraVirtualPath(unittest.TestCase):
    """A 1-LoRA chain (or N LoRAs with all but one disabled) must NOT take
    optimize_merge's single-LoRA fast path: load_lora_for_models looks up
    trainer-format keys, virtual items carry model-target keys, so it would
    load NOTHING — and with the originals stripped, the output model would
    silently lose the LoRA entirely."""

    def _fake_model(self, patches, applied):
        return _pipeline_model(patches, applied)

    def test_single_virtual_lora_goes_through_pipeline(self):
        key = "layer.weight"
        applied = {}
        model = self._fake_model(
            _chain_patches((0.9, {key: _adapter(rank=4, out_dim=16, in_dim=16)})),
            applied)
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": key}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            result = node.execute_inline(model, output_strength=1.0)
        out = result["result"] if isinstance(result, dict) else result
        self.assertEqual(calls, [])               # fast path NOT taken
        self.assertTrue(applied.get("patches"))   # pipeline re-applied patches
        self.assertIn("chain lora #1", out[2])

    def test_disabled_down_to_single_also_uses_pipeline(self):
        key = "layer.weight"
        applied = {}
        model = self._fake_model(
            _chain_patches((0.9, {key: _adapter(rank=4, out_dim=16, in_dim=16)}),
                           (0.5, {key: _adapter(rank=4, out_dim=16, in_dim=16)})),
            applied)
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": key}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            node.execute_inline(model, output_strength=1.0, enabled_2=False)
        self.assertEqual(calls, [])
        self.assertTrue(applied.get("patches"))

    def test_plain_single_lora_stack_still_takes_fast_path(self):
        # Regression net for the guard: trainer-format single-LoRA stacks
        # keep the fast path, and the realistic stub CAN match their keys.
        model = _FakePatcher({})
        model.model = types.SimpleNamespace(
            layer=types.SimpleNamespace(weight=torch.zeros(16, 16)))
        stack = [{"name": "A", "strength": 1.0,
                  "lora": {"alias_layer.lora_up.weight": torch.randn(16, 4),
                           "alias_layer.lora_down.weight": torch.randn(4, 16),
                           "alias_layer.alpha": torch.tensor(4.0)}}]
        node = lora_optimizer.LoRAOptimizer()
        node._get_model_keys = lambda m: {"alias_layer": "layer.weight"}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            node.optimize_merge(model, stack, 1.0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["matched"])


class TestRegistration(unittest.TestCase):
    def test_node_registered(self):
        self.assertIn("LoRAOptimizerInline", lora_optimizer.NODE_CLASS_MAPPINGS)
        self.assertIs(lora_optimizer.NODE_CLASS_MAPPINGS["LoRAOptimizerInline"],
                      lora_optimizer.LoRAOptimizerInline)
        self.assertIn("LoRAOptimizerInline",
                      lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS["LoRAOptimizerInline"],
            "LoRA Optimizer (Inline Chain)")


def _advanced_settings(**overrides):
    """Full OPTIMIZER_SETTINGS advanced-mode dict with deterministic,
    test-friendly engine knobs (no merge cache, no compression SVD, CPU SVD).
    The settings input is the node's supported surface for engine kwargs —
    execute_inline deliberately does not forward loose **kwargs to
    optimize_merge."""
    settings = {
        "mode": "advanced",
        "auto_strength": "disabled", "auto_strength_floor": -1.0,
        "optimization_mode": "per_prefix",
        "sparsification": "disabled", "sparsification_density": 0.7,
        "dare_dampening": 0.0,
        "merge_strategy_override": "",
        "merge_refinement": "none", "strategy_set": "full",
        "normalize_keys": "disabled",
        "architecture_preset": "dit", "decision_smoothing": 0.25,
        "smooth_slerp_gate": False,
        "star_eta": 100.0, "tame_layers": 0.0, "tame_threshold": 0.3,
        "cache_patches": "disabled", "patch_compression": "disabled",
        "svd_device": "cpu", "free_vram_between_passes": "disabled",
        "vram_budget": 0.0,
    }
    settings.update(overrides)
    return settings


class TestMultiLoraEndToEnd(unittest.TestCase):
    """Captured MULTI-LoRA chains through the REAL optimize_merge: two rank-4
    adapters on the same 16x16 layer key, different loader strengths. The
    single-LoRA virtual path is covered above; this closes the >= 2-LoRA gap
    (full Pass 1 conflict analysis + Pass 2 group merge on virtual items)."""

    KEY = "layer.weight"

    @staticmethod
    def _det_mats(seed_row):
        """Deterministic rank-2 up/down matrices (each linspace reshape has
        rows/columns affine in the index, so the span is {1, arange} — rank 2;
        deterministic on purpose, no randn luck)."""
        up = torch.linspace(-1.0 + seed_row, 1.0 + seed_row, 16 * 4).reshape(16, 4)
        down = torch.linspace(0.5 - seed_row, -0.5 + seed_row, 4 * 16).reshape(4, 16)
        return up, down

    def _run(self, chain, settings=None, output_strength=1.0, **exec_kwargs):
        applied = {}
        model = _pipeline_model(_chain_patches(*chain), applied)
        # Snapshot the loader entries BEFORE execute_inline runs — comparing
        # a post-run snapshot to the post-run state would be tautological and
        # could never catch the node mutating its input model's patch lists.
        orig_entries = {k: list(v) for k, v in model.patches.items()}
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": self.KEY}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            result = node.execute_inline(model, output_strength=output_strength,
                                         settings=settings, **exec_kwargs)
        out = result["result"] if isinstance(result, dict) else result
        return model, applied, calls, out, orig_entries

    def test_two_lora_chain_weighted_sum_is_numerically_exact(self):
        up_a, down_a = self._det_mats(0.0)
        up_b, down_b = self._det_mats(0.25)
        chain = (
            (1.0, {self.KEY: _adapter(up=up_a, down=down_a)}),
            (0.7, {self.KEY: _adapter(up=up_b, down=down_b)}),
        )
        # weighted_sum override makes the expected result exact:
        # merged = 1.0 * (up_a @ down_a) + 0.7 * (up_b @ down_b)
        settings = _advanced_settings(optimization_mode="global",
                                      merge_strategy_override="weighted_sum")
        model, applied, calls, out, orig_entries = self._run(
            chain, settings=settings)

        # (a) report opens with the chain fingerprints, engine report follows
        report = out[2]
        self.assertIn("#1: 1 keys, rank 4, loader strength 1.00", report)
        self.assertIn("#2: 1 keys, rank 4, loader strength 0.70", report)
        self.assertIn("chain lora #1", report)

        # (b) merged patches re-applied via add_patches on a clone
        self.assertEqual(calls, [])                      # no fast-path bypass
        self.assertIn("patches", applied)
        self.assertIsNot(applied["patcher"], model)
        self.assertIs(out[0], applied["patcher"])        # patched clone returned
        self.assertAlmostEqual(applied["strength"], 1.0)

        # (c) the applied patch is the exact weighted combination
        self.assertEqual(set(applied["patches"]), {self.KEY})
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        expected = up_a @ down_a + 0.7 * (up_b @ down_b)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5),
                        f"max err {(got - expected).abs().max().item()}")

        # (d) the original input model still carries its loader entries
        # (orig_entries snapshotted pre-run in _run, so this has teeth)
        self.assertEqual(model.patches[self.KEY], orig_entries[self.KEY])
        self.assertEqual(len(model.patches[self.KEY]), 2)

    def test_output_strength_scales_at_apply_not_in_patches(self):
        # Same exact-weighted-sum setup, but output_strength=0.5: the master
        # volume must ride on add_patches ONCE and never ALSO be baked into
        # the patch tensors (which would double-scale the merged result).
        up_a, down_a = self._det_mats(0.0)
        up_b, down_b = self._det_mats(0.25)
        chain = (
            (1.0, {self.KEY: _adapter(up=up_a, down=down_a)}),
            (0.7, {self.KEY: _adapter(up=up_b, down=down_b)}),
        )
        settings = _advanced_settings(optimization_mode="global",
                                      merge_strategy_override="weighted_sum")
        model, applied, calls, out, orig_entries = self._run(
            chain, settings=settings, output_strength=0.5)

        self.assertEqual(calls, [])
        self.assertAlmostEqual(applied["strength"], 0.5)
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        # UNscaled weighted sum — identical to the output_strength=1.0 case
        expected = up_a @ down_a + 0.7 * (up_b @ down_b)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5),
                        f"max err {(got - expected).abs().max().item()}")

    def test_conflicting_loras_complete_through_conflict_analysis(self):
        # Opposite-sign directions built deliberately: B is A negated with a
        # small deterministic perturbation (near-antiparallel, cos ~ -1) so
        # the per_prefix conflict path really engages without relying on a
        # degenerate exactly-antiparallel corner.
        up_a, down_a = self._det_mats(0.0)
        down_b = down_a.clone()
        down_b[0] += 0.05
        chain = (
            (1.0, {self.KEY: _adapter(up=up_a, down=down_a)}),
            (0.7, {self.KEY: _adapter(up=-up_a, down=down_b)}),
        )
        model, applied, calls, out, orig_entries = self._run(
            chain, settings=_advanced_settings())

        self.assertEqual(calls, [])
        report = out[2]
        self.assertIn("#1: 1 keys, rank 4, loader strength 1.00", report)
        self.assertIn("#2: 1 keys, rank 4, loader strength 0.70", report)
        # merged patches exist and are finite/nonzero for the shared key
        self.assertEqual(set(applied["patches"]), {self.KEY})
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        self.assertEqual(tuple(got.shape), (16, 16))
        self.assertTrue(torch.isfinite(got).all())
        self.assertGreater(got.abs().max().item(), 0.0)
        # direction teeth: A (strength 1.0) beats the weaker near-antiparallel
        # B (0.7), so any sane conflict resolution must land on A's side of
        # the axis — positive cosine alignment with A's diff.
        diff_a = (up_a @ down_a).flatten()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), diff_a, dim=0).item()
        self.assertGreater(cos, 0.0,
                           f"merged patch flipped to the weaker LoRA's "
                           f"direction (cos={cos:.4f})")
        # original chain untouched on the input model (pre-run snapshot)
        self.assertEqual(model.patches[self.KEY], orig_entries[self.KEY])
        self.assertEqual(len(model.patches[self.KEY]), 2)


if __name__ == "__main__":
    unittest.main()
