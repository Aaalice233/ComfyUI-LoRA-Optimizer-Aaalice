"""Tests for the inline chain-filter optimizer node."""
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


if __name__ == "__main__":
    unittest.main()
