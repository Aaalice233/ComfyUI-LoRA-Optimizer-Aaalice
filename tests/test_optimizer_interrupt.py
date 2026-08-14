import os
import sys
import threading
import time
import unittest

import torch

_TEST_DIR = os.path.dirname(__file__)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
try:
    from tests import test_lora_optimizer as _HELPER
except ImportError:
    import test_lora_optimizer as _HELPER
lora_optimizer = _HELPER.lora_optimizer
chunked_merge = sys.modules[lora_optimizer.ExecutionPlanner.__module__]
sys.modules.setdefault("lora_optimizer", lora_optimizer)


class NativeInterrupt(Exception):
    pass


class InterruptControllerTests(unittest.TestCase):
    def setUp(self):
        controller_module = sys.modules[lora_optimizer.InterruptController.__module__]
        self.model_management = controller_module.comfy.model_management
        self.old_exception = getattr(
            self.model_management, "InterruptProcessingException", None)
        self.old_checker = getattr(
            self.model_management,
            "throw_exception_if_processing_interrupted", None)
        self.model_management.InterruptProcessingException = NativeInterrupt

    def tearDown(self):
        if self.old_exception is None:
            delattr(self.model_management, "InterruptProcessingException")
        else:
            self.model_management.InterruptProcessingException = self.old_exception
        if self.old_checker is None:
            if hasattr(self.model_management,
                       "throw_exception_if_processing_interrupted"):
                delattr(self.model_management,
                        "throw_exception_if_processing_interrupted")
        else:
            self.model_management.throw_exception_if_processing_interrupted = self.old_checker

    def test_native_exception_is_rethrown_and_shared_event_is_set(self):
        original = NativeInterrupt("cancel")

        def checker():
            raise original

        self.model_management.throw_exception_if_processing_interrupted = checker
        controller = lora_optimizer.InterruptController()
        with self.assertRaises(NativeInterrupt) as raised:
            controller.check()
        self.assertIs(raised.exception, original)
        self.assertTrue(controller.event.is_set())

    def test_other_workers_observe_consumed_native_interrupt(self):
        self.model_management.throw_exception_if_processing_interrupted = lambda: None
        controller = lora_optimizer.InterruptController()
        controller.cancel()
        with self.assertRaises(NativeInterrupt):
            controller.check()

    def test_progress_is_monotonic_and_finishes_exactly(self):
        class Bar:
            def __init__(self):
                self.value = 0
                self.updates = []

            def update(self, amount):
                self.updates.append(amount)
                self.value += amount

        optimizer = lora_optimizer.LoRAOptimizer()
        bar = Bar()
        optimizer._progress_state = {
            "bar": bar,
            "total": 3000,
            "value": 0,
            "targets": {},
            "decision": False,
            "lock": threading.Lock(),
        }
        optimizer._progress_update("pass1", "x", 0.25)
        optimizer._progress_update("pass1", "x", 0.2)
        optimizer._progress_update("pass1", "x", 1.0)
        optimizer._progress_decision()
        optimizer._progress_update("pass2", "x", 0.5)
        optimizer._progress_finish()
        self.assertTrue(all(amount > 0 for amount in bar.updates))
        self.assertEqual(bar.value, 3000)

    def test_optimizer_aborts_before_cache_or_patch_application(self):
        calls = {"n": 0}

        def checker():
            calls["n"] += 1
            if calls["n"] == 1:
                raise NativeInterrupt("cancel")

        self.model_management.throw_exception_if_processing_interrupted = checker
        optimizer = lora_optimizer.LoRAOptimizer()
        optimizer._merge_cache = {"sentinel": object()}
        with self.assertRaises(NativeInterrupt):
            optimizer.optimize_merge(None, [], 1.0)
        self.assertEqual(list(optimizer._merge_cache), ["sentinel"])

    def test_save_node_checks_native_interrupt_before_writing(self):
        self.model_management.throw_exception_if_processing_interrupted = (
            lambda: (_ for _ in ()).throw(NativeInterrupt("cancel")))
        with self.assertRaises(NativeInterrupt):
            lora_optimizer.SaveMergedLoRA().save_lora(None, "", "cancelled")

    def test_tiled_merge_interrupts_between_rows(self):
        checks = {"n": 0}

        def checker():
            checks["n"] += 1
            if checks["n"] >= 4:
                raise NativeInterrupt("cancel")

        self.model_management.throw_exception_if_processing_interrupted = checker
        optimizer = lora_optimizer.LoRAOptimizer()
        optimizer._interrupt_controller = lora_optimizer.InterruptController()
        source = chunked_merge.DenseDiffSource(torch.ones(64, 16), (64, 16))
        group = {
            "sources": {0: source},
            "eff_strengths": {0: 1.0},
            "target_shape": torch.Size((64, 16)),
        }
        plan = chunked_merge.ExecutionPlan(
            "tiled_gpu", torch.device("cpu"), 8, 0, 16 * 1024 ** 2)
        with self.assertRaises(NativeInterrupt):
            optimizer._merge_group_sources_tiled(
                group, [{"strength": 1.0, "conflict_mode": "all"}],
                plan, "weighted_sum", 0.6, "total", [False])

    def test_multi_slerp_karcher_iteration_is_interruptible(self):
        sources = {}
        active = []
        for index in range(3):
            sources[index] = chunked_merge.LoRADiffSource(
                torch.randn(64, 4), torch.randn(4, 64), 4.0, None, (64, 64))
            active.append({"strength": 1.0})
        source_group = {
            "sources": sources,
            "eff_strengths": {index: 1.0 for index in sources},
            "target_shape": torch.Size((64, 64)),
            "target_key": "karcher",
            "label_prefix": "karcher",
            "storage_dtype": torch.float32,
        }
        plan = chunked_merge.ExecutionPlan(
            "cpu", torch.device("cpu"), 4, 0, 0, "test")
        checks = {"n": 0}

        def checker():
            checks["n"] += 1
            if checks["n"] >= 90:
                raise NativeInterrupt("cancel")

        self.model_management.throw_exception_if_processing_interrupted = checker
        optimizer = lora_optimizer.LoRAOptimizer()
        optimizer._interrupt_controller = lora_optimizer.InterruptController()
        with self.assertRaises(NativeInterrupt):
            optimizer._merge_group_sources_tiled(
                source_group, active, plan, "slerp", 0.6, "total",
                [False, False, False])

    def test_chunked_svd_interrupts_between_operator_tiles(self):
        checks = {"n": 0}

        def checker():
            checks["n"] += 1
            if checks["n"] >= 3:
                raise NativeInterrupt("cancel")

        self.model_management.throw_exception_if_processing_interrupted = checker
        controller = lora_optimizer.InterruptController()
        dense = torch.randn(64, 32)
        with self.assertRaises(NativeInterrupt):
            chunked_merge.chunked_randomized_svd(
                lambda start, end, _device: dense[start:end], dense.shape, 4,
                torch.device("cpu"), 8, controller)


if __name__ == "__main__":
    unittest.main()
