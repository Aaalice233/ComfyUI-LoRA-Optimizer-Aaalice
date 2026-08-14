import os
import sys
import unittest
from unittest import mock

import torch

_TEST_DIR = os.path.dirname(__file__)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
try:
    from tests import test_lora_optimizer as _HELPER
except ImportError:
    import test_lora_optimizer as _HELPER
lora_optimizer = _HELPER.lora_optimizer
sys.modules.setdefault("lora_optimizer", lora_optimizer)
chunked_merge = sys.modules[lora_optimizer.ExecutionPlanner.__module__]
from chunked_merge import ExecutionPlan


class ChunkSourceTests(unittest.TestCase):
    def test_low_rank_rows_match_full_matrix(self):
        torch.manual_seed(1)
        up = torch.randn(13, 4)
        down = torch.randn(4, 17)
        source = chunked_merge.LoRADiffSource(up, down, 2.0, None, (13, 17))
        expected = (up @ down) * 0.5
        actual = torch.cat([
            source.materialize_rows(0, 5, torch.device("cpu")),
            source.materialize_rows(5, 13, torch.device("cpu")),
        ])
        torch.testing.assert_close(actual, expected)

    def test_locon_mid_rows_match_full_einsum(self):
        torch.manual_seed(2)
        rank, mid_rank, out_dim, in_dim = 3, 2, 9, 7
        up = torch.randn(rank, out_dim)
        down = torch.randn(mid_rank, in_dim, 1, 1)
        mid = torch.randn(rank, mid_rank, 3, 3)
        source = chunked_merge.LoRADiffSource(
            up, down, mid_rank, mid, (out_dim, in_dim, 3, 3))
        expected = torch.einsum(
            "i j k l, j r, i p -> p r k l", mid, down.flatten(1), up)
        actual = torch.cat([
            source.materialize_rows(0, 4, torch.device("cpu")),
            source.materialize_rows(4, out_dim, torch.device("cpu")),
        ]).reshape_as(expected)
        torch.testing.assert_close(actual, expected)

    def test_loha_rows_match_full_matrix(self):
        torch.manual_seed(3)
        w1a, w1b = torch.randn(11, 3), torch.randn(3, 8)
        w2a, w2b = torch.randn(11, 2), torch.randn(2, 8)
        source = chunked_merge.LoHaDiffSource(
            w1a, w1b, 3.0, w2a, w2b, None, None, (11, 8))
        expected = (w1a @ w1b) * (w2a @ w2b)
        actual = torch.cat([
            source.materialize_rows(0, 6, torch.device("cpu")),
            source.materialize_rows(6, 11, torch.device("cpu")),
        ])
        torch.testing.assert_close(actual, expected)

    def test_lokr_rows_match_full_kronecker_product(self):
        torch.manual_seed(4)
        w1 = torch.randn(3, 2)
        w2 = torch.randn(4, 5)
        source = chunked_merge.LoKrDiffSource(
            w1, w2, 1.0, None, None, None, None, None, (12, 10))
        expected = torch.kron(w1, w2)
        actual = torch.cat([
            source.materialize_rows(0, 5, torch.device("cpu")),
            source.materialize_rows(5, 12, torch.device("cpu")),
        ])
        torch.testing.assert_close(actual, expected)

    def test_sum_dense_and_scaled_sources_preserve_alias_semantics(self):
        first = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        second = torch.ones_like(first)
        source = chunked_merge.ScaledDiffSource(
            chunked_merge.SumDiffSource([
                chunked_merge.DenseDiffSource(first, first.shape),
                chunked_merge.DenseDiffSource(second, second.shape),
            ]),
            -0.25,
        )
        expected = (first + second) * -0.25
        actual = torch.cat([
            source.materialize_rows(0, 2, torch.device("cpu")),
            source.materialize_rows(2, 5, torch.device("cpu")),
        ])
        torch.testing.assert_close(actual, expected)


class ExecutionPlannerTests(unittest.TestCase):
    def setUp(self):
        self.cuda = torch.device("cuda")

    def test_selects_full_gpu_when_peak_fits(self):
        planner = lora_optimizer.ExecutionPlanner(lambda _device: 4 * 1024 ** 3)
        plan = planner.plan(
            self.cuda, (128, 128), 2, 4, factor_bytes=1024)
        self.assertEqual(plan.mode, "full_gpu")
        self.assertEqual(plan.rows_per_tile, 128)

    def test_selects_tiled_gpu_for_oversized_group(self):
        planner = lora_optimizer.ExecutionPlanner(lambda _device: 4 * 1024 ** 3)
        plan = planner.plan(
            self.cuda, (8192, 8192), 9, 14,
            factor_bytes=64 * 1024 ** 2)
        self.assertEqual(plan.mode, "tiled_gpu")
        self.assertGreaterEqual(plan.rows_per_tile, 1)
        self.assertLess(plan.rows_per_tile, 8192)
        self.assertLessEqual(
            plan.rows_per_tile * 8192 * torch.float32.itemsize,
            128 * 1024 ** 2)

    def test_cpu_is_kept_for_cpu_only_execution(self):
        planner = lora_optimizer.ExecutionPlanner(lambda _device: 0)
        plan = planner.plan(
            torch.device("cpu"), (128, 128), 2, 4, factor_bytes=0)
        self.assertEqual(plan.mode, "cpu")

    def test_live_vram_recheck_only_shrinks_tile_rows(self):
        free = {"value": 2 * 1024 ** 3}
        planner = lora_optimizer.ExecutionPlanner(
            lambda _device: free["value"])
        plan = planner.plan(
            self.cuda, (8192, 8192), 9, 13,
            factor_bytes=64 * 1024 ** 2)
        original_rows = plan.rows_per_tile
        free["value"] = 1024 ** 3
        shrunk_rows = planner.shrink_rows(
            plan, (8192, 8192), 13,
            factor_bytes=64 * 1024 ** 2)
        self.assertGreaterEqual(original_rows, shrunk_rows)
        self.assertGreater(shrunk_rows, 0)
        free["value"] = 512 * 1024 ** 2
        self.assertEqual(
            planner.shrink_rows(
                plan, (8192, 8192), 13,
                factor_bytes=64 * 1024 ** 2),
            0,
        )

    def test_environment_override_is_bounded(self):
        with mock.patch.dict(os.environ, {"LORA_OPTIMIZER_TILE_MB": "16"}):
            override = lora_optimizer.ExecutionPlanner._workset_override()
        self.assertEqual(override, 16 * 1024 ** 2)


class ChunkedLinearAlgebraTests(unittest.TestCase):
    def test_randomized_svd_reconstructs_low_rank_source(self):
        torch.manual_seed(5)
        up = torch.randn(40, 3)
        down = torch.randn(3, 32)
        source = chunked_merge.LoRADiffSource(up, down, 3.0, None, (40, 32))
        controller = lora_optimizer.InterruptController()
        u, s, vh = chunked_merge.chunked_randomized_svd(
            lambda start, end, device: source.materialize_rows(start, end, device),
            (40, 32), 3, torch.device("cpu"), 7, controller, niter=2, seed=7,
        )
        reconstructed = (u * s.unsqueeze(0)) @ vh
        torch.testing.assert_close(reconstructed, up @ down, rtol=1e-4, atol=1e-5)

    def test_logical_random_is_independent_of_physical_slices(self):
        whole = chunked_merge.deterministic_random_like(
            (2_500_000,), torch.device("cpu"), 123, 0, 0)
        split = torch.cat([
            chunked_merge.deterministic_random_like(
                (777_777,), torch.device("cpu"), 123, 0, 0),
            chunked_merge.deterministic_random_like(
                (1_022_223,), torch.device("cpu"), 123, 0, 777_777),
            chunked_merge.deterministic_random_like(
                (700_000,), torch.device("cpu"), 123, 0, 1_800_000),
        ])
        torch.testing.assert_close(split, whole, rtol=0, atol=0)


class TiledMergeRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.sources = {
            index: chunked_merge.LoRADiffSource(
                torch.randn(32, 3), torch.randn(3, 16), 3.0, None, (32, 16))
            for index in range(3)
        }
        self.strengths = {0: 0.8, 1: -0.35, 2: 0.55}
        self.group = {
            "sources": self.sources,
            "eff_strengths": self.strengths,
            "target_shape": torch.Size((32, 16)),
        }
        self.active = [
            {"strength": self.strengths[index], "preserve": False,
             "conflict_mode": "all"}
            for index in range(3)
        ]
        self.optimizer = lora_optimizer.LoRAOptimizer()
        self.optimizer._interrupt_controller = lora_optimizer.InterruptController()
        self.plan = ExecutionPlan(
            "tiled_gpu", torch.device("cpu"), 7, 0, 16 * 1024 ** 2)

    def _full(self, mode, density=0.6):
        entries = [
            (self.sources[index].materialize_full(torch.device("cpu")),
             self.strengths[index])
            for index in range(3)
        ]
        return self.optimizer._merge_diffs(
            entries, mode, density=density, majority_sign_method="total",
            compute_device=torch.device("cpu"), keep_on_gpu=False,
            preserve_flags=[False, False, False])

    def test_weighted_modes_match_full_path(self):
        for mode in ("weighted_sum", "weighted_average", "normalize"):
            with self.subTest(mode=mode):
                tiled = self.optimizer._merge_group_sources_tiled(
                    self.group, self.active, self.plan, mode, 0.6, "total",
                    [False, False, False])
                torch.testing.assert_close(tiled, self._full(mode), rtol=1e-5, atol=1e-6)

    def test_two_way_slerp_matches_full_path(self):
        group = {
            "sources": {0: self.sources[0], 1: self.sources[1]},
            "eff_strengths": {0: self.strengths[0], 1: self.strengths[1]},
            "target_shape": torch.Size((32, 16)),
        }
        active = self.active[:2]
        tiled = self.optimizer._merge_group_sources_tiled(
            group, active, self.plan, "slerp", 0.6, "total", [False, False])
        entries = [
            (self.sources[index].materialize_full(torch.device("cpu")),
             self.strengths[index]) for index in range(2)
        ]
        full = self.optimizer._merge_diffs(
            entries, "slerp", compute_device=torch.device("cpu"),
            preserve_flags=[False, False])
        torch.testing.assert_close(tiled, full, rtol=1e-5, atol=1e-6)

    def test_multi_way_slerp_matches_full_path(self):
        tiled = self.optimizer._merge_group_sources_tiled(
            self.group, self.active, self.plan, "slerp", 0.6, "total",
            [False, False, False])
        torch.testing.assert_close(tiled, self._full("slerp"), rtol=1e-5, atol=1e-6)

    def test_ties_matches_full_path(self):
        tiled = self.optimizer._merge_group_sources_tiled(
            self.group, self.active, self.plan, "ties", 0.6, "total",
            [False, False, False])
        torch.testing.assert_close(tiled, self._full("ties"), rtol=1e-5, atol=1e-6)

    def test_sparse_output_is_independent_of_physical_tile_rows(self):
        for sparsification in ("dare", "della", "dare_conflict", "della_conflict"):
            first = self.optimizer._merge_group_sources_tiled(
                self.group, self.active, self.plan, "weighted_average", 0.6, "total",
                [False, False, False], sparsification=sparsification,
                sparsification_density=0.7, dare_dampening=0.1)
            other_plan = ExecutionPlan("tiled_gpu", torch.device("cpu"), 5, 0, 0)
            second = self.optimizer._merge_group_sources_tiled(
                self.group, self.active, other_plan, "weighted_average", 0.6, "total",
                [False, False, False], sparsification=sparsification,
                sparsification_density=0.7, dare_dampening=0.1)
            torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-6)

    def test_conflict_modes_are_supported_by_tiled_merge(self):
        self.active[0]["conflict_mode"] = "low_conflict"
        result = self.optimizer._merge_group_sources_tiled(
            self.group, self.active, self.plan, "weighted_sum", 0.6, "total",
            [False, False, False])
        self.assertEqual(tuple(result.shape), tuple(self.group["target_shape"]))
        self.assertTrue(torch.isfinite(result).all())

    def test_refine_and_full_refinement_are_tiled(self):
        for refinement in ("refine", "full"):
            result = self.optimizer._merge_group_sources_tiled(
                self.group, self.active, self.plan, "weighted_average", 0.6,
                "total", [False, False, False], merge_refinement=refinement)
            self.assertEqual(tuple(result.shape), tuple(self.group["target_shape"]))
            self.assertTrue(torch.isfinite(result).all())

    def test_star_is_applied_without_dense_materialization(self):
        result = self.optimizer._merge_group_sources_tiled(
            self.group, self.active, self.plan, "weighted_average", 0.6,
            "total", [False, False, False], star_eta=70.0)
        self.assertEqual(tuple(result.shape), tuple(self.group["target_shape"]))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
