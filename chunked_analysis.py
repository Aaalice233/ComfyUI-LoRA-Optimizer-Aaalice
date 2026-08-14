"""Streaming Pass 1 analysis and shared tiled execution helpers."""

import logging
import math
import zlib

import torch

try:
    from .chunked_merge import (
        CallableDiffSource,
        ScaledDiffSource,
        chunked_randomized_svd,
        iter_row_ranges,
    )
except ImportError:
    from chunked_merge import (
        CallableDiffSource,
        ScaledDiffSource,
        chunked_randomized_svd,
        iter_row_ranges,
    )


class ChunkedAnalysisMixin:
    def _materialize_source_tile(self, sources, eff_strengths, active_loras,
                                 start, end, device, merge_refinement):
        tiles = {index: source.materialize_rows(start, end, device)
                 for index, source in sources.items()}
        if len(tiles) <= 1 or not any(
                active_loras[index].get("conflict_mode", "all") != "all"
                for index in tiles):
            return tiles
        indices = sorted(tiles)
        if merge_refinement != "none":
            sign_sum = torch.zeros(end - start, device=device, dtype=torch.float32)
            for index in indices:
                effective = tiles[index] if eff_strengths[index] >= 0 else -tiles[index]
                sign_sum.add_(effective.sum(dim=1).sign())
            majority = torch.where(sign_sum >= 0, 1.0, -1.0).unsqueeze(1)
        else:
            sign_sum = torch.zeros_like(tiles[indices[0]], dtype=torch.float32)
            for index in indices:
                effective = tiles[index] if eff_strengths[index] >= 0 else -tiles[index]
                sign_sum.add_(effective.sign())
            majority = torch.where(sign_sum >= 0, 1.0, -1.0)
        for index in indices:
            mode = active_loras[index].get("conflict_mode", "all")
            if mode == "all":
                continue
            effective = tiles[index] if eff_strengths[index] >= 0 else -tiles[index]
            if mode == "low_conflict":
                tiles[index].mul_((effective * majority) > 0)
            elif mode == "high_conflict":
                tiles[index].mul_((effective * majority) < 0)
        return tiles

    def _iter_live_rows(self, plan, target_shape, buffer_count, factor_bytes=0):
        rows = int(target_shape[0])
        start = 0
        while start < rows:
            self._interrupt_check()
            if plan.mode == "tiled_gpu":
                live_rows = self._execution_planner.shrink_rows(
                    plan, target_shape, buffer_count, factor_bytes)
                if live_rows < 1:
                    estimated = math.prod(int(value) for value in target_shape) * 4
                    raise RuntimeError(
                        f"Tiled GPU target {tuple(target_shape)} cannot fit one safe row "
                        f"(one dense target is {estimated / (1024 ** 2):.1f} MiB)")
            else:
                live_rows = plan.rows_per_tile
            end = min(rows, start + live_rows)
            yield start, end
            start = end

    @torch.no_grad()
    def _star_sources_tiled(self, sources, shape, plan, eta):
        if eta is None or eta >= 100.0:
            return sources
        rows = int(shape[0])
        cols = math.prod(int(v) for v in shape[1:])
        cleaned = {}
        for index, source in sources.items():
            self._interrupt_check()
            if source.rank <= 0:
                raise ValueError("STAR tiled execution requires a bounded-rank source")
            rank = min(max(1, source.rank), rows, cols)
            u, singular, vh = chunked_randomized_svd(
                source.materialize_rows, shape, rank, plan.device,
                plan.rows_per_tile, self._interrupt_controller, niter=2,
                seed=4200 + index)
            total = singular.sum()
            if total <= 0:
                cleaned[index] = source
                continue
            kept_rank = int((torch.cumsum(singular, 0)
                             < (eta / 100.0) * total).sum().item()) + 1
            kept_rank = max(1, min(kept_rank, singular.numel()))
            kept = singular[:kept_rank]
            scaled = kept * (total / kept.sum())
            source_u = u[:, :kept_rank]
            source_vh = vh[:kept_rank]

            def materialize(start, end, device, left=source_u,
                            values=scaled, right=source_vh):
                return ((left[start:end].to(device) * values.to(device).unsqueeze(0))
                        @ right.to(device))

            cleaned[index] = CallableDiffSource(
                shape, materialize, factor_bytes=source.factor_bytes,
                rank=kept_rank)
        return cleaned

    @torch.no_grad()
    def _analyze_target_group_tiled(self, source_group, active_loras, model, clip,
                                    plan, merge_refinement, n_magnitude_samples):
        self._interrupt_check()
        sources = dict(source_group["sources"])
        eff_strengths = source_group["eff_strengths"]
        indices = sorted(sources)
        controller = self._interrupt_controller
        rows = int(source_group["target_shape"][0])
        cols = math.prod(int(v) for v in source_group["target_shape"][1:])
        tile_rows = plan.rows_per_tile
        factor_bytes = sum(source.factor_bytes for source in sources.values())
        if factor_bytes <= plan.workset_bytes:
            for source in sources.values():
                source.stage(plan.device)
        sources = self._star_sources_tiled(
            sources, source_group["target_shape"], plan,
            getattr(self, "_star_eta", 100.0))

        # TAME is a global scalar transform, so determine it before conflict masks.
        tame = getattr(self, "_tame_layers", 0.0)
        if tame > 0.0:
            base_norm = self._resolve_base_norm(
                source_group["target_key"], source_group["is_clip"], model, clip)
            if base_norm:
                for index in indices:
                    if active_loras[index].get("preserve", False):
                        continue
                    total = 0.0
                    for start, end in self._iter_live_rows(
                            plan, source_group["target_shape"], 3, factor_bytes):
                        controller.check()
                        tile = sources[index].materialize_rows(start, end, plan.device)
                        value = torch.linalg.vector_norm(tile).item()
                        total += float(value) * float(value)
                    scale = self._tame_scale(
                        math.sqrt(total), base_norm,
                        getattr(self, "_tame_threshold", 0.3), tame)
                    if scale != 1.0:
                        sources[index] = ScaledDiffSource(sources[index], scale)

        pair_count = min(100000, rows * cols)
        pair_generator = torch.Generator().manual_seed(42)
        pair_indices = (torch.arange(rows * cols) if pair_count == rows * cols else
                        torch.randint(0, rows * cols, (pair_count,), generator=pair_generator))
        pair_samples = {index: torch.empty(pair_count, dtype=torch.float32) for index in indices}
        magnitude_indices = {}
        magnitude_samples = {index: torch.empty(min(n_magnitude_samples, rows * cols),
                                                 dtype=torch.float32) for index in indices}
        seed = zlib.crc32(source_group["label_prefix"].encode("utf-8")) & 0xFFFFFFFF
        mag_generator = torch.Generator().manual_seed(seed)
        for index in indices:
            count = magnitude_samples[index].numel()
            magnitude_indices[index] = (torch.arange(rows * cols) if count == rows * cols
                                        else torch.randint(0, rows * cols, (count,),
                                                           generator=mag_generator))

        norm_sq = {index: 0.0 for index in indices}
        start = 0
        factor_bytes = sum(source.factor_bytes for source in sources.values())
        while start < rows:
            controller.check()
            live_rows = self._execution_planner.shrink_rows(
                plan, source_group["target_shape"], len(indices) + 4, factor_bytes)
            if live_rows < 1:
                raise RuntimeError(
                    f"LoRA Optimizer tiled GPU cannot fit one row for {source_group['target_key']} "
                    f"shape={tuple(source_group['target_shape'])}")
            end = min(rows, start + live_rows)
            tiles = self._materialize_source_tile(
                sources, eff_strengths, active_loras, start, end, plan.device,
                merge_refinement)
            flat_start = start * cols
            flat_end = end * cols
            pair_mask = (pair_indices >= flat_start) & (pair_indices < flat_end)
            pair_positions = pair_mask.nonzero(as_tuple=False).flatten()
            pair_local = (pair_indices[pair_mask] - flat_start).to(plan.device)
            for index, tile in tiles.items():
                value = torch.linalg.vector_norm(tile).item()
                norm_sq[index] += float(value) * float(value)
                if pair_positions.numel():
                    pair_samples[index][pair_positions] = tile.flatten()[pair_local].cpu()
                mag_idx = magnitude_indices[index]
                mag_mask = (mag_idx >= flat_start) & (mag_idx < flat_end)
                mag_positions = mag_mask.nonzero(as_tuple=False).flatten()
                if mag_positions.numel():
                    local = (mag_idx[mag_mask] - flat_start).to(plan.device)
                    magnitude_samples[index][mag_positions] = tile.flatten()[local].abs().cpu()
            self._progress_update(
                "pass1", source_group["label_prefix"], end / rows)
            del tiles
            start = end

        bases = {}
        for index in indices:
            controller.check()
            def materialize(start, end, device, wanted=index):
                return self._materialize_source_tile(
                    sources, eff_strengths, active_loras, start, end, device,
                    merge_refinement)[wanted]
            rank = min(8, rows, cols)
            u, _s, vh = chunked_randomized_svd(
                materialize, source_group["target_shape"], rank, plan.device,
                tile_rows, controller, niter=1, seed=42)
            bases[index] = {"left": u, "right": vh.T}

        partial_stats = []
        for index in indices:
            norm = math.sqrt(norm_sq[index])
            partial_stats.append((index, source_group["rank_sums"].get(index, 0),
                                  norm * abs(active_loras[index]["strength"]),
                                  norm_sq[index]))

        pair_conflicts = {}
        for ai in range(len(indices)):
            for bi in range(ai + 1, len(indices)):
                i, j = indices[ai], indices[bi]
                sample_i = pair_samples[i] if eff_strengths[i] >= 0 else -pair_samples[i]
                sample_j = pair_samples[j] if eff_strengths[j] >= 0 else -pair_samples[j]
                pair_conflicts[(i, j)] = self._sample_pair_metrics(
                    sample_i, sample_j, basis_a=bases[i], basis_b=bases[j], device=None)

        weighted_magnitudes = [magnitude_samples[index] * abs(eff_strengths[index])
                               for index in indices]
        if self._execution_stats is not None:
            if plan.mode == "tiled_gpu":
                self._execution_stats["tiled_gpu"].add(source_group["target_key"])
                self._execution_stats["tile_rows"].append(tile_rows)
            else:
                self._execution_stats["cpu"].add(source_group["target_key"])
        return (
            source_group["label_prefix"], partial_stats, pair_conflicts,
            weighted_magnitudes,
            (source_group["target_key"], source_group["is_clip"]),
            source_group["skip_count"], source_group["raw_n_loras"], norm_sq,
        )


