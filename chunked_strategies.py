"""Streaming Pass 2 merge strategies for oversized targets."""

import math
import zlib

import torch

try:
    from .chunked_merge import (
        CallableDiffSource,
        ScaledDiffSource,
        chunked_randomized_svd,
        deterministic_random_like,
        iter_row_ranges,
    )
except ImportError:
    from chunked_merge import (
        CallableDiffSource,
        ScaledDiffSource,
        chunked_randomized_svd,
        deterministic_random_like,
        iter_row_ranges,
    )


class ChunkedStrategyMixin:
    def _stream_ties_selection(self, source, signed_scale, density, plan, controller):
        rows, cols = source.rows, source.cols
        total = rows * cols
        keep = max(1, int(total * density))
        if keep >= total:
            return None, None
        tile_ranges = lambda: self._iter_live_rows(
            plan, source.target_shape, 3, source.factor_bytes)
        max_value = 0.0
        bins = 4096
        for start, end in tile_ranges():
            controller.check()
            tile = source.materialize_rows(start, end, plan.device)
            if signed_scale < 0:
                tile.neg_()
            max_value = max(max_value, float(tile.abs().max().item()))
        if max_value <= 0.0:
            return float("inf"), torch.empty(0, dtype=torch.long)
        histogram = torch.zeros(bins, dtype=torch.int64)
        for start, end in tile_ranges():
            controller.check()
            tile = source.materialize_rows(start, end, plan.device)
            histogram.add_(torch.histc(tile.abs(), bins=bins, min=0.0,
                                       max=max_value).to(torch.int64).cpu())
        cumulative = 0
        threshold_bin = bins - 1
        for index in range(bins - 1, -1, -1):
            count = int(histogram[index])
            if cumulative + count >= keep:
                threshold_bin = index
                break
            cumulative += count
        low = max_value * threshold_bin / bins
        high = max_value * (threshold_bin + 1) / bins
        candidate_values = []
        candidate_indices = []
        for start, end in tile_ranges():
            controller.check()
            tile = source.materialize_rows(start, end, plan.device).abs().flatten()
            mask = (tile >= low) & (tile <= high)
            local = mask.nonzero(as_tuple=False).flatten()
            if local.numel():
                candidate_values.append(tile[local].cpu())
                candidate_indices.append(local.cpu() + start * cols)
        remaining = keep - cumulative
        values = torch.cat(candidate_values) if candidate_values else torch.empty(0)
        global_indices = (torch.cat(candidate_indices) if candidate_indices
                          else torch.empty(0, dtype=torch.long))
        order = sorted(range(values.numel()),
                       key=lambda i: (-float(values[i]), int(global_indices[i])))
        selected = global_indices[torch.tensor(order[:remaining], dtype=torch.long)]
        strict_threshold = high
        if selected.numel():
            strict_threshold = float(values[torch.tensor(order[remaining - 1])])
        return strict_threshold, selected

    @torch.no_grad()
    def _merge_group_sources_tiled(self, source_group, active_loras, plan, mode,
                                    density, majority_sign_method, preserve_flags,
                                    sparsification="disabled", sparsification_density=0.7,
                                    dare_dampening=0.0, merge_refinement="none",
                                    model=None, clip=None, star_eta=100.0,
                                    tame_layers=0.0, tame_threshold=0.3):
        sources = self._star_sources_tiled(
            source_group["sources"], source_group["target_shape"], plan, star_eta)
        strengths = source_group["eff_strengths"]
        rows = int(source_group["target_shape"][0])
        progress_key = source_group.get(
            "label_prefix", source_group.get("target_key", tuple(source_group["target_shape"])))
        factor_bytes = sum(source.factor_bytes for source in sources.values())
        buffer_count = len(sources) + 4 if mode == "ties" else 4

        def tile_ranges():
            for start, end in self._iter_live_rows(
                    plan, source_group["target_shape"], buffer_count, factor_bytes):
                yield start, end
                self._progress_update("pass2", progress_key, end / rows)

        if tame_layers > 0.0:
            base_norm = self._resolve_base_norm(
                source_group["target_key"], source_group["is_clip"], model, clip)
            if base_norm is not None:
                scaled_sources = {}
                for index, source in sources.items():
                    norm_sq = 0.0
                    for start, end in tile_ranges():
                        self._interrupt_check()
                        tile = source.materialize_rows(start, end, plan.device)
                        tile_norm = float(torch.linalg.vector_norm(tile).item())
                        norm_sq += tile_norm * tile_norm
                    scale = self._tame_scale(
                        math.sqrt(norm_sq), base_norm, tame_threshold,
                        tame_layers * min(abs(strengths[index]), 1.0))
                    scaled_sources[index] = ScaledDiffSource(source, scale)
                sources = scaled_sources
        if any(active_loras[index].get("conflict_mode", "all") != "all"
               for index in sources):
            raw_sources = sources
            tile_cache = {"key": None, "tiles": None}

            def materialize_group(start, end, device):
                key = (start, end, str(device))
                if tile_cache["key"] != key:
                    tile_cache["key"] = key
                    tile_cache["tiles"] = self._materialize_source_tile(
                        raw_sources, strengths, active_loras, start, end, device, "none")
                return tile_cache["tiles"]

            sources = {
                index: CallableDiffSource(
                    source_group["target_shape"],
                    lambda start, end, device, wanted=index:
                        materialize_group(start, end, device)[wanted],
                    factor_bytes=raw_sources[index].factor_bytes,
                    rank=raw_sources[index].rank)
                for index in raw_sources
            }
        indices = sorted(sources)
        controller = self._interrupt_controller
        shape = source_group["target_shape"]
        rows = int(shape[0])
        cols = math.prod(int(v) for v in shape[1:])
        output = torch.empty((rows, cols), device="cpu", dtype=torch.float32)
        blend_indices = [i for i, preserve in zip(indices, preserve_flags) if not preserve]
        preserve_indices = [i for i, preserve in zip(indices, preserve_flags) if preserve]
        if not blend_indices:
            blend_indices = []

        if sparsification != "disabled" and blend_indices:
            raw_sources = sources
            conflict_only = sparsification in ("dare_conflict", "della_conflict")
            conflict_cache = {"key": None, "mask": None}
            conflict_fraction = 0.0
            if conflict_only and mode != "ties":
                conflict_count = 0
                total_count = 0
                for start, end in tile_ranges():
                    controller.check()
                    has_positive = torch.zeros((end - start, cols), device=plan.device,
                                               dtype=torch.bool)
                    has_negative = torch.zeros_like(has_positive)
                    for index in blend_indices:
                        tile = raw_sources[index].materialize_rows(start, end, plan.device)
                        if strengths[index] < 0:
                            tile.neg_()
                        has_positive.logical_or_(tile > 0)
                        has_negative.logical_or_(tile < 0)
                    conflict_count += int((has_positive & has_negative).sum().item())
                    total_count += has_positive.numel()
                conflict_fraction = conflict_count / max(total_count, 1)
                if conflict_fraction > 0.40:
                    conflict_only = False
                    sparsification = "disabled"
                    self._sparsification_skipped = getattr(
                        self, "_sparsification_skipped", 0) + 1

            def conflict_mask(start, end, device):
                key = (start, end, str(device))
                if conflict_cache["key"] != key:
                    positive = torch.zeros((end - start, cols), device=device,
                                           dtype=torch.bool)
                    negative = torch.zeros_like(positive)
                    for wanted in blend_indices:
                        tile = raw_sources[wanted].materialize_rows(start, end, device)
                        if strengths[wanted] < 0:
                            tile.neg_()
                        positive.logical_or_(tile > 0)
                        negative.logical_or_(tile < 0)
                    conflict_cache["key"] = key
                    conflict_cache["mask"] = positive & negative
                return conflict_cache["mask"]

            if sparsification != "disabled":
                transformed = dict(sources)
                for index in blend_indices:
                    source = raw_sources[index]

                    def materialize_sparse(start, end, device, wanted=index, base=source):
                        tile = base.materialize_rows(start, end, device)
                        random = deterministic_random_like(
                            tile.shape, tile.device, 42, wanted, start * cols)
                        if sparsification.startswith("dare"):
                            q = (sparsification_density
                                 + dare_dampening * (1.0 - sparsification_density))
                            sparse = tile * (random < sparsification_density) * (1.0 / q)
                        else:
                            ascending = tile.abs().argsort(dim=1).argsort(dim=1).float()
                            ranks = (cols - 1) - ascending
                            p_min = max((1.0 - sparsification_density) - 0.15, 0.0)
                            drop = (p_min + (0.3 / cols) * ranks).clamp(0.0, 1.0)
                            keep = 1.0 - drop
                            mask = random < keep
                            sparse = tile * mask / keep.clamp(min=1e-6)
                        if conflict_only:
                            return torch.where(conflict_mask(start, end, device), sparse, tile)
                        return sparse

                    transformed[index] = CallableDiffSource(
                        shape, materialize_sparse,
                        factor_bytes=source.factor_bytes,
                        rank=0)
                sources = transformed

        selfish_materializer = None
        if merge_refinement != "none" and len(blend_indices) >= 2 and mode != "ties":
            pre_refine_sources = sources
            tall_cache = {"key": None, "tiles": None, "selfish": None}

            def materialize_tall(start, end, device):
                key = (start, end, str(device))
                if tall_cache["key"] != key:
                    tiles = {index: pre_refine_sources[index].materialize_rows(
                        start, end, device) for index in blend_indices}
                    merged = torch.zeros((end - start, cols), device=device)
                    contributions = {}
                    masks = {}
                    for index in blend_indices:
                        contribution = tiles[index] * strengths[index]
                        contributions[index] = contribution
                        merged.add_(contribution)
                    agreement = torch.zeros_like(merged)
                    for index in blend_indices:
                        mask = contributions[index].abs() >= (
                            merged - contributions[index]).abs()
                        masks[index] = mask
                        agreement.add_(mask.float())
                    selfish = torch.zeros_like(merged)
                    consensus = {}
                    for index in blend_indices:
                        mask = masks[index] & (agreement == 1)
                        selfish.add_(contributions[index] * mask)
                        consensus[index] = torch.where(mask, torch.zeros_like(tiles[index]),
                                                       tiles[index])
                    tall_cache.update(key=key, tiles=consensus, selfish=selfish)
                return tall_cache["tiles"], tall_cache["selfish"]

            sources = dict(sources)
            for index in blend_indices:
                source = pre_refine_sources[index]
                sources[index] = CallableDiffSource(
                    shape,
                    lambda start, end, device, wanted=index:
                        materialize_tall(start, end, device)[0][wanted],
                    factor_bytes=source.factor_bytes,
                    rank=0)
            selfish_materializer = lambda start, end, device: materialize_tall(
                start, end, device)[1]

            gram = torch.zeros((len(blend_indices), len(blend_indices)), dtype=torch.float64)
            for start, end in tile_ranges():
                controller.check()
                flats = [sources[index].materialize_rows(start, end, plan.device).flatten()
                         for index in blend_indices]
                for i in range(len(flats)):
                    for j in range(i + 1):
                        value = float(torch.dot(flats[i], flats[j]).item())
                        gram[i, j] += value
                        if i != j:
                            gram[j, i] += value
            coefficients = []
            for i in range(len(blend_indices)):
                magnitude = math.sqrt(max(float(gram[i, i]), 0.0))
                coeff = torch.zeros(len(blend_indices), dtype=torch.float64)
                if magnitude > 1e-8:
                    coeff[i] = 1.0 / magnitude
                for prior in coefficients:
                    projection = float(coeff @ gram @ prior)
                    coeff -= projection * prior
                norm = math.sqrt(max(float(coeff @ gram @ coeff), 0.0))
                if norm > 1e-8:
                    coeff /= norm
                else:
                    coeff.zero_()
                coefficients.append(coeff)
            orthogonal_bases = sources
            refined_sources = dict(orthogonal_bases)
            for position, index in enumerate(blend_indices):
                source = sources[index]
                magnitude = math.sqrt(max(float(gram[position, position]), 0.0))
                coeff = coefficients[position] * magnitude

                def materialize_ortho(start, end, device, weights=coeff):
                    result = torch.zeros((end - start, cols), device=device)
                    for source_pos, source_index in enumerate(blend_indices):
                        weight = float(weights[source_pos])
                        if weight:
                            result.add_(orthogonal_bases[source_index].materialize_rows(
                                start, end, device), alpha=weight)
                    return result

                refined_sources[index] = CallableDiffSource(
                    shape, materialize_ortho,
                    factor_bytes=source.factor_bytes, rank=0)
            sources = refined_sources

            if merge_refinement == "full":
                aligned_inputs = sources
                total_weight = sum(abs(strengths[index]) for index in blend_indices)

                def materialize_target(start, end, device):
                    target = torch.zeros((end - start, cols), device=device)
                    if total_weight > 0:
                        for source_index in blend_indices:
                            target.add_(aligned_inputs[source_index].materialize_rows(
                                start, end, device),
                                alpha=abs(strengths[source_index]) / total_weight)
                    return target

                target_sum = torch.zeros(cols, device=plan.device)
                source_sums = {index: torch.zeros(cols, device=plan.device)
                               for index in blend_indices}
                for start, end in tile_ranges():
                    controller.check()
                    target_sum.add_(materialize_target(start, end, plan.device).sum(dim=0))
                    for index in blend_indices:
                        source_sums[index].add_(aligned_inputs[index].materialize_rows(
                            start, end, plan.device).sum(dim=0))
                target_mean = target_sum / max(rows, 1)
                aligned_sources = dict(aligned_inputs)
                for position, index in enumerate(blend_indices):
                    controller.check()
                    source_mean = source_sums[index] / max(rows, 1)
                    if cols > 32:
                        projection_rank = min(24, cols - 1)
                        generator = torch.Generator(device=plan.device)
                        generator.manual_seed(1729 + position)
                        projection = torch.linalg.qr(torch.randn(
                            cols, projection_rank, device=plan.device,
                            generator=generator, dtype=torch.float32)).Q
                        covariance = torch.zeros((projection_rank, projection_rank),
                                                 device=plan.device)
                    else:
                        projection = None
                        covariance = torch.zeros((cols, cols), device=plan.device)
                    for start, end in tile_ranges():
                        controller.check()
                        source_tile = aligned_inputs[index].materialize_rows(
                            start, end, plan.device) - source_mean
                        target_tile = materialize_target(
                            start, end, plan.device) - target_mean
                        if projection is not None:
                            source_tile = source_tile @ projection
                            target_tile = target_tile @ projection
                        covariance.add_(source_tile.T @ target_tile)
                    u, _singular, vh = torch.linalg.svd(covariance)
                    rotation = u @ vh
                    base = aligned_inputs[index]

                    def materialize_aligned(start, end, device, source=base,
                                            p=projection, r=rotation):
                        tile = source.materialize_rows(start, end, device)
                        if p is None:
                            return tile @ r
                        projected = tile @ p
                        return tile + ((projected @ r - projected) @ p.T)

                    aligned_sources[index] = CallableDiffSource(
                        shape, materialize_aligned,
                        factor_bytes=base.factor_bytes, rank=0)
                sources = aligned_sources

        norms = {}
        if mode == "slerp":
            for index in blend_indices:
                total = 0.0
                for start, end in tile_ranges():
                    controller.check()
                    tile = sources[index].materialize_rows(start, end, plan.device)
                    value = torch.linalg.vector_norm(tile).item()
                    total += float(value) * float(value)
                norms[index] = math.sqrt(total)
            total_weight = sum(abs(strengths[i]) for i in blend_indices)
            target_norm = (sum(norms[i] * abs(strengths[i]) for i in blend_indices)
                           / total_weight) if total_weight else 0.0
            if len(blend_indices) == 2 and total_weight:
                first, second = sorted(blend_indices,
                                       key=lambda i: abs(strengths[i]), reverse=True)
                dot = 0.0
                for start, end in tile_ranges():
                    controller.check()
                    a = sources[first].materialize_rows(start, end, plan.device)
                    b = sources[second].materialize_rows(start, end, plan.device)
                    if strengths[first] < 0:
                        a.neg_()
                    if strengths[second] < 0:
                        b.neg_()
                    dot += float(torch.dot(a.flatten(), b.flatten()).item())
                denom = norms[first] * norms[second]
                cosine = max(-1.0, min(1.0, dot / denom)) if denom > 0 else 1.0
                theta = math.acos(cosine)
                fraction = abs(strengths[second]) / total_weight
                if theta < 1e-6:
                    coef_a, coef_b = 1.0 - fraction, fraction
                else:
                    sin_theta = math.sin(theta)
                    coef_a = math.sin((1.0 - fraction) * theta) / sin_theta
                    coef_b = math.sin(fraction * theta) / sin_theta
                current_sq = 0.0
                for start, end in tile_ranges():
                    controller.check()
                    a = sources[first].materialize_rows(start, end, plan.device)
                    b = sources[second].materialize_rows(start, end, plan.device)
                    if strengths[first] < 0:
                        a.neg_()
                    if strengths[second] < 0:
                        b.neg_()
                    tile = a.mul(coef_a).add_(b, alpha=coef_b)
                    current_sq += float(torch.linalg.vector_norm(tile).item()) ** 2
                    output[start:end].copy_(tile.cpu())
                current = math.sqrt(current_sq)
                if current > 1e-8:
                    output.mul_(target_norm / current)
            elif len(blend_indices) >= 3 and total_weight:
                weights = {i: abs(strengths[i]) / total_weight for i in blend_indices}
                for start, end in tile_ranges():
                    controller.check()
                    tile = torch.zeros((end - start, cols), device=plan.device)
                    for index in blend_indices:
                        value = sources[index].materialize_rows(start, end, plan.device)
                        if strengths[index] < 0:
                            value.neg_()
                        if norms[index] > 1e-12:
                            tile.add_(value, alpha=weights[index] / norms[index])
                    output[start:end].copy_(tile.cpu())
                m_norm = torch.linalg.vector_norm(output).item()
                if m_norm > 1e-8:
                    output.div_(m_norm)
                else:
                    first = blend_indices[0]
                    for start, end in tile_ranges():
                        value = sources[first].materialize_rows(start, end, plan.device)
                        if strengths[first] < 0:
                            value.neg_()
                        output[start:end].copy_((value / max(norms[first], 1e-12)).cpu())
                for _ in range(8):
                    controller.check()
                    cosines = {i: 0.0 for i in blend_indices}
                    for start, end in tile_ranges():
                        m_tile = output[start:end].to(plan.device)
                        for index in blend_indices:
                            value = sources[index].materialize_rows(start, end, plan.device)
                            if strengths[index] < 0:
                                value.neg_()
                            if norms[index] > 1e-12:
                                cosines[index] += float(torch.dot(
                                    value.flatten() / norms[index], m_tile.flatten()).item())
                    tangent = torch.empty_like(output)
                    tangent_sq = 0.0
                    for start, end in tile_ranges():
                        controller.check()
                        m_tile = output[start:end].to(plan.device)
                        tile = torch.zeros_like(m_tile)
                        radial = 0.0
                        for index in blend_indices:
                            cosine = max(-1.0 + 1e-7, min(1.0 - 1e-7, cosines[index]))
                            theta = math.acos(cosine)
                            coef = (0.0 if theta < 1e-7 else
                                    weights[index] * theta / math.sin(theta))
                            value = sources[index].materialize_rows(start, end, plan.device)
                            if strengths[index] < 0:
                                value.neg_()
                            if norms[index] > 1e-12:
                                tile.add_(value, alpha=coef / norms[index])
                            radial += coef * cosine
                        tile.add_(m_tile, alpha=-radial)
                        tangent_sq += float(torch.linalg.vector_norm(tile).item()) ** 2
                        tangent[start:end].copy_(tile.cpu())
                    tangent_norm = math.sqrt(tangent_sq)
                    if tangent_norm < 1e-7:
                        break
                    output.mul_(math.cos(tangent_norm)).add_(
                        tangent, alpha=math.sin(tangent_norm) / tangent_norm)
                    output.div_(torch.linalg.vector_norm(output).clamp(min=1e-12))
                output.mul_(target_norm)
            else:
                output.zero_()
        elif mode == "consensus":
            abs_weights = {index: abs(strengths[index]) for index in blend_indices}
            total_weight = sum(abs_weights.values())
            input_norms = {}
            for index in blend_indices:
                total = 0.0
                for start, end in tile_ranges():
                    controller.check()
                    tile = sources[index].materialize_rows(start, end, plan.device)
                    value = torch.linalg.vector_norm(tile).item()
                    total += float(value) * float(value)
                input_norms[index] = math.sqrt(total) * abs_weights[index]
            merged_sq = 0.0
            for start, end in tile_ranges():
                controller.check()
                numerator = torch.zeros((end - start, cols), device=plan.device)
                denominator = torch.zeros_like(numerator)
                for index in blend_indices:
                    tile = sources[index].materialize_rows(start, end, plan.device)
                    importance = tile.square()
                    numerator.add_(tile * strengths[index] * importance)
                    denominator.add_(importance, alpha=abs_weights[index])
                tile = torch.where(denominator > 0, numerator / denominator,
                                   torch.zeros_like(numerator))
                merged_sq += float(torch.linalg.vector_norm(tile).item()) ** 2
                output[start:end].copy_(tile.cpu())
            merged_norm = math.sqrt(merged_sq)
            target_norm = (sum(input_norms.values()) / total_weight
                           if total_weight > 0 else 0.0)
            if merged_norm > 1e-8:
                output.mul_(target_norm / merged_norm)

            if min(rows, cols) >= 4:
                rank_budget = min(rows, cols, 128)
                u, singular, vh = chunked_randomized_svd(
                    lambda start, end, device: output[start:end].to(device),
                    shape, rank_budget, plan.device, plan.rows_per_tile,
                    controller, niter=2, seed=42)
                singular_sum = singular.sum()
                if singular_sum > 1e-10:
                    probability = (singular / singular_sum).clamp(min=1e-10)
                    effective_rank = max(1, min(
                        int(math.exp(float(-(probability * probability.log()).sum())) + 0.5),
                        rank_budget))
                    positions = torch.arange(rank_budget, dtype=torch.float32)
                    gate = torch.sigmoid(
                        4.0 * (positions - effective_rank)
                        * (-1.0 / max(effective_rank, 1)))
                    gated = singular * gate
                    pre_norm = torch.linalg.vector_norm(output).item()
                    post_sq = 0.0
                    for start, end in tile_ranges():
                        controller.check()
                        tile = ((u[start:end].to(plan.device)
                                 * gated.to(plan.device).unsqueeze(0))
                                @ vh.to(plan.device))
                        post_sq += float(torch.linalg.vector_norm(tile).item()) ** 2
                        output[start:end].copy_(tile.cpu())
                    post_norm = math.sqrt(post_sq)
                    if post_norm > 1e-8:
                        output.mul_(pre_norm / post_norm)
        elif mode == "ties":
            selections = {}
            for index in blend_indices:
                if sparsification == "disabled":
                    selections[index] = self._stream_ties_selection(
                        sources[index], strengths[index], density, plan, controller)
                else:
                    selections[index] = (-1.0, torch.empty(0, dtype=torch.long))
            for start, end in tile_ranges():
                controller.check()
                trimmed = []
                abs_weights = []
                for index in blend_indices:
                    tile = sources[index].materialize_rows(start, end, plan.device)
                    if strengths[index] < 0:
                        tile.neg_()
                    threshold, boundary = selections[index]
                    if threshold is not None:
                        flat = tile.flatten()
                        global_index = (torch.arange(flat.numel(), device=plan.device)
                                        + start * cols)
                        mask = flat.abs() > threshold
                        if boundary is not None and boundary.numel():
                            mask |= torch.isin(global_index, boundary.to(plan.device))
                        tile = (flat * mask).reshape_as(tile)
                    trimmed.append(tile)
                    abs_weights.append(abs(strengths[index]))
                if trimmed:
                    majority = self._ties_elect_sign(trimmed, majority_sign_method)
                    result = self._ties_disjoint_merge(trimmed, abs_weights, majority)
                else:
                    result = torch.zeros((end - start, cols), device=plan.device)
                output[start:end].copy_(result.cpu())
        else:
            total_weight = sum(abs(strengths[i]) for i in blend_indices)
            if mode == "weighted_average" and total_weight > 0:
                scale = 1.0 / total_weight
            elif mode == "normalize":
                energy = math.sqrt(sum(strengths[i] ** 2 for i in blend_indices))
                scale = 1.0 / energy if energy > 0 else 1.0
            else:
                scale = 1.0
            for start, end in tile_ranges():
                controller.check()
                tile = torch.zeros((end - start, cols), device=plan.device)
                for index in blend_indices:
                    tile.add_(sources[index].materialize_rows(start, end, plan.device),
                              alpha=strengths[index] * scale)
                output[start:end].copy_(tile.cpu())

        if selfish_materializer is not None:
            for start, end in tile_ranges():
                controller.check()
                tile = output[start:end].to(plan.device)
                tile.add_(selfish_materializer(start, end, plan.device))
                output[start:end].copy_(tile.cpu())

        if preserve_indices:
            for start, end in tile_ranges():
                controller.check()
                tile = output[start:end].to(plan.device)
                for index in preserve_indices:
                    tile.add_(sources[index].materialize_rows(start, end, plan.device),
                              alpha=strengths[index])
                output[start:end].copy_(tile.cpu())
        return output.reshape(shape)


