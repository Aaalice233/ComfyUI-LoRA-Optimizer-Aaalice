"""Row-addressable source parsing for the tiled optimizer integration."""

import torch

from comfy.weight_adapter.lora import LoRAAdapter
from comfy.weight_adapter.loha import LoHaAdapter
from comfy.weight_adapter.lokr import LoKrAdapter

try:
    from .chunked_merge import (
        DenseDiffSource,
        LoHaDiffSource,
        LoKrDiffSource,
        LoRADiffSource,
        SumDiffSource,
    )
except ImportError:
    from chunked_merge import (
        DenseDiffSource,
        LoHaDiffSource,
        LoKrDiffSource,
        LoRADiffSource,
        SumDiffSource,
    )


class ChunkedSourceMixin:
    def _payload_diff_source(self, payload, target_shape):
        """Normalize a captured/virtual payload into a row-addressable source."""
        if isinstance(payload, torch.Tensor):
            try:
                return DenseDiffSource(payload.reshape(target_shape), target_shape)
            except RuntimeError:
                return None
        if isinstance(payload, LoHaAdapter):
            weights = payload.weights
            if len(weights) >= 7:
                w1a, w1b, alpha, w2a, w2b, t1, t2 = weights[:7]
                return LoHaDiffSource(w1a, w1b, alpha, w2a, w2b, t1, t2, target_shape)
        if isinstance(payload, LoKrAdapter):
            weights = payload.weights
            if len(weights) >= 8:
                w1, w2, alpha, w1a, w1b, w2a, w2b, t2 = weights[:8]
                return LoKrDiffSource(w1, w2, alpha, w1a, w1b, w2a, w2b, t2,
                                      target_shape, adapter_scaling=True)
        if isinstance(payload, LoRAAdapter):
            weights = payload.weights
            dora_scale = weights[4] if len(weights) > 4 else None
            if dora_scale is not None:
                # DoRA is not a plain additive diff; preserve the existing full
                # expansion semantics rather than pretending it is linear.
                return None
            if len(weights) >= 4:
                up, down, alpha, mid = weights[:4]
                return LoRADiffSource(up, down, alpha, mid, target_shape)
        if isinstance(payload, tuple) and len(payload) > 1 and payload[0] == "diff":
            values = payload[1]
            tensor = values[0] if isinstance(values, (tuple, list)) else values
            if isinstance(tensor, torch.Tensor):
                try:
                    return DenseDiffSource(tensor.reshape(target_shape), target_shape)
                except RuntimeError:
                    return None
        return None

    def _dict_alt_diff_source(self, lora_dict, alias, target_shape):
        if self._has_lokr_keys(lora_dict, alias):
            p = alias
            alpha = lora_dict.get(f"{p}.alpha")
            alpha = alpha.item() if alpha is not None else None
            return LoKrDiffSource(
                lora_dict.get(f"{p}.lokr_w1"), lora_dict.get(f"{p}.lokr_w2"), alpha,
                lora_dict.get(f"{p}.lokr_w1_a"), lora_dict.get(f"{p}.lokr_w1_b"),
                lora_dict.get(f"{p}.lokr_w2_a"), lora_dict.get(f"{p}.lokr_w2_b"),
                lora_dict.get(f"{p}.lokr_t2"), target_shape)
        if self._has_loha_keys(lora_dict, alias):
            p = alias
            alpha = lora_dict.get(f"{p}.alpha")
            w1b = lora_dict[f"{p}.hada_w1_b"]
            alpha = alpha.item() if alpha is not None else int(w1b.shape[0])
            return LoHaDiffSource(
                lora_dict[f"{p}.hada_w1_a"], w1b, alpha,
                lora_dict[f"{p}.hada_w2_a"], lora_dict[f"{p}.hada_w2_b"],
                lora_dict.get(f"{p}.hada_t1"), lora_dict.get(f"{p}.hada_t2"),
                target_shape)
        return None

    def _prepare_group_sources(self, target_group, active_loras, model, clip,
                               auto_scale=1.0):
        """Resolve one target into lightweight sources without dense expansion."""
        target_key = self._group_target_key(target_group)
        is_clip = target_group["is_clip"]
        try:
            target_shape = self._resolve_target_shape(target_key, is_clip, model, clip)
        except (AttributeError, RuntimeError, IndexError):
            return None

        sources = {}
        skip_count = 0
        for index, item in enumerate(active_loras):
            self._interrupt_check()
            contributions = []
            if item.get("_precomputed_diffs"):
                payload = item["lora"].get(target_key)
                if payload is None and isinstance(target_key, tuple):
                    payload = item["lora"].get(target_key[0])
                if payload is not None:
                    source = self._payload_diff_source(payload, target_shape)
                    if source is None:
                        # Unknown third-party payload semantics cannot be inferred
                        # safely from shape alone; retain the established CPU path.
                        return {"unsupported": True, "reason": "unknown third-party payload"}
                    contributions.append(source)
            else:
                for alias in target_group["aliases"]:
                    info = self._get_lora_key_info(item["lora"], alias)
                    if info is not None:
                        up, down, alpha, mid = info
                        source_out = (int(up.reshape(up.shape[0], -1).shape[1])
                                      if mid is not None else int(up.shape[0]))
                        if source_out != int(target_shape[0]):
                            self._note_shape_mismatch(
                                item, target_key, source_out, int(target_shape[0]))
                            continue
                        contributions.append(
                            LoRADiffSource(up, down, alpha, mid, target_shape))
                    else:
                        source = self._dict_alt_diff_source(item["lora"], alias, target_shape)
                        if source is not None:
                            contributions.append(source)
            if contributions:
                sources[index] = contributions[0] if len(contributions) == 1 else SumDiffSource(contributions)
            else:
                skip_count += 1

        raw_n = len(sources)
        filtered = {}
        eff_strengths = {}
        is_audio_group = self._target_is_audio(target_group)
        for index, source in sources.items():
            key_filter = active_loras[index].get("key_filter", "all")
            if key_filter == "shared_only" and raw_n < 2:
                continue
            if key_filter == "unique_only" and raw_n != 1:
                continue
            if key_filter == "audio_only" and not is_audio_group:
                continue
            if key_filter == "no_audio" and is_audio_group:
                continue
            filtered[index] = source
            eff_strengths[index] = self._resolve_branch_strength(
                active_loras[index], is_clip) * auto_scale

        rank_bound = None
        if filtered and all(source.rank_bound_known for source in filtered.values()):
            rank_bound = sum(source.rank for source in filtered.values())
        storage_dtype = next((source.storage_dtype for source in filtered.values()), None)
        return {
            "label_prefix": target_group["label_prefix"],
            "target_key": target_key,
            "is_clip": is_clip,
            "raw_n_loras": raw_n,
            "sources": filtered,
            "eff_strengths": eff_strengths,
            "rank_sums": {index: source.rank for index, source in filtered.items()},
            "rank_bound": rank_bound,
            "target_shape": target_shape,
            "storage_dtype": storage_dtype,
            "skip_count": skip_count,
        }



