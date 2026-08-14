#!/usr/bin/env python3
"""Synthetic benchmark for LoRA Optimizer full/tiled/CPU merge paths."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PLUGIN))

import torch
from safetensors import safe_open

from chunked_merge import ExecutionPlanner, InterruptController, LoRADiffSource, iter_row_ranges


def rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return 0


def make_sources(rows, cols, rank, count, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sources = []
    weights = []
    for index in range(count):
        up = torch.randn(rows, rank, generator=generator, dtype=torch.float32)
        down = torch.randn(rank, cols, generator=generator, dtype=torch.float32)
        sources.append(LoRADiffSource(up, down, rank, None, (rows, cols)))
        weights.append(1.0 / count)
    return sources, weights, []


def make_real_sources(directory, count):
    grouped = defaultdict(list)
    for path in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for down_key in keys:
                if down_key.endswith(".lora_A.weight"):
                    up_key = down_key.replace(".lora_A.weight", ".lora_B.weight")
                elif down_key.endswith(".lora_down.weight"):
                    up_key = down_key.replace(".lora_down.weight", ".lora_up.weight")
                else:
                    continue
                if up_key not in keys:
                    continue
                down_shape = handle.get_slice(down_key).get_shape()
                up_shape = handle.get_slice(up_key).get_shape()
                if len(down_shape) == 2 and len(up_shape) == 2:
                    grouped[(up_shape[0], down_shape[1])].append(
                        (path, down_key, up_key))
    eligible = [(shape, entries) for shape, entries in grouped.items()
                if len({entry[0] for entry in entries}) >= count]
    if not eligible:
        raise RuntimeError(
            f"No common 2D LoRA target exists across {count} files in {directory}")
    shape, entries = max(eligible, key=lambda item: item[0][0] * item[0][1])
    selected = []
    seen = set()
    for entry in entries:
        if entry[0] not in seen:
            seen.add(entry[0])
            selected.append(entry)
        if len(selected) == count:
            break
    sources = []
    names = []
    for path, down_key, up_key in selected:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            down = handle.get_tensor(down_key)
            up = handle.get_tensor(up_key)
        rank = int(down.shape[0])
        sources.append(LoRADiffSource(up, down, rank, None, shape))
        names.append(path.name)
    return sources, [1.0 / count] * count, names


def merge_full(sources, weights, device, controller):
    values = []
    for source in sources:
        controller.check()
        values.append(source.materialize_full(device))
    output = torch.zeros_like(values[0])
    for value, weight in zip(values, weights):
        controller.check()
        output.add_(value, alpha=weight)
    return output.cpu(), len(sources)


def merge_tiled(sources, weights, device, tile_rows, controller):
    rows, cols = sources[0].target_shape
    output = torch.empty(rows, cols, device="cpu", dtype=torch.float32)
    tile_count = 0
    for start, end in iter_row_ranges(rows, tile_rows):
        controller.check()
        merged = torch.zeros(end - start, cols, device=device, dtype=torch.float32)
        for source, weight in zip(sources, weights):
            controller.check()
            merged.add_(source.materialize_rows(start, end, device), alpha=weight)
        output[start:end].copy_(merged.cpu())
        tile_count += 1
    return output, tile_count


def analyze_full(sources, device, controller):
    diffs = []
    for source in sources:
        controller.check()
        diffs.append(source.materialize_rows(0, source.target_shape[0], device))
    norms = [float(torch.linalg.vector_norm(diff).item()) for diff in diffs]
    pair_dot = float(torch.sum(diffs[0] * diffs[1]).item()) if len(diffs) > 1 else 0.0
    return norms, pair_dot


def analyze_tiled(sources, device, tile_rows, controller):
    norm_sq = [0.0] * len(sources)
    pair_dot = 0.0
    rows = sources[0].target_shape[0]
    for start, end in iter_row_ranges(rows, tile_rows):
        controller.check()
        tiles = [source.materialize_rows(start, end, device) for source in sources]
        for index, tile in enumerate(tiles):
            norm_sq[index] += float(torch.sum(tile * tile).item())
        if len(tiles) > 1:
            pair_dot += float(torch.sum(tiles[0] * tiles[1]).item())
    return [value ** 0.5 for value in norm_sq], pair_dot


def numerical_error(sources, weights, output, tile_rows):
    max_abs = 0.0
    error_sq = 0.0
    reference_sq = 0.0
    for start, end in iter_row_ranges(sources[0].rows, tile_rows):
        expected = torch.zeros(end - start, sources[0].cols)
        for source, weight in zip(sources, weights):
            expected.add_(source.materialize_rows(start, end, torch.device("cpu")), alpha=weight)
        error = output[start:end] - expected
        max_abs = max(max_abs, float(error.abs().max().item()))
        error_sq += float(torch.sum(error.double().square()).item())
        reference_sq += float(torch.sum(expected.double().square()).item())
    return max_abs, (error_sq / max(reference_sq, 1e-30)) ** 0.5


def run(args):
    if args.mode == "cpu":
        device = torch.device("cpu")
    elif args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm GPU is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.real_lora_dir:
        sources, weights, source_names = make_real_sources(
            args.real_lora_dir, args.loras)
        args.rows, args.cols = (int(value) for value in sources[0].target_shape)
        args.rank = max(source.rank for source in sources)
    else:
        sources, weights, source_names = make_sources(
            args.rows, args.cols, args.rank, args.loras, args.seed)
    factor_bytes = sum(source.factor_bytes for source in sources)
    if device.type == "cuda":
        for source in sources:
            source.stage(device)
    planner = ExecutionPlanner()
    plan = planner.plan(
        device, (args.rows, args.cols), args.loras, args.loras + 3,
        factor_bytes=factor_bytes)
    tile_rows = args.tile_rows or (
        plan.rows_per_tile if plan.mode == "tiled_gpu" else min(args.rows, 256))

    controller = InterruptController()
    if args.cancel_after > 0:
        timer = threading.Timer(args.cancel_after, controller.cancel)
        timer.start()
    else:
        timer = None

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    rss_start = rss_bytes()
    started = time.perf_counter()
    cancelled = False
    output = None
    norms = None
    pair_dot = None
    tiles = 0
    analysis_seconds = 0.0
    merge_seconds = 0.0
    try:
        analysis_started = time.perf_counter()
        if args.mode == "full_gpu":
            if device.type != "cuda":
                raise RuntimeError("full_gpu requires --device gpu")
            norms, pair_dot = analyze_full(sources, device, controller)
        else:
            norms, pair_dot = analyze_tiled(sources, device, tile_rows, controller)
        analysis_seconds = time.perf_counter() - analysis_started

        merge_started = time.perf_counter()
        if args.mode == "full_gpu":
            output, tiles = merge_full(sources, weights, device, controller)
        else:
            output, tiles = merge_tiled(sources, weights, device, tile_rows, controller)
        merge_seconds = time.perf_counter() - merge_started
    except BaseException as error:
        if controller.event.is_set():
            cancelled = True
        else:
            raise
    finally:
        if timer is not None:
            timer.cancel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if output is not None:
        max_abs_error, relative_l2_error = numerical_error(
            sources, weights, output, tile_rows)
    else:
        max_abs_error = None
        relative_l2_error = None

    result = {
        "mode": args.mode,
        "device": str(device),
        "shape": [args.rows, args.cols],
        "rank": args.rank,
        "loras": args.loras,
        "source_files": source_names,
        "planner_mode": plan.mode,
        "planner_reason": plan.reason,
        "tile_rows": tile_rows,
        "tile_count": tiles,
        "wall_seconds": round(elapsed, 4),
        "stage_seconds": {
            "analysis": round(analysis_seconds, 4),
            "merge": round(merge_seconds, 4),
        },
        "cancelled": cancelled,
        "cancel_latency_seconds": (
            round(max(0.0, elapsed - args.cancel_after), 4)
            if cancelled and args.cancel_after > 0 else None),
        "cpu_rss_delta_mib": round((rss_bytes() - rss_start) / 1024 ** 2, 2),
        "gpu_peak_allocated_mib": (
            round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2)
            if device.type == "cuda" else 0),
        "gpu_peak_reserved_mib": (
            round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 2)
            if device.type == "cuda" else 0),
        "analysis_norm_mean": (sum(norms) / len(norms)) if norms else None,
        "analysis_first_pair_dot": pair_dot,
        "output_norm": float(torch.linalg.vector_norm(output).item()) if output is not None else None,
        "max_abs_error": max_abs_error,
        "relative_l2_error": relative_l2_error,
    }
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full_gpu", "tiled_gpu", "cpu"), default="tiled_gpu")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=8192)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--loras", type=int, default=9)
    parser.add_argument("--real-lora-dir", type=str, default="")
    parser.add_argument("--tile-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cancel-after", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
