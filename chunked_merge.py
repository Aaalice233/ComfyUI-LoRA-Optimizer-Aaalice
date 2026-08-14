"""Private tiled tensor engine for LoRA Optimizer.

This module deliberately has no node, workflow, model-key, or patcher knowledge.
It owns row-addressable diff sources, GPU execution planning, cooperative ComfyUI
interruption, deterministic logical-block randomness, and streamed tensor math.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import threading
from typing import Callable, Iterable, Sequence

import torch
import comfy.model_management

_MIB = 1024 * 1024
_LOGICAL_RANDOM_ELEMENTS = 1_048_576


class InterruptController:
    """Execution-scoped bridge from ComfyUI's one-shot flag to all workers."""

    def __init__(self):
        self.event = threading.Event()

    def check(self):
        exception_type = getattr(
            comfy.model_management, "InterruptProcessingException", RuntimeError)
        if self.event.is_set():
            raise exception_type()
        checker = getattr(
            comfy.model_management, "throw_exception_if_processing_interrupted", None)
        if checker is None:
            return
        try:
            checker()
        except exception_type:
            self.event.set()
            raise

    def cancel(self):
        self.event.set()


@dataclass(frozen=True)
class ExecutionPlan:
    mode: str
    device: torch.device
    rows_per_tile: int
    estimated_full_bytes: int
    workset_bytes: int
    reason: str = ""

    @property
    def tiled(self):
        return self.mode == "tiled_gpu"


class ExecutionPlanner:
    """Plan one target without relying on OOM recovery."""

    def __init__(self, free_memory: Callable[[torch.device], int] | None = None):
        self.free_memory = free_memory or comfy.model_management.get_free_memory

    @staticmethod
    def _total_memory(device: torch.device, free_bytes: int):
        if device.type == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(device).total_memory)
        return free_bytes

    @staticmethod
    def _workset_override():
        raw = os.environ.get("LORA_OPTIMIZER_TILE_MB", "").strip()
        if not raw:
            return None
        try:
            return max(16, min(512, int(float(raw)))) * _MIB
        except ValueError:
            return None

    def plan(self, device, target_shape, contributor_count, buffer_count,
             factor_bytes=0, chunkable=True, force_cpu=False, reason=""):
        cpu = torch.device("cpu")
        rows = max(1, int(target_shape[0]))
        cols = math.prod(int(v) for v in target_shape[1:]) if len(target_shape) > 1 else 1
        if force_cpu or device is None or torch.device(device).type != "cuda":
            tile_rows = min(rows, max(1, (128 * _MIB) // max(1, cols * torch.float32.itemsize)))
            if tile_rows >= 8:
                tile_rows = max(8, (tile_rows // 8) * 8)
            return ExecutionPlan("cpu", cpu, tile_rows, 0, 0,
                                 reason or "GPU unavailable or CPU explicitly selected")

        device = torch.device(device)
        dense_bytes = rows * cols * torch.float32.itemsize
        estimated_full = dense_bytes * max(1, int(buffer_count)) + int(factor_bytes)
        free_bytes = int(self.free_memory(device))
        total_bytes = self._total_memory(device, free_bytes)
        reserve = max(512 * _MIB, int(total_bytes * 0.10))
        usable = max(0, free_bytes - reserve - int(factor_bytes))

        if estimated_full <= int(usable * 0.80):
            return ExecutionPlan("full_gpu", device, rows, estimated_full,
                                 estimated_full, "full target fits with safety reserve")

        if not chunkable:
            return ExecutionPlan("cpu", cpu, rows, estimated_full, 0,
                                 reason or "payload does not support row materialization")

        override = self._workset_override()
        workset = override if override is not None else min(512 * _MIB, usable // 2)
        workset = min(workset, usable)
        bytes_per_row = cols * torch.float32.itemsize * max(1, int(buffer_count))
        tile_rows = workset // max(1, bytes_per_row)
        single_dense_cap_rows = (128 * _MIB) // max(1, cols * torch.float32.itemsize)
        tile_rows = min(tile_rows, max(1, single_dense_cap_rows), rows)
        if tile_rows >= 8:
            tile_rows = max(8, (tile_rows // 8) * 8)
        if tile_rows < 1:
            return ExecutionPlan("cpu", cpu, rows, estimated_full, workset,
                                 reason or "one output row does not fit the safe GPU workset")
        return ExecutionPlan("tiled_gpu", device, tile_rows, estimated_full,
                             tile_rows * bytes_per_row, "full target exceeds safe GPU budget")

    def shrink_rows(self, plan: ExecutionPlan, target_shape, buffer_count, factor_bytes=0):
        """Re-check live free VRAM before a tile and only shrink its physical size."""
        if plan.mode != "tiled_gpu":
            return plan.rows_per_tile
        free_bytes = int(self.free_memory(plan.device))
        total_bytes = self._total_memory(plan.device, free_bytes)
        reserve = max(512 * _MIB, int(total_bytes * 0.10))
        usable = max(0, free_bytes - reserve - int(factor_bytes))
        cols = math.prod(int(v) for v in target_shape[1:]) if len(target_shape) > 1 else 1
        bytes_per_row = cols * torch.float32.itemsize * max(1, int(buffer_count))
        rows = min(plan.rows_per_tile, usable // max(1, bytes_per_row))
        if rows >= 8:
            rows = max(8, (rows // 8) * 8)
        return max(0, int(rows))


class DiffSource:
    """A target-shaped diff that can reconstruct a contiguous output-row range."""

    def __init__(self, target_shape, storage_dtype, rank=1, rank_bound_known=False):
        self.target_shape = torch.Size(target_shape)
        self.storage_dtype = storage_dtype
        self.rank = int(rank)
        self.rank_bound_known = bool(rank_bound_known)

    @property
    def rows(self):
        return int(self.target_shape[0])

    @property
    def cols(self):
        return math.prod(int(v) for v in self.target_shape[1:]) if len(self.target_shape) > 1 else 1

    @property
    def factor_bytes(self):
        return 0

    def materialize_rows(self, start, end, device):
        raise NotImplementedError

    def stage(self, device):
        return self

    def release(self):
        return None

    def materialize_full(self, device):
        return self.materialize_rows(0, self.rows, device).reshape(self.target_shape)


class CallableDiffSource(DiffSource):
    """Row source backed by a group-aware materializer."""

    def __init__(self, target_shape, materialize, factor_bytes=0, rank=0):
        super().__init__(target_shape, torch.float32, rank)
        self.materialize = materialize
        self._factor_bytes = int(factor_bytes)

    @property
    def factor_bytes(self):
        return self._factor_bytes

    def materialize_rows(self, start, end, device):
        return self.materialize(start, end, device).reshape(
            end - start, self.cols).float().clone()


class DenseDiffSource(DiffSource):
    def __init__(self, tensor, target_shape=None):
        shape = torch.Size(target_shape or tensor.shape)
        super().__init__(shape, tensor.dtype, rank=1, rank_bound_known=False)
        self.tensor = tensor

    @property
    def factor_bytes(self):
        return self.tensor.numel() * self.tensor.element_size()

    def materialize_rows(self, start, end, device):
        return self.tensor.reshape(self.rows, self.cols)[start:end].to(
            device=device, dtype=torch.float32)


class LoRADiffSource(DiffSource):
    def __init__(self, mat_up, mat_down, alpha, mid, target_shape):
        rank = int(mat_down.shape[0])
        linear = mid is None and len(target_shape) == 2 and mat_up.dim() == 2 and mat_down.dim() == 2
        super().__init__(target_shape, mat_up.dtype, rank=rank, rank_bound_known=linear)
        self.mat_up = mat_up
        self.mat_down = mat_down
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.mid = mid

    @property
    def factor_bytes(self):
        tensors = (self.mat_up, self.mat_down, self.mid)
        return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))

    def stage(self, device):
        self.mat_up = self.mat_up.to(device=device, dtype=torch.float32)
        self.mat_down = self.mat_down.to(device=device, dtype=torch.float32)
        if self.mid is not None:
            self.mid = self.mid.to(device=device, dtype=torch.float32)
        return self

    def materialize_rows(self, start, end, device):
        down = self.mat_down.to(device=device, dtype=torch.float32)
        if self.mid is not None:
            down = down.reshape(down.shape[0], -1)
            up_full = self.mat_up.to(device=device, dtype=torch.float32)
            up_full = up_full.reshape(up_full.shape[0], -1)
            up = up_full[:, start:end]
            mid = self.mid.to(device=device, dtype=torch.float32)
            diff = torch.einsum("i j k l, j r, i p -> p r k l", mid, down, up)
        else:
            up = self.mat_up[start:end].to(device=device, dtype=torch.float32)
            diff = torch.mm(up.flatten(1), down.flatten(1))
        diff = diff.reshape(end - start, self.cols)
        diff.mul_(self.alpha / max(1, int(self.mat_down.shape[0])))
        return diff


class LoHaDiffSource(DiffSource):
    def __init__(self, w1a, w1b, alpha, w2a, w2b, t1, t2, target_shape):
        rank = int(w1b.shape[0])
        super().__init__(target_shape, w1a.dtype, rank=rank, rank_bound_known=False)
        self.w1a = w1a
        self.w1b = w1b
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.w2a = w2a
        self.w2b = w2b
        self.t1 = t1
        self.t2 = t2

    @property
    def factor_bytes(self):
        tensors = (self.w1a, self.w1b, self.w2a, self.w2b, self.t1, self.t2)
        return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))

    def stage(self, device):
        for name in ("w1a", "w1b", "w2a", "w2b", "t1", "t2"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device=device, dtype=torch.float32))
        return self

    def materialize_rows(self, start, end, device):
        w1b = self.w1b.to(device=device, dtype=torch.float32)
        w2b = self.w2b.to(device=device, dtype=torch.float32)
        if self.t1 is not None:
            w1a = self.w1a[:, start:end].to(device=device, dtype=torch.float32)
            w2a = self.w2a[:, start:end].to(device=device, dtype=torch.float32)
            t1 = self.t1.to(device=device, dtype=torch.float32)
            t2 = self.t2.to(device=device, dtype=torch.float32)
            m1 = torch.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
            m2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)
        else:
            w1a = self.w1a[start:end].to(device=device, dtype=torch.float32)
            w2a = self.w2a[start:end].to(device=device, dtype=torch.float32)
            m1 = torch.mm(w1a, w1b)
            m2 = torch.mm(w2a, w2b)
        return (m1.mul_(m2).mul_(self.alpha / max(1, self.rank))).reshape(end - start, self.cols)


class LoKrDiffSource(DiffSource):
    def __init__(self, w1, w2, alpha, w1a, w1b, w2a, w2b, t2, target_shape,
                 adapter_scaling=False):
        ref = w1 if w1 is not None else (w1a if w1a is not None else w2a)
        rank = int(w1b.shape[0] if w1 is None else (w2b.shape[0] if w2 is None else 1))
        super().__init__(target_shape, ref.dtype, rank=rank, rank_bound_known=False)
        self.w1 = w1
        self.w2 = w2
        self.alpha = float(alpha) if alpha is not None else None
        self.w1a = w1a
        self.w1b = w1b
        self.w2a = w2a
        self.w2b = w2b
        self.t2 = t2
        self.adapter_scaling = bool(adapter_scaling)

    @property
    def factor_bytes(self):
        tensors = (self.w1, self.w2, self.w1a, self.w1b, self.w2a, self.w2b, self.t2)
        return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))

    def _factors(self, device):
        dim = None
        if self.w1 is None:
            rank1 = int(self.w1b.shape[0])
            dim = rank1
            w1 = torch.mm(self.w1a.to(device, torch.float32), self.w1b.to(device, torch.float32))
            if self.adapter_scaling and self.alpha is not None:
                w1.mul_(self.alpha / rank1)
        else:
            w1 = self.w1.to(device, torch.float32)
        if self.w2 is None:
            rank2 = int(self.w2b.shape[0])
            dim = rank2
            if self.t2 is None:
                w2 = torch.mm(self.w2a.to(device, torch.float32), self.w2b.to(device, torch.float32))
            else:
                w2 = torch.einsum("i j k l, j r, i p -> p r k l",
                                  self.t2.to(device, torch.float32),
                                  self.w2b.to(device, torch.float32),
                                  self.w2a.to(device, torch.float32))
            if self.adapter_scaling and self.alpha is not None:
                w2.mul_(self.alpha / rank2)
        else:
            w2 = self.w2.to(device, torch.float32)
        scale = 1.0 if self.adapter_scaling else (
            self.alpha / dim if self.alpha is not None and dim is not None else 1.0)
        return w1, w2, scale

    def stage(self, device):
        for name in ("w1", "w2", "w1a", "w1b", "w2a", "w2b", "t2"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device=device, dtype=torch.float32))
        return self

    def materialize_rows(self, start, end, device):
        w1, w2, scale = self._factors(device)
        out2 = int(w2.shape[0])
        rows = []
        for row in range(start, end):
            row1, row2 = divmod(row, out2)
            rows.append(torch.kron(w1[row1], w2[row2]).reshape(-1))
        result = torch.stack(rows, dim=0)
        result.mul_(scale)
        if result.shape[1] != self.cols:
            raise RuntimeError(f"LoKr row width {result.shape[1]} does not match target width {self.cols}")
        return result


class SumDiffSource(DiffSource):
    def __init__(self, sources: Sequence[DiffSource]):
        if not sources:
            raise ValueError("SumDiffSource requires at least one contribution")
        shape = sources[0].target_shape
        if any(source.target_shape != shape for source in sources):
            raise ValueError("All alias contributions must share the target shape")
        rank = sum(source.rank for source in sources)
        known = all(source.rank_bound_known for source in sources)
        super().__init__(shape, sources[0].storage_dtype, rank, known)
        self.sources = list(sources)

    @property
    def factor_bytes(self):
        return sum(source.factor_bytes for source in self.sources)

    def stage(self, device):
        for source in self.sources:
            source.stage(device)
        return self

    def materialize_rows(self, start, end, device):
        result = None
        for source in self.sources:
            value = source.materialize_rows(start, end, device)
            if result is None:
                result = value
            else:
                result.add_(value)
        return result


class ScaledDiffSource(DiffSource):
    def __init__(self, source: DiffSource, scale: float):
        super().__init__(source.target_shape, source.storage_dtype,
                         source.rank, source.rank_bound_known)
        self.source = source
        self.scale = float(scale)

    @property
    def factor_bytes(self):
        return self.source.factor_bytes

    def stage(self, device):
        self.source.stage(device)
        return self

    def materialize_rows(self, start, end, device):
        value = self.source.materialize_rows(start, end, device)
        value.mul_(self.scale)
        return value


def iter_row_ranges(rows: int, rows_per_tile: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < rows:
        end = min(rows, start + rows_per_tile)
        yield start, end
        start = end


def logical_block_seed(base_seed: int, source_index: int, flat_start: int):
    block = int(flat_start) // _LOGICAL_RANDOM_ELEMENTS
    value = (int(base_seed) & 0xFFFFFFFFFFFFFFFF) ^ (source_index * 0x9E3779B97F4A7C15)
    value ^= block * 0xBF58476D1CE4E5B9
    value ^= value >> 30
    value *= 0xBF58476D1CE4E5B9
    value ^= value >> 27
    value *= 0x94D049BB133111EB
    value ^= value >> 31
    return value & 0x7FFFFFFFFFFFFFFF


def deterministic_random_like(shape, device, base_seed, source_index, flat_start=0):
    """Random values stable across physical tile sizes via fixed logical blocks."""
    total = math.prod(int(v) for v in shape)
    output = torch.empty(total, device=device, dtype=torch.float32)
    written = 0
    global_pos = int(flat_start)
    while written < total:
        block_offset = global_pos % _LOGICAL_RANDOM_ELEMENTS
        count = min(total - written, _LOGICAL_RANDOM_ELEMENTS - block_offset)
        generator = torch.Generator(device=device)
        generator.manual_seed(logical_block_seed(base_seed, source_index, global_pos))
        if block_offset:
            torch.rand(block_offset, generator=generator, device=device)
        output[written:written + count] = torch.rand(count, generator=generator, device=device)
        written += count
        global_pos += count
    return output.reshape(shape)


def stable_norm_sq(source: DiffSource, device, rows_per_tile, controller: InterruptController,
                   transform=None):
    total = 0.0
    for start, end in iter_row_ranges(source.rows, rows_per_tile):
        controller.check()
        value = source.materialize_rows(start, end, device)
        if transform is not None:
            value = transform(value, start, end)
        norm = torch.linalg.vector_norm(value).item()
        total += float(norm) * float(norm)
        del value
    return total


def streamed_dot(source_a: DiffSource, source_b: DiffSource, device, rows_per_tile,
                 controller: InterruptController, transform=None):
    total = 0.0
    for start, end in iter_row_ranges(source_a.rows, rows_per_tile):
        controller.check()
        a = source_a.materialize_rows(start, end, device)
        b = source_b.materialize_rows(start, end, device)
        if transform is not None:
            a, b = transform(a, b, start, end)
        total += float(torch.dot(a.flatten(), b.flatten()).item())
        del a, b
    return total


def chunked_randomized_svd(materialize_rows, shape, rank, device, rows_per_tile,
                           controller: InterruptController, niter=2, seed=0):
    """Interruptible randomized SVD of a row-materialized linear operator."""
    rows = int(shape[0])
    cols = math.prod(int(v) for v in shape[1:]) if len(shape) > 1 else 1
    rank = max(1, min(int(rank), rows, cols))
    q = min(rank + max(8, rank // 4), rows, cols)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    omega = torch.randn(cols, q, generator=generator, device=device, dtype=torch.float32)

    def apply_right(matrix):
        out = torch.empty(rows, matrix.shape[1], device="cpu", dtype=torch.float32)
        for start, end in iter_row_ranges(rows, rows_per_tile):
            controller.check()
            tile = materialize_rows(start, end, device)
            out[start:end].copy_((tile @ matrix).cpu())
            del tile
        return out

    def apply_left(matrix_cpu):
        out = torch.zeros(cols, matrix_cpu.shape[1], device=device, dtype=torch.float32)
        for start, end in iter_row_ranges(rows, rows_per_tile):
            controller.check()
            tile = materialize_rows(start, end, device)
            out.add_(tile.T @ matrix_cpu[start:end].to(device))
            del tile
        return out

    y = apply_right(omega)
    for _ in range(max(0, int(niter))):
        controller.check()
        z = apply_left(y)
        y = apply_right(z)
        del z
    controller.check()
    q_cpu = torch.linalg.qr(y, mode="reduced").Q
    b = torch.zeros(q_cpu.shape[1], cols, device=device, dtype=torch.float32)
    for start, end in iter_row_ranges(rows, rows_per_tile):
        controller.check()
        tile = materialize_rows(start, end, device)
        b.add_(q_cpu[start:end].to(device).T @ tile)
        del tile
    controller.check()
    u_hat, singular, vh = torch.linalg.svd(b, full_matrices=False)
    u = q_cpu @ u_hat[:, :rank].cpu()
    return u, singular[:rank].cpu(), vh[:rank].cpu()
