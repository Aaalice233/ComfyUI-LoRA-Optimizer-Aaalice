"""Persistent, content-addressed cache for completed LoRA merge patches."""

import hashlib
import json
import logging
import os
import shutil
import struct
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open

import folder_paths
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.weight_adapter.loha import LoHaAdapter
from comfy.weight_adapter.lokr import LoKrAdapter


_CACHE_FORMAT_VERSION = 1
_HASH_INDEX_VERSION = 1
_DEFAULT_CACHE_GB = 20.0
_MIN_FREE_BYTES = 512 * 1024 * 1024
_ADAPTER_TYPES = {
    "lora": LoRAAdapter,
    "loha": LoHaAdapter,
    "lokr": LoKrAdapter,
}
_SAFETENSORS_DTYPES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}
for _torch_name, _safe_name in (
        ("uint16", "U16"), ("uint32", "U32"), ("uint64", "U64"),
        ("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _torch_name):
        _SAFETENSORS_DTYPES[getattr(torch, _torch_name)] = _safe_name


class PersistentCacheUnsupported(Exception):
    pass


def _check(controller):
    if controller is not None:
        controller.check()


def _write_safetensors(path, tensors, metadata, controller):
    header = {}
    offset = 0
    for name, tensor in tensors.items():
        _check(controller)
        dtype = _SAFETENSORS_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise PersistentCacheUnsupported(
                f"unsupported cached tensor dtype: {tensor.dtype}")
        size = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header["__metadata__"] = metadata
    encoded = json.dumps(
        header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)

    with open(path, "xb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for tensor in tensors.values():
            byte_view = tensor.reshape(-1).view(torch.uint8).numpy()
            raw = memoryview(byte_view)
            for start in range(0, len(raw), 8 * 1024 * 1024):
                _check(controller)
                handle.write(raw[start:start + 8 * 1024 * 1024])
        handle.flush()
        os.fsync(handle.fileno())
    _check(controller)


def _encode_json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _encode_json_value(item)
            for key, item in value.items()
        }
    raise PersistentCacheUnsupported(f"unsupported metadata value: {type(value).__name__}")


def _decode_json_value(value):
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_decode_json_value(item) for item in value["__tuple__"])
        return {key: _decode_json_value(item) for key, item in value.items()}
    return value


def _encode_patch_value(value, tensors, name_prefix):
    if isinstance(value, torch.Tensor):
        tensor_name = f"{name_prefix}_{len(tensors):06d}"
        tensors[tensor_name] = value.detach().to("cpu").contiguous()
        return {"tensor": tensor_name}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"value": value}
    if isinstance(value, tuple):
        return {"tuple": [
            _encode_patch_value(item, tensors, name_prefix)
            for item in value
        ]}
    if isinstance(value, list):
        return {"list": [
            _encode_patch_value(item, tensors, name_prefix)
            for item in value
        ]}
    raise PersistentCacheUnsupported(f"unsupported patch value: {type(value).__name__}")


def _decode_patch_value(value, tensors):
    if "tensor" in value:
        return tensors[value["tensor"]]
    if "value" in value:
        return value["value"]
    if "tuple" in value:
        return tuple(_decode_patch_value(item, tensors) for item in value["tuple"])
    if "list" in value:
        return [_decode_patch_value(item, tensors) for item in value["list"]]
    raise ValueError("invalid persistent patch value")


def _encode_patch_map(patches, tensors, branch, controller):
    encoded = []
    for index, (target_key, patch) in enumerate(patches.items()):
        _check(controller)
        entry = {"key": _encode_json_value(target_key)}
        if isinstance(patch, LoRAAdapter):
            entry["kind"] = "lora"
            value = patch.weights
        elif isinstance(patch, LoHaAdapter):
            entry["kind"] = "loha"
            value = patch.weights
        elif isinstance(patch, LoKrAdapter):
            entry["kind"] = "lokr"
            value = patch.weights
        elif isinstance(patch, (tuple, list)):
            entry["kind"] = "patch"
            value = patch
        else:
            raise PersistentCacheUnsupported(
                f"unsupported {branch} patch type: {type(patch).__name__}")
        entry["payload"] = _encode_patch_value(
            value, tensors, f"{branch}_{index:06d}")
        encoded.append(entry)
    return encoded


def _decode_patch_map(entries, tensors, controller):
    patches = {}
    for entry in entries:
        _check(controller)
        target_key = _decode_json_value(entry["key"])
        payload = _decode_patch_value(entry["payload"], tensors)
        adapter_type = _ADAPTER_TYPES.get(entry["kind"])
        if adapter_type is not None:
            payload = adapter_type(set(), payload)
        elif entry["kind"] != "patch":
            raise ValueError(f"unknown persistent patch type: {entry['kind']}")
        patches[target_key] = payload
    return patches


class PersistentPatchCache:
    def __init__(self, cache_dir=None, max_cache_gb=None):
        if cache_dir is None:
            cache_dir = os.path.join(
                folder_paths.get_user_directory(), "lora_optimizer_cache")
        self.cache_dir = Path(cache_dir)
        configured_gb = max_cache_gb
        if configured_gb is None:
            try:
                configured_gb = float(os.environ.get(
                    "LORA_OPTIMIZER_CACHE_GB", _DEFAULT_CACHE_GB))
            except ValueError:
                configured_gb = _DEFAULT_CACHE_GB
        self.max_cache_bytes = int(max(1.0, min(float(configured_gb), 200.0)) * (1024 ** 3))
        self._lock = threading.RLock()
        self._hash_index = None

    def _ensure_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def hash_index_path(self):
        return self.cache_dir / "source_hashes.json"

    def cache_path(self, cache_key):
        return self.cache_dir / f"{cache_key}.safetensors"

    def _load_hash_index(self):
        if self._hash_index is not None:
            return self._hash_index
        try:
            with self.hash_index_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if loaded.get("version") != _HASH_INDEX_VERSION:
                loaded = {"version": _HASH_INDEX_VERSION, "files": {}}
        except (FileNotFoundError, OSError, ValueError, TypeError):
            loaded = {"version": _HASH_INDEX_VERSION, "files": {}}
        self._hash_index = loaded
        return loaded

    def _save_hash_index(self, controller):
        _check(controller)
        self._ensure_dir()
        path = self.hash_index_path
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self._hash_index, handle, ensure_ascii=False,
                          separators=(",", ":"), sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            _check(controller)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _hash_file(self, path, controller):
        stat = os.stat(path)
        identity = os.path.normcase(os.path.abspath(path))
        with self._lock:
            index = self._load_hash_index()
            cached = index["files"].get(identity)
            if (cached is not None
                    and cached.get("size") == stat.st_size
                    and cached.get("mtime_ns") == stat.st_mtime_ns):
                return cached["sha256"]

        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                _check(controller)
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        result = digest.hexdigest()
        with self._lock:
            index = self._load_hash_index()
            index["files"][identity] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": result,
            }
            self._save_hash_index(controller)
        return result

    @staticmethod
    def _stack_entry(entry):
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            return {
                "name": str(entry[0]),
                "strength": float(entry[1]),
                "clip_strength": (None if len(entry) <= 2 or entry[2] is None
                                  else float(entry[2])),
                "conflict_mode": entry[3] if len(entry) > 3 else "all",
                "key_filter": entry[4] if len(entry) > 4 else "all",
                "preserve": bool(entry[5]) if len(entry) > 5 else False,
            }
        if isinstance(entry, dict):
            if entry.get("_precomputed_diffs") is not None or entry.get("lora") is not None:
                raise PersistentCacheUnsupported(
                    "in-memory or virtual LoRA payload has no stable file identity")
            return {
                "name": str(entry.get("name", "")),
                "strength": float(entry.get("strength", 0.0)),
                "clip_strength": (None if entry.get("clip_strength") is None
                                  else float(entry["clip_strength"])),
                "conflict_mode": entry.get("conflict_mode", "all"),
                "key_filter": entry.get("key_filter", "all"),
                "preserve": bool(entry.get("preserve", False)),
            }
        raise PersistentCacheUnsupported(
            f"unsupported LoRA stack entry: {type(entry).__name__}")

    def source_digest(self, lora_stack, controller=None):
        entries = []
        for raw_entry in lora_stack:
            _check(controller)
            entry = self._stack_entry(raw_entry)
            name = entry["name"]
            path = folder_paths.get_full_path("loras", name)
            if path is None and os.path.isfile(name):
                path = os.path.abspath(name)
            if path is None or not os.path.isfile(path):
                raise PersistentCacheUnsupported(
                    f"LoRA file cannot be resolved: {name}")
            entry["content_sha256"] = self._hash_file(path, controller)
            entries.append(entry)
        encoded = json.dumps(entries, ensure_ascii=False,
                             separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def model_signature(model, clip, controller=None):
        digest = hashlib.sha256()
        for label, root in (
                ("model", getattr(model, "model", None)),
                ("clip", getattr(clip, "cond_stage_model", None))):
            _check(controller)
            if root is None:
                digest.update(f"{label}:none".encode())
                continue
            cls = root.__class__
            digest.update(
                f"{label}:{cls.__module__}.{cls.__qualname__}".encode())
            state_dict_method = getattr(root, "state_dict", None)
            if not callable(state_dict_method):
                continue
            try:
                state = state_dict_method()
            except (AttributeError, RuntimeError, TypeError):
                continue
            for key, value in sorted(state.items()):
                _check(controller)
                shape = tuple(getattr(value, "shape", ()))
                dtype = str(getattr(value, "dtype", "unknown"))
                digest.update(f"|{key}:{shape}:{dtype}".encode())
            del state
        return digest.hexdigest()

    @staticmethod
    def build_key(config_key, source_digest, model_signature, algorithm_version):
        payload = (
            f"format={_CACHE_FORMAT_VERSION}|algo={algorithm_version}|"
            f"config={config_key}|source={source_digest}|model={model_signature}")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, cache_key, controller=None):
        path = self.cache_path(cache_key)
        if not path.is_file():
            return None
        _check(controller)
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("cache_format") != str(_CACHE_FORMAT_VERSION):
                    return None
                manifest = json.loads(metadata["manifest"])
                tensor_names = set(manifest.get("tensor_names", []))
                tensors = {}
                for name in tensor_names:
                    _check(controller)
                    tensors[name] = handle.get_tensor(name)
            model_patches = _decode_patch_map(
                manifest["model_patches"], tensors, controller)
            clip_patches = _decode_patch_map(
                manifest["clip_patches"], tensors, controller)
            key_map = {}
            for entry in manifest.get("key_map", []):
                key_map[_decode_json_value(entry["key"])] = _decode_json_value(entry["value"])
            info = _decode_json_value(manifest["lora_info"])
            lora_data = dict(info)
            lora_data["model_patches"] = model_patches
            lora_data["clip_patches"] = clip_patches
            lora_data["key_map"] = key_map
            try:
                os.utime(path, None)
            except OSError:
                pass
            return {
                "model_patches": model_patches,
                "clip_patches": clip_patches,
                "report": manifest.get("report", ""),
                "lora_data": lora_data,
                "path": str(path),
            }
        except Exception as error:
            _check(controller)
            logging.warning(
                "[LoRA Optimizer Cache] Ignoring unreadable cache %s: %s",
                path, error)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def save(self, cache_key, model_patches, clip_patches, report,
             lora_data, controller=None):
        _check(controller)
        tensors = {}
        try:
            encoded_model = _encode_patch_map(
                model_patches, tensors, "model", controller)
            encoded_clip = _encode_patch_map(
                clip_patches, tensors, "clip", controller)
            key_map = [
                {
                    "key": _encode_json_value(key),
                    "value": _encode_json_value(value),
                }
                for key, value in lora_data.get("key_map", {}).items()
            ]
            lora_info = {
                key: value for key, value in lora_data.items()
                if key not in ("model_patches", "clip_patches", "key_map")
            }
            manifest = {
                "tensor_names": sorted(tensors),
                "model_patches": encoded_model,
                "clip_patches": encoded_clip,
                "key_map": key_map,
                "lora_info": _encode_json_value(lora_info),
                "report": report,
            }
        except PersistentCacheUnsupported as error:
            logging.info("[LoRA Optimizer Cache] Persistent cache skipped: %s", error)
            return None

        tensor_bytes = sum(tensor.numel() * tensor.element_size()
                           for tensor in tensors.values())
        if tensor_bytes > self.max_cache_bytes:
            logging.warning(
                "[LoRA Optimizer Cache] Merge patch is %.1f MiB, larger than the "
                "configured cache limit of %.1f MiB; persistent cache write skipped",
                tensor_bytes / (1024 ** 2),
                self.max_cache_bytes / (1024 ** 2))
            return None
        self._ensure_dir()
        try:
            free_bytes = shutil.disk_usage(self.cache_dir).free
        except OSError:
            free_bytes = tensor_bytes + _MIN_FREE_BYTES
        if free_bytes < tensor_bytes + _MIN_FREE_BYTES:
            logging.warning(
                "[LoRA Optimizer Cache] Need %.1f MiB but only %.1f MiB is free; "
                "persistent cache write skipped",
                (tensor_bytes + _MIN_FREE_BYTES) / (1024 ** 2),
                free_bytes / (1024 ** 2))
            return None

        path = self.cache_path(cache_key)
        temporary = path.with_name(
            f".{path.stem}.{os.getpid()}.{threading.get_ident()}."
            f"{time.time_ns()}.tmp.safetensors")
        metadata = {
            "cache_format": str(_CACHE_FORMAT_VERSION),
            "manifest": json.dumps(
                manifest, ensure_ascii=False, separators=(",", ":"),
                sort_keys=True),
        }
        committed = False
        try:
            _check(controller)
            _write_safetensors(temporary, tensors, metadata, controller)
            _check(controller)
            os.replace(temporary, path)
            committed = True
            _check(controller)
            self._prune(path, controller)
            _check(controller)
            logging.info(
                "[LoRA Optimizer Cache] Saved persistent merge cache: %s (%.1f MiB)",
                path, path.stat().st_size / (1024 ** 2))
            return str(path)
        except BaseException:
            if committed:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _prune(self, keep_path, controller=None):
        _check(controller)
        files = []
        total = 0
        for path in self.cache_dir.glob("*.safetensors"):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size
        if total <= self.max_cache_bytes:
            return
        for _mtime, size, path in sorted(files):
            _check(controller)
            if path == keep_path:
                continue
            try:
                path.unlink()
                total -= size
            except OSError:
                continue
            if total <= self.max_cache_bytes:
                break
