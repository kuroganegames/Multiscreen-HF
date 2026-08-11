#!/usr/bin/env python
"""Independently review raw Level 1 core requalification evidence.

This program intentionally uses only the Python standard library.  It does not
import Transformers, PyTorch, the training harnesses, or the archive verifier.
Every input path is explicit and absolute.  The successful output is a
deterministic, path-free JSON report whose artifact hashes bind it to the raw
evidence that was reviewed.

Tokenizer reload checks are produced by the separate tokenizer verifier and
are supplied as repeatable ``LOGICAL_NAME=PATH`` arguments.  Required logical
names are ``p0_3_psi8``, ``p0_3_psi16``, ``p0_4_psi8``, and
``p0_4_psi16``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "multiscreen-level1-raw-evidence-review-v1"
P0_4_LANE_SCHEMA_VERSION = "multiscreen-p0-4-lane-review-v1"
TOKENIZER_SCHEMA_VERSION = "multiscreen-tokenizer-reload-check-v1"
ENVIRONMENT_SCHEMA_VERSION = "multiscreen-level1-environment-v1"
REPOSITORY_CHECK_SCHEMA_VERSION = "multiscreen-level1-repository-check-v1"
TOKENIZER_NAMES = (
    "p0_3_psi8",
    "p0_3_psi16",
    "p0_4_psi8",
    "p0_4_psi16",
)
TOKENIZER_CHECKPOINT_IDENTIFIERS = {
    "p0_3_psi8": "p0-3-psi8-checkpoint",
    "p0_3_psi16": "p0-3-psi16-checkpoint",
    "p0_4_psi8": "p0-4-psi8-checkpoint",
    "p0_4_psi16": "p0-4-psi16-checkpoint",
}
TOKENIZER_CHECKED_FIELDS = (
    "full_vocabulary_mapping",
    "vocab_size",
    "tokenizer_length",
    "added_vocabulary_mapping",
    "added_tokens_decoder",
    "special_tokens_map",
    "special_tokens_map_extended",
    "special_token_attributes",
    "special_token_ids",
    "all_special_tokens",
    "all_special_ids",
    "model_input_names",
    "padding_side",
    "model_max_length",
    "truncation_side",
    "probe_encodings_without_special_tokens",
    "probe_encodings_with_special_tokens",
    "probe_decodings",
    "special_token_boundary_encodings",
    "special_token_boundary_decodings",
    "tokenizer_class",
    "is_fast",
    "checkpoint_reload_origin",
    "source_normalization",
)
C3_ROW_MANIFEST_SHA256 = (
    "942f9b3397ff7073342973082efa4cddf3ace16bc7e3d180c827df3203243831"
)
C3_TOKENIZER_ASSET_SIZES = {
    "merges.txt": 456_318,
    "tokenizer.json": 1_355_256,
    "tokenizer_config.json": 26,
    "vocab.json": 1_042_301,
}
C3_PACKED_CHUNK_SHA256 = (
    "371d6dd52faa5b4278469eea935708ad99113621bb4ce3eed31cb8e519b00076",
    "4b4e51618393b9deaa9cc0b6dc0b7b093fd9fef6567ee5f4e15fb2745a9ccccb",
    "b1bbe605c93eb17d8b36f9f3cbbc73e2e4c5096cb447325fcea1e318420dbb7e",
    "e13251a530b7eb5636ad426ffe6a5b42923121f223a2089f41936e5a4ac2c39a",
    "ef05aa2f9860867e77617c88bbf693d222f7b61561bd5795e06c33a237ea4735",
    "75f04cd07a7ae0e927ae86eec630110a069805e36acb6fc4c7cf49db28661656",
    "e34beac42f4ca180661ab289a0cd79237b12c7e6de399396b821742347e5b175",
    "ba9a311adeba69458424a36591fdaef5b2582db8074b61c79f942006b4404e18",
    "4229f001a2d3967bc4a2f803cad2e79b0ae9ed48b1892896e7e77f49fd432873",
    "c1cdaefbf45852700c0b58db5a212acd20ce616a68ba1f50ddb1d7e9c2f9b010",
    "98bbdc89c73fbec7fb96a24bc73a8cd73af51bc8184f79846677aab007c49c3d",
    "aec3c2e7ea78f3ed7c7c6f77299b21164c5dcb9d4cb9f5b1e19c05ed9f2534c2",
    "0c749304c428d82d2d2333bde432cdd2617afb24f8aabc6716977cc3b0f22a72",
    "ac023f84f579520ede3aa139fc459eb2d17cff6f3a9a0558b330df94495d5eda",
)
P0_3_STEPS = {8: 40, 16: 25}
P0_3_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
P0_4_STEPS = 50
P0_4_EVENT_SEQUENCE = (
    "run_start",
    "preflight_complete",
    *("train_step" for _ in range(P0_4_STEPS)),
    "training_complete",
    "save_reload_check",
    "cache_split_check",
    "generation_check",
    "run_complete",
)
C3_LANES = (
    ("c3_psi8_operational", 8, "operational"),
    ("c3_psi8_peak_exposure", 8, "peak-exposure"),
    ("c3_psi16_operational", 16, "operational"),
    ("c3_psi16_peak_exposure", 16, "peak-exposure"),
)
REQUIRED_COMMAND_NAMES = (
    "environment-tf4576",
    "environment-tf5141",
    "environment-cuda0",
    "offline-cache-preflight",
    "repository-hygiene",
    "syntax-level1",
    "level1-evidence-support-tests",
    "tokenizer-reload-tests-tf4576",
    "tokenizer-reload-tests-tf5141",
    "validation-evidence-tests",
    "json-validation",
    "workflow-yaml",
    "markdown-links",
    "c1-architecture",
    "c1-initialization",
    "c1-packed-data",
    "c1-manifest",
    "c2-position-cache",
    "gradient-checkpointing-tf4576",
    "gradient-checkpointing-tf5141",
    "formula-units",
    "oracle-selfcheck",
    "oracle-smoke",
    "p0-1-cpu-fp32",
    "p0-1-cuda-bf16",
    "p0-2-cpu-fp32",
    "p0-2-cuda-bf16",
    "c3-contracts-tf4576",
    "c3-contracts-tf5141",
    "c3-contract-cli",
    "c3-data",
    "c3-psi8-operational",
    "c3-psi8-peak-exposure",
    "c3-psi16-operational",
    "c3-psi16-peak-exposure",
    "p0-3-checkpointed",
    "p0-3-tokenizer-psi8",
    "p0-3-tokenizer-psi16",
    "p0-4-psi8-preflight",
    "p0-4-psi8",
    "p0-4-tokenizer-psi8",
    "p0-4-review-psi8",
    "p0-4-psi16-preflight",
    "p0-4-psi16",
    "p0-4-tokenizer-psi16",
    "repository-hygiene-final",
)

CPU_COMMAND_NAMES = frozenset(
    {
        "environment-tf4576",
        "environment-tf5141",
        "offline-cache-preflight",
        "repository-hygiene",
        "syntax-level1",
        "level1-evidence-support-tests",
        "tokenizer-reload-tests-tf4576",
        "tokenizer-reload-tests-tf5141",
        "validation-evidence-tests",
        "json-validation",
        "workflow-yaml",
        "markdown-links",
        "formula-units",
        "oracle-selfcheck",
        "oracle-smoke",
        "p0-1-cpu-fp32",
        "p0-2-cpu-fp32",
        "c1-architecture",
        "c1-initialization",
        "c1-packed-data",
        "c1-manifest",
        "c3-contract-cli",
        "c3-data",
        "p0-3-tokenizer-psi8",
        "p0-3-tokenizer-psi16",
        "p0-4-psi8-preflight",
        "p0-4-psi16-preflight",
        "p0-4-tokenizer-psi8",
        "p0-4-review-psi8",
        "p0-4-tokenizer-psi16",
        "repository-hygiene-final",
    }
)
CUDA_COMMAND_NAMES = frozenset(
    {
        "environment-cuda0",
        "p0-1-cuda-bf16",
        "p0-2-cuda-bf16",
        "c2-position-cache",
        "gradient-checkpointing-tf4576",
        "gradient-checkpointing-tf5141",
        "c3-contracts-tf4576",
        "c3-contracts-tf5141",
        "c3-psi8-operational",
        "c3-psi8-peak-exposure",
        "c3-psi16-operational",
        "c3-psi16-peak-exposure",
        "p0-3-checkpointed",
        "p0-4-psi8",
        "p0-4-psi16",
    }
)
if CPU_COMMAND_NAMES & CUDA_COMMAND_NAMES or (
    CPU_COMMAND_NAMES | CUDA_COMMAND_NAMES
) != frozenset(REQUIRED_COMMAND_NAMES):
    raise RuntimeError("internal command CPU/CUDA classification is incomplete")

HERMETIC_FIXED_ENVIRONMENT = (
    "PATH=/usr/bin:/bin",
    "LANG=C.UTF-8",
    "LC_ALL=C.UTF-8",
    "TZ=UTC",
    "HF_DATASETS_DISABLE_PROGRESS_BARS=1",
    "HF_DATASETS_OFFLINE=1",
    "HF_HUB_DISABLE_TELEMETRY=1",
    "HF_HUB_OFFLINE=1",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONHASHSEED=0",
    "PYTHONNOUSERSITE=1",
    "PYTHONOPTIMIZE=0",
    "PYTHONUNBUFFERED=1",
    "PYTHONUTF8=1",
    "TOKENIZERS_PARALLELISM=false",
    "TRANSFORMERS_OFFLINE=1",
)
PYTHONPATH_DOT_COMMAND_NAMES = frozenset(
    {
        "level1-evidence-support-tests",
        "tokenizer-reload-tests-tf4576",
        "tokenizer-reload-tests-tf5141",
        "validation-evidence-tests",
        "c1-manifest",
        "p0-3-tokenizer-psi8",
        "p0-3-tokenizer-psi16",
        "p0-4-psi8-preflight",
        "p0-4-psi16-preflight",
        "p0-4-tokenizer-psi8",
        "p0-4-tokenizer-psi16",
    }
)
PYTHONPATH_ORACLE_COMMAND_NAMES = frozenset(
    {
        "formula-units",
        "oracle-selfcheck",
        "oracle-smoke",
        "c3-contracts-tf4576",
        "c3-contracts-tf5141",
        "c3-contract-cli",
        "c3-data",
        "c3-psi8-operational",
        "c3-psi8-peak-exposure",
        "c3-psi16-operational",
        "c3-psi16-peak-exposure",
    }
)
PYTHONPATH_FULL_COMMAND_NAMES = frozenset(
    {
        "c1-architecture",
        "c1-initialization",
        "c1-packed-data",
        "c2-position-cache",
        "gradient-checkpointing-tf4576",
        "gradient-checkpointing-tf5141",
        "p0-1-cpu-fp32",
        "p0-1-cuda-bf16",
        "p0-2-cpu-fp32",
        "p0-2-cuda-bf16",
        "p0-3-checkpointed",
        "p0-4-psi8",
        "p0-4-psi16",
    }
)
if (
    PYTHONPATH_DOT_COMMAND_NAMES & PYTHONPATH_ORACLE_COMMAND_NAMES
    or PYTHONPATH_DOT_COMMAND_NAMES & PYTHONPATH_FULL_COMMAND_NAMES
    or PYTHONPATH_ORACLE_COMMAND_NAMES & PYTHONPATH_FULL_COMMAND_NAMES
    or not (
        PYTHONPATH_DOT_COMMAND_NAMES
        | PYTHONPATH_ORACLE_COMMAND_NAMES
        | PYTHONPATH_FULL_COMMAND_NAMES
        | {"syntax-level1"}
    ).issubset(REQUIRED_COMMAND_NAMES)
):
    raise RuntimeError("internal command environment-suffix classification is invalid")
REQUIRED_ENVIRONMENT_NAMES = ("runtime-tf4576", "runtime-tf5141")
COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
P0_4_PSI8_REVIEW_COMMAND_NAMES = (
    "environment-tf4576",
    "environment-cuda0",
    "offline-cache-preflight",
    "p0-4-psi8-preflight",
    "p0-4-psi8",
    "p0-4-tokenizer-psi8",
)
P0_4_PSI8_REVIEW_ENVIRONMENT_NAMES = ("runtime-tf4576",)
TF5141_COMMAND_NAMES = frozenset(
    {
        "environment-tf5141", "tokenizer-reload-tests-tf5141",
        "gradient-checkpointing-tf5141", "c3-contracts-tf5141",
    }
)
COMMAND_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,78}[a-z0-9])?$")
ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"(?:Z|[+]00:00)$")
P0_3_DATA_CONTRACT_STDOUT_RE = re.compile(
    r"^\[P0-3\] data_contract sha256=([0-9a-f]{64})$"
)
P0_4_DATA_CONTRACT_STDOUT_RE = re.compile(
    r"^\[P0-4\] data_contract sha256=([0-9a-f]{64})$"
)
P0_3_DATASET_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{16,64}$")
P0_3_TOKENIZER_CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
P0_3_STEP_RE = re.compile(
    r"^\[P0-3\]\[Psi=(8|16)\] step=(\d{4})/(\d+) "
    r"loss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?) "
    r"grad_norm=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
P0_3_PROBE_RE = re.compile(
    r"^\[P0-3\]\[Psi=(8|16)\] probe_loss "
    r"initial=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?) "
    r"final=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?) "
    r"drop=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?) "
    r"rel=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)%$"
)


class ReviewError(ValueError):
    """Raised when raw evidence does not satisfy the review contract."""


def _fail(message: str) -> None:
    raise ReviewError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _pretty_canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    _fail(f"non-finite JSON number is forbidden: {value}")


def _decode_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"{label} is not valid UTF-8: {exc}") from exc
    if text.startswith("\ufeff"):
        _fail(f"{label} must not contain a UTF-8 BOM")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejector,
            parse_constant=_reject_constant,
        )
    except ReviewError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ReviewError(f"malformed JSON in {label}: {exc}") from exc
    _validate_finite_tree(value, label=label)
    return value


def _validate_finite_tree(value: Any, *, label: str) -> None:
    if type(value) is float and not math.isfinite(value):
        _fail(f"{label} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, label=f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, label=f"{label}.{key}")


def _check_no_symlink_components(path: Path, *, label: str) -> None:
    current = path
    while True:
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                _fail(f"{label} contains a symlink component: {current}")
        if current.parent == current:
            break
        current = current.parent


def _absolute_path(value: str | os.PathLike[str], *, label: str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        _fail(f"{label} must be a text path")
    path = Path(raw)
    if not path.is_absolute():
        _fail(f"{label} must be an explicit absolute path")
    _check_no_symlink_components(path, label=label)
    resolved = path.resolve(strict=False)
    if raw != os.fspath(resolved):
        _fail(f"{label} must be a lexical canonical absolute path")
    return resolved


def _safe_root(value: str | os.PathLike[str], *, label: str) -> Path:
    root = _absolute_path(value, label=label)
    if not root.is_dir():
        _fail(f"{label} is not a directory: {root}")
    for directory, names, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *filenames):
            candidate = base / name
            if candidate.is_symlink():
                _fail(f"{label} contains a symlink: {candidate}")
    return root


def _safe_file(value: str | os.PathLike[str], *, label: str) -> Path:
    path = _absolute_path(value, label=label)
    if not path.is_file():
        _fail(f"{label} is not a regular file: {path}")
    return path


def _child_file(root: Path, relative: str, *, label: str) -> Path:
    candidate = root / relative
    _check_no_symlink_components(candidate, label=label)
    resolved = candidate.resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        _fail(f"{label} escapes its explicit root")
    if not resolved.is_file():
        _fail(f"missing required file {label}: {resolved}")
    return resolved


def _child_directory(root: Path, relative: str, *, label: str) -> Path:
    candidate = root / relative
    _check_no_symlink_components(candidate, label=label)
    resolved = candidate.resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        _fail(f"{label} escapes its explicit root")
    if not resolved.is_dir():
        _fail(f"missing required directory {label}: {resolved}")
    return resolved


def _stable_read_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ReviewError(f"could not open {label} safely: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} is not a regular file")
        if before.st_nlink != 1:
            _fail(f"{label} must not have hard links")
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(fd, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if identity_before != identity_after or len(raw) != after.st_size:
            _fail(f"{label} changed while it was read")
        return raw
    finally:
        os.close(fd)


def _read_bytes(path: Path, *, label: str, hashes: dict[str, str]) -> bytes:
    raw = _stable_read_bytes(path, label=label)
    if label in hashes:
        _fail(f"duplicate artifact label: {label}")
    hashes[label] = _sha256_bytes(raw)
    return raw

def _load_json(path: Path, *, label: str, hashes: dict[str, str]) -> Any:
    return _decode_json_bytes(_read_bytes(path, label=label, hashes=hashes), label=label)


def _load_jsonl(path: Path, *, label: str, hashes: dict[str, str]) -> list[dict[str, Any]]:
    raw = _read_bytes(path, label=label, hashes=hashes)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"{label} is not valid UTF-8: {exc}") from exc
    if not raw or not raw.endswith(b"\n"):
        _fail(f"{label} must be non-empty and newline-terminated")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            _fail(f"{label} contains a blank JSONL record at line {line_number}")
        value = _decode_json_bytes(line.encode("utf-8"), label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            _fail(f"{label}:{line_number} must be a JSON object")
        events.append(value)
    return events


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a JSON array")
    return value


def _at(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            _fail(f"missing required field: {dotted}")
        current = current[part]
    return current


def _exact_bool(value: Any, *, label: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be an unambiguous JSON boolean")
    if expected is not None and value is not expected:
        _fail(f"{label} must be {str(expected).lower()}")
    return value


def _exact_int(
    value: Any,
    *,
    label: str,
    expected: int | None = None,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an unambiguous JSON integer")
    if expected is not None and value != expected:
        _fail(f"{label} must be {expected}, got {value}")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}, got {value}")
    return value


def _finite_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be an unambiguous JSON number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    if positive and result <= 0:
        _fail(f"{label} must be positive")
    if minimum is not None and result < minimum:
        _fail(f"{label} must be at least {minimum}")
    return result


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty JSON string")
    return value


def _equals(value: Any, expected: Any, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(f"{label} must equal {expected!r}, got {value!r}")


def _exact_values(
    value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for dotted, expected_value in expected.items():
        _equals(_at(value, dotted), expected_value, label=f"{label}.{dotted}")


def _review_cuda_memory(
    value: Mapping[str, Any], *, label: str, total_bytes: int
) -> dict[str, int]:
    fields = (
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    reviewed = {
        field: _exact_int(
            _at(value, field), label=f"{label}.{field}", minimum=0
        )
        for field in fields
    }
    allocated = reviewed["allocated_bytes"]
    reserved = reviewed["reserved_bytes"]
    peak_allocated = reviewed["peak_allocated_bytes"]
    peak_reserved = reviewed["peak_reserved_bytes"]
    if allocated > reserved:
        _fail(f"{label} allocated bytes exceed reserved bytes")
    if peak_allocated < allocated or peak_reserved < reserved:
        _fail(f"{label} peak memory is below current memory")
    if peak_allocated > peak_reserved:
        _fail(f"{label} peak allocated bytes exceed peak reserved bytes")
    if reserved > total_bytes or peak_reserved > total_bytes:
        _fail(f"{label} memory exceeds total GPU memory")
    return reviewed


def _close(actual: float, expected: float, *, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=atol):
        _fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _failure_artifact_check(root: Path, *, label: str) -> None:
    forbidden_exact = {
        "failure.json",
        "p0-4_failed.md",
        "p0-4_diagnostic_complete.md",
    }
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            lowered = filename.casefold()
            if lowered in forbidden_exact or "failure" in lowered or "failed" in lowered:
                _fail(f"{label} contains a failure/diagnostic artifact: {Path(directory) / filename}")


def _validate_embedded_directory(value: Any, expected: Path, *, label: str) -> None:
    raw = _nonempty_string(value, label=label)
    candidate = _absolute_path(raw, label=label)
    if candidate != expected:
        _fail(f"{label} must identify the expected artifact directory")


def _review_p0_3_stdout(
    stdout_path: Path,
    *,
    hashes: dict[str, str],
) -> dict[str, Any]:
    raw = _read_bytes(stdout_path, label="p0_3.stdout", hashes=hashes)
    if not raw or not raw.endswith(b"\n"):
        _fail("P0-3 stdout must be non-empty and newline-terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"P0-3 stdout is not valid UTF-8: {exc}") from exc
    if "\x00" in text or "\r" in text:
        _fail("P0-3 stdout must be lossless UTF-8 text with LF newlines")
    if re.search(r"\b(?:nan|[+-]?inf(?:inity)?)\b", text, flags=re.IGNORECASE):
        _fail("P0-3 stdout contains a non-finite numeric indicator")
    if re.search(r"(?:Traceback \(most recent call last\)|\bFAILED\b|\bfailure\b)", text, flags=re.IGNORECASE):
        _fail("P0-3 stdout contains a failure indicator")

    lines = text.splitlines()
    steps: list[dict[str, Any]] = []
    probes: dict[int, dict[str, float]] = {}
    header_positions: dict[int, int] = {}
    probe_positions: dict[int, int] = {}
    data_contract_events: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        data_contract_match = P0_3_DATA_CONTRACT_STDOUT_RE.fullmatch(line)
        if data_contract_match:
            data_contract_events.append((index, data_contract_match.group(1)))
            continue
        if line.startswith("[P0-3] data_contract"):
            _fail(f"malformed P0-3 data-contract digest at stdout line {index + 1}")
        header = re.match(r"^\[P0-3\] Psi=(8|16) .* steps=(\d+) device=(\S+) amp=(\S+)$", line)
        if header:
            psi = int(header.group(1))
            if psi in header_positions:
                _fail(f"duplicate P0-3 Psi={psi} run header")
            if int(header.group(2)) != P0_3_STEPS[psi]:
                _fail(f"P0-3 Psi={psi} header has the wrong step count")
            if not header.group(3).startswith("cuda") or header.group(4) not in {"bf16", "bfloat16"}:
                _fail(f"P0-3 Psi={psi} header is not CUDA bf16")
            header_positions[psi] = index
            continue
        match = P0_3_STEP_RE.match(line)
        if match:
            psi = int(match.group(1))
            loss = float(match.group(4))
            grad = float(match.group(5))
            if not math.isfinite(loss) or not math.isfinite(grad):
                _fail("P0-3 stdout contains a non-finite step metric")
            steps.append(
                {
                    "psi": psi,
                    "step": int(match.group(2)),
                    "total": int(match.group(3)),
                    "loss": loss,
                    "grad_norm": grad,
                    "line": index,
                }
            )
            continue
        probe_match = P0_3_PROBE_RE.match(line)
        if probe_match:
            psi = int(probe_match.group(1))
            if psi in probes:
                _fail(f"duplicate P0-3 Psi={psi} probe event")
            values = [float(probe_match.group(i)) for i in range(2, 6)]
            if not all(math.isfinite(item) for item in values):
                _fail("P0-3 stdout contains a non-finite probe metric")
            probes[psi] = {
                "initial": values[0],
                "final": values[1],
                "drop": values[2],
                "relative_percent": values[3],
            }
            probe_positions[psi] = index
            continue
        if line.startswith("[P0-3][Psi=") and ("step=" in line or "probe_loss" in line):
            _fail(f"malformed P0-3 raw event at stdout line {index + 1}")

    expected = [
        (psi, step, P0_3_STEPS[psi])
        for psi in (8, 16)
        for step in range(1, P0_3_STEPS[psi] + 1)
    ]
    actual = [(item["psi"], item["step"], item["total"]) for item in steps]
    if actual != expected:
        _fail("P0-3 stdout has missing, duplicate, extra, or out-of-order step events")
    if set(header_positions) != {8, 16} or set(probes) != {8, 16}:
        _fail("P0-3 stdout is missing a required run header or probe event")
    if len(data_contract_events) != 1:
        _fail("P0-3 stdout must contain exactly one data-contract digest event")
    data_contract_position, data_contract_sha256 = data_contract_events[0]
    if data_contract_position >= header_positions[8]:
        _fail("P0-3 data-contract digest must precede both training runs")
    if not (
        header_positions[8] < steps[0]["line"] < probe_positions[8]
        < header_positions[16] < steps[-1]["line"] < probe_positions[16]
    ):
        _fail("P0-3 stdout run, step, and probe events are out of order")
    if text.count("P0-3 TinyStories stability checks passed.") != 1:
        _fail("P0-3 stdout must contain exactly one success completion line")
    completion_position = next(
        index for index, line in enumerate(lines)
        if line == "P0-3 TinyStories stability checks passed."
    )
    if completion_position <= probe_positions[16]:
        _fail("P0-3 success line appears before raw events completed")
    return {
        "data_contract_sha256": data_contract_sha256,
        "step_event_count": len(steps),
        "steps": steps,
        "probes": probes,
        "line_count": len(lines),
    }


def _review_p0_3_data_contract(
    root: Path,
    *,
    hashes: dict[str, str],
) -> dict[str, Any]:
    path = _child_file(root, "data_contract.json", label="P0-3 data contract")
    raw = _read_bytes(path, label="p0_3.data_contract", hashes=hashes)
    decoded = _decode_json_bytes(raw, label="P0-3 data contract")
    if raw != _canonical_bytes(decoded):
        _fail("P0-3 data contract must use canonical JSON bytes")
    contract = _mapping(decoded, label="P0-3 data contract")
    if set(contract) != {"packing", "schema_version", "source", "status", "tokenizer"}:
        _fail("P0-3 data contract fields are incomplete or ambiguous")
    _exact_values(
        contract,
        {
            "schema_version": "multiscreen-p0-3-data-contract-v1",
            "status": "recorded",
        },
        label="P0-3 data contract",
    )

    source = _mapping(_at(contract, "source"), label="P0-3 data contract source")
    if set(source) != {
        "algorithm",
        "data_dir",
        "data_files",
        "dataset_config",
        "dataset_fingerprint",
        "dataset_name",
        "max_texts",
        "revision",
        "selected_text_count",
        "selected_text_manifest_sha256",
        "selected_text_utf8_bytes",
        "source_kind",
        "text_column",
        "text_file",
        "train_split",
    }:
        _fail("P0-3 data contract source fields are incomplete or ambiguous")
    _exact_values(
        source,
        {
            "algorithm": "sha256-length-framed-utf8-texts-v1",
            "data_dir": None,
            "data_files": None,
            "dataset_config": None,
            "dataset_name": "roneneldan/TinyStories",
            "max_texts": 20_000,
            "revision": P0_3_DATASET_REVISION,
            "selected_text_count": 20_000,
            "source_kind": "huggingface_dataset",
            "text_column": "text",
            "text_file": None,
            "train_split": "train[:20000]",
        },
        label="P0-3 data contract source",
    )
    dataset_fingerprint = _nonempty_string(
        _at(source, "dataset_fingerprint"),
        label="P0-3 data contract source.dataset_fingerprint",
    )
    if P0_3_DATASET_FINGERPRINT_RE.fullmatch(dataset_fingerprint) is None:
        _fail(
            "P0-3 data contract dataset fingerprint must be 16-64 lowercase "
            "hexadecimal characters"
        )
    dataset_fingerprint_sha256 = hashlib.sha256(
        dataset_fingerprint.encode("utf-8")
    ).hexdigest()
    selected_text_sha = _nonempty_string(
        _at(source, "selected_text_manifest_sha256"),
        label="P0-3 data contract source.selected_text_manifest_sha256",
    )
    if HEX64_RE.fullmatch(selected_text_sha) is None:
        _fail("P0-3 selected-text manifest must be lowercase SHA-256")
    _exact_int(
        _at(source, "selected_text_utf8_bytes"),
        label="P0-3 data contract source.selected_text_utf8_bytes",
        minimum=1,
    )

    packing = _mapping(_at(contract, "packing"), label="P0-3 data contract packing")
    if set(packing) != {
        "algorithm",
        "chunk_count",
        "chunk_size",
        "eos_token_id",
        "legacy_shifted_labels",
        "max_train_tokens",
        "packed_token_stream_sha256",
        "return_labels_are_shifted",
        "seq_len",
        "usable_token_count",
    }:
        _fail("P0-3 data contract packing fields are incomplete or ambiguous")
    _exact_values(
        packing,
        {
            "algorithm": "sha256-uint32-le-packed-token-stream-v1",
            "chunk_count": 2_032,
            "chunk_size": 129,
            "legacy_shifted_labels": True,
            "max_train_tokens": 262_144,
            "return_labels_are_shifted": True,
            "seq_len": 128,
            "usable_token_count": 262_128,
        },
        label="P0-3 data contract packing",
    )
    _exact_int(
        _at(packing, "eos_token_id"),
        label="P0-3 data contract packing.eos_token_id",
        expected=2,
    )
    _exact_bool(
        _at(packing, "legacy_shifted_labels"),
        label="P0-3 data contract packing.legacy_shifted_labels",
        expected=True,
    )
    _exact_bool(
        _at(packing, "return_labels_are_shifted"),
        label="P0-3 data contract packing.return_labels_are_shifted",
        expected=True,
    )
    packed_token_sha = _nonempty_string(
        _at(packing, "packed_token_stream_sha256"),
        label="P0-3 data contract packing.packed_token_stream_sha256",
    )
    if HEX64_RE.fullmatch(packed_token_sha) is None:
        _fail("P0-3 packed-token stream manifest must be lowercase SHA-256")
    if _at(packing, "chunk_count") * _at(packing, "chunk_size") != _at(
        packing, "usable_token_count"
    ):
        _fail("P0-3 packed-token count differs from its chunk shape")

    tokenizer = _mapping(
        _at(contract, "tokenizer"), label="P0-3 data contract tokenizer"
    )
    if set(tokenizer) != {
        "class",
        "counts",
        "hashes",
        "is_fast",
        "operationalization",
    }:
        _fail("P0-3 data contract tokenizer fields are incomplete or ambiguous")
    tokenizer_class = _nonempty_string(
        _at(tokenizer, "class"), label="P0-3 data contract tokenizer.class"
    )
    if P0_3_TOKENIZER_CLASS_RE.fullmatch(tokenizer_class) is None:
        _fail("P0-3 data contract tokenizer class is not a path-free identifier")
    tokenizer_is_fast = _exact_bool(
        _at(tokenizer, "is_fast"), label="P0-3 data contract tokenizer.is_fast"
    )

    counts = _mapping(
        _at(tokenizer, "counts"), label="P0-3 data contract tokenizer.counts"
    )
    expected_count_fields = {
        "added_vocabulary",
        "all_special_tokens",
        "probes",
        "special_token_boundary_probes",
        "tokenizer_length",
        "vocab_size",
        "vocabulary",
    }
    if set(counts) != expected_count_fields:
        _fail("P0-3 data contract tokenizer counts are incomplete or ambiguous")
    normalized_counts = {
        field: _exact_int(
            _at(counts, field),
            label=f"P0-3 data contract tokenizer.counts.{field}",
            minimum=0,
        )
        for field in sorted(expected_count_fields)
    }
    for field in ("tokenizer_length", "vocab_size", "vocabulary"):
        if normalized_counts[field] != 768:
            _fail(f"P0-3 data contract tokenizer.counts.{field} must equal 768")
    if normalized_counts["all_special_tokens"] < 1:
        _fail("P0-3 data contract tokenizer must have a special token")
    if normalized_counts["probes"] != 5:
        _fail("P0-3 data contract tokenizer must bind five probes")
    if normalized_counts["special_token_boundary_probes"] != (
        normalized_counts["all_special_tokens"] * 7
    ):
        _fail("P0-3 data contract tokenizer boundary-probe count is inconsistent")

    hash_fields = _mapping(
        _at(tokenizer, "hashes"), label="P0-3 data contract tokenizer.hashes"
    )
    expected_hash_fields = {
        "probe_manifest_sha256",
        "special_tokens_manifest_sha256",
        "vocabulary_manifest_sha256",
    }
    if set(hash_fields) != expected_hash_fields:
        _fail("P0-3 data contract tokenizer hashes are incomplete or ambiguous")
    normalized_hashes: dict[str, str] = {}
    for field in sorted(expected_hash_fields):
        digest = _nonempty_string(
            _at(hash_fields, field),
            label=f"P0-3 data contract tokenizer.hashes.{field}",
        )
        if HEX64_RE.fullmatch(digest) is None:
            _fail(
                f"P0-3 data contract tokenizer.hashes.{field} must be lowercase SHA-256"
            )
        normalized_hashes[field] = digest

    operationalization = _mapping(
        _at(tokenizer, "operationalization"),
        label="P0-3 data contract tokenizer.operationalization",
    )
    if set(operationalization) != {
        "model_input_names",
        "model_max_length",
        "padding_side",
        "truncation_side",
    }:
        _fail("P0-3 data contract tokenizer operationalization is incomplete")
    model_inputs = _list(
        _at(operationalization, "model_input_names"),
        label="P0-3 data contract tokenizer.operationalization.model_input_names",
    )
    if not model_inputs or any(
        not isinstance(item, str)
        or P0_3_TOKENIZER_CLASS_RE.fullmatch(item) is None
        for item in model_inputs
    ):
        _fail("P0-3 data contract tokenizer model_input_names are invalid")
    model_max_length = _exact_int(
        _at(operationalization, "model_max_length"),
        label="P0-3 data contract tokenizer.operationalization.model_max_length",
        minimum=1,
    )
    padding_side = _nonempty_string(
        _at(operationalization, "padding_side"),
        label="P0-3 data contract tokenizer.operationalization.padding_side",
    )
    truncation_side = _nonempty_string(
        _at(operationalization, "truncation_side"),
        label="P0-3 data contract tokenizer.operationalization.truncation_side",
    )
    if padding_side not in {"left", "right"} or truncation_side not in {
        "left",
        "right",
    }:
        _fail("P0-3 data contract tokenizer side configuration is invalid")

    tokenizer_projection = {
        "class": tokenizer_class,
        "counts": normalized_counts,
        "hashes": normalized_hashes,
        "is_fast": tokenizer_is_fast,
        "operationalization": {
            "model_input_names": list(model_inputs),
            "model_max_length": model_max_length,
            "padding_side": padding_side,
            "truncation_side": truncation_side,
        },
    }
    return {
        "sha256": _sha256_bytes(raw),
        "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
        "selected_text_manifest_sha256": selected_text_sha,
        "packed_token_stream_sha256": packed_token_sha,
        "tokenizer_projection": tokenizer_projection,
    }


def _review_p0_3(
    root_value: str | os.PathLike[str],
    stdout_value: str | os.PathLike[str],
    *,
    hashes: dict[str, str],
) -> dict[str, Any]:
    root = _safe_root(root_value, label="P0-3 root")
    _failure_artifact_check(root, label="P0-3 root")
    data_contract = _review_p0_3_data_contract(root, hashes=hashes)
    marker = _child_file(root, "P0-3_COMPLETE.md", label="P0-3 completion marker")
    marker_raw = _read_bytes(marker, label="p0_3.completion_marker", hashes=hashes)
    try:
        marker_text = marker_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"P0-3 marker is not UTF-8: {exc}") from exc
    if "Passed." not in marker_text:
        _fail("P0-3 completion marker does not record Passed")

    stdout_path = _safe_file(stdout_value, label="P0-3 stdout")
    stdout = _review_p0_3_stdout(stdout_path, hashes=hashes)
    if stdout["data_contract_sha256"] != data_contract["sha256"]:
        _fail("P0-3 stdout data-contract digest does not match the contract file")
    results_path = _child_file(root, "p0_3_results.json", label="P0-3 results")
    results = _list(
        _load_json(results_path, label="p0_3.results", hashes=hashes),
        label="P0-3 results",
    )
    if len(results) != 2:
        _fail("P0-3 results must contain exactly Psi=8 and Psi=16")

    reviewed: list[dict[str, Any]] = []
    for index, psi in enumerate((8, 16)):
        label = f"P0-3 Psi={psi}"
        metric = _mapping(results[index], label=label)
        per_path = _child_file(root, f"psi{psi}/p0_3_metrics.json", label=f"{label} metrics")
        per_metric = _mapping(
            _load_json(per_path, label=f"p0_3.psi{psi}.metrics", hashes=hashes),
            label=f"{label} metrics",
        )
        aggregate_contract_sha = _nonempty_string(
            _at(metric, "data_contract_sha256"),
            label=f"{label}.data_contract_sha256",
        )
        per_contract_sha = _nonempty_string(
            _at(per_metric, "data_contract_sha256"),
            label=f"{label} per-Psi data_contract_sha256",
        )
        if aggregate_contract_sha != per_contract_sha:
            _fail(f"{label} aggregate and per-Psi data-contract references differ")
        if (
            aggregate_contract_sha != data_contract["sha256"]
            or per_contract_sha != data_contract["sha256"]
        ):
            _fail(f"{label} metrics do not bind the canonical P0-3 data contract")
        if per_metric != metric:
            _fail(f"{label} aggregate and per-Psi metrics differ")
        _exact_int(_at(metric, "psi"), label=f"{label}.psi", expected=psi)
        steps = _exact_int(
            _at(metric, "steps"), label=f"{label}.steps", expected=P0_3_STEPS[psi]
        )
        device = _nonempty_string(_at(metric, "device"), label=f"{label}.device")
        if not device.startswith("cuda"):
            _fail(f"{label} must run on CUDA")
        if _at(metric, "amp_dtype") not in {"bf16", "bfloat16"}:
            _fail(f"{label} must use bf16 autocast")
        _exact_bool(
            _at(metric, "gradient_checkpointing"),
            label=f"{label}.gradient_checkpointing",
            expected=True,
        )
        kwargs = _mapping(
            _at(metric, "gradient_checkpointing_kwargs"),
            label=f"{label}.gradient_checkpointing_kwargs",
        )
        if kwargs != {"use_reentrant": False}:
            _fail(f"{label} must use non-reentrant gradient checkpointing")
        _exact_int(_at(metric, "seq_len"), label=f"{label}.seq_len", expected=128)
        _exact_int(_at(metric, "batch_size"), label=f"{label}.batch_size", expected=4)
        _exact_int(
            _at(metric, "tokens_per_step"),
            label=f"{label}.tokens_per_step",
            expected=512,
        )
        _exact_int(
            _at(metric, "approx_tokens_seen"),
            label=f"{label}.approx_tokens_seen",
            expected=steps * 512,
        )

        initial = _finite_number(_at(metric, "initial_probe_loss"), label=f"{label}.initial_probe_loss")
        final = _finite_number(_at(metric, "final_probe_loss"), label=f"{label}.final_probe_loss")
        absolute = _finite_number(_at(metric, "abs_loss_drop"), label=f"{label}.abs_loss_drop", positive=True)
        relative = _finite_number(_at(metric, "rel_loss_drop"), label=f"{label}.rel_loss_drop", positive=True)
        if not initial > final:
            _fail(f"{label} probe loss did not decrease")
        _close(absolute, initial - final, label=f"{label}.abs_loss_drop")
        _close(relative, absolute / max(abs(initial), 1e-12), label=f"{label}.rel_loss_drop")
        for name in (
            "train_loss_first",
            "train_loss_last",
            "train_loss_min",
            "grad_norm_max",
            "save_load_logits_max_abs",
            "cache_split_logits_max_abs",
        ):
            minimum = 0.0 if "max_abs" in name or "grad_norm" in name else None
            _finite_number(_at(metric, name), label=f"{label}.{name}", minimum=minimum)
        # The harness checks every element with atol + rtol * abs(expected).
        # A scalar maximum absolute difference cannot independently reconstruct
        # that pass/fail decision, so do not incorrectly compare max_abs to atol.
        generation = _mapping(_at(metric, "generation"), label=f"{label}.generation")
        prompt_len = _exact_int(
            _at(generation, "prompt_len"), label=f"{label}.generation.prompt_len", minimum=1
        )
        generated_len = _exact_int(
            _at(generation, "generated_len"), label=f"{label}.generation.generated_len", minimum=1
        )
        if generated_len <= prompt_len:
            _fail(f"{label} generation did not append a token")
        checkpoint = _child_directory(root, f"psi{psi}", label=f"{label} checkpoint")
        _validate_embedded_directory(
            _at(metric, "checkpoint_dir"), checkpoint, label=f"{label}.checkpoint_dir"
        )

        raw_steps = [item for item in stdout["steps"] if item["psi"] == psi]
        raw_probe = stdout["probes"][psi]
        if len(raw_steps) != steps:
            _fail(f"{label} stdout step count differs from metrics")
        _close(
            raw_steps[0]["loss"],
            round(float(_at(metric, "train_loss_first")), 4),
            label=f"{label} first stdout loss",
            atol=5.1e-5,
        )
        _close(
            raw_steps[-1]["loss"],
            round(float(_at(metric, "train_loss_last")), 4),
            label=f"{label} last stdout loss",
            atol=5.1e-5,
        )
        _close(raw_probe["initial"], round(initial, 4), label=f"{label} stdout initial probe", atol=5.1e-5)
        _close(raw_probe["final"], round(final, 4), label=f"{label} stdout final probe", atol=5.1e-5)
        _close(
            min(item["loss"] for item in raw_steps),
            round(float(_at(metric, "train_loss_min")), 4),
            label=f"{label} minimum stdout loss",
            atol=5.1e-5,
        )
        _close(
            max(item["grad_norm"] for item in raw_steps),
            round(float(_at(metric, "grad_norm_max")), 4),
            label=f"{label} maximum stdout grad norm",
            atol=5.1e-5,
        )
        _close(raw_probe["drop"], round(absolute, 4), label=f"{label} stdout probe drop", atol=5.1e-5)
        _close(
            raw_probe["relative_percent"],
            round(relative * 100.0, 4),
            label=f"{label} stdout relative probe drop",
            atol=5.1e-5,
        )
        reviewed.append(
            {
                "psi": psi,
                "optimizer_steps": steps,
                "step_events": len(raw_steps),
                "probe_loss_decreased": True,
                "checkpointing_non_reentrant": True,
                "save_reload_checked": True,
                "cache_checked": True,
                "generation_checked": True,
            }
        )
    return {
        "status": "passed",
        "data_contract": data_contract,
        "stdout_step_event_count": stdout["step_event_count"],
        "stdout_line_count": stdout["line_count"],
        "runs": reviewed,
    }


def _review_p0_4_tokenizer_projection(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    tokenizer = _mapping(value, label=label)
    if set(tokenizer) != {
        "class",
        "counts",
        "hashes",
        "is_fast",
        "operationalization",
    }:
        _fail(f"{label} fields are incomplete or ambiguous")
    tokenizer_class = _nonempty_string(_at(tokenizer, "class"), label=f"{label}.class")
    if tokenizer_class != "GPT2TokenizerFast":
        _fail(f"{label}.class must be GPT2TokenizerFast")
    tokenizer_is_fast = _exact_bool(
        _at(tokenizer, "is_fast"), label=f"{label}.is_fast", expected=True
    )

    counts = _mapping(_at(tokenizer, "counts"), label=f"{label}.counts")
    count_fields = {
        "added_vocabulary",
        "all_special_tokens",
        "probes",
        "special_token_boundary_probes",
        "tokenizer_length",
        "vocab_size",
        "vocabulary",
    }
    if set(counts) != count_fields:
        _fail(f"{label}.counts fields are incomplete or ambiguous")
    normalized_counts = {
        field: _exact_int(
            _at(counts, field), label=f"{label}.counts.{field}", minimum=0
        )
        for field in sorted(count_fields)
    }
    for field in ("tokenizer_length", "vocab_size", "vocabulary"):
        if normalized_counts[field] != 50_257:
            _fail(f"{label}.counts.{field} must equal 50257")
    if normalized_counts["all_special_tokens"] < 1:
        _fail(f"{label} must bind at least one special token")
    if normalized_counts["probes"] != 5:
        _fail(f"{label} must bind five tokenizer probes")
    if normalized_counts["special_token_boundary_probes"] != (
        normalized_counts["all_special_tokens"] * 7
    ):
        _fail(f"{label} special-token boundary probe count is inconsistent")

    hashes = _mapping(_at(tokenizer, "hashes"), label=f"{label}.hashes")
    hash_fields = {
        "probe_manifest_sha256",
        "special_tokens_manifest_sha256",
        "vocabulary_manifest_sha256",
    }
    if set(hashes) != hash_fields:
        _fail(f"{label}.hashes fields are incomplete or ambiguous")
    normalized_hashes: dict[str, str] = {}
    for field in sorted(hash_fields):
        digest = _nonempty_string(_at(hashes, field), label=f"{label}.hashes.{field}")
        if HEX64_RE.fullmatch(digest) is None:
            _fail(f"{label}.hashes.{field} must be lowercase SHA-256")
        normalized_hashes[field] = digest

    operationalization = _mapping(
        _at(tokenizer, "operationalization"),
        label=f"{label}.operationalization",
    )
    if set(operationalization) != {
        "model_input_names",
        "model_max_length",
        "padding_side",
        "truncation_side",
    }:
        _fail(f"{label}.operationalization fields are incomplete or ambiguous")
    model_input_names = _list(
        _at(operationalization, "model_input_names"),
        label=f"{label}.operationalization.model_input_names",
    )
    if model_input_names != ["input_ids", "attention_mask"]:
        _fail(f"{label}.operationalization.model_input_names differs from GPT-2")
    model_max_length = _exact_int(
        _at(operationalization, "model_max_length"),
        label=f"{label}.operationalization.model_max_length",
        expected=4_096,
    )
    padding_side = _nonempty_string(
        _at(operationalization, "padding_side"),
        label=f"{label}.operationalization.padding_side",
    )
    truncation_side = _nonempty_string(
        _at(operationalization, "truncation_side"),
        label=f"{label}.operationalization.truncation_side",
    )
    if padding_side != "right" or truncation_side != "right":
        _fail(f"{label}.operationalization must use right padding and truncation")
    return {
        "class": tokenizer_class,
        "counts": normalized_counts,
        "hashes": normalized_hashes,
        "is_fast": tokenizer_is_fast,
        "operationalization": {
            "model_input_names": list(model_input_names),
            "model_max_length": model_max_length,
            "padding_side": padding_side,
            "truncation_side": truncation_side,
        },
    }


def _review_p0_4_data_contract(
    root: Path,
    *,
    logical: str,
    hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _child_file(root, "data_contract.json", label=f"{logical} data contract")
    raw = _read_bytes(path, label=f"{logical}.data_contract", hashes=hashes)
    contract = _mapping(
        _decode_json_bytes(raw, label=f"{logical} data contract"),
        label=f"{logical} data contract",
    )
    if raw != _canonical_bytes(contract):
        _fail(f"{logical} data contract must use canonical JSON bytes")
    if set(contract) != {"packing", "schema_version", "source", "status", "tokenizer"}:
        _fail(f"{logical} data contract fields are incomplete or ambiguous")
    _equals(
        _at(contract, "schema_version"),
        "multiscreen-p0-4-data-contract-v1",
        label=f"{logical}.data_contract.schema_version",
    )
    _equals(_at(contract, "status"), "recorded", label=f"{logical}.data_contract.status")

    source = _mapping(_at(contract, "source"), label=f"{logical}.data_contract.source")
    if set(source) != {
        "algorithm",
        "data_dir",
        "data_files",
        "dataset_config",
        "dataset_fingerprint",
        "dataset_name",
        "max_texts",
        "revision",
        "revision_resolution",
        "selected_text_count",
        "selected_text_manifest_sha256",
        "selected_text_utf8_bytes",
        "source_kind",
        "streaming",
        "text_column",
        "text_file",
        "train_split",
    }:
        _fail(f"{logical} data contract source fields are incomplete or ambiguous")
    _exact_values(
        source,
        {
            "algorithm": "sha256-length-framed-utf8-texts-v1",
            "data_dir": None,
            "data_files": None,
            "dataset_config": None,
            "dataset_name": "roneneldan/TinyStories",
            "max_texts": 20_000,
            "revision": None,
            "revision_resolution": "default_ref",
            "selected_text_count": 20_000,
            "source_kind": "huggingface_dataset",
            "streaming": False,
            "text_column": "text",
            "text_file": None,
            "train_split": "train[:20000]",
        },
        label=f"{logical}.data_contract.source",
    )
    fingerprint = _nonempty_string(
        _at(source, "dataset_fingerprint"),
        label=f"{logical}.data_contract.source.dataset_fingerprint",
    )
    if P0_3_DATASET_FINGERPRINT_RE.fullmatch(fingerprint) is None:
        _fail(f"{logical} dataset fingerprint is not lowercase hexadecimal")
    text_manifest = _nonempty_string(
        _at(source, "selected_text_manifest_sha256"),
        label=f"{logical}.data_contract.source.selected_text_manifest_sha256",
    )
    if HEX64_RE.fullmatch(text_manifest) is None:
        _fail(f"{logical} selected-text manifest must be lowercase SHA-256")
    _exact_int(
        _at(source, "selected_text_utf8_bytes"),
        label=f"{logical}.data_contract.source.selected_text_utf8_bytes",
        minimum=1,
    )

    packing = _mapping(_at(contract, "packing"), label=f"{logical}.data_contract.packing")
    if set(packing) != {
        "algorithm",
        "chunk_count",
        "chunk_size",
        "eos_token_id",
        "legacy_shifted_labels",
        "max_train_tokens",
        "packed_token_stream_sha256",
        "return_labels_are_shifted",
        "seq_len",
        "usable_token_count",
    }:
        _fail(f"{logical} data contract packing fields are incomplete or ambiguous")
    _exact_values(
        packing,
        {
            "algorithm": "sha256-uint32-le-packed-token-stream-v1",
            "chunk_count": 128,
            "chunk_size": 4_097,
            "eos_token_id": 50_256,
            "legacy_shifted_labels": True,
            "max_train_tokens": 524_416,
            "return_labels_are_shifted": True,
            "seq_len": 4_096,
            "usable_token_count": 524_416,
        },
        label=f"{logical}.data_contract.packing",
    )
    if _at(packing, "chunk_count") * _at(packing, "chunk_size") != _at(
        packing, "usable_token_count"
    ):
        _fail(f"{logical} packed-token count differs from its chunk shape")
    packed_manifest = _nonempty_string(
        _at(packing, "packed_token_stream_sha256"),
        label=f"{logical}.data_contract.packing.packed_token_stream_sha256",
    )
    if HEX64_RE.fullmatch(packed_manifest) is None:
        _fail(f"{logical} packed-token manifest must be lowercase SHA-256")

    tokenizer = _review_p0_4_tokenizer_projection(
        _at(contract, "tokenizer"),
        label=f"{logical}.data_contract.tokenizer",
    )
    digest = _sha256_bytes(raw)
    reference = {
        "file": "data_contract.json",
        "schema_version": "multiscreen-p0-4-data-contract-v1",
        "sha256": digest,
    }
    return dict(contract), {
        "dataset_fingerprint_sha256": _sha256_bytes(fingerprint.encode("utf-8")),
        "packed_token_stream_sha256": packed_manifest,
        "reference": reference,
        "selected_text_manifest_sha256": text_manifest,
        "tokenizer_projection": tokenizer,
    }


def _review_p0_4_lane(
    root_value: str | os.PathLike[str],
    *,
    psi: int,
    expected_cache: Path,
    hashes: dict[str, str],
) -> dict[str, Any]:
    logical = f"p0_4_psi{psi}"
    root = _safe_root(root_value, label=f"P0-4 Psi={psi} root")
    _failure_artifact_check(root, label=f"P0-4 Psi={psi} root")
    data_contract, reviewed_data_contract = _review_p0_4_data_contract(
        root,
        logical=logical,
        hashes=hashes,
    )
    marker = _child_file(root, "P0-4_COMPLETE.md", label=f"P0-4 Psi={psi} marker")
    marker_raw = _read_bytes(marker, label=f"{logical}.completion_marker", hashes=hashes)
    try:
        marker_text = marker_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"P0-4 Psi={psi} marker is not UTF-8: {exc}") from exc
    if "Passed." not in marker_text:
        _fail(f"P0-4 Psi={psi} marker does not record Passed")

    summary_path = _child_file(root, "summary.json", label=f"P0-4 Psi={psi} summary")
    metrics_path = _child_file(root, "metrics.jsonl", label=f"P0-4 Psi={psi} metrics")
    summary = _mapping(
        _load_json(summary_path, label=f"{logical}.summary", hashes=hashes),
        label=f"P0-4 Psi={psi} summary",
    )
    events = _load_jsonl(metrics_path, label=f"{logical}.metrics", hashes=hashes)
    timestamp_utc = _nonempty_string(
        _at(summary, "timestamp_utc"), label=f"{logical}.timestamp_utc"
    )
    _parse_utc(timestamp_utc, label=f"{logical}.timestamp_utc")
    sequence = tuple(event.get("event") for event in events)
    if sequence != P0_4_EVENT_SEQUENCE:
        _fail(f"P0-4 Psi={psi} has missing, duplicate, extra, or out-of-order events")
    if len(events) != 57:
        _fail(f"P0-4 Psi={psi} must contain exactly 57 events")

    if set(summary) != {
        "stage",
        "status",
        "timestamp_utc",
        "qualification",
        "environment",
        "settings",
        "model",
        "data",
        "training",
        "checks",
    }:
        _fail(f"{logical}.summary fields are incomplete or ambiguous")
    _equals(_at(summary, "stage"), "P0-4", label=f"{logical}.stage")
    _equals(_at(summary, "status"), "passed", label=f"{logical}.status")
    qualification = _mapping(
        _at(summary, "qualification"), label=f"{logical}.qualification"
    )
    if set(qualification) != {"qualified", "conditions"}:
        _fail(f"{logical}.qualification fields are incomplete or ambiguous")
    _exact_bool(
        _at(qualification, "qualified"),
        label=f"{logical}.qualified",
        expected=True,
    )
    conditions = _mapping(
        _at(qualification, "conditions"), label=f"{logical}.conditions"
    )
    required_conditions = {
        "gpt2_vocab_50257",
        "context_4096",
        "cuda_device",
        "bf16_amp",
        "optimizer_steps_at_least_50",
    }
    if set(conditions) != required_conditions:
        _fail(f"P0-4 Psi={psi} qualification conditions are incomplete or ambiguous")
    for name in sorted(required_conditions):
        _exact_bool(conditions[name], label=f"{logical}.qualification.{name}", expected=True)

    settings = _mapping(_at(summary, "settings"), label=f"{logical}.settings")
    expected_setting_fields = {
        "allow_cpu",
        "amp_dtype",
        "batch_size",
        "betas",
        "cache_atol",
        "cache_dir",
        "cache_rtol",
        "cache_tokens",
        "config_dir",
        "data_dir",
        "data_files",
        "dataset_config",
        "dataset_name",
        "device",
        "eps",
        "expected_vocab_size",
        "fused_adamw",
        "grad_accum",
        "gradient_checkpointing",
        "log_every",
        "lr",
        "max_grad_norm",
        "max_new_tokens",
        "max_texts",
        "max_train_tokens",
        "min_loss_drop",
        "min_rel_loss_drop",
        "num_workers",
        "output_dir",
        "probe_replay_every",
        "prompt",
        "reload_atol",
        "reload_rtol",
        "reload_tokens",
        "repo_root",
        "revision",
        "seed",
        "seq_len",
        "steps",
        "streaming",
        "text_column",
        "text_file",
        "tokenizer_name",
        "tokenizer_use_fast",
        "train_split",
        "weight_decay",
    }
    if set(settings) != expected_setting_fields:
        _fail(f"{logical}.settings fields are incomplete or ambiguous")
    reviewer_checkout = Path(__file__).resolve().parents[1]
    expected_config = (
        reviewer_checkout / f"configs/p0_4_multiscreen_psi{psi}_gpt2_ctx4096"
    )
    _equals(
        _at(settings, "repo_root"),
        os.fspath(reviewer_checkout),
        label=f"{logical}.settings.repo_root",
    )
    _equals(
        _at(settings, "config_dir"),
        os.fspath(expected_config),
        label=f"{logical}.settings.config_dir",
    )
    _equals(
        _at(settings, "cache_dir"),
        os.fspath(expected_cache),
        label=f"{logical}.settings.cache_dir",
    )
    _exact_values(
        settings,
        {
            "dataset_name": "roneneldan/TinyStories",
            "dataset_config": None,
            "train_split": "train[:20000]",
            "text_column": "text",
            "text_file": None,
            "data_files": None,
            "data_dir": None,
            "revision": None,
            "streaming": False,
            "min_loss_drop": 0.01,
            "min_rel_loss_drop": 0.001,
            "reload_tokens": 16,
            "cache_tokens": 24,
            "prompt": "Once upon a time",
            "max_new_tokens": 8,
            "num_workers": 0,
        },
        label=f"{logical}.settings",
    )
    _exact_int(_at(settings, "expected_vocab_size"), label=f"{logical}.expected_vocab_size", expected=50257)
    _exact_int(_at(settings, "seq_len"), label=f"{logical}.seq_len", expected=4096)
    _exact_int(_at(settings, "steps"), label=f"{logical}.steps", expected=50)
    _exact_int(_at(settings, "batch_size"), label=f"{logical}.batch_size", expected=1)
    _exact_int(_at(settings, "grad_accum"), label=f"{logical}.grad_accum", expected=8)
    _validate_embedded_directory(
        _at(settings, "output_dir"), root, label=f"{logical}.settings.output_dir"
    )
    _nonempty_string(_at(settings, "cache_dir"), label=f"{logical}.settings.cache_dir")
    _equals(_at(settings, "tokenizer_name"), "gpt2", label=f"{logical}.tokenizer_name")
    _exact_bool(_at(settings, "tokenizer_use_fast"), label=f"{logical}.tokenizer_use_fast", expected=True)
    _exact_int(_at(settings, "max_texts"), label=f"{logical}.max_texts", expected=20000)
    _exact_int(_at(settings, "max_train_tokens"), label=f"{logical}.max_train_tokens", expected=524416)
    _close(_finite_number(_at(settings, "lr"), label=f"{logical}.lr"), 0.0006, label=f"{logical}.lr")
    _close(_finite_number(_at(settings, "weight_decay"), label=f"{logical}.weight_decay"), 0.0, label=f"{logical}.weight_decay")
    betas = _list(_at(settings, "betas"), label=f"{logical}.betas")
    if len(betas) != 2:
        _fail(f"{logical}.betas must contain exactly two values")
    _close(_finite_number(betas[0], label=f"{logical}.betas[0]"), 0.9, label=f"{logical}.betas[0]")
    _close(_finite_number(betas[1], label=f"{logical}.betas[1]"), 0.95, label=f"{logical}.betas[1]")
    _close(_finite_number(_at(settings, "eps"), label=f"{logical}.eps"), 1e-8, label=f"{logical}.eps")
    _close(_finite_number(_at(settings, "max_grad_norm"), label=f"{logical}.max_grad_norm"), 1.0, label=f"{logical}.max_grad_norm")
    _exact_bool(_at(settings, "fused_adamw"), label=f"{logical}.fused_adamw", expected=True)
    _exact_int(_at(settings, "probe_replay_every"), label=f"{logical}.probe_replay_every", expected=4)
    _exact_int(_at(settings, "seed"), label=f"{logical}.seed", expected=42)
    _exact_int(_at(settings, "log_every"), label=f"{logical}.log_every", expected=1)
    for name, expected in (("reload_atol", 1e-5), ("reload_rtol", 1e-5), ("cache_atol", 0.03), ("cache_rtol", 0.03)):
        _close(_finite_number(_at(settings, name), label=f"{logical}.{name}"), expected, label=f"{logical}.{name}")
    _equals(_at(settings, "amp_dtype"), "bf16", label=f"{logical}.amp_dtype")
    _exact_bool(_at(settings, "gradient_checkpointing"), label=f"{logical}.gradient_checkpointing", expected=True)
    _exact_bool(_at(settings, "allow_cpu"), label=f"{logical}.allow_cpu", expected=False)
    _equals(_at(settings, "device"), "cuda:0", label=f"{logical}.device")

    environment = _mapping(_at(summary, "environment"), label=f"{logical}.environment")
    if set(environment) != {
        "python",
        "platform",
        "torch",
        "transformers",
        "datasets",
        "device",
        "cuda_available",
        "cuda_version",
        "gpu_name",
        "gpu_total_memory_bytes",
        "bf16_supported",
    }:
        _fail(f"{logical}.environment fields are incomplete or ambiguous")
    _nonempty_string(_at(environment, "python"), label=f"{logical}.environment.python")
    _nonempty_string(_at(environment, "platform"), label=f"{logical}.environment.platform")
    _equals(_at(environment, "device"), "cuda:0", label=f"{logical}.environment.device")
    _equals(
        _at(environment, "device"),
        _at(settings, "device"),
        label=f"{logical}.environment.device_binding",
    )
    _exact_bool(_at(environment, "cuda_available"), label=f"{logical}.cuda_available", expected=True)
    _exact_bool(_at(environment, "bf16_supported"), label=f"{logical}.bf16_supported", expected=True)
    gpu_total_memory_bytes = _exact_int(
        _at(environment, "gpu_total_memory_bytes"),
        label=f"{logical}.environment.gpu_total_memory_bytes",
        minimum=1,
    )

    model = _mapping(_at(summary, "model"), label=f"{logical}.model")
    if set(model) != {
        "psi",
        "parameter_count",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "key_dim",
        "value_dim",
        "max_position_embeddings",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "dense_similarity_one_layer_lower_bound_bytes",
    }:
        _fail(f"{logical}.model fields are incomplete or ambiguous")
    _exact_int(_at(model, "psi"), label=f"{logical}.model.psi", expected=psi)
    _exact_int(
        _at(model, "parameter_count"),
        label=f"{logical}.parameter_count",
        expected={8: 4_134_146, 16: 27_546_626}[psi],
    )
    _exact_int(_at(model, "key_dim"), label=f"{logical}.key_dim", expected=16)
    _exact_int(_at(model, "value_dim"), label=f"{logical}.value_dim", expected=64)
    _exact_int(
        _at(model, "dense_similarity_one_layer_lower_bound_bytes"),
        label=f"{logical}.dense_similarity_one_layer_lower_bound_bytes",
        expected={8: 268_435_456, 16: 536_870_912}[psi],
    )
    _exact_int(_at(model, "vocab_size"), label=f"{logical}.model.vocab_size", expected=50257)
    _exact_int(_at(model, "max_position_embeddings"), label=f"{logical}.max_position_embeddings", expected=4096)
    _exact_bool(_at(model, "gradient_checkpointing"), label=f"{logical}.model.gradient_checkpointing", expected=True)
    if _at(model, "gradient_checkpointing_kwargs") != {"use_reentrant": False}:
        _fail(f"P0-4 Psi={psi} must use non-reentrant checkpointing")
    _equals(_at(environment, "transformers"), "4.57.6", label=f"{logical}.environment.transformers")
    _equals(_at(environment, "datasets"), "5.0.1", label=f"{logical}.environment.datasets")
    _equals(
        _at(environment, "torch"),
        "2.7.1+cu128",
        label=f"{logical}.environment.torch",
    )
    _equals(
        _at(environment, "cuda_version"),
        "12.8",
        label=f"{logical}.environment.cuda_version",
    )
    _nonempty_string(
        _at(environment, "gpu_name"), label=f"{logical}.environment.gpu_name"
    )
    data = _mapping(_at(summary, "data"), label=f"{logical}.data")
    if set(data) != {
        "data_contract",
        "max_train_tokens",
        "packed_chunks",
        "seq_len",
        "source",
        "texts_loaded",
        "tokenizer_class",
        "tokenizer_vocab_size",
        "train_split",
    }:
        _fail(f"{logical}.data fields are incomplete or ambiguous")
    _exact_values(
        data,
        {
            "source": "roneneldan/TinyStories",
            "train_split": "train[:20000]",
            "texts_loaded": 20_000,
            "packed_chunks": 128,
            "seq_len": 4_096,
            "max_train_tokens": 524_416,
            "tokenizer_class": "GPT2TokenizerFast",
            "tokenizer_vocab_size": 50_257,
            "data_contract": reviewed_data_contract["reference"],
        },
        label=f"{logical}.data",
    )
    if (
        _at(data, "tokenizer_class")
        != reviewed_data_contract["tokenizer_projection"]["class"]
    ):
        _fail(f"{logical} summary tokenizer class differs from its data contract")
    _exact_int(_at(data, "tokenizer_vocab_size"), label=f"{logical}.tokenizer_vocab_size", expected=50257)
    _exact_int(_at(data, "seq_len"), label=f"{logical}.data.seq_len", expected=4096)

    training = _mapping(_at(summary, "training"), label=f"{logical}.training")
    if set(training) != {
        "optimizer_steps",
        "gradient_accumulation_steps",
        "microbatch_size",
        "effective_batch_tokens",
        "initial_probe_loss",
        "final_probe_loss",
        "abs_loss_drop",
        "rel_loss_drop",
        "train_loss_first",
        "train_loss_last",
        "train_loss_min",
        "grad_norm_max",
        "elapsed_sec",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }:
        _fail(f"{logical}.training fields are incomplete or ambiguous")
    _review_cuda_memory(
        training, label=f"{logical}.training", total_bytes=gpu_total_memory_bytes
    )
    training_elapsed = _finite_number(
        _at(training, "elapsed_sec"), label=f"{logical}.training.elapsed_sec", minimum=0.0
    )
    _exact_int(_at(training, "optimizer_steps"), label=f"{logical}.optimizer_steps", expected=50)
    _exact_int(_at(model, "hidden_size"), label=f"{logical}.hidden_size", expected=psi * psi)
    _exact_int(_at(model, "num_hidden_layers"), label=f"{logical}.num_hidden_layers", expected=psi)
    _exact_int(_at(model, "num_attention_heads"), label=f"{logical}.num_attention_heads", expected=psi)
    _exact_int(_at(model, "parameter_count"), label=f"{logical}.parameter_count", expected={8: 4_134_146, 16: 27_546_626}[psi])
    _exact_int(_at(training, "gradient_accumulation_steps"), label=f"{logical}.training.grad_accum", expected=8)
    _exact_int(_at(training, "microbatch_size"), label=f"{logical}.training.microbatch_size", expected=1)
    initial = _finite_number(_at(training, "initial_probe_loss"), label=f"{logical}.initial_probe_loss")
    final = _finite_number(_at(training, "final_probe_loss"), label=f"{logical}.final_probe_loss")
    absolute = _finite_number(_at(training, "abs_loss_drop"), label=f"{logical}.abs_loss_drop", positive=True)
    relative = _finite_number(_at(training, "rel_loss_drop"), label=f"{logical}.rel_loss_drop", positive=True)
    if not initial > final:
        _fail(f"P0-4 Psi={psi} probe loss did not decrease")
    _close(absolute, initial - final, label=f"{logical}.abs_loss_drop")
    _close(relative, absolute / max(abs(initial), 1e-12), label=f"{logical}.rel_loss_drop")
    if absolute < 0.01 and relative < 0.001:
        _fail(f"P0-4 Psi={psi} probe loss did not meet either fixed threshold")
    for name in ("train_loss_first", "train_loss_last", "train_loss_min", "grad_norm_max"):
        _finite_number(_at(training, name), label=f"{logical}.{name}", minimum=0.0 if name == "grad_norm_max" else None)
    _exact_int(_at(training, "effective_batch_tokens"), label=f"{logical}.effective_batch_tokens", expected=32768)

    checks = _mapping(_at(summary, "checks"), label=f"{logical}.checks")
    if set(checks) != {"save_reload", "cache", "generation"}:
        _fail(f"{logical}.checks fields are incomplete or ambiguous")
    reload_check = _mapping(_at(checks, "save_reload"), label=f"{logical}.save_reload")
    if set(reload_check) != {
        "checkpoint_dir",
        "loaded_logits_max_abs",
        "reload_check_tokens",
    }:
        _fail(f"{logical}.save_reload fields are incomplete or ambiguous")
    checkpoint = _child_directory(root, "checkpoint", label=f"P0-4 Psi={psi} checkpoint")
    _validate_embedded_directory(_at(reload_check, "checkpoint_dir"), checkpoint, label=f"{logical}.checkpoint_dir")
    _finite_number(_at(reload_check, "loaded_logits_max_abs"), label=f"{logical}.loaded_logits_max_abs", minimum=0.0)
    _exact_int(_at(reload_check, "reload_check_tokens"), label=f"{logical}.reload_check_tokens", expected=16)
    cache = _mapping(_at(checks, "cache"), label=f"{logical}.cache")
    if set(cache) != {
        "cache_split_logits_max_abs",
        "cache_check_tokens",
        "cache_split",
    }:
        _fail(f"{logical}.cache fields are incomplete or ambiguous")
    _finite_number(_at(cache, "cache_split_logits_max_abs"), label=f"{logical}.cache_split_logits_max_abs", minimum=0.0)
    _exact_int(_at(cache, "cache_check_tokens"), label=f"{logical}.cache_check_tokens", expected=24)
    _exact_int(_at(cache, "cache_split"), label=f"{logical}.cache_split", expected=12)
    # Successful harness completion proves elementwise atol+rtol assertions;
    # max_abs alone is not an equivalent pass/fail criterion.
    generation = _mapping(_at(checks, "generation"), label=f"{logical}.generation")
    if set(generation) != {"generated_len", "prompt", "prompt_len", "sample_text"}:
        _fail(f"{logical}.generation fields are incomplete or ambiguous")
    _equals(_at(generation, "prompt"), "Once upon a time", label=f"{logical}.generation.prompt")
    sample_text = _nonempty_string(_at(generation, "sample_text"), label=f"{logical}.generation.sample_text")
    if len(sample_text) > 400:
        _fail(f"{logical}.generation.sample_text exceeds the producer bound")
    prompt_len = _exact_int(_at(generation, "prompt_len"), label=f"{logical}.prompt_len", minimum=1)
    generated_len = _exact_int(_at(generation, "generated_len"), label=f"{logical}.generated_len", minimum=1)
    if not prompt_len < generated_len <= prompt_len + 8:
        _fail(f"P0-4 Psi={psi} generation did not append a token")

    run_start, preflight = events[0], events[1]
    if set(run_start) != {
        "event", "stage", "timestamp_utc", "settings", "environment", "data_contract"
    }:
        _fail(f"{logical}.run_start fields are incomplete or ambiguous")
    if set(preflight) != {"event", "stage", "timestamp_utc", "model", "data"}:
        _fail(f"{logical}.preflight_complete fields are incomplete or ambiguous")
    if run_start.get("settings") != settings or run_start.get("environment") != environment:
        _fail(f"P0-4 Psi={psi} run_start differs from summary")
    if run_start.get("data_contract") != reviewed_data_contract["reference"]:
        _fail(f"P0-4 Psi={psi} run_start data-contract reference differs")
    if data.get("data_contract") != reviewed_data_contract["reference"]:
        _fail(f"P0-4 Psi={psi} preflight data-contract reference differs")
    if preflight.get("model") != model or preflight.get("data") != data:
        _fail(f"P0-4 Psi={psi} preflight differs from summary")
    previous_elapsed = -1.0
    step_events = events[2:52]
    event_times: list[dt.datetime] = []
    for event_index, event in enumerate(events):
        _equals(_at(event, "stage"), "P0-4", label=f"{logical}.event[{event_index}].stage")
        event_times.append(
            _parse_utc(
                _at(event, "timestamp_utc"),
                label=f"{logical}.event[{event_index}].timestamp_utc",
            )
        )
    if any(later < earlier for earlier, later in zip(event_times, event_times[1:])):
        _fail(f"P0-4 Psi={psi} event timestamps are out of order")
    expected_step_fields = {
        "event",
        "stage",
        "timestamp_utc",
        "optimizer_step",
        "optimizer_steps",
        "mean_loss",
        "micro_losses",
        "grad_norm",
        "elapsed_sec",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    previous_peak_allocated = -1
    previous_peak_reserved = -1
    for index, event in enumerate(step_events, start=1):
        if set(event) != expected_step_fields:
            _fail(f"{logical}.train_step fields are incomplete or ambiguous")
        event_memory = _review_cuda_memory(
            event,
            label=f"{logical}.step[{index}]",
            total_bytes=gpu_total_memory_bytes,
        )
        if (
            event_memory["peak_allocated_bytes"] < previous_peak_allocated
            or event_memory["peak_reserved_bytes"] < previous_peak_reserved
        ):
            _fail(f"{logical} peak CUDA memory is not monotonic across steps")
        previous_peak_allocated = event_memory["peak_allocated_bytes"]
        previous_peak_reserved = event_memory["peak_reserved_bytes"]
        _equals(_at(event, "stage"), "P0-4", label=f"{logical}.event[{index}].stage")
        _exact_int(_at(event, "optimizer_step"), label=f"{logical}.optimizer_step", expected=index)
        _exact_int(_at(event, "optimizer_steps"), label=f"{logical}.optimizer_steps", expected=50)
        micro = _list(_at(event, "micro_losses"), label=f"{logical}.micro_losses")
        if len(micro) != 8:
            _fail(f"P0-4 Psi={psi} step {index} must contain 8 micro losses")
        micro_values = [
            _finite_number(item, label=f"{logical}.step[{index}].micro_losses")
            for item in micro
        ]
        mean = _finite_number(_at(event, "mean_loss"), label=f"{logical}.step[{index}].mean_loss")
        _close(mean, sum(micro_values) / 8.0, label=f"{logical}.step[{index}].mean_loss")
        _finite_number(_at(event, "grad_norm"), label=f"{logical}.step[{index}].grad_norm", minimum=0.0)
        elapsed = _finite_number(_at(event, "elapsed_sec"), label=f"{logical}.step[{index}].elapsed_sec", minimum=0.0)
        if elapsed < previous_elapsed:
            _fail(f"P0-4 Psi={psi} elapsed times are out of order")
        previous_elapsed = elapsed
    if (
        _at(training, "peak_allocated_bytes") < previous_peak_allocated
        or _at(training, "peak_reserved_bytes") < previous_peak_reserved
    ):
        _fail(f"{logical} training summary peak memory is below a step peak")
    observed_losses = [float(_at(event, "mean_loss")) for event in step_events]
    observed_grads = [float(_at(event, "grad_norm")) for event in step_events]
    _close(float(_at(training, "train_loss_first")), observed_losses[0], label=f"{logical}.train_loss_first")
    _close(float(_at(training, "train_loss_last")), observed_losses[-1], label=f"{logical}.train_loss_last")
    _close(float(_at(training, "train_loss_min")), min(observed_losses), label=f"{logical}.train_loss_min")
    _close(float(_at(training, "grad_norm_max")), max(observed_grads), label=f"{logical}.grad_norm_max")
    if training_elapsed < previous_elapsed:
        _fail(f"P0-4 Psi={psi} training elapsed time precedes the last step")
    training_event = events[52]
    if set(training_event) != {"event", "stage", "timestamp_utc"} | set(training):
        _fail(f"{logical}.training_complete fields are incomplete or ambiguous")
    for name, value in training.items():
        if training_event[name] != value:
            _fail(f"P0-4 Psi={psi} training_complete differs at {name}")
    for event, expected, event_name in (
        (events[53], reload_check, "save_reload_check"),
        (events[54], cache, "cache_split_check"),
        (events[55], generation, "generation_check"),
    ):
        if set(event) != {"event", "stage", "timestamp_utc"} | set(expected):
            _fail(f"{logical}.{event_name} fields are incomplete or ambiguous")
        _equals(_at(event, "event"), event_name, label=f"{logical}.{event_name}.event")
        for name, value in expected.items():
            if event[name] != value:
                _fail(f"P0-4 Psi={psi} check event differs at {name}")
    run_complete = events[56]
    if set(run_complete) != {
        "event",
        "stage",
        "status",
        "qualification",
        "timestamp_utc",
        "data_contract",
    }:
        _fail(f"{logical}.run_complete fields are incomplete or ambiguous")
    _equals(_at(run_complete, "status"), "passed", label=f"{logical}.run_complete.status")
    if run_complete.get("data_contract") != reviewed_data_contract["reference"]:
        _fail(f"P0-4 Psi={psi} run_complete data-contract reference differs")
    if run_complete.get("qualification") != summary["qualification"]:
        _fail(f"P0-4 Psi={psi} run_complete qualification differs from summary")
    if _at(run_complete, "timestamp_utc") != timestamp_utc:
        _fail(f"P0-4 Psi={psi} run_complete timestamp differs from summary")
    return {
        "data_contract": reviewed_data_contract,
        "psi": psi,
        "status": "passed",
        "event_count": len(events),
        "optimizer_step_events": len(step_events),
        "probe_loss_decreased": True,
        "checkpointing_non_reentrant": True,
        "save_reload_checked": True,
        "cache_checked": True,
        "generation_checked": True,
        "qualified": True,
        "timestamp_utc": timestamp_utc,
    }


def _review_c3_data(
    root_value: str | os.PathLike[str],
    *,
    hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _safe_root(root_value, label="C3 data root")
    _failure_artifact_check(root, label="C3 data root")
    marker_names = sorted(path.name for path in root.glob("P0_5_C3_*_COMPLETE.json"))
    if marker_names != ["P0_5_C3_DATA_CONTRACT_COMPLETE.json"]:
        _fail("C3 data root must contain exactly its data-contract completion marker")
    data = _mapping(
        _load_json(
            _child_file(root, "data_contract.json", label="C3 data contract"),
            label="c3_data.data_contract",
            hashes=hashes,
        ),
        label="C3 data contract",
    )
    marker = _mapping(
        _load_json(
            _child_file(
                root,
                "P0_5_C3_DATA_CONTRACT_COMPLETE.json",
                label="C3 data marker",
            ),
            label="c3_data.completion_marker",
            hashes=hashes,
        ),
        label="C3 data marker",
    )
    expected_top_fields = {
        "accounting",
        "checked_accounting_match",
        "hashes",
        "limitations",
        "mode",
        "packing",
        "qualification",
        "source",
        "stage",
        "status",
        "tokenizer",
    }
    if set(data) != expected_top_fields:
        _fail("C3 data contract top-level fields are incomplete or ambiguous")
    tokenizer = _mapping(_at(data, "tokenizer"), label="c3_data.tokenizer")
    if set(tokenizer) != {
        "asset_manifest_sha256",
        "assets",
        "document_tokenization_truncation",
        "eos_token_id",
        "repository",
        "revision",
        "tokenizer_class",
        "vocab_size",
    }:
        _fail("C3 data tokenizer fields are incomplete or ambiguous")
    assets = _mapping(_at(tokenizer, "assets"), label="c3_data.tokenizer.assets")
    expected_asset_hashes = {
        "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
        "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
        "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    }
    if set(assets) != set(expected_asset_hashes):
        _fail("C3 data tokenizer asset inventory is incomplete or ambiguous")
    for filename, expected_digest in sorted(expected_asset_hashes.items()):
        asset = _mapping(
            assets[filename], label=f"c3_data.tokenizer.assets.{filename}"
        )
        if set(asset) != {"filename", "revision", "sha256", "size_bytes"}:
            _fail(f"C3 tokenizer asset {filename} fields are incomplete or ambiguous")
        _equals(
            _at(asset, "filename"),
            filename,
            label=f"c3_data.tokenizer.assets.{filename}.filename",
        )
        _equals(
            _at(asset, "revision"),
            "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            label=f"c3_data.tokenizer.assets.{filename}.revision",
        )
        _equals(
            _at(asset, "sha256"),
            expected_digest,
            label=f"c3_data.tokenizer.assets.{filename}.sha256",
        )
        _exact_int(
            _at(asset, "size_bytes"),
            label=f"c3_data.tokenizer.assets.{filename}.size_bytes",
            expected=C3_TOKENIZER_ASSET_SIZES[filename],
        )
    source = _mapping(_at(data, "source"), label="c3_data.source")
    if set(source) != {
        "canonical_hub_loader",
        "data_files",
        "datasets_version",
        "family",
        "file",
        "full_fingerprint",
        "local_parquet_fingerprint_not_accepted",
        "repository",
        "revision",
        "row_manifest_sha256",
        "row_records",
        "selection",
        "source_scope",
        "split",
        "streaming",
    }:
        _fail("C3 data source fields are incomplete or ambiguous")
    source_file = _mapping(_at(source, "file"), label="c3_data.source.file")
    if set(source_file) != {"filename", "revision", "sha256", "size_bytes"}:
        _fail("C3 data source file fields are incomplete or ambiguous")
    selection = _mapping(_at(source, "selection"), label="c3_data.source.selection")
    if set(selection) != {"fingerprint", "kind", "row_count", "start", "stop"}:
        _fail("C3 data source selection fields are incomplete or ambiguous")
    packing = _mapping(_at(data, "packing"), label="c3_data.packing")
    if set(packing) != {
        "discard_incomplete_tail_only",
        "document_handling",
        "legacy_shifted_labels",
        "return_labels_are_shifted",
    }:
        _fail("C3 data packing fields are incomplete or ambiguous")
    accounting = _mapping(_at(data, "accounting"), label="c3_data.accounting")
    if set(accounting) != {
        "concatenated_tokens",
        "discarded_tail_tokens",
        "eos_tokens",
        "nonempty_documents",
        "packed_chunks",
        "prediction_tokens_per_chunk",
        "selected_rows",
        "sequence_length",
        "stored_chunk_tokens",
        "text_tokens",
        "usable_tokens",
    }:
        _fail("C3 data accounting fields are incomplete or ambiguous")
    data_hashes = _mapping(_at(data, "hashes"), label="c3_data.hashes")
    if set(data_hashes) != {
        "packed_chunk_sha256",
        "row_manifest_sha256",
        "token_encoding",
        "token_stream_sha256",
    }:
        _fail("C3 data hash fields are incomplete or ambiguous")
    expected_limitations = [
        "The source is a pinned third-party SlimPajama reupload test shard.",
        "It is not claimed byte-identical to the paper corpus or representative of its train split.",
        "No raw source text is retained in this output.",
    ]
    if _at(data, "limitations") != expected_limitations:
        _fail("C3 data limitations differ from the fixed producer contract")
    _exact_values(
        data,
        {
            "stage": "P0.5-C3",
            "status": "passed",
            "mode": "data",
            "qualification": "diagnostic_pinned_slimpajama_family_shard",
            "tokenizer.repository": "gpt2",
            "tokenizer.revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            "tokenizer.asset_manifest_sha256": "07c45937a89b33f30016aef5b3982f13f25bf2c6ba940c535d1b5daa90459a71",
            "tokenizer.vocab_size": 50257,
            "tokenizer.eos_token_id": 50256,
            "tokenizer.tokenizer_class": "GPT2TokenizerFast",
            "tokenizer.document_tokenization_truncation": False,
            "source.family": "SlimPajama",
            "source.repository": "gmongaras/SlimPajama-627B_Reupload",
            "source.revision": "c34c22dbb10ae6b264a2f357a909d1a537141b36",
            "source.data_files": {"test": "data/test-00000-of-00030.parquet"},
            "source.split": "test",
            "source.file.filename": "data/test-00000-of-00030.parquet",
            "source.file.size_bytes": 43263929,
            "source.file.sha256": "d9a83d59b72f4c303f0c0e46d0e73a8446eabb56b9aa5fd992347c358ab65743",
            "source.file.revision": "c34c22dbb10ae6b264a2f357a909d1a537141b36",
            "source.source_scope": "third-party reupload test shard; not claimed byte-identical to the paper corpus or representative of its train split",
            "packing.document_handling": "eos_concatenated_continuous_stream",
            "packing.legacy_shifted_labels": True,
            "packing.return_labels_are_shifted": True,
            "packing.discard_incomplete_tail_only": True,
            "source.datasets_version": "5.0.1",
            "source.full_fingerprint": "507a47fcec5cbfdc",
            "source.selection.kind": "contiguous_rows",
            "source.selection.start": 0,
            "source.selection.stop": 64,
            "source.selection.row_count": 64,
            "source.selection.fingerprint": "f1e6c1c09434a7e4",
            "source.row_manifest_sha256": C3_ROW_MANIFEST_SHA256,
            "source.streaming": False,
            "source.canonical_hub_loader": True,
            "source.local_parquet_fingerprint_not_accepted": True,
            "accounting.selected_rows": 64,
            "accounting.nonempty_documents": 64,
            "accounting.text_tokens": 58645,
            "accounting.eos_tokens": 64,
            "accounting.concatenated_tokens": 58709,
            "accounting.packed_chunks": 14,
            "accounting.usable_tokens": 57358,
            "accounting.discarded_tail_tokens": 1351,
            "accounting.sequence_length": 4096,
            "accounting.stored_chunk_tokens": 4097,
            "accounting.prediction_tokens_per_chunk": 4096,
            "hashes.row_manifest_sha256": C3_ROW_MANIFEST_SHA256,
            "hashes.token_stream_sha256": "3232bc3996272d563b6cc4e63a8d7a7d3769c7ec33e74d3d008d97cd290d7496",
            "hashes.token_encoding": "uint32_little_endian",
            "checked_accounting_match": True,
        },
        label="c3_data",
    )
    rows = _list(_at(data, "source.row_records"), label="c3_data.source.row_records")
    if len(rows) != 64:
        _fail("C3 data row manifest must contain exactly 64 entries")
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, label=f"c3_data.row[{index}]")
        if set(row) != {"row_index", "sha256", "utf8_bytes"}:
            _fail(f"C3 data row {index} fields are incomplete or ambiguous")
        _exact_int(
            _at(row, "row_index"),
            label=f"c3_data.row[{index}].row_index",
            expected=index,
        )
        digest = _nonempty_string(
            _at(row, "sha256"), label=f"c3_data.row[{index}].sha256"
        )
        if not HEX64_RE.fullmatch(digest):
            _fail("C3 data row digest must be a lowercase SHA-256")
        _exact_int(
            _at(row, "utf8_bytes"),
            label=f"c3_data.row[{index}].utf8_bytes",
            minimum=0,
        )
    row_manifest_sha256 = _sha256_bytes(_canonical_bytes(rows))
    source_manifest_sha256 = _at(data, "source.row_manifest_sha256")
    hashes_manifest_sha256 = _at(data, "hashes.row_manifest_sha256")
    if (
        row_manifest_sha256 != C3_ROW_MANIFEST_SHA256
        or source_manifest_sha256 != row_manifest_sha256
        or hashes_manifest_sha256 != row_manifest_sha256
    ):
        _fail("C3 data row records do not match both fixed manifest hashes")
    chunk_hashes = _list(
        _at(data, "hashes.packed_chunk_sha256"),
        label="c3_data.hashes.packed_chunk_sha256",
    )
    if chunk_hashes != list(C3_PACKED_CHUNK_SHA256):
        _fail("C3 data packed-chunk digests differ from the fixed ordered manifest")
    if set(marker) != {"stage", "status", "mode", "timestamp_utc"}:
        _fail("C3 data marker fields are incomplete or ambiguous")
    _exact_values(
        marker,
        {"stage": "P0.5-C3", "status": "passed", "mode": "data"},
        label="c3_data.marker",
    )
    _parse_utc(_at(marker, "timestamp_utc"), label="c3_data.marker.timestamp_utc")
    return dict(data), {
        "status": "passed",
        "selected_rows": 64,
        "packed_chunks": 14,
        "dataset_file_sha256": _at(data, "source.file.sha256"),
        "row_manifest_sha256": _at(data, "hashes.row_manifest_sha256"),
        "token_stream_sha256": _at(data, "hashes.token_stream_sha256"),
    }


def _review_c3_lane(
    root_value: str | os.PathLike[str],
    *,
    logical: str,
    psi: int,
    mode: str,
    expected_data: Mapping[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    root = _safe_root(root_value, label=f"C3 {logical} root")
    _failure_artifact_check(root, label=f"C3 {logical} root")
    expected_marker = (
        "P0_5_C3_OPERATIONAL_COMPLETE.json"
        if mode == "operational"
        else "P0_5_C3_PEAK_EXPOSURE_COMPLETE.json"
    )
    found_markers = sorted(path.name for path in root.glob("P0_5_C3_*_COMPLETE.json"))
    if found_markers != [expected_marker]:
        _fail(f"C3 {logical} must contain exactly the correct completion marker")
    marker_path = _child_file(root, expected_marker, label=f"C3 {logical} marker")
    summary_path = _child_file(root, "summary.json", label=f"C3 {logical} summary")
    metrics_path = _child_file(root, "metrics.jsonl", label=f"C3 {logical} metrics")
    marker = _mapping(
        _load_json(marker_path, label=f"{logical}.completion_marker", hashes=hashes),
        label=f"C3 {logical} marker",
    )
    summary = _mapping(
        _load_json(summary_path, label=f"{logical}.summary", hashes=hashes),
        label=f"C3 {logical} summary",
    )
    events = _load_jsonl(metrics_path, label=f"{logical}.metrics", hashes=hashes)
    if set(summary) != {
        "stage",
        "status",
        "mode",
        "qualification",
        "timestamp_utc",
        "psi",
        "parameter_count",
        "model",
        "optimizer",
        "scheduler",
        "training",
        "data",
        "environment",
        "memory",
        "limitations",
    }:
        _fail(f"{logical}.summary fields are incomplete or ambiguous")
    timestamp_utc = _nonempty_string(
        _at(summary, "timestamp_utc"), label=f"{logical}.timestamp_utc"
    )
    _parse_utc(timestamp_utc, label=f"{logical}.timestamp_utc")

    _equals(_at(summary, "stage"), "P0.5-C3", label=f"{logical}.stage")
    _equals(_at(summary, "status"), "diagnostic_passed", label=f"{logical}.status")
    _equals(_at(summary, "mode"), mode, label=f"{logical}.mode")
    _exact_int(_at(summary, "psi"), label=f"{logical}.psi", expected=psi)
    _exact_int(
        _at(summary, "parameter_count"),
        label=f"{logical}.parameter_count",
        expected={8: 4_134_146, 16: 27_546_626}[psi],
    )
    expected_qualification = (
        "diagnostic_only_reduced_warmup_and_learning_rate"
        if mode == "operational"
        else "diagnostic_only_bounded_exact_peak_exposure"
    )
    _equals(_at(summary, "qualification"), expected_qualification, label=f"{logical}.qualification")
    model = _mapping(_at(summary, "model"), label=f"{logical}.model")
    if set(model) != {
        "vocab_size",
        "sequence_length",
        "mipe_position_mode",
        "mipe_compute_dtype",
        "softmask_compute_dtype",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "tie_word_embeddings",
    }:
        _fail(f"{logical}.model fields are incomplete or ambiguous")
    _exact_int(_at(model, "vocab_size"), label=f"{logical}.vocab_size", expected=50257)
    _exact_int(_at(model, "sequence_length"), label=f"{logical}.sequence_length", expected=4096)
    _equals(_at(model, "mipe_position_mode"), "paper_absolute", label=f"{logical}.mipe_position_mode")
    _equals(_at(model, "mipe_compute_dtype"), "fp32", label=f"{logical}.mipe_compute_dtype")
    _equals(_at(model, "softmask_compute_dtype"), "fp32", label=f"{logical}.softmask_compute_dtype")
    _exact_bool(_at(model, "gradient_checkpointing"), label=f"{logical}.gradient_checkpointing", expected=True)
    if _at(model, "gradient_checkpointing_kwargs") != {"use_reentrant": False}:
        _fail(f"C3 {logical} must use non-reentrant checkpointing")
    _exact_bool(_at(model, "tie_word_embeddings"), label=f"{logical}.tie_word_embeddings", expected=True)

    optimizer = _mapping(_at(summary, "optimizer"), label=f"{logical}.optimizer")
    if set(optimizer) != {
        "name",
        "betas",
        "weight_decay",
        "eps",
        "eps_source",
        "fused",
        "gradient_clipping",
    }:
        _fail(f"{logical}.optimizer fields are incomplete or ambiguous")
    _equals(_at(optimizer, "name"), "AdamW", label=f"{logical}.optimizer.name")
    _equals(
        _at(optimizer, "eps_source"),
        "repository_operationalization_paper_unspecified",
        label=f"{logical}.optimizer.eps_source",
    )
    _exact_bool(_at(optimizer, "gradient_clipping"), label=f"{logical}.optimizer.gradient_clipping", expected=False)
    _close(_finite_number(_at(optimizer, "weight_decay"), label=f"{logical}.weight_decay"), 0.0, label=f"{logical}.weight_decay")
    scheduler = _mapping(_at(summary, "scheduler"), label=f"{logical}.scheduler")
    if set(scheduler) != {
        "name",
        "paper_warmup_steps",
        "paper_peak_learning_rate",
        "executed_warmup_steps",
        "executed_peak_learning_rate",
        "observed_learning_rates",
        "diagnostic_reduced_from_paper",
    }:
        _fail(f"{logical}.scheduler fields are incomplete or ambiguous")
    _equals(
        _at(scheduler, "name"),
        "linear_warmup_then_constant",
        label=f"{logical}.scheduler.name",
    )
    _exact_int(
        _at(scheduler, "paper_warmup_steps"),
        label=f"{logical}.paper_warmup_steps",
        expected=4096,
    )
    _exact_int(
        _at(scheduler, "executed_warmup_steps"),
        label=f"{logical}.executed_warmup_steps",
        expected=2 if mode == "operational" else 1,
    )
    _close(
        _finite_number(
            _at(scheduler, "executed_peak_learning_rate"),
            label=f"{logical}.executed_peak_learning_rate",
        ),
        0.0006 if mode == "operational" else 0.0625,
        label=f"{logical}.executed_peak_learning_rate",
    )
    diagnostic_reduced_from_paper = (
        _at(scheduler, "executed_warmup_steps")
        != _at(scheduler, "paper_warmup_steps")
        or _at(scheduler, "executed_peak_learning_rate")
        != _at(scheduler, "paper_peak_learning_rate")
    )
    _exact_bool(
        _at(scheduler, "diagnostic_reduced_from_paper"),
        label=f"{logical}.diagnostic_reduced_from_paper",
        expected=diagnostic_reduced_from_paper,
    )
    betas = _list(_at(optimizer, "betas"), label=f"{logical}.optimizer.betas")
    if len(betas) != 2:
        _fail(f"{logical}.optimizer.betas must contain exactly two values")
    _close(_finite_number(betas[0], label=f"{logical}.optimizer.betas[0]"), 0.9, label=f"{logical}.optimizer.betas[0]")
    _close(_finite_number(betas[1], label=f"{logical}.optimizer.betas[1]"), 0.95, label=f"{logical}.optimizer.betas[1]")
    _close(_finite_number(_at(optimizer, "eps"), label=f"{logical}.optimizer.eps"), 1e-8, label=f"{logical}.optimizer.eps")
    _exact_bool(_at(optimizer, "fused"), label=f"{logical}.optimizer.fused", expected=False)
    _close(_finite_number(_at(scheduler, "paper_peak_learning_rate"), label=f"{logical}.paper_peak_lr"), 0.0625, label=f"{logical}.paper_peak_lr")
    training = _mapping(_at(summary, "training"), label=f"{logical}.training")
    if set(training) != {
        "optimizer_steps",
        "world_size",
        "microbatch_size",
        "sequences_per_optimizer_step",
        "gradient_accumulation_steps",
        "effective_tokens_per_optimizer_step",
        "paper_global_batch_tokens",
        "local_to_paper_batch_ratio",
        "losses_finite",
        "gradients_finite",
        "parameters_finite",
        "optimizer_updates_nonzero",
        "post_update_loss",
        "loss_decrease_required",
    }:
        _fail(f"{logical}.training fields are incomplete or ambiguous")
    expected_steps = 3 if mode == "operational" else 1
    expected_accum = 2 if mode == "operational" else 1
    expected_lrs = [0.0003, 0.0006, 0.0006] if mode == "operational" else [0.0625]
    _exact_int(_at(training, "optimizer_steps"), label=f"{logical}.optimizer_steps", expected=expected_steps)
    _exact_int(_at(training, "microbatch_size"), label=f"{logical}.microbatch_size", expected=1)
    _exact_int(_at(training, "gradient_accumulation_steps"), label=f"{logical}.grad_accum", expected=expected_accum)
    _exact_int(_at(training, "world_size"), label=f"{logical}.world_size", expected=1)
    _exact_int(
        _at(training, "sequences_per_optimizer_step"),
        label=f"{logical}.sequences_per_optimizer_step",
        expected=expected_accum,
    )
    expected_effective_tokens = 4096 * expected_accum
    _exact_int(
        _at(training, "effective_tokens_per_optimizer_step"),
        label=f"{logical}.effective_tokens_per_optimizer_step",
        expected=expected_effective_tokens,
    )
    _exact_int(
        _at(training, "paper_global_batch_tokens"),
        label=f"{logical}.paper_global_batch_tokens",
        expected=4_194_304,
    )
    _close(
        _finite_number(
            _at(training, "local_to_paper_batch_ratio"),
            label=f"{logical}.local_to_paper_batch_ratio",
        ),
        expected_effective_tokens / 4_194_304,
        label=f"{logical}.local_to_paper_batch_ratio",
    )
    _exact_bool(
        _at(training, "loss_decrease_required"),
        label=f"{logical}.loss_decrease_required",
        expected=False,
    )
    for name in ("losses_finite", "gradients_finite", "parameters_finite", "optimizer_updates_nonzero"):
        _exact_bool(_at(training, name), label=f"{logical}.{name}", expected=True)
    _finite_number(_at(training, "post_update_loss"), label=f"{logical}.post_update_loss")
    observed = _list(_at(scheduler, "observed_learning_rates"), label=f"{logical}.observed_learning_rates")
    if len(observed) != len(expected_lrs):
        _fail(f"C3 {logical} has the wrong LR event count")
    for index, expected_lr in enumerate(expected_lrs):
        _close(_finite_number(observed[index], label=f"{logical}.observed_lr[{index}]"), expected_lr, label=f"{logical}.observed_lr[{index}]")

    environment = _mapping(_at(summary, "environment"), label=f"{logical}.environment")
    if set(environment) != {
        "python",
        "platform",
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
        "device",
        "cuda_runtime",
        "gpu_name",
        "gpu_total_memory_bytes",
        "bf16_supported",
    }:
        _fail(f"{logical}.environment fields are incomplete or ambiguous")
    _nonempty_string(_at(environment, "python"), label=f"{logical}.environment.python")
    _nonempty_string(_at(environment, "platform"), label=f"{logical}.environment.platform")
    _equals(_at(environment, "device"), "cuda:0", label=f"{logical}.environment.device")
    _exact_bool(_at(environment, "bf16_supported"), label=f"{logical}.bf16_supported", expected=True)
    gpu_total_memory_bytes = _exact_int(
        _at(environment, "gpu_total_memory_bytes"),
        label=f"{logical}.environment.gpu_total_memory_bytes",
        minimum=1,
    )
    if len(events) != expected_steps or any(event.get("event") != "optimizer_step" for event in events):
        _fail(f"C3 {logical} has missing, duplicate, extra, or out-of-order metric events")
    _equals(_at(environment, "transformers"), "4.57.6", label=f"{logical}.environment.transformers")
    _equals(_at(environment, "datasets"), "5.0.1", label=f"{logical}.environment.datasets")
    _equals(_at(environment, "huggingface_hub"), "0.34.3", label=f"{logical}.environment.huggingface_hub")
    _equals(
        _at(environment, "torch"),
        "2.7.1+cu128",
        label=f"{logical}.environment.torch",
    )
    _equals(
        _at(environment, "cuda_runtime"),
        "12.8",
        label=f"{logical}.environment.cuda_runtime",
    )
    _nonempty_string(
        _at(environment, "gpu_name"), label=f"{logical}.environment.gpu_name"
    )
    memory = _mapping(_at(summary, "memory"), label=f"{logical}.memory")
    if set(memory) != {
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }:
        _fail(f"{logical}.memory fields are incomplete or ambiguous")
    _review_cuda_memory(
        memory, label=f"{logical}.memory", total_bytes=gpu_total_memory_bytes
    )
    expected_limitations = [
        "This is a bounded dense-reference workstation diagnostic.",
        "It does not reproduce the paper global batch, duration, corpus, quality, or efficiency.",
        "The peak-exposure mode requires finite updates, not loss decrease or model quality.",
    ]
    if _at(summary, "limitations") != expected_limitations:
        _fail(f"{logical}.limitations differ from the fixed producer contract")
    if _at(summary, "data") != expected_data:
        _fail(f"C3 {logical} data identity differs from the reviewed pinned data contract")
    deltas: list[float] = []
    expected_event_fields = {
        "event",
        "stage",
        "mode",
        "psi",
        "optimizer_step_zero_based",
        "learning_rate",
        "mean_micro_loss",
        "micro_losses",
        "gradient_l2_norm",
        "tracked_gradient_abs",
        "tracked_parameter",
        "tracked_parameter_delta",
        "gradient_clipping_applied",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    previous_peak_allocated = -1
    previous_peak_reserved = -1
    for index, (event, expected_lr) in enumerate(zip(events, expected_lrs, strict=True)):
        if set(event) != expected_event_fields:
            _fail(f"{logical}.optimizer_step fields are incomplete or ambiguous")
        event_memory = _review_cuda_memory(
            event,
            label=f"{logical}.event[{index}]",
            total_bytes=gpu_total_memory_bytes,
        )
        if (
            event_memory["peak_allocated_bytes"] < previous_peak_allocated
            or event_memory["peak_reserved_bytes"] < previous_peak_reserved
        ):
            _fail(f"{logical} peak CUDA memory is not monotonic across events")
        previous_peak_allocated = event_memory["peak_allocated_bytes"]
        previous_peak_reserved = event_memory["peak_reserved_bytes"]
        _equals(_at(event, "stage"), "P0.5-C3", label=f"{logical}.event[{index}].stage")
        _equals(_at(event, "mode"), mode, label=f"{logical}.event[{index}].mode")
        _exact_int(_at(event, "psi"), label=f"{logical}.event[{index}].psi", expected=psi)
        _exact_int(_at(event, "optimizer_step_zero_based"), label=f"{logical}.event[{index}].step", expected=index)
        _close(_finite_number(_at(event, "learning_rate"), label=f"{logical}.event[{index}].lr"), expected_lr, label=f"{logical}.event[{index}].lr")
        micro = _list(_at(event, "micro_losses"), label=f"{logical}.event[{index}].micro_losses")
        if len(micro) != expected_accum:
            _fail(f"C3 {logical} event {index} has the wrong micro-loss count")
        micro_values = [_finite_number(item, label=f"{logical}.event[{index}].micro_loss") for item in micro]
        mean = _finite_number(_at(event, "mean_micro_loss"), label=f"{logical}.event[{index}].mean_micro_loss")
        _close(mean, sum(micro_values) / len(micro_values), label=f"{logical}.event[{index}].mean_micro_loss")
        _finite_number(_at(event, "gradient_l2_norm"), label=f"{logical}.event[{index}].gradient_l2_norm", positive=True)
        _finite_number(_at(event, "tracked_gradient_abs"), label=f"{logical}.event[{index}].tracked_gradient_abs", positive=True)
        _nonempty_string(
            _at(event, "tracked_parameter"), label=f"{logical}.event[{index}].tracked_parameter"
        )
        delta = _finite_number(_at(event, "tracked_parameter_delta"), label=f"{logical}.event[{index}].tracked_parameter_delta")
        if delta == 0.0:
            _fail(f"C3 {logical} event {index} did not change the tracked parameter")
        deltas.append(delta)
        _exact_bool(_at(event, "gradient_clipping_applied"), label=f"{logical}.event[{index}].gradient_clipping", expected=False)
    if (
        _at(memory, "peak_allocated_bytes") < previous_peak_allocated
        or _at(memory, "peak_reserved_bytes") < previous_peak_reserved
    ):
        _fail(f"{logical} summary peak memory is below an optimizer-step peak")
    if mode == "peak-exposure":
        _close(deltas[0], -0.0625, label=f"{logical}.peak_parameter_delta")

    expected_marker_value = {
        "stage": "P0.5-C3",
        "status": "diagnostic_passed",
        "mode": mode,
        "psi": psi,
        "timestamp_utc": _at(summary, "timestamp_utc"),
    }
    if marker != expected_marker_value:
        _fail(f"C3 {logical} completion marker does not match its summary")
    return {
        "logical_name": logical,
        "psi": psi,
        "mode": mode,
        "status": "diagnostic_passed",
        "event_count": len(events),
        "observed_learning_rates": expected_lrs,
        "gradient_clipping_applied": False,
        "parameter_updates_nonzero": True,
        "timestamp_utc": timestamp_utc,
    }


def _safe_identifier(value: Any, *, label: str) -> str:
    result = _nonempty_string(value, label=label)
    if result.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", result):
        _fail(f"{label} must not expose an absolute path")
    if ".." in Path(result).parts:
        _fail(f"{label} must not contain parent traversal")
    return result


def _review_tokenizer_reports(
    reports: Mapping[str, str | os.PathLike[str]],
    *,
    hashes: dict[str, str],
    required_names: Sequence[str] = TOKENIZER_NAMES,
) -> dict[str, Any]:
    if not required_names or any(name not in TOKENIZER_NAMES for name in required_names):
        _fail("tokenizer report required-name selection is invalid")
    if set(reports) != set(required_names):
        missing = sorted(set(required_names) - set(reports))
        extra = sorted(set(reports) - set(required_names))
        _fail(f"tokenizer reports must be exact; missing={missing}, extra={extra}")
    reviewed: list[dict[str, Any]] = []
    for logical in required_names:
        path = _safe_file(reports[logical], label=f"tokenizer report {logical}")
        report = _mapping(
            _load_json(path, label=f"tokenizer_reload.{logical}", hashes=hashes),
            label=f"tokenizer report {logical}",
        )
        expected_top_fields = {
            "checked_fields",
            "checkpoint",
            "counts",
            "hashes",
            "logical_name",
            "operationalization",
            "schema_version",
            "source",
            "source_normalization",
            "status",
            "versions",
        }
        if set(report) != expected_top_fields:
            _fail(f"{logical} tokenizer report fields are incomplete or ambiguous")
        _equals(_at(report, "schema_version"), TOKENIZER_SCHEMA_VERSION, label=f"{logical}.schema_version")
        _equals(_at(report, "status"), "passed", label=f"{logical}.status")
        _equals(_at(report, "logical_name"), logical, label=f"{logical}.logical_name")
        source = _mapping(_at(report, "source"), label=f"{logical}.source")
        checkpoint = _mapping(_at(report, "checkpoint"), label=f"{logical}.checkpoint")
        if set(source) != {"class", "identifier", "is_fast"}:
            _fail(f"{logical}.source fields are incomplete or ambiguous")
        if set(checkpoint) != {
            "class",
            "identifier",
            "is_fast",
            "reload_method",
            "reloaded_from_checkpoint",
        }:
            _fail(f"{logical}.checkpoint fields are incomplete or ambiguous")
        source_identifier = _safe_identifier(
            _at(source, "identifier"), label=f"{logical}.source.identifier"
        )
        checkpoint_identifier = _safe_identifier(
            _at(checkpoint, "identifier"), label=f"{logical}.checkpoint.identifier"
        )
        expected_source_identifier = (
            "tinystories-spm768" if logical.startswith("p0_3") else "gpt2"
        )
        expected_checkpoint_identifier = TOKENIZER_CHECKPOINT_IDENTIFIERS[logical]
        if source_identifier != expected_source_identifier:
            _fail(f"{logical}.source.identifier differs from the fixed lane identity")
        if checkpoint_identifier != expected_checkpoint_identifier:
            _fail(f"{logical}.checkpoint.identifier differs from the fixed lane identity")
        source_class = _nonempty_string(_at(source, "class"), label=f"{logical}.source.class")
        checkpoint_class = _nonempty_string(_at(checkpoint, "class"), label=f"{logical}.checkpoint.class")
        if source_class != checkpoint_class:
            _fail(f"{logical} tokenizer class changed after reload")
        source_fast = _exact_bool(_at(source, "is_fast"), label=f"{logical}.source.is_fast")
        checkpoint_fast = _exact_bool(_at(checkpoint, "is_fast"), label=f"{logical}.checkpoint.is_fast")
        if source_fast != checkpoint_fast:
            _fail(f"{logical} tokenizer fast/backend status changed after reload")
        _exact_bool(
            _at(checkpoint, "reloaded_from_checkpoint"),
            label=f"{logical}.checkpoint.reloaded_from_checkpoint",
            expected=True,
        )
        _nonempty_string(
            _at(checkpoint, "reload_method"),
            label=f"{logical}.checkpoint.reload_method",
        )
        normalization = _mapping(
            _at(report, "source_normalization"),
            label=f"{logical}.source_normalization",
        )
        if set(normalization) != {
            "pad_token_from_eos",
            "padding_side",
            "model_max_length",
        }:
            _fail(f"{logical}.source_normalization fields are incomplete or ambiguous")
        expected_normalization = (
            {
                "pad_token_from_eos": False,
                "padding_side": None,
                "model_max_length": None,
            }
            if logical.startswith("p0_3")
            else {
                "pad_token_from_eos": True,
                "padding_side": "right",
                "model_max_length": 4096,
            }
        )
        if normalization != expected_normalization:
            _fail(f"{logical}.source_normalization does not match its run contract")
        operational = _mapping(
            _at(report, "operationalization"),
            label=f"{logical}.operationalization",
        )
        if set(operational) != {
            "model_input_names",
            "padding_side",
            "truncation_side",
            "model_max_length",
        }:
            _fail(f"{logical}.operationalization fields are incomplete or ambiguous")
        model_inputs = _list(
            _at(operational, "model_input_names"),
            label=f"{logical}.operationalization.model_input_names",
        )
        if not model_inputs or any(
            not isinstance(item, str) or not item for item in model_inputs
        ):
            _fail(f"{logical}.operationalization.model_input_names is invalid")
        padding_side = _nonempty_string(
            _at(operational, "padding_side"),
            label=f"{logical}.operationalization.padding_side",
        )
        truncation_side = _nonempty_string(
            _at(operational, "truncation_side"),
            label=f"{logical}.operationalization.truncation_side",
        )
        model_max_length = _exact_int(
            _at(operational, "model_max_length"),
            label=f"{logical}.operationalization.model_max_length",
            minimum=1,
        )
        if logical.startswith("p0_4") and (
            model_inputs != ["input_ids", "attention_mask"]
            or padding_side != "right"
            or truncation_side != "right"
            or model_max_length != 4096
        ):
            _fail(f"{logical}.operationalization does not match strict P0-4")
        expected_vocab = 768 if logical.startswith("p0_3") else 50257
        counts = _mapping(_at(report, "counts"), label=f"{logical}.counts")
        expected_count_fields = {
            "vocabulary",
            "vocab_size",
            "tokenizer_length",
            "added_vocabulary",
            "all_special_tokens",
            "probes",
            "special_token_boundary_probes",
        }
        if set(counts) != expected_count_fields:
            _fail(f"{logical} tokenizer counts are incomplete or ambiguous")
        _exact_int(_at(counts, "vocabulary"), label=f"{logical}.counts.vocabulary", expected=expected_vocab)
        _exact_int(_at(counts, "vocab_size"), label=f"{logical}.counts.vocab_size", expected=expected_vocab)
        _exact_int(_at(counts, "tokenizer_length"), label=f"{logical}.counts.tokenizer_length", expected=expected_vocab)
        _exact_int(_at(counts, "added_vocabulary"), label=f"{logical}.counts.added_vocabulary", minimum=0)
        special_count = _exact_int(
            _at(counts, "all_special_tokens"),
            label=f"{logical}.counts.all_special_tokens",
            minimum=1,
        )
        _exact_int(_at(counts, "probes"), label=f"{logical}.counts.probes", expected=5)
        _exact_int(
            _at(counts, "special_token_boundary_probes"),
            label=f"{logical}.counts.special_token_boundary_probes",
            expected=special_count * 7,
        )
        hash_fields = _mapping(_at(report, "hashes"), label=f"{logical}.hashes")
        expected_hashes = {
            "vocabulary_manifest_sha256",
            "special_tokens_manifest_sha256",
            "probe_manifest_sha256",
        }
        if set(hash_fields) != expected_hashes:
            _fail(f"{logical} tokenizer hash proof is incomplete or has ambiguous fields")
        for name in sorted(expected_hashes):
            value = _nonempty_string(hash_fields[name], label=f"{logical}.hashes.{name}")
            if not HEX64_RE.fullmatch(value):
                _fail(f"{logical}.hashes.{name} must be a lowercase SHA-256")
        checked_fields = _list(_at(report, "checked_fields"), label=f"{logical}.checked_fields")
        if checked_fields != list(TOKENIZER_CHECKED_FIELDS):
            _fail(f"{logical}.checked_fields does not match the complete verifier contract")
        versions = _mapping(_at(report, "versions"), label=f"{logical}.versions")
        if set(versions) != {"verifier", "python", "transformers", "tokenizers"}:
            _fail(f"{logical}.versions fields are incomplete or ambiguous")
        expected_versions = {
            "verifier": "1.0.0",
            "python": "3.12.11",
            "transformers": "4.57.6",
            "tokenizers": "0.22.0",
        }
        if versions != expected_versions:
            _fail(f"{logical}.versions differs from the recorded tf4576 lane")
        reviewed_report: dict[str, Any] = {
            "logical_name": logical,
            "vocabulary_size": expected_vocab,
            "vocabulary_manifest_sha256": hash_fields[
                "vocabulary_manifest_sha256"
            ],
            "special_tokens_manifest_sha256": hash_fields[
                "special_tokens_manifest_sha256"
            ],
            "probe_manifest_sha256": hash_fields["probe_manifest_sha256"],
            "reloaded_from_checkpoint": True,
        }
        reviewed_report["source_projection"] = {
            "class": source_class,
            "counts": {
                field: counts[field] for field in sorted(expected_count_fields)
            },
            "hashes": {
                field: hash_fields[field] for field in sorted(expected_hashes)
            },
            "is_fast": source_fast,
            "operationalization": {
                "model_input_names": list(model_inputs),
                "model_max_length": model_max_length,
                "padding_side": padding_side,
                "truncation_side": truncation_side,
            },
        }
        reviewed.append(reviewed_report)
    return {"status": "passed", "report_count": len(reviewed), "reports": reviewed}


def _parse_utc(value: Any, *, label: str) -> dt.datetime:
    text = _nonempty_string(value, label=label)
    if not UTC_RE.search(text):
        _fail(f"{label} must explicitly use UTC")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReviewError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.utcoffset() != dt.timedelta(0):
        _fail(f"{label} must use UTC")
    return parsed


def _runner_canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _run_relative(run_root: Path, value: Path, *, label: str) -> str:
    try:
        relative = value.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise ReviewError(f"{label} must be inside the recorder run root") from exc
    if not relative or relative.startswith("../") or "\\" in relative:
        _fail(f"{label} is not a canonical run-root-relative path")
    return relative


def _argv_option(argv: Sequence[str], option: str, *, label: str) -> str:
    values: list[str] = []
    for index, argument in enumerate(argv):
        if argument == option:
            if index + 1 >= len(argv):
                _fail(f"{label} has an option without a value: {option}")
            values.append(argv[index + 1])
        elif argument.startswith(option + "="):
            values.append(argument[len(option) + 1 :])
    if len(values) != 1 or not values[0]:
        _fail(f"{label} must contain exactly one {option} value")
    return values[0]


def _command_environment_suffix(name: str, run_root: Path) -> tuple[str, ...]:
    if name == "syntax-level1":
        return (
            "PYTHONPATH=.:oracle:third_party/multiscreen-pytorch",
            f"PYTHONPYCACHEPREFIX={run_root / 'pycache/syntax-level1'}",
        )
    if name in PYTHONPATH_DOT_COMMAND_NAMES:
        return ("PYTHONPATH=.",)
    if name in PYTHONPATH_ORACLE_COMMAND_NAMES:
        return ("PYTHONPATH=.:oracle",)
    if name in PYTHONPATH_FULL_COMMAND_NAMES:
        return ("PYTHONPATH=.:oracle:third_party/multiscreen-pytorch",)
    return ()


def _expected_hermetic_environment(
    *,
    name: str,
    run_root: Path,
) -> tuple[str, ...]:
    if name in CPU_COMMAND_NAMES:
        device_environment = ("CUDA_VISIBLE_DEVICES=",)
    elif name in CUDA_COMMAND_NAMES:
        device_environment = (
            "CUDA_DEVICE_ORDER=PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES=0",
        )
    else:  # Defensive even though the ledger name set is checked separately.
        _fail(f"command {name} has no fixed CPU/CUDA environment classification")
    return (
        f"HOME={run_root}",
        *HERMETIC_FIXED_ENVIRONMENT,
        *device_environment,
        *_command_environment_suffix(name, run_root),
    )


def _review_hermetic_command_argv(
    argv: Sequence[str],
    *,
    name: str,
    run_root: Path,
) -> int:
    if len(argv) < 3 or tuple(argv[:2]) != ("/usr/bin/env", "-i"):
        _fail(f"command {name} must start with the fixed hermetic /usr/bin/env -i prefix")
    assignment_end = 2
    while assignment_end < len(argv) and ENV_ASSIGNMENT_RE.fullmatch(
        argv[assignment_end]
    ):
        assignment_end += 1
    observed = tuple(argv[2:assignment_end])
    expected = _expected_hermetic_environment(name=name, run_root=run_root)
    if observed != expected:
        _fail(
            f"command {name} hermetic environment assignments are missing, extra, "
            "duplicated, reordered, or use the wrong CPU/CUDA selection"
        )
    if assignment_end >= len(argv):
        _fail(f"command {name} hermetic environment has no child executable")
    return assignment_end


def _review_offline_cache_command_argv(
    argv: Sequence[str],
    *,
    executable_index: int,
) -> tuple[str, Path]:
    tail = tuple(argv[executable_index:])
    if len(tail) != 8 or tail[1:4] != (
        "-P",
        "-B",
        "scripts/check_level1_offline_cache.py",
    ):
        _fail(
            "offline-cache-preflight must use the exact TF4 Python -P -B checker tail"
        )
    executable, _, _, _, repo_option, repo_value, cache_option, cache_value = tail
    if repo_option != "--repo-root" or cache_option != "--cache-dir":
        _fail("offline-cache-preflight checker options are missing or reordered")
    if not Path(executable).is_absolute():
        _fail("offline-cache-preflight Python executable must be absolute")
    repository = _absolute_path(repo_value, label="offline-cache-preflight repo root")
    expected_repository = Path(__file__).resolve().parents[1]
    if repository != expected_repository:
        _fail("offline-cache-preflight repo root differs from the reviewer checkout")
    cache = _absolute_path(cache_value, label="offline-cache-preflight explicit cache")
    if os.fspath(cache) != cache_value:
        _fail("offline-cache-preflight cache path must be canonical")
    return executable, cache


def _tracked_python_files(repository: Path) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repository), "ls-files", "*.py"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ReviewError("could not enumerate the tracked Python syntax set") from exc
    if result.returncode != 0:
        _fail("could not enumerate the tracked Python syntax set")
    try:
        text = result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReviewError("tracked Python syntax set is not UTF-8") from exc
    if not text or not text.endswith("\n") or "\r" in text:
        _fail("tracked Python syntax set is empty or non-canonical")
    files = tuple(text[:-1].split("\n"))
    if len(files) != len(set(files)) or tuple(sorted(files)) != files:
        _fail("tracked Python syntax set is duplicated or unordered")
    for value in files:
        candidate = Path(value)
        if (
            not value.endswith(".py")
            or not value
            or candidate.is_absolute()
            or ".." in candidate.parts
            or "\\" in value
            or any(character.isspace() for character in value)
        ):
            _fail("tracked Python syntax set contains an unsafe argument")
    return files


def _unit_test_tail(
    python: str,
    pattern: str,
    *,
    suppress_site: bool,
) -> tuple[str, ...]:
    return (
        python,
        *(("-S",) if suppress_site else ()),
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        pattern,
        "-v",
    )


def _expected_command_tails(
    *,
    required_names: Sequence[str],
    repository: Path,
    run_root: Path,
    cache: Path,
    tf4576_python: str,
    tf5141_python: str | None,
) -> dict[str, tuple[str, ...]]:
    tf5141_names = {
        "environment-tf5141",
        "tokenizer-reload-tests-tf5141",
        "gradient-checkpointing-tf5141",
        "c3-contracts-tf5141",
    }
    if set(required_names) & tf5141_names and tf5141_python is None:
        _fail("the full matrix is missing its exact TF5141 Python binding")
    tf5141 = tf5141_python or ""
    repo = os.fspath(repository)
    p03 = os.fspath(run_root / "artifacts/p0-3")
    p04_8 = os.fspath(run_root / "artifacts/p0-4/psi8")
    p04_16 = os.fspath(run_root / "artifacts/p0-4/psi16")
    c3_data = os.fspath(run_root / "artifacts/c3/data")
    cache_value = os.fspath(cache)

    def repository_check(check: str) -> tuple[str, ...]:
        return (
            tf4576_python,
            "-P",
            "-S",
            "-B",
            "scripts/check_level1_repository.py",
            "--repo-root",
            repo,
            "--check",
            check,
        )

    tails: dict[str, tuple[str, ...]] = {
        "environment-tf4576": (
            tf4576_python,
            "scripts/report_level1_environment.py",
            "--lane",
            "tf4576",
        ),
        "environment-tf5141": (
            tf5141,
            "scripts/report_level1_environment.py",
            "--lane",
            "tf5141",
        ),
        "environment-cuda0": (
            tf4576_python,
            "scripts/report_level1_environment.py",
            "--lane",
            "cuda0",
        ),
        "offline-cache-preflight": (
            tf4576_python,
            "-P",
            "-B",
            "scripts/check_level1_offline_cache.py",
            "--repo-root",
            repo,
            "--cache-dir",
            cache_value,
        ),
        "repository-hygiene": repository_check("hygiene"),
        "syntax-level1": (
            tf4576_python,
            "-m",
            "py_compile",
            *_tracked_python_files(repository),
        ),
        "level1-evidence-support-tests": _unit_test_tail(
            tf4576_python, "test_level1_*.py", suppress_site=True
        ),
        "tokenizer-reload-tests-tf4576": _unit_test_tail(
            tf4576_python, "test_tokenizer_reload_check.py", suppress_site=False
        ),
        "tokenizer-reload-tests-tf5141": _unit_test_tail(
            tf5141, "test_tokenizer_reload_check.py", suppress_site=False
        ),
        "validation-evidence-tests": _unit_test_tail(
            tf4576_python, "test_validation_evidence*.py", suppress_site=True
        ),
        "json-validation": repository_check("json"),
        "workflow-yaml": repository_check("workflow-yaml"),
        "markdown-links": repository_check("markdown-links"),
        "c1-architecture": _unit_test_tail(
            tf4576_python, "test_paper_architecture_contract.py", suppress_site=False
        ),
        "c1-initialization": _unit_test_tail(
            tf4576_python, "test_paper_initialization_contract.py", suppress_site=False
        ),
        "c1-packed-data": _unit_test_tail(
            tf4576_python, "test_packed_text_contract.py", suppress_site=False
        ),
        "c1-manifest": (
            tf4576_python,
            "scripts/generate_paper_scale_manifest.py",
            "--check",
            "docs/validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json",
        ),
        "c2-position-cache": _unit_test_tail(
            tf4576_python, "test_mipe_position_cache_contract.py", suppress_site=False
        ),
        "gradient-checkpointing-tf4576": _unit_test_tail(
            tf4576_python, "test_gradient_checkpointing_contract.py", suppress_site=False
        ),
        "gradient-checkpointing-tf5141": _unit_test_tail(
            tf5141, "test_gradient_checkpointing_contract.py", suppress_site=False
        ),
        "formula-units": (tf4576_python, "oracle/test_formula_units.py"),
        "oracle-selfcheck": (
            tf4576_python,
            "oracle/test_paper_math_oracle_selfcheck.py",
        ),
        "oracle-smoke": (
            tf4576_python,
            "oracle/test_paper_math_oracle_smoke.py",
        ),
        "p0-1-cpu-fp32": (
            tf4576_python,
            "oracle/test_against_hf_port.py",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--seed",
            "1234",
            "--rtol",
            "1e-5",
            "--atol",
            "1e-5",
        ),
        "p0-1-cuda-bf16": (
            tf4576_python,
            "oracle/test_against_hf_port.py",
            "--device",
            "cuda:0",
            "--dtype",
            "bf16",
            "--seed",
            "1234",
            "--rtol",
            "0.03",
            "--atol",
            "0.03",
        ),
        "p0-2-cpu-fp32": (
            tf4576_python,
            "p0_2_three_way_minimal/test_three_way_minimal.py",
            "--reference-root",
            "third_party/multiscreen-pytorch",
            "--hf-root",
            ".",
            "--oracle-root",
            "oracle",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--seed",
            "4321",
            "--rtol",
            "1e-5",
            "--atol",
            "1e-5",
        ),
        "p0-2-cuda-bf16": (
            tf4576_python,
            "p0_2_three_way_minimal/test_three_way_minimal.py",
            "--reference-root",
            "third_party/multiscreen-pytorch",
            "--hf-root",
            ".",
            "--oracle-root",
            "oracle",
            "--device",
            "cuda:0",
            "--dtype",
            "bf16",
            "--seed",
            "4321",
            "--rtol",
            "0.03",
            "--atol",
            "0.03",
        ),
        "c3-contracts-tf4576": _unit_test_tail(
            tf4576_python, "test_paper_training_contract.py", suppress_site=False
        ),
        "c3-contracts-tf5141": _unit_test_tail(
            tf5141, "test_paper_training_contract.py", suppress_site=False
        ),
        "c3-contract-cli": (
            tf4576_python,
            "scripts/p0_5_c3_paper_training_contract.py",
            "--manifest",
            "configs/p0_5_c3_paper_training_contract.json",
            "--mode",
            "contract",
        ),
        "c3-data": (
            tf4576_python,
            "scripts/p0_5_c3_paper_training_contract.py",
            "--manifest",
            "configs/p0_5_c3_paper_training_contract.json",
            "--mode",
            "data",
            "--cache-dir",
            cache_value,
            "--output-dir",
            c3_data,
        ),
        "p0-3-checkpointed": (
            tf4576_python,
            "scripts/p0_3_tinystories_stability.py",
            "--repo-root",
            repo,
            "--tokenizer-path",
            "tokenizers/tinystories_spm768",
            "--cache-dir",
            cache_value,
            "--dataset-name",
            "roneneldan/TinyStories",
            "--revision",
            P0_3_DATASET_REVISION,
            "--train-split",
            "train[:20000]",
            "--text-column",
            "text",
            "--max-texts",
            "20000",
            "--max-train-tokens",
            "262144",
            "--seq-len",
            "128",
            "--psi",
            "8",
            "16",
            "--steps-per-psi",
            "8:40,16:25",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cuda:0",
            "--amp-dtype",
            "bf16",
            "--model-compute-dtype",
            "fp32",
            "--key-dim",
            "16",
            "--value-dim",
            "64",
            "--gradient-checkpointing",
            "true",
            "--mipe-threshold",
            "256",
            "--initializer-range",
            "0.1",
            "--learning-rate",
            "0.0006",
            "--weight-decay",
            "0",
            "--max-grad-norm",
            "1",
            "--seed",
            "42",
            "--log-every",
            "1",
            "--train-probe-every",
            "4",
            "--min-loss-drop",
            "0.01",
            "--min-rel-loss-drop",
            "0.001",
            "--reload-atol",
            "1e-5",
            "--reload-rtol",
            "1e-5",
            "--cache-atol",
            "0.03",
            "--cache-rtol",
            "0.03",
            "--prompt",
            "Once upon a time",
            "--max-new-tokens",
            "12",
            "--output-dir",
            p03,
        ),
        "p0-3-tokenizer-psi8": (
            tf4576_python,
            "scripts/check_tokenizer_reload.py",
            "--source-tokenizer",
            "tokenizers/tinystories_spm768",
            "--checkpoint",
            os.fspath(run_root / "artifacts/p0-3/psi8"),
            "--logical-name",
            "p0_3_psi8",
            "--source-id",
            "tinystories-spm768",
            "--checkpoint-id",
            "p0-3-psi8-checkpoint",
            "--output",
            os.fspath(run_root / "artifacts/p0-3/tokenizer-reload-psi8.json"),
        ),
        "p0-3-tokenizer-psi16": (
            tf4576_python,
            "scripts/check_tokenizer_reload.py",
            "--source-tokenizer",
            "tokenizers/tinystories_spm768",
            "--checkpoint",
            os.fspath(run_root / "artifacts/p0-3/psi16"),
            "--logical-name",
            "p0_3_psi16",
            "--source-id",
            "tinystories-spm768",
            "--checkpoint-id",
            "p0-3-psi16-checkpoint",
            "--output",
            os.fspath(run_root / "artifacts/p0-3/tokenizer-reload-psi16.json"),
        ),
        "p0-4-psi8-preflight": (
            tf4576_python,
            "scripts/p0_4_gpt2_context4096_smoke.py",
            "--repo-root",
            repo,
            "--config-dir",
            "configs/p0_4_multiscreen_psi8_gpt2_ctx4096",
            "--validate-config-only",
        ),
        "p0-4-psi16-preflight": (
            tf4576_python,
            "scripts/p0_4_gpt2_context4096_smoke.py",
            "--repo-root",
            repo,
            "--config-dir",
            "configs/p0_4_multiscreen_psi16_gpt2_ctx4096",
            "--validate-config-only",
        ),
        "p0-4-review-psi8": (
            tf4576_python,
            "-P",
            "-S",
            "-B",
            "scripts/review_level1_requalification.py",
            "--mode",
            "p0-4-lane",
            "--tested-commit",
            "__TESTED_COMMIT__",
            "--command-ledger",
            os.fspath(run_root / "commands.jsonl"),
            "--p0-4-root",
            p04_8,
            "--tokenizer-reload-report",
            f"p0_4_psi8={p04_8}/tokenizer-reload.json",
            "--output",
            f"{p04_8}/raw-review.json",
        ),
        "repository-hygiene-final": repository_check("hygiene"),
    }

    for name, mode, psi in (
        ("c3-psi8-operational", "operational", "8"),
        ("c3-psi8-peak-exposure", "peak-exposure", "8"),
        ("c3-psi16-operational", "operational", "16"),
        ("c3-psi16-peak-exposure", "peak-exposure", "16"),
    ):
        output = os.fspath(run_root / f"artifacts/c3/cuda/psi{psi}/{mode}")
        tails[name] = (
            tf4576_python,
            "scripts/p0_5_c3_paper_training_contract.py",
            "--manifest",
            "configs/p0_5_c3_paper_training_contract.json",
            "--mode",
            mode,
            "--psi",
            psi,
            "--device",
            "cuda:0",
            "--cache-dir",
            cache_value,
            "--output-dir",
            output,
        )

    def p0_4_run(psi: str, output: str) -> tuple[str, ...]:
        return (
            tf4576_python,
            "scripts/p0_4_gpt2_context4096_smoke.py",
            "--repo-root",
            repo,
            "--config-dir",
            f"configs/p0_4_multiscreen_psi{psi}_gpt2_ctx4096",
            "--output-dir",
            output,
            "--tokenizer-name-or-path",
            "gpt2",
            "--cache-dir",
            cache_value,
            "--dataset-name",
            "roneneldan/TinyStories",
            "--train-split",
            "train[:20000]",
            "--text-column",
            "text",
            "--streaming",
            "false",
            "--max-texts",
            "20000",
            "--max-train-tokens",
            "524416",
            "--seq-len",
            "4096",
            "--steps",
            "50",
            "--microbatch-size",
            "1",
            "--gradient-accumulation-steps",
            "8",
            "--learning-rate",
            "0.0006",
            "--weight-decay",
            "0",
            "--max-grad-norm",
            "1",
            "--amp-dtype",
            "bf16",
            "--gradient-checkpointing",
            "true",
            "--fused-adamw",
            "true",
            "--device",
            "cuda:0",
            "--allow-cpu",
            "false",
            "--num-workers",
            "0",
            "--seed",
            "42",
            "--log-every",
            "1",
        )

    def p0_4_tokenizer(psi: str, output: str) -> tuple[str, ...]:
        return (
            tf4576_python,
            "scripts/check_tokenizer_reload.py",
            "--source-tokenizer",
            "gpt2",
            "--cache-dir",
            cache_value,
            "--checkpoint",
            f"{output}/checkpoint",
            "--logical-name",
            f"p0_4_psi{psi}",
            "--source-id",
            "gpt2",
            "--checkpoint-id",
            f"p0-4-psi{psi}-checkpoint",
            "--source-pad-token-from-eos",
            "--source-padding-side",
            "right",
            "--source-model-max-length",
            "4096",
            "--output",
            f"{output}/tokenizer-reload.json",
        )

    tails["p0-4-psi8"] = p0_4_run("8", p04_8)
    tails["p0-4-psi16"] = p0_4_run("16", p04_16)
    tails["p0-4-tokenizer-psi8"] = p0_4_tokenizer("8", p04_8)
    tails["p0-4-tokenizer-psi16"] = p0_4_tokenizer("16", p04_16)

    unknown = set(required_names) - set(tails)
    if unknown:
        _fail(f"internal exact command-tail contract is incomplete: {sorted(unknown)}")
    return {name: tails[name] for name in required_names}


def _bind_exact_command_tails(
    command_pairs: Mapping[str, tuple[Mapping[str, Any], bytes]],
    *,
    required_names: Sequence[str],
    run_root: Path,
    tested_commit: str,
) -> tuple[dict[str, tuple[str, ...]], Path]:
    def observed_tail(name: str) -> tuple[list[str], int, tuple[str, ...]]:
        if name not in command_pairs:
            _fail(f"exact command-tail binding is missing {name}")
        record = command_pairs[name][0]
        argv = _list(_at(record, "argv"), label=f"command[{name}].argv")
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            _fail(f"command {name} argv must contain non-empty strings")
        executable_index = _review_hermetic_command_argv(
            argv, name=name, run_root=run_root
        )
        return argv, executable_index, tuple(argv[executable_index:])

    _, _, tf4576_tail = observed_tail("environment-tf4576")
    if (
        len(tf4576_tail) != 4
        or tf4576_tail[1:]
        != ("scripts/report_level1_environment.py", "--lane", "tf4576")
        or not Path(tf4576_tail[0]).is_absolute()
    ):
        _fail("environment-tf4576 must bind the exact absolute TF4576 Python tail")
    tf4576_python = tf4576_tail[0]

    tf5141_python: str | None = None
    if "environment-tf5141" in required_names:
        _, _, tf5141_tail = observed_tail("environment-tf5141")
        if (
            len(tf5141_tail) != 4
            or tf5141_tail[1:]
            != ("scripts/report_level1_environment.py", "--lane", "tf5141")
            or not Path(tf5141_tail[0]).is_absolute()
        ):
            _fail("environment-tf5141 must bind the exact absolute TF5141 Python tail")
        tf5141_python = tf5141_tail[0]

    _, _, cuda_tail = observed_tail("environment-cuda0")
    if cuda_tail != (
        tf4576_python,
        "scripts/report_level1_environment.py",
        "--lane",
        "cuda0",
    ):
        _fail("environment-cuda0 must use the exact TF4576 Python tail")

    offline_argv, offline_index, _ = observed_tail("offline-cache-preflight")
    offline_python, cache = _review_offline_cache_command_argv(
        offline_argv, executable_index=offline_index
    )
    if offline_python != tf4576_python:
        _fail("offline-cache-preflight must use the exact TF4576 Python")

    expected = _expected_command_tails(
        required_names=required_names,
        repository=Path(__file__).resolve().parents[1],
        run_root=run_root,
        cache=cache,
        tf4576_python=tf4576_python,
        tf5141_python=tf5141_python,
    )
    review_tail = expected.get("p0-4-review-psi8")
    if review_tail is not None:
        expected["p0-4-review-psi8"] = tuple(
            tested_commit if value == "__TESTED_COMMIT__" else value
            for value in review_tail
        )
    return expected, cache

def _review_runtime(
    value: Any,
    *,
    label: str,
    expected_python_version: str,
) -> None:
    runtime = _mapping(value, label=label)
    operating_system = _mapping(_at(runtime, "operating_system"), label=f"{label}.operating_system")
    python = _mapping(_at(runtime, "python"), label=f"{label}.python")
    recorder = _mapping(_at(runtime, "recorder"), label=f"{label}.recorder")
    if set(runtime) != {"operating_system", "python", "recorder"}:
        _fail(f"{label} fields are incomplete or ambiguous")
    if set(operating_system) != {
        "libc_name", "libc_version", "machine", "release", "system"
    }:
        _fail(f"{label}.operating_system fields are incomplete or ambiguous")
    _equals(
        _at(operating_system, "system"),
        "Linux",
        label=f"{label}.operating_system.system",
    )
    for field in ("libc_name", "libc_version", "machine", "release"):
        item = _at(operating_system, field)
        if item is not None:
            _nonempty_string(item, label=f"{label}.operating_system.{field}")
    if set(python) != {
        "assertions_enabled",
        "cache_tag",
        "compiler",
        "implementation",
        "optimization_level",
        "version",
    }:
        _fail(f"{label}.python fields are incomplete or ambiguous")
    _equals(
        _at(python, "implementation"),
        "CPython",
        label=f"{label}.python.implementation",
    )
    for field in ("cache_tag", "compiler"):
        item = _at(python, field)
        if item is not None:
            _nonempty_string(item, label=f"{label}.python.{field}")
    if set(recorder) != {"name", "version"}:
        _fail(f"{label}.recorder fields are incomplete or ambiguous")
    _equals(_at(python, "version"), expected_python_version, label=f"{label}.python.version")
    _exact_int(
        _at(python, "optimization_level"),
        label=f"{label}.python.optimization_level",
        expected=0,
    )
    _exact_bool(
        _at(python, "assertions_enabled"),
        label=f"{label}.python.assertions_enabled",
        expected=True,
    )
    _equals(
        _at(recorder, "name"),
        "run_level1_requalification_command.py",
        label=f"{label}.recorder.name",
    )
    _equals(_at(recorder, "version"), "1.0.0", label=f"{label}.recorder.version")


def _review_record_common(record: Mapping[str, Any], *, name: str) -> None:
    _equals(
        _at(record, "format_version"),
        "level1-requalification-command-record-v1",
        label=f"record[{name}].format_version",
    )
    _equals(_at(record, "name"), name, label=f"record[{name}].name")
    if COMMAND_NAME_RE.fullmatch(name) is None:
        _fail(f"record name is not canonical: {name}")
    cwd = _mapping(_at(record, "cwd"), label=f"record[{name}].cwd")
    if set(cwd) != {"base", "path"}:
        _fail(f"record[{name}].cwd fields are incomplete or ambiguous")
    _equals(_at(cwd, "base"), "repository_root", label=f"record[{name}].cwd.base")
    cwd_path = _nonempty_string(_at(cwd, "path"), label=f"record[{name}].cwd.path")
    if cwd_path != "." and (
        cwd_path.startswith("/")
        or "\\" in cwd_path
        or any(part in {"", ".", ".."} for part in cwd_path.split("/"))
    ):
        _fail(f"record[{name}].cwd.path is not canonical repository-relative")
    started = _parse_utc(_at(record, "started_at_utc"), label=f"record[{name}].started_at_utc")
    ended = _parse_utc(_at(record, "ended_at_utc"), label=f"record[{name}].ended_at_utc")
    if ended < started:
        _fail(f"record {name} ended before it started")
    duration_ns = _exact_int(
        _at(record, "duration_ns"), label=f"record[{name}].duration_ns", minimum=0
    )
    duration_seconds = _finite_number(
        _at(record, "duration_seconds"),
        label=f"record[{name}].duration_seconds",
        minimum=0.0,
    )
    _close(
        duration_seconds,
        round(duration_ns / 1_000_000_000, 9),
        label=f"record[{name}].duration_seconds",
        atol=5e-10,
    )
    expected_python_version = (
        "3.12.10"
        if name == "runtime-tf5141" or name in TF5141_COMMAND_NAMES
        else "3.12.11"
    )
    _review_runtime(_at(record, "runtime"), label=f"record[{name}].runtime", expected_python_version=expected_python_version)


def _load_runner_jsonl(
    path: Path,
    *,
    label: str,
    hashes: dict[str, str],
    bind_hash: bool,
) -> tuple[list[dict[str, Any]], list[bytes]]:
    raw = (
        _read_bytes(path, label=label, hashes=hashes)
        if bind_hash
        else _stable_read_bytes(path, label=label)
    )
    if not raw or not raw.endswith(b"\n"):
        _fail(f"{label} must be non-empty and newline-terminated")
    lines = raw.splitlines(keepends=True)
    values: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        value = _mapping(
            _decode_json_bytes(line, label=f"{label}:{index + 1}"),
            label=f"{label}:{index + 1}",
        )
        if line != _runner_canonical_bytes(value):
            _fail(f"{label}:{index + 1} is not a canonical recorder record")
        values.append(dict(value))
    return values, lines



def _canonical_command_log_object(raw: bytes, *, name: str) -> Mapping[str, Any]:
    if not raw or raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
        _fail(f"command {name} semantic log must contain exactly one JSON line")
    value = _mapping(
        _decode_json_bytes(raw, label=f"command log {name}"),
        label=f"command log {name}",
    )
    if raw != _runner_canonical_bytes(value):
        _fail(f"command {name} semantic log is not canonical JSON")
    return value


def _review_environment_command_log(name: str, raw: bytes) -> dict[str, Any]:
    report = _canonical_command_log_object(raw, name=name)
    _exact_values(
        report,
        {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "status": "passed",
            "tool_version": "1.0.0",
        },
        label=f"environment log {name}",
    )
    expected_packages: dict[str, dict[str, str | None]] = {
        "tf4576": {
            "PyYAML": "6.0.1",
            "accelerate": "1.6.0",
            "datasets": "5.0.1",
            "huggingface-hub": "0.34.3",
            "numpy": "1.26.4",
            "pyarrow": "25.0.0",
            "safetensors": "0.5.3",
            "sentencepiece": "0.2.0",
            "tokenizers": "0.22.0",
            "torch": "2.7.1+cu128",
            "transformers": "4.57.6",
            "trl": "1.9.2",
        },
        "tf5141": {
            "PyYAML": "6.0.2",
            "accelerate": None,
            "datasets": None,
            "huggingface-hub": "1.27.0",
            "numpy": "2.3.2",
            "pyarrow": None,
            "safetensors": "0.8.0",
            "sentencepiece": "0.2.0",
            "tokenizers": "0.22.2",
            "torch": "2.8.0",
            "transformers": "5.14.1",
            "trl": None,
        },
    }
    expected_runtime = {
        "tf4576": {"torch": "2.7.1+cu128", "transformers": "4.57.6"},
        "tf5141": {"torch": "2.8.0+cu128", "transformers": "5.14.1"},
    }
    expected_python = {
        "tf4576": {
            "assertions_enabled": True,
            "implementation": "CPython",
            "optimization_level": 0,
            "version": "3.12.11",
        },
        "tf5141": {
            "assertions_enabled": True,
            "implementation": "CPython",
            "optimization_level": 0,
            "version": "3.12.10",
        },
    }
    if name == "environment-cuda0":
        if set(report) != {
            "cuda",
            "lane",
            "nvidia_smi",
            "packages",
            "python",
            "runtime",
            "schema_version",
            "selection",
            "status",
            "tool_version",
        }:
            _fail("environment-cuda0 fields are incomplete or ambiguous")
        _equals(_at(report, "lane"), "cuda0", label="environment-cuda0.lane")
        if _at(report, "packages") != expected_packages["tf4576"]:
            _fail("environment-cuda0 packages differ from the fixed tf4576 lane")
        if _at(report, "runtime") != expected_runtime["tf4576"]:
            _fail("environment-cuda0 runtime differs from the fixed tf4576 lane")
        if _at(report, "python") != expected_python["tf4576"]:
            _fail("environment-cuda0 python differs from the fixed tf4576 lane")
        selection = _mapping(
            _at(report, "selection"), label="environment-cuda0.selection"
        )
        if selection != {
            "cuda_visible_devices": "0",
            "logical_device": "cuda:0",
        }:
            _fail("environment-cuda0 did not bind logical cuda:0 exclusively")
        cuda = _mapping(_at(report, "cuda"), label="environment-cuda0.cuda")
        if set(cuda) != {
            "allocated_memory_bytes",
            "bf16_supported",
            "capability",
            "cudnn_version",
            "device_count",
            "device_name",
            "free_memory_bytes",
            "reserved_memory_bytes",
            "runtime_version",
            "total_memory_bytes",
        }:
            _fail("environment-cuda0 CUDA fields are incomplete or ambiguous")
        _exact_bool(
            _at(cuda, "bf16_supported"),
            label="environment-cuda0.cuda.bf16_supported",
            expected=True,
        )
        if _at(cuda, "capability") != [12, 0]:
            _fail("environment-cuda0 CUDA capability differs from the fixed device")
        _exact_int(
            _at(cuda, "cudnn_version"),
            label="environment-cuda0.cuda.cudnn_version",
            expected=90701,
        )
        _exact_int(
            _at(cuda, "device_count"),
            label="environment-cuda0.cuda.device_count",
            expected=1,
        )
        device_name = _nonempty_string(
            _at(cuda, "device_name"), label="environment-cuda0.cuda.device_name"
        )
        _equals(
            _at(cuda, "runtime_version"),
            "12.8",
            label="environment-cuda0.cuda.runtime_version",
        )
        total_memory = _exact_int(
            _at(cuda, "total_memory_bytes"),
            label="environment-cuda0.cuda.total_memory_bytes",
            minimum=1,
        )
        for field in (
            "allocated_memory_bytes",
            "free_memory_bytes",
            "reserved_memory_bytes",
        ):
            observed = _exact_int(
                _at(cuda, field), label=f"environment-cuda0.cuda.{field}", minimum=0
            )
            if observed > total_memory:
                _fail(f"environment-cuda0.cuda.{field} exceeds total memory")

        nvidia = _mapping(
            _at(report, "nvidia_smi"), label="environment-cuda0.nvidia_smi"
        )
        if set(nvidia) != {
            "compute_capability",
            "device_name",
            "driver_version",
            "memory_free_mib",
            "memory_total_mib",
            "other_compute_process_count",
            "other_compute_used_memory_mib",
            "physical_index",
            "reporter_compute_process_present",
            "reporter_used_memory_mib",
        }:
            _fail("environment-cuda0 nvidia-smi fields are incomplete or ambiguous")
        _equals(
            _at(nvidia, "compute_capability"),
            "12.0",
            label="environment-cuda0.nvidia_smi.compute_capability",
        )
        _equals(
            _at(nvidia, "device_name"),
            device_name,
            label="environment-cuda0.nvidia_smi.device_name",
        )
        _nonempty_string(
            _at(nvidia, "driver_version"),
            label="environment-cuda0.nvidia_smi.driver_version",
        )
        _exact_int(
            _at(nvidia, "physical_index"),
            label="environment-cuda0.nvidia_smi.physical_index",
            expected=0,
        )
        nvidia_total = _exact_int(
            _at(nvidia, "memory_total_mib"),
            label="environment-cuda0.nvidia_smi.memory_total_mib",
            minimum=1,
        )
        nvidia_free = _exact_int(
            _at(nvidia, "memory_free_mib"),
            label="environment-cuda0.nvidia_smi.memory_free_mib",
            minimum=0,
        )
        if nvidia_free > nvidia_total:
            _fail("environment-cuda0 nvidia-smi free memory exceeds total memory")
        for field in (
            "other_compute_process_count",
            "other_compute_used_memory_mib",
            "reporter_used_memory_mib",
        ):
            _exact_int(
                _at(nvidia, field),
                label=f"environment-cuda0.nvidia_smi.{field}",
                minimum=0,
            )
        reporter_present = _exact_bool(
            _at(nvidia, "reporter_compute_process_present"),
            label="environment-cuda0.nvidia_smi.reporter_compute_process_present",
        )
        if not reporter_present and _at(nvidia, "reporter_used_memory_mib") != 0:
            _fail("absent reporter process must have zero recorded GPU memory")
        return {
            "lane": "cuda0",
            "status": "passed",
            "bf16_supported": True,
            "capability": [12, 0],
            "python_version": expected_python["tf4576"]["version"],
            "total_memory_bytes": total_memory,
            "other_compute_process_count": _at(
                nvidia, "other_compute_process_count"
            ),
        }

    expected = {
        "environment-tf4576": {
            "lane": "tf4576",
            "python": expected_python["tf4576"],
            "packages": expected_packages["tf4576"],
            "runtime": expected_runtime["tf4576"],
        },
        "environment-tf5141": {
            "lane": "tf5141",
            "python": expected_python["tf5141"],
            "packages": expected_packages["tf5141"],
            "runtime": expected_runtime["tf5141"],
        },
    }
    if name not in expected:
        _fail(f"unsupported semantic environment command: {name}")
    if set(report) != {
        "lane",
        "packages",
        "python",
        "runtime",
        "schema_version",
        "status",
        "tool_version",
    }:
        _fail(f"{name} fields are incomplete or ambiguous")
    for field, expected_value in expected[name].items():
        if report[field] != expected_value:
            _fail(f"{name}.{field} differs from the fixed compatibility lane")
    return {
        "lane": expected[name]["lane"],
        "status": "passed",
        "python_version": expected[name]["python"]["version"],
        "torch": expected[name]["runtime"]["torch"],
        "transformers": expected[name]["runtime"]["transformers"],
    }

def _review_repository_command_log(
    name: str,
    raw: bytes,
    *,
    tested_commit: str,
) -> dict[str, Any]:
    report = _canonical_command_log_object(raw, name=name)
    expected_check = {
        "json-validation": "json",
        "workflow-yaml": "workflow-yaml",
        "markdown-links": "markdown-links",
        "repository-hygiene": "hygiene",
        "repository-hygiene-final": "hygiene",
    }[name]
    if set(report) != {"check", "format_version", "head_commit", "result", "status"}:
        _fail(f"{name} report fields are incomplete or ambiguous")
    _equals(_at(report, "head_commit"), tested_commit, label=f"{name}.head_commit")
    _exact_values(
        report,
        {
            "check": expected_check,
            "format_version": REPOSITORY_CHECK_SCHEMA_VERSION,
            "status": "passed",
        },
        label=name,
    )
    result = _mapping(_at(report, "result"), label=f"{name}.result")
    common_fields = {
        "artifact_count",
        "artifact_manifest_sha256",
        "bytes_checked",
    }
    artifact_count = _exact_int(
        _at(result, "artifact_count"),
        label=f"{name}.artifact_count",
        minimum=1,
    )
    bytes_checked = _exact_int(
        _at(result, "bytes_checked"),
        label=f"{name}.bytes_checked",
        minimum=1,
    )
    digest = _nonempty_string(
        _at(result, "artifact_manifest_sha256"),
        label=f"{name}.artifact_manifest_sha256",
    )
    if HEX64_RE.fullmatch(digest) is None:
        _fail(f"{name}.artifact_manifest_sha256 must be lowercase SHA-256")

    expected_extra_fields: dict[str, set[str]] = {
        "json": set(),
        "markdown-links": {"link_count", "local_link_count"},
        "workflow-yaml": {"job_count", "parser"},
    }
    if expected_check != "hygiene":
        expected_fields = common_fields | expected_extra_fields[expected_check]
        if set(result) != expected_fields:
            _fail(f"{name} aggregate fields are incomplete or ambiguous")
        for field in sorted(expected_extra_fields[expected_check] - {"parser"}):
            _exact_int(_at(result, field), label=f"{name}.{field}", minimum=0)
        if expected_check == "workflow-yaml":
            _exact_int(_at(result, "job_count"), label=f"{name}.job_count", minimum=1)
            _equals(
                _at(result, "parser"),
                "stdlib_github_actions_restricted_v2",
                label=f"{name}.parser",
            )
        return {
            "check": expected_check,
            "status": "passed",
            "artifact_count": artifact_count,
            "artifact_manifest_sha256": digest,
            "bytes_checked": bytes_checked,
        }

    expected_hygiene_fields = common_fields | {
        "gitlink_count",
        "head_commit",
        "index_diff_check",
        "maximum_tracked_file_bytes",
        "privacy",
        "submodules",
        "worktree",
        "worktree_diff_check",
    }
    if set(result) != expected_hygiene_fields:
        _fail(f"{name} hygiene fields are incomplete or ambiguous")
    _equals(_at(result, "head_commit"), tested_commit, label=f"{name}.head_commit")
    _equals(
        _at(result, "index_diff_check"),
        "passed",
        label=f"{name}.index_diff_check",
    )
    _equals(
        _at(result, "worktree_diff_check"),
        "passed",
        label=f"{name}.worktree_diff_check",
    )
    _exact_int(
        _at(result, "maximum_tracked_file_bytes"),
        label=f"{name}.maximum_tracked_file_bytes",
        expected=5 * 1024 * 1024,
    )
    _exact_int(
        _at(result, "gitlink_count"), label=f"{name}.gitlink_count", minimum=0
    )
    privacy = _mapping(_at(result, "privacy"), label=f"{name}.privacy")
    if set(privacy) != {
        "artifact_count",
        "artifact_manifest_sha256",
        "bytes_checked",
        "fixture_exemption_artifact_count",
        "rules",
        "status",
    }:
        _fail(f"{name}.privacy fields are incomplete or ambiguous")
    privacy_artifact_count = _exact_int(
        _at(privacy, "artifact_count"),
        label=f"{name}.privacy.artifact_count",
        minimum=1,
    )
    privacy_bytes_checked = _exact_int(
        _at(privacy, "bytes_checked"),
        label=f"{name}.privacy.bytes_checked",
        minimum=1,
    )
    privacy_digest = _nonempty_string(
        _at(privacy, "artifact_manifest_sha256"),
        label=f"{name}.privacy.artifact_manifest_sha256",
    )
    if HEX64_RE.fullmatch(privacy_digest) is None:
        _fail(f"{name}.privacy.artifact_manifest_sha256 must be lowercase SHA-256")
    fixture_exemptions = _exact_int(
        _at(privacy, "fixture_exemption_artifact_count"),
        label=f"{name}.privacy.fixture_exemption_artifact_count",
        minimum=0,
    )
    if fixture_exemptions > privacy_artifact_count:
        _fail(f"{name}.privacy fixture exemptions exceed scanned artifacts")
    privacy_rules = [
        "account-and-repository-root",
        "credential-token",
        "credential-url",
        "file-uri",
        "private-user-path",
        "shareable-secret-assignment",
    ]
    if _at(privacy, "rules") != privacy_rules:
        _fail(f"{name}.privacy.rules differ from the required ordered rule set")
    _equals(_at(privacy, "status"), "passed", label=f"{name}.privacy.status")
    if privacy_artifact_count > artifact_count or privacy_bytes_checked > bytes_checked:
        _fail(f"{name}.privacy scan exceeds the tracked-artifact aggregate")
    worktree = _mapping(_at(result, "worktree"), label=f"{name}.worktree")
    expected_empty_sha = hashlib.sha256(b"").hexdigest()
    if worktree != {
        "clean": True,
        "porcelain_byte_count": 0,
        "porcelain_sha256": expected_empty_sha,
    }:
        _fail(f"{name} does not prove a clean losslessly hashed worktree")
    submodules = _mapping(_at(result, "submodules"), label=f"{name}.submodules")
    if set(submodules) != {"byte_count", "record_count", "sha256", "states"}:
        _fail(f"{name}.submodules fields are incomplete or ambiguous")
    byte_count = _exact_int(
        _at(submodules, "byte_count"),
        label=f"{name}.submodules.byte_count",
        minimum=0,
    )
    record_count = _exact_int(
        _at(submodules, "record_count"),
        label=f"{name}.submodules.record_count",
        minimum=0,
    )
    submodule_sha = _nonempty_string(
        _at(submodules, "sha256"), label=f"{name}.submodules.sha256"
    )
    if HEX64_RE.fullmatch(submodule_sha) is None:
        _fail(f"{name}.submodules.sha256 must be lowercase SHA-256")
    states = _mapping(_at(submodules, "states"), label=f"{name}.submodules.states")
    if set(states) != {"clean", "conflict", "missing", "modified"}:
        _fail(f"{name}.submodules.states fields are incomplete or ambiguous")
    state_counts = {
        field: _exact_int(
            states[field], label=f"{name}.submodules.states.{field}", minimum=0
        )
        for field in sorted(states)
    }
    if sum(state_counts.values()) != record_count:
        _fail(f"{name}.submodules record count differs from its state counts")
    return {
        "check": "hygiene",
        "status": "passed",
        "artifact_count": artifact_count,
        "artifact_manifest_sha256": digest,
        "bytes_checked": bytes_checked,
        "head_commit": tested_commit,
        "privacy": {
            "artifact_count": privacy_artifact_count,
            "artifact_manifest_sha256": privacy_digest,
            "bytes_checked": privacy_bytes_checked,
            "fixture_exemption_artifact_count": fixture_exemptions,
            "rules": privacy_rules,
            "status": "passed",
        },
        "submodule_byte_count": byte_count,
        "submodule_record_count": record_count,
        "submodule_sha256": submodule_sha,
    }



def _review_path_free_offline_tree(
    value: Any,
    *,
    repository: Path,
    cache: Path,
    label: str,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _review_path_free_offline_tree(
                item,
                repository=repository,
                cache=cache,
                label=f"{label}.{key}",
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _review_path_free_offline_tree(
                item,
                repository=repository,
                cache=cache,
                label=f"{label}[{index}]",
            )
        return
    if not isinstance(value, str):
        return
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail(f"{label} contains a control character")
    lowered = value.lower()
    repository_value = os.fspath(repository)
    cache_value = os.fspath(cache)
    if repository_value in value or cache_value in value:
        _fail(f"{label} exposes the explicit repository or cache path")
    if (
        value.startswith(("/", "\\", "~"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or lowered.startswith("file:")
        or re.search(r"(?:^|\s)(?:/[A-Za-z0-9_.-]|[A-Za-z]:[\\/]|file:)", value, re.I)
    ):
        _fail(f"{label} exposes an absolute path or file URI")


def _review_offline_tinystories_check(
    value: Any,
    *,
    label: str,
    revision: str,
) -> dict[str, Any]:
    report = _mapping(value, label=label)
    expected_fields = {
        "fingerprint_nonempty",
        "fingerprint_sha256",
        "repository",
        "revision",
        "row_count",
        "split",
        "text_column",
        "text_rows",
    }
    if set(report) != expected_fields:
        _fail(f"{label} fields are incomplete or ambiguous")
    _exact_bool(
        _at(report, "fingerprint_nonempty"),
        label=f"{label}.fingerprint_nonempty",
        expected=True,
    )
    fingerprint = _nonempty_string(
        _at(report, "fingerprint_sha256"), label=f"{label}.fingerprint_sha256"
    )
    if HEX64_RE.fullmatch(fingerprint) is None:
        _fail(f"{label}.fingerprint_sha256 must be lowercase SHA-256")
    _equals(
        _at(report, "repository"),
        "roneneldan/TinyStories",
        label=f"{label}.repository",
    )
    _equals(_at(report, "revision"), revision, label=f"{label}.revision")
    _exact_int(_at(report, "row_count"), label=f"{label}.row_count", expected=20_000)
    _equals(_at(report, "split"), "train[:20000]", label=f"{label}.split")
    _equals(_at(report, "text_column"), "text", label=f"{label}.text_column")
    _exact_int(_at(report, "text_rows"), label=f"{label}.text_rows", expected=20_000)
    return {
        "fingerprint_sha256": fingerprint,
        "repository": "roneneldan/TinyStories",
        "revision": revision,
        "row_count": 20_000,
    }


def _review_offline_cache_command_log(
    raw: bytes,
    *,
    repository: Path,
    cache: Path,
) -> dict[str, Any]:
    name = "offline-cache-preflight"
    report = _canonical_command_log_object(raw, name=name)
    _review_path_free_offline_tree(
        report,
        repository=repository,
        cache=cache,
        label="offline-cache-preflight",
    )
    if set(report) != {
        "cache",
        "checks",
        "offline_environment",
        "schema_version",
        "status",
    }:
        _fail("offline-cache-preflight fields are incomplete or ambiguous")
    _equals(
        _at(report, "schema_version"),
        "multiscreen-level1-offline-cache-v1",
        label="offline-cache-preflight.schema_version",
    )
    _equals(_at(report, "status"), "passed", label="offline-cache-preflight.status")

    cache_report = _mapping(
        _at(report, "cache"), label="offline-cache-preflight.cache"
    )
    if set(cache_report) != {"explicit", "path_recorded", "single_cache"}:
        _fail("offline-cache-preflight.cache fields are incomplete or ambiguous")
    _exact_bool(
        _at(cache_report, "explicit"),
        label="offline-cache-preflight.cache.explicit",
        expected=True,
    )
    _exact_bool(
        _at(cache_report, "path_recorded"),
        label="offline-cache-preflight.cache.path_recorded",
        expected=False,
    )
    _exact_bool(
        _at(cache_report, "single_cache"),
        label="offline-cache-preflight.cache.single_cache",
        expected=True,
    )

    offline_environment = _mapping(
        _at(report, "offline_environment"),
        label="offline-cache-preflight.offline_environment",
    )
    expected_offline_environment = {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if offline_environment != expected_offline_environment:
        _fail("offline-cache-preflight offline flags differ from the exact contract")

    checks = _mapping(_at(report, "checks"), label="offline-cache-preflight.checks")
    if set(checks) != {
        "p0_3_tinystories",
        "p0_4_gpt2_tokenizer",
        "p0_4_tinystories",
        "p0_5_c3",
    }:
        _fail("offline-cache-preflight check set differs from the fixed matrix")
    p0_3 = _review_offline_tinystories_check(
        _at(checks, "p0_3_tinystories"),
        label="offline-cache-preflight.p0_3_tinystories",
        revision=P0_3_DATASET_REVISION,
    )
    p0_4_dataset = _review_offline_tinystories_check(
        _at(checks, "p0_4_tinystories"),
        label="offline-cache-preflight.p0_4_tinystories",
        revision="default",
    )

    tokenizer = _mapping(
        _at(checks, "p0_4_gpt2_tokenizer"),
        label="offline-cache-preflight.p0_4_gpt2_tokenizer",
    )
    if set(tokenizer) != {
        "eos_token_id",
        "identity_projection",
        "repository",
        "revision",
        "use_fast",
        "vocab_size",
    }:
        _fail("offline-cache-preflight GPT-2 tokenizer fields are incomplete")
    _exact_int(
        _at(tokenizer, "eos_token_id"),
        label="offline-cache-preflight.p0_4_gpt2_tokenizer.eos_token_id",
        expected=50_256,
    )
    _equals(
        _at(tokenizer, "repository"),
        "gpt2",
        label="offline-cache-preflight.p0_4_gpt2_tokenizer.repository",
    )
    _equals(
        _at(tokenizer, "revision"),
        "default",
        label="offline-cache-preflight.p0_4_gpt2_tokenizer.revision",
    )
    _exact_bool(
        _at(tokenizer, "use_fast"),
        label="offline-cache-preflight.p0_4_gpt2_tokenizer.use_fast",
        expected=True,
    )
    _exact_int(
        _at(tokenizer, "vocab_size"),
        label="offline-cache-preflight.p0_4_gpt2_tokenizer.vocab_size",
        expected=50_257,
    )
    tokenizer_projection = _review_p0_4_tokenizer_projection(
        _at(tokenizer, "identity_projection"),
        label="offline-cache-preflight.p0_4_gpt2_tokenizer.identity_projection",
    )

    c3 = _mapping(_at(checks, "p0_5_c3"), label="offline-cache-preflight.p0_5_c3")
    if set(c3) != {"dataset", "manifest_sha256", "tokenizer"}:
        _fail("offline-cache-preflight C3 fields are incomplete or ambiguous")
    _equals(
        _at(c3, "manifest_sha256"),
        "480c127a8db02acb839d49e55d3a468cf452e816a04780d1b2a9fa8fe2c16060",
        label="offline-cache-preflight.p0_5_c3.manifest_sha256",
    )
    c3_dataset = _mapping(
        _at(c3, "dataset"), label="offline-cache-preflight.p0_5_c3.dataset"
    )
    expected_c3_dataset = {
        "data_files": {"test": "data/test-00000-of-00030.parquet"},
        "file_sha256": "d9a83d59b72f4c303f0c0e46d0e73a8446eabb56b9aa5fd992347c358ab65743",
        "file_size_bytes": 43_263_929,
        "full_fingerprint": "507a47fcec5cbfdc",
        "repository": "gmongaras/SlimPajama-627B_Reupload",
        "revision": "c34c22dbb10ae6b264a2f357a909d1a537141b36",
        "row_manifest_sha256": C3_ROW_MANIFEST_SHA256,
        "selected_fingerprint": "f1e6c1c09434a7e4",
        "selected_rows": 64,
        "split": "test",
        "text_column": "text",
    }
    if c3_dataset != expected_c3_dataset:
        _fail("offline-cache-preflight C3 dataset identity differs from the manifest")
    _exact_int(
        _at(c3_dataset, "file_size_bytes"),
        label="offline-cache-preflight.p0_5_c3.dataset.file_size_bytes",
        expected=43_263_929,
    )
    _exact_int(
        _at(c3_dataset, "selected_rows"),
        label="offline-cache-preflight.p0_5_c3.dataset.selected_rows",
        expected=64,
    )
    c3_tokenizer = _mapping(
        _at(c3, "tokenizer"), label="offline-cache-preflight.p0_5_c3.tokenizer"
    )
    expected_c3_tokenizer = {
        "asset_manifest_sha256": "07c45937a89b33f30016aef5b3982f13f25bf2c6ba940c535d1b5daa90459a71",
        "eos_token_id": 50_256,
        "repository": "gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "vocab_size": 50_257,
    }
    if c3_tokenizer != expected_c3_tokenizer:
        _fail("offline-cache-preflight C3 tokenizer identity differs from the manifest")
    _exact_int(
        _at(c3_tokenizer, "eos_token_id"),
        label="offline-cache-preflight.p0_5_c3.tokenizer.eos_token_id",
        expected=50_256,
    )
    _exact_int(
        _at(c3_tokenizer, "vocab_size"),
        label="offline-cache-preflight.p0_5_c3.tokenizer.vocab_size",
        expected=50_257,
    )
    return {
        "cache": {"explicit": True, "path_recorded": False, "single_cache": True},
        "checks": {
            "p0_3_tinystories": p0_3,
            "p0_4_gpt2_tokenizer": {
                "repository": "gpt2",
                "identity_projection": tokenizer_projection,
                "revision": "default",
                "vocab_size": 50_257,
            },
            "p0_4_tinystories": p0_4_dataset,
            "p0_5_c3": {
                "dataset_revision": expected_c3_dataset["revision"],
                "row_manifest_sha256": expected_c3_dataset["row_manifest_sha256"],
                "manifest_sha256": _at(c3, "manifest_sha256"),
                "tokenizer_revision": expected_c3_tokenizer["revision"],
            },
        },
        "offline_environment": expected_offline_environment,
        "schema_version": "multiscreen-level1-offline-cache-v1",
        "status": "passed",
    }


def _review_p0_4_command_stdout(raw: bytes, *, name: str) -> dict[str, Any]:
    if not raw or not raw.endswith(b"\n"):
        _fail(f"command {name} lossless stdout must be newline-terminated")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"command {name} lossless stdout is not UTF-8: {exc}") from exc
    digests: list[str] = []
    for line in text.splitlines():
        if "[P0-4] data_contract" not in line:
            continue
        match = P0_4_DATA_CONTRACT_STDOUT_RE.fullmatch(line)
        if match is None:
            _fail(f"command {name} has a malformed P0-4 data-contract digest")
        digests.append(match.group(1))
    if len(digests) != 1:
        _fail(f"command {name} must contain exactly one P0-4 data-contract digest")
    return {"data_contract_sha256": digests[0]}


def _review_semantic_command_logs(
    log_raws: Mapping[str, bytes],
    *,
    tested_commit: str,
    repository: Path,
    cache: Path,
) -> dict[str, Any]:
    environment: dict[str, Any] = {}
    for name in (
        "environment-tf4576",
        "environment-tf5141",
        "environment-cuda0",
    ):
        if name in log_raws:
            environment[name] = _review_environment_command_log(name, log_raws[name])
    repository_checks: dict[str, Any] = {}
    for name in (
        "json-validation",
        "workflow-yaml",
        "markdown-links",
        "repository-hygiene",
        "repository-hygiene-final",
    ):
        if name in log_raws:
            repository_checks[name] = _review_repository_command_log(
                name, log_raws[name], tested_commit=tested_commit
            )
    if (
        "repository-hygiene" in repository_checks
        and "repository-hygiene-final" in repository_checks
    ):
        first = dict(repository_checks["repository-hygiene"])
        final = dict(repository_checks["repository-hygiene-final"])
        for value in (first, final):
            value.pop("status", None)
        if first != final:
            _fail("initial and final repository hygiene identities differ")
    offline_cache = None
    p0_4: dict[str, Any] = {}
    for name, logical in (
        ("p0-4-psi8", "p0_4_psi8"),
        ("p0-4-psi16", "p0_4_psi16"),
    ):
        if name in log_raws:
            p0_4[logical] = _review_p0_4_command_stdout(log_raws[name], name=name)
    if "offline-cache-preflight" in log_raws:
        offline_cache = _review_offline_cache_command_log(
            log_raws["offline-cache-preflight"],
            repository=repository,
            cache=cache,
        )
    return {
        "environment": dict(sorted(environment.items())),
        "offline_cache": offline_cache,
        "p0_4": dict(sorted(p0_4.items())),
        "repository": dict(sorted(repository_checks.items())),
    }


def _review_command_ledger(
    ledger_value: str | os.PathLike[str],
    *,
    required_names: Sequence[str],
    required_environment_names: Sequence[str],
    tested_commit: str,
    expected_options: Mapping[str, Mapping[str, str]],
    expected_absent_paths: Mapping[str, Sequence[Path]],
    expected_log_paths: Mapping[str, Path],
    bind_ledger_hashes: bool,
    allow_extra_records: bool,
    hashes: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    if COMMIT_RE.fullmatch(tested_commit) is None:
        _fail("tested commit must be a full lowercase 40- or 64-hex identifier")
    ledger_path = _safe_file(ledger_value, label="command ledger")
    if ledger_path.name != "commands.jsonl":
        _fail("command ledger must be the recorder's commands.jsonl")
    run_root = _safe_root(ledger_path.parent, label="recorder run root")
    if stat.S_IMODE(run_root.stat().st_mode) != 0o700:
        _fail("recorder run root must have private mode 0700")
    for directory_name in ("logs", "records"):
        control_directory = _child_directory(
            run_root, directory_name, label=f"runner {directory_name} directory"
        )
        if stat.S_IMODE(control_directory.stat().st_mode) != 0o700:
            _fail(f"runner {directory_name} directory must have private mode 0700")
    marker = _mapping(
        _load_json(
            _child_file(
                run_root,
                ".level1-requalification-run.json",
                label="recorder run marker",
            ),
            label="runner.run_marker",
            hashes=hashes,
        ),
        label="recorder run marker",
    )
    if set(marker) != {"created_at_utc", "format_version", "repository", "tool_version"}:
        _fail("recorder run marker fields are incomplete or ambiguous")
    _equals(
        _at(marker, "format_version"),
        "level1-requalification-run-v1",
        label="runner.marker.format_version",
    )
    _parse_utc(_at(marker, "created_at_utc"), label="runner.marker.created_at_utc")
    _nonempty_string(_at(marker, "tool_version"), label="runner.marker.tool_version")
    repository = _mapping(_at(marker, "repository"), label="runner.marker.repository")
    if set(repository) != {"head_commit", "worktree_path_sha256"}:
        _fail("runner marker repository identity is incomplete or ambiguous")
    _equals(_at(repository, "head_commit"), tested_commit, label="runner.marker.head_commit")
    worktree_digest = _nonempty_string(
        _at(repository, "worktree_path_sha256"), label="runner.marker.worktree_path_sha256"
    )
    if not HEX64_RE.fullmatch(worktree_digest):
        _fail("runner marker worktree path digest is not lowercase SHA-256")
    reviewer_checkout = Path(__file__).resolve().parents[1]
    if not reviewer_checkout.is_absolute():
        _fail("reviewer checkout path is not absolute")
    _check_no_symlink_components(reviewer_checkout, label="reviewer checkout")
    expected_worktree_digest = _sha256_bytes(os.fsencode(reviewer_checkout))
    _equals(
        worktree_digest,
        expected_worktree_digest,
        label="runner.marker.worktree_path_sha256",
    )

    commands, command_lines = _load_runner_jsonl(
        ledger_path,
        label="runner.commands_ledger",
        hashes=hashes,
        bind_hash=bind_ledger_hashes,
    )
    environment_path = _child_file(
        run_root, "environment.jsonl", label="runner environment ledger"
    )
    environments, environment_lines = _load_runner_jsonl(
        environment_path,
        label="runner.environment_ledger",
        hashes=hashes,
        bind_hash=bind_ledger_hashes,
    )
    names = [
        _nonempty_string(command.get("name"), label=f"command[{index}].name")
        for index, command in enumerate(commands)
    ]
    if len(set(names)) != len(names):
        _fail("command ledger contains duplicate names")
    missing = sorted(set(required_names) - set(names))
    extra = sorted(set(names) - set(required_names))
    if missing or (extra and not allow_extra_records):
        _fail(
            "command ledger names differ from the fixed required matrix: "
            f"missing={missing}, extra={extra}"
        )
    observed_required_order = [name for name in names if name in set(required_names)]
    if observed_required_order != list(required_names):
        _fail("command ledger order differs from the fixed TESTING matrix")
    command_pairs = {
        command["name"]: (command, line)
        for command, line in zip(commands, command_lines, strict=True)
    }
    expected_command_tails, offline_cache = _bind_exact_command_tails(
        command_pairs,
        required_names=required_names,
        run_root=run_root,
        tested_commit=tested_commit,
    )
    record_hashes: dict[str, str] = {}
    log_hashes: dict[str, str] = {}
    log_raws: dict[str, bytes] = {}
    command_times: dict[str, tuple[dt.datetime, dt.datetime]] = {}
    for name in required_names:
        command, ledger_line = command_pairs[name]
        expected_fields = {
            "argv", "cwd", "duration_ns", "duration_seconds", "ended_at_utc",
            "exit_code", "format_version", "log", "name", "preconditions",
            "record_type", "returncode", "runtime", "started_at_utc",
            "termination_signal",
        }
        if set(command) != expected_fields:
            _fail(f"command record {name} fields are incomplete or indicate an error")
        _review_record_common(command, name=name)
        command_times[name] = (
            _parse_utc(_at(command, "started_at_utc"), label=f"command[{name}].started_at_utc"),
            _parse_utc(_at(command, "ended_at_utc"), label=f"command[{name}].ended_at_utc"),
        )
        _equals(_at(command, "record_type"), "command", label=f"command[{name}].record_type")
        argv = _list(_at(command, "argv"), label=f"command[{name}].argv")
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            _fail(f"command {name} argv must contain non-empty strings")
        executable_index = _review_hermetic_command_argv(
            argv, name=name, run_root=run_root
        )
        observed_tail = tuple(argv[executable_index:])
        if observed_tail != expected_command_tails[name]:
            _fail(
                f"command {name} child argv tail differs from the fixed TESTING matrix"
            )
        _exact_int(_at(command, "exit_code"), label=f"command[{name}].exit_code", expected=0)
        _exact_int(_at(command, "returncode"), label=f"command[{name}].returncode", expected=0)
        if _at(command, "termination_signal") is not None:
            _fail(f"command {name} has a termination signal")
        preconditions = _mapping(
            _at(command, "preconditions"), label=f"command[{name}].preconditions"
        )
        if set(preconditions) != {"absent_paths"}:
            _fail(f"command {name} precondition fields are incomplete or ambiguous")
        absent_paths = _list(
            _at(preconditions, "absent_paths"), label=f"command[{name}].absent_paths"
        )
        expected_absent = sorted(
            _run_relative(run_root, path, label=f"command {name} absent path")
            for path in expected_absent_paths.get(name, ())
        )
        if absent_paths != expected_absent:
            _fail(f"command {name} fresh-output preconditions do not match")
        for option, expected_value in expected_options.get(name, {}).items():
            if _argv_option(argv, option, label=f"command {name}") != expected_value:
                _fail(f"command {name} {option} does not bind the reviewed artifact")

        log = _mapping(_at(command, "log"), label=f"command[{name}].log")
        if set(log) != {"path", "sha256", "size_bytes"}:
            _fail(f"command {name} log fields are incomplete or ambiguous")
        expected_log_relative = f"logs/{name}.log"
        _equals(_at(log, "path"), expected_log_relative, label=f"command[{name}].log.path")
        log_path = _child_file(run_root, expected_log_relative, label=f"command {name} log")
        if name in expected_log_paths and log_path != expected_log_paths[name]:
            _fail(f"command {name} log path is not the reviewed lossless stdout log")
        log_raw = _read_bytes(log_path, label=f"runner.log.{name}", hashes=hashes)
        log_digest = _sha256_bytes(log_raw)
        _exact_int(_at(log, "size_bytes"), label=f"command[{name}].log.size_bytes", expected=len(log_raw))
        _equals(_at(log, "sha256"), log_digest, label=f"command[{name}].log.sha256")
        log_hashes[name] = log_digest
        log_raws[name] = log_raw
        record_path = _child_file(
            run_root, f"records/{name}.json", label=f"command {name} record"
        )
        record_raw = _read_bytes(
            record_path, label=f"runner.record.{name}", hashes=hashes
        )
        if record_raw != ledger_line:
            _fail(f"command {name} named record differs from commands.jsonl")
        record_hashes[name] = _sha256_bytes(record_raw)

    environment_names = [
        _nonempty_string(record.get("name"), label=f"environment[{index}].name")
        for index, record in enumerate(environments)
    ]
    if len(set(environment_names)) != len(environment_names):
        _fail("environment ledger contains duplicate names")
    missing_environment = sorted(set(required_environment_names) - set(environment_names))
    extra_environment = sorted(set(environment_names) - set(required_environment_names))
    if missing_environment or (extra_environment and not allow_extra_records):
        _fail("environment ledger names differ from the fixed required matrix")
    environment_pairs = {
        record["name"]: (record, line)
        for record, line in zip(environments, environment_lines, strict=True)
    }
    environment_times: dict[str, tuple[dt.datetime, dt.datetime]] = {}
    for name in required_environment_names:
        record, ledger_line = environment_pairs[name]
        expected_fields = {
            "cwd", "duration_ns", "duration_seconds", "ended_at_utc",
            "format_version", "name", "record_type", "repository", "runtime",
            "started_at_utc",
        }
        if set(record) != expected_fields:
            _fail(f"environment record {name} fields are incomplete or ambiguous")
        _review_record_common(record, name=name)
        environment_times[name] = (
            _parse_utc(_at(record, "started_at_utc"), label=f"environment[{name}].started_at_utc"),
            _parse_utc(_at(record, "ended_at_utc"), label=f"environment[{name}].ended_at_utc"),
        )
        _equals(_at(record, "record_type"), "environment", label=f"environment[{name}].record_type")
        _equals(
            _at(record, "repository.head_commit"),
            tested_commit,
            label=f"environment[{name}].repository.head_commit",
        )
        record_path = _child_file(
            run_root, f"records/{name}.json", label=f"environment {name} record"
        )
        record_raw = _read_bytes(
            record_path, label=f"runner.record.{name}", hashes=hashes
        )
        if record_raw != ledger_line:
            _fail(f"environment {name} named record differs from environment.jsonl")
        record_hashes[name] = _sha256_bytes(record_raw)

    semantic_logs = _review_semantic_command_logs(
        log_raws,
        tested_commit=tested_commit,
        repository=Path(__file__).resolve().parents[1],
        cache=offline_cache,
    )

    def require_sequence(sequence: Sequence[str], *, label: str) -> None:
        if any(name not in command_times for name in sequence):
            _fail(f"{label} is missing a required command timestamp")
        for earlier_name, later_name in zip(sequence, sequence[1:]):
            if not command_times[earlier_name][1] < command_times[later_name][0]:
                _fail(
                    f"{label} is out of order: {earlier_name} must finish "
                    f"before {later_name} starts"
                )

    ordering_checks: list[str] = []
    full_boundary_names = {
        "environment-tf4576",
        "environment-tf5141",
        "environment-cuda0",
        "offline-cache-preflight",
        "repository-hygiene",
        "syntax-level1",
        "p0-4-tokenizer-psi16",
        "repository-hygiene-final",
    }
    if full_boundary_names <= set(required_names):
        require_sequence(
            (
                "environment-tf4576",
                "environment-tf5141",
                "environment-cuda0",
                "offline-cache-preflight",
                "repository-hygiene",
                "syntax-level1",
            ),
            label="environment and initial hygiene commands",
        )
        if set(REQUIRED_ENVIRONMENT_NAMES) <= set(environment_times):
            require_environment_sequence = tuple(REQUIRED_ENVIRONMENT_NAMES)
            for earlier_name, later_name in zip(
                require_environment_sequence, require_environment_sequence[1:]
            ):
                if not environment_times[earlier_name][1] < environment_times[later_name][0]:
                    _fail("runtime environment records are out of order")
            if not environment_times[require_environment_sequence[-1]][1] < command_times[
                "environment-tf4576"
            ][0]:
                _fail("runtime environment records must precede environment commands")

        initial_hygiene_end = command_times["repository-hygiene"][1]
        final_hygiene_start = command_times["repository-hygiene-final"][0]
        if not initial_hygiene_end < command_times["syntax-level1"][0]:
            _fail("initial repository hygiene must finish before syntax-level1 starts")
        if not command_times["p0-4-tokenizer-psi16"][1] < final_hygiene_start:
            _fail(
                "P0-4 Psi=16 tokenizer review must finish before final hygiene starts"
            )
        bracket_exclusions = {
            "environment-tf4576",
            "environment-tf5141",
            "environment-cuda0",
            "offline-cache-preflight",
            "repository-hygiene",
            "repository-hygiene-final",
        }
        for name in set(required_names) - bracket_exclusions:
            started, ended = command_times[name]
            if not initial_hygiene_end < started or not ended < final_hygiene_start:
                _fail(f"command {name} falls outside the clean-worktree hygiene bracket")
        require_sequence(
            (
                "c3-data",
                "c3-psi8-operational",
                "c3-psi8-peak-exposure",
                "c3-psi16-operational",
                "c3-psi16-peak-exposure",
            ),
            label="C3 data and CUDA lanes",
        )
        require_sequence(
            (
                "p0-3-checkpointed",
                "p0-3-tokenizer-psi8",
                "p0-3-tokenizer-psi16",
            ),
            label="P0-3 run and tokenizer reloads",
        )
        require_sequence(
            (
                "p0-4-psi8-preflight",
                "p0-4-psi8",
                "p0-4-tokenizer-psi8",
                "p0-4-review-psi8",
                "p0-4-psi16-preflight",
                "p0-4-psi16",
                "p0-4-tokenizer-psi16",
            ),
            label="strict P0-4 Psi=8 review boundary and Psi=16 run",
        )
        ordering_checks.extend(
            [
                "runtime-environment-records-before-environment-commands",
                "all-matrix-commands-inside-hygiene-bracket",
                "c3-data-and-four-lanes-ordered",
                "p0-3-run-and-tokenizer-reloads-ordered",
                "p0-4-psi8-reviewed-before-psi16",
            ]
        )
    elif set(P0_4_PSI8_REVIEW_COMMAND_NAMES) <= set(required_names):
        require_sequence(
            (
                "environment-tf4576",
                "environment-cuda0",
                "offline-cache-preflight",
                "p0-4-psi8-preflight",
                "p0-4-psi8",
                "p0-4-tokenizer-psi8",
            ),
            label="P0-4 Psi=8 focused-review environment and inputs",
        )
        if (
            "runtime-tf4576" in environment_times
            and not environment_times["runtime-tf4576"][1]
            < command_times["environment-tf4576"][0]
        ):
            _fail("runtime-tf4576 record must precede its environment command")
        ordering_checks.append("p0-4-psi8-run-and-tokenizer-ordered")

    return ({
        "status": "passed",
        "observed_command_count": len(commands),
        "reviewed_command_count": len(required_names),
        "required_command_count": len(required_names),
        "required_commands": sorted(required_names),
        "observed_environment_record_count": len(environments),
        "reviewed_environment_record_count": len(required_environment_names),
        "required_environment_records": sorted(required_environment_names),
        "tested_commit": tested_commit,
        "run_marker_sha256": hashes["runner.run_marker"],
        "record_sha256": dict(sorted(record_hashes.items())),
        "log_sha256": dict(sorted(log_hashes.items())),
        "semantic_logs": semantic_logs,
        "ordering_checks": ordering_checks,
    }, offline_cache)

def _bind_run_artifact(
    run_root: Path,
    value: str | os.PathLike[str],
    relative: str,
    *,
    label: str,
) -> Path:
    actual = _absolute_path(value, label=label)
    expected = (run_root / relative).resolve(strict=False)
    if actual != expected:
        _fail(f"{label} must use the fixed recorder run-root layout: {relative}")
    return actual


def _full_runner_contract(
    *,
    command_ledger: str | os.PathLike[str],
    tested_commit: str,
    p0_3_root: str | os.PathLike[str],
    p0_3_stdout: str | os.PathLike[str],
    p0_4_psi8_root: str | os.PathLike[str],
    p0_4_psi16_root: str | os.PathLike[str],
    p0_4_psi8_review: str | os.PathLike[str],
    c3_data_root: str | os.PathLike[str],
    c3_roots: Mapping[str, str | os.PathLike[str]],
    tokenizer_reports: Mapping[str, str | os.PathLike[str]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, tuple[Path, ...]],
    dict[str, Path],
]:
    ledger_path = _safe_file(command_ledger, label="command ledger")
    run_root = _safe_root(ledger_path.parent, label="recorder run root")
    p03 = _bind_run_artifact(
        run_root, p0_3_root, "artifacts/p0-3", label="P0-3 root"
    )
    p03_stdout = _bind_run_artifact(
        run_root,
        p0_3_stdout,
        "logs/p0-3-checkpointed.log",
        label="P0-3 lossless stdout",
    )
    p04_8 = _bind_run_artifact(
        run_root, p0_4_psi8_root, "artifacts/p0-4/psi8", label="P0-4 Psi=8 root"
    )
    p04_16 = _bind_run_artifact(
        run_root,
        p0_4_psi16_root,
        "artifacts/p0-4/psi16",
        label="P0-4 Psi=16 root",
    )
    p04_review = _bind_run_artifact(
        run_root,
        p0_4_psi8_review,
        "artifacts/p0-4/psi8/raw-review.json",
        label="P0-4 Psi=8 focused review",
    )
    c3_data = _bind_run_artifact(
        run_root, c3_data_root, "artifacts/c3/data", label="C3 data root"
    )
    c3_expected = {
        "c3_psi8_operational": "artifacts/c3/cuda/psi8/operational",
        "c3_psi8_peak_exposure": "artifacts/c3/cuda/psi8/peak-exposure",
        "c3_psi16_operational": "artifacts/c3/cuda/psi16/operational",
        "c3_psi16_peak_exposure": "artifacts/c3/cuda/psi16/peak-exposure",
    }
    if set(c3_roots) != set(c3_expected):
        _fail("C3 root mapping differs from the fixed four-lane matrix")
    c3_paths = {
        name: _bind_run_artifact(
            run_root, c3_roots[name], relative, label=f"C3 {name} root"
        )
        for name, relative in c3_expected.items()
    }
    tokenizer_expected = {
        "p0_3_psi8": "artifacts/p0-3/tokenizer-reload-psi8.json",
        "p0_3_psi16": "artifacts/p0-3/tokenizer-reload-psi16.json",
        "p0_4_psi8": "artifacts/p0-4/psi8/tokenizer-reload.json",
        "p0_4_psi16": "artifacts/p0-4/psi16/tokenizer-reload.json",
    }
    if set(tokenizer_reports) != set(tokenizer_expected):
        _fail("tokenizer report mapping differs from the fixed four-report matrix")
    tokenizer_paths = {
        name: _bind_run_artifact(
            run_root,
            tokenizer_reports[name],
            relative,
            label=f"tokenizer report {name}",
        )
        for name, relative in tokenizer_expected.items()
    }

    expected_options: dict[str, dict[str, str]] = {
        "c3-data": {"--mode": "data", "--output-dir": os.fspath(c3_data)},
        "p0-3-checkpointed": {
            "--output-dir": os.fspath(p03),
            "--log-every": "1",
            "--revision": P0_3_DATASET_REVISION,
        },
        "p0-3-tokenizer-psi8": {
            "--logical-name": "p0_3_psi8",
            "--checkpoint": os.fspath(p03 / "psi8"),
            "--output": os.fspath(tokenizer_paths["p0_3_psi8"]),
            "--source-id": "tinystories-spm768",
            "--checkpoint-id": "p0-3-psi8-checkpoint",
        },
        "p0-3-tokenizer-psi16": {
            "--logical-name": "p0_3_psi16",
            "--checkpoint": os.fspath(p03 / "psi16"),
            "--output": os.fspath(tokenizer_paths["p0_3_psi16"]),
            "--source-id": "tinystories-spm768",
            "--checkpoint-id": "p0-3-psi16-checkpoint",
        },
        "p0-4-psi8": {"--output-dir": os.fspath(p04_8)},
        "p0-4-tokenizer-psi8": {
            "--logical-name": "p0_4_psi8",
            "--checkpoint": os.fspath(p04_8 / "checkpoint"),
            "--output": os.fspath(tokenizer_paths["p0_4_psi8"]),
            "--source-id": "gpt2",
            "--checkpoint-id": "p0-4-psi8-checkpoint",
        },
        "p0-4-review-psi8": {
            "--mode": "p0-4-lane",
            "--tested-commit": tested_commit,
            "--p0-4-root": os.fspath(p04_8),
            "--tokenizer-reload-report": f"p0_4_psi8={tokenizer_paths['p0_4_psi8']}",
            "--command-ledger": os.fspath(ledger_path),
            "--output": os.fspath(p04_review),
        },
        "p0-4-psi16": {"--output-dir": os.fspath(p04_16)},
        "p0-4-tokenizer-psi16": {
            "--logical-name": "p0_4_psi16",
            "--checkpoint": os.fspath(p04_16 / "checkpoint"),
            "--output": os.fspath(tokenizer_paths["p0_4_psi16"]),
            "--source-id": "gpt2",
            "--checkpoint-id": "p0-4-psi16-checkpoint",
        },
    }
    for logical, psi, mode in C3_LANES:
        expected_options[logical.replace("_", "-")] = {
            "--mode": mode,
            "--psi": str(psi),
            "--output-dir": os.fspath(c3_paths[logical]),
        }

    expected_absent_paths: dict[str, tuple[Path, ...]] = {
        "syntax-level1": (run_root / "pycache/syntax-level1",),
        "c3-data": (c3_data,),
        "p0-3-checkpointed": (p03,),
        "p0-3-tokenizer-psi8": (tokenizer_paths["p0_3_psi8"],),
        "p0-3-tokenizer-psi16": (tokenizer_paths["p0_3_psi16"],),
        "p0-4-psi8": (p04_8,),
        "p0-4-tokenizer-psi8": (tokenizer_paths["p0_4_psi8"],),
        "p0-4-review-psi8": (p04_review,),
        "p0-4-psi16": (p04_16,),
        "p0-4-tokenizer-psi16": (tokenizer_paths["p0_4_psi16"],),
    }
    for logical, _, _ in C3_LANES:
        expected_absent_paths[logical.replace("_", "-")] = (c3_paths[logical],)
    return expected_options, expected_absent_paths, {
        "p0-3-checkpointed": p03_stdout
    }


def _p0_4_lane_runner_contract(
    *,
    command_ledger: str | os.PathLike[str],
    p0_4_root: str | os.PathLike[str],
    tokenizer_report: str | os.PathLike[str],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, tuple[Path, ...]],
    dict[str, Path],
]:
    ledger_path = _safe_file(command_ledger, label="command ledger")
    run_root = _safe_root(ledger_path.parent, label="recorder run root")
    p04 = _bind_run_artifact(
        run_root, p0_4_root, "artifacts/p0-4/psi8", label="P0-4 Psi=8 root"
    )
    report = _bind_run_artifact(
        run_root,
        tokenizer_report,
        "artifacts/p0-4/psi8/tokenizer-reload.json",
        label="P0-4 Psi=8 tokenizer report",
    )
    return (
        {
            "p0-4-psi8": {"--output-dir": os.fspath(p04)},
            "p0-4-tokenizer-psi8": {
                "--logical-name": "p0_4_psi8",
                "--checkpoint": os.fspath(p04 / "checkpoint"),
                "--output": os.fspath(report),
                "--source-id": "gpt2",
                "--checkpoint-id": "p0-4-psi8-checkpoint",
            },
        },
        {
            "p0-4-psi8": (p04,),
            "p0-4-tokenizer-psi8": (report,),
        },
        {},
    )


def review_p0_4_lane_inputs(
    *,
    p0_4_root: str | os.PathLike[str],
    tokenizer_reports: Mapping[str, str | os.PathLike[str]],
    command_ledger: str | os.PathLike[str],
    tested_commit: str,
) -> dict[str, Any]:
    """Review the fresh Psi=8 lane before any Psi=16 command may start."""

    if set(tokenizer_reports) != {"p0_4_psi8"}:
        _fail("focused P0-4 review requires exactly p0_4_psi8 tokenizer evidence")
    hashes: dict[str, str] = {}
    expected_options, expected_absent, expected_logs = _p0_4_lane_runner_contract(
        command_ledger=command_ledger,
        p0_4_root=p0_4_root,
        tokenizer_report=tokenizer_reports["p0_4_psi8"],
    )
    ledger, expected_cache = _review_command_ledger(
        command_ledger,
        required_names=P0_4_PSI8_REVIEW_COMMAND_NAMES,
        required_environment_names=P0_4_PSI8_REVIEW_ENVIRONMENT_NAMES,
        tested_commit=tested_commit,
        expected_options=expected_options,
        expected_absent_paths=expected_absent,
        expected_log_paths=expected_logs,
        bind_ledger_hashes=False,
        allow_extra_records=True,
        hashes=hashes,
    )
    p0_4 = _review_p0_4_lane(
        p0_4_root,
        psi=8,
        expected_cache=expected_cache,
        hashes=hashes,
    )
    tokenizer = _review_tokenizer_reports(
        tokenizer_reports,
        hashes=hashes,
        required_names=("p0_4_psi8",),
    )
    p0_4["cross_bindings"] = _review_p0_4_cross_bindings(
        p0_4_runs=[p0_4],
        tokenizer=tokenizer,
        ledger=ledger,
    )
    artifact_hashes = dict(sorted(hashes.items()))
    material = {
        "artifact_hashes": artifact_hashes,
        "p0_4": p0_4,
        "tested_commit": tested_commit,
        "tokenizer_reload": tokenizer,
    }
    report = {
        "schema_version": P0_4_LANE_SCHEMA_VERSION,
        "mode": "p0-4-lane",
        "status": "passed",
        "psi": 8,
        "tested_commit": tested_commit,
        "p0_4": p0_4,
        "tokenizer_reload": tokenizer,
        "command_ledger": ledger,
        "aggregate": {
            "artifact_count": len(artifact_hashes),
            "artifact_hashes": artifact_hashes,
            "review_material_sha256": _sha256_bytes(_canonical_bytes(material)),
        },
    }
    _validate_finite_tree(report, label="P0-4 focused review report")
    return report


def _review_p0_4_focused_report(
    report_value: str | os.PathLike[str],
    *,
    tested_commit: str,
    expected_p0_4: Mapping[str, Any],
    expected_tokenizer_report: Mapping[str, Any],
    current_hashes: Mapping[str, str],
    hashes: dict[str, str],
) -> dict[str, Any]:
    path = _safe_file(report_value, label="P0-4 Psi=8 focused review")
    report = _mapping(
        _load_json(path, label="p0_4_psi8.focused_review", hashes=hashes),
        label="P0-4 Psi=8 focused review",
    )
    if set(report) != {
        "aggregate",
        "command_ledger",
        "mode",
        "p0_4",
        "psi",
        "schema_version",
        "status",
        "tested_commit",
        "tokenizer_reload",
    }:
        _fail("P0-4 Psi=8 focused review fields are incomplete or ambiguous")
    _exact_values(
        report,
        {
            "schema_version": P0_4_LANE_SCHEMA_VERSION,
            "mode": "p0-4-lane",
            "status": "passed",
            "psi": 8,
            "tested_commit": tested_commit,
        },
        label="P0-4 Psi=8 focused review",
    )
    if _at(report, "p0_4") != expected_p0_4:
        _fail("P0-4 Psi=8 focused review lane summary differs from raw evidence")
    if _at(report, "tokenizer_reload") != expected_tokenizer_report:
        _fail("P0-4 Psi=8 focused tokenizer summary differs from raw evidence")
    ledger = _mapping(_at(report, "command_ledger"), label="focused command ledger")
    _equals(_at(ledger, "status"), "passed", label="focused ledger.status")
    _equals(_at(ledger, "tested_commit"), tested_commit, label="focused ledger.tested_commit")
    if set(_at(ledger, "required_commands")) != set(P0_4_PSI8_REVIEW_COMMAND_NAMES):
        _fail("focused ledger required commands differ from the fixed Psi=8 matrix")
    if set(_at(ledger, "required_environment_records")) != set(
        P0_4_PSI8_REVIEW_ENVIRONMENT_NAMES
    ):
        _fail("focused ledger environment records differ from the fixed Psi=8 matrix")

    aggregate = _mapping(_at(report, "aggregate"), label="focused aggregate")
    if set(aggregate) != {
        "artifact_count",
        "artifact_hashes",
        "review_material_sha256",
    }:
        _fail("P0-4 Psi=8 focused aggregate fields are incomplete or ambiguous")
    focused_hashes = _mapping(_at(aggregate, "artifact_hashes"), label="focused hashes")
    allowed_labels = {
        "p0_4_psi8.completion_marker",
        "p0_4_psi8.data_contract",
        "p0_4_psi8.metrics",
        "p0_4_psi8.summary",
        "tokenizer_reload.p0_4_psi8",
        "runner.run_marker",
        *(f"runner.log.{name}" for name in P0_4_PSI8_REVIEW_COMMAND_NAMES),
        *(
            f"runner.record.{name}"
            for name in (
                *P0_4_PSI8_REVIEW_COMMAND_NAMES,
                *P0_4_PSI8_REVIEW_ENVIRONMENT_NAMES,
            )
        ),
    }
    if set(focused_hashes) != allowed_labels:
        _fail("P0-4 Psi=8 focused review artifact set is incomplete or ambiguous")
    for label in sorted(allowed_labels):
        digest = focused_hashes[label]
        if not isinstance(digest, str) or HEX64_RE.fullmatch(digest) is None:
            _fail("focused review contains an invalid artifact digest")
        if current_hashes.get(label) != digest:
            _fail(f"P0-4 Psi=8 focused artifact changed after review: {label}")
    _exact_int(
        _at(aggregate, "artifact_count"),
        label="focused aggregate.artifact_count",
        expected=len(allowed_labels),
    )
    material = {
        "artifact_hashes": dict(focused_hashes),
        "p0_4": _at(report, "p0_4"),
        "tested_commit": tested_commit,
        "tokenizer_reload": _at(report, "tokenizer_reload"),
    }
    expected_digest = _sha256_bytes(_canonical_bytes(material))
    _equals(
        _at(aggregate, "review_material_sha256"),
        expected_digest,
        label="focused aggregate.review_material_sha256",
    )
    return {
        "status": "passed",
        "psi": 8,
        "artifact_count": len(allowed_labels),
        "review_material_sha256": expected_digest,
    }


def _review_p0_3_cross_bindings(
    *,
    p0_3: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    data_contract = _mapping(
        _at(p0_3, "data_contract"), label="P0-3 reviewed data contract"
    )
    tokenizer_projection = _mapping(
        _at(data_contract, "tokenizer_projection"),
        label="P0-3 data-contract tokenizer projection",
    )
    tokenizer_reports = _list(
        _at(tokenizer, "reports"), label="reviewed tokenizer reports"
    )
    p0_3_reports = {
        _at(report, "logical_name"): _mapping(
            report, label="P0-3 tokenizer reload projection"
        )
        for report in tokenizer_reports
        if isinstance(report, Mapping)
        and report.get("logical_name") in {"p0_3_psi8", "p0_3_psi16"}
    }
    if set(p0_3_reports) != {"p0_3_psi8", "p0_3_psi16"}:
        _fail("P0-3 tokenizer reload projections are missing or ambiguous")
    for logical in ("p0_3_psi8", "p0_3_psi16"):
        source_projection = _mapping(
            _at(p0_3_reports[logical], "source_projection"),
            label=f"{logical} source projection",
        )
        if source_projection != tokenizer_projection:
            _fail(
                f"{logical} source tokenizer projection differs from the P0-3 "
                "data contract"
            )

    semantic_logs = _mapping(
        _at(ledger, "semantic_logs"), label="command-ledger semantic logs"
    )
    offline_cache = _mapping(
        _at(semantic_logs, "offline_cache"), label="offline-cache semantic review"
    )
    offline_checks = _mapping(
        _at(offline_cache, "checks"), label="offline-cache semantic checks"
    )
    offline_p0_3 = _mapping(
        _at(offline_checks, "p0_3_tinystories"),
        label="offline-cache P0-3 TinyStories identity",
    )
    offline_fingerprint_sha256 = _nonempty_string(
        _at(offline_p0_3, "fingerprint_sha256"),
        label="offline-cache P0-3 fingerprint_sha256",
    )
    contract_fingerprint_sha256 = _nonempty_string(
        _at(data_contract, "dataset_fingerprint_sha256"),
        label="P0-3 data-contract dataset_fingerprint_sha256",
    )
    if offline_fingerprint_sha256 != contract_fingerprint_sha256:
        _fail(
            "offline-cache P0-3 fingerprint does not bind the actual P0-3 "
            "data contract"
        )
    projection_sha256 = _sha256_bytes(_canonical_bytes(tokenizer_projection))
    return {
        "dataset_fingerprint_sha256": contract_fingerprint_sha256,
        "offline_cache_fingerprint_match": True,
        "stdout_data_contract_sha256": _at(data_contract, "sha256"),
        "tokenizer_projection_sha256": projection_sha256,
        "tokenizer_reload_source_matches": ["p0_3_psi8", "p0_3_psi16"],
    }


def _review_p0_4_cross_bindings(
    *,
    p0_4_runs: Sequence[Mapping[str, Any]],
    tokenizer: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    expected_logicals = [f"p0_4_psi{_at(run, 'psi')}" for run in p0_4_runs]
    if expected_logicals not in (
        ["p0_4_psi8"],
        ["p0_4_psi16"],
        ["p0_4_psi8", "p0_4_psi16"],
    ):
        _fail("P0-4 cross-binding received an invalid lane set")
    contracts = [
        _mapping(_at(run, "data_contract"), label=f"{logical}.reviewed data contract")
        for logical, run in zip(expected_logicals, p0_4_runs, strict=True)
    ]
    if len(contracts) == 2 and contracts[0] != contracts[1]:
        _fail("P0-4 Psi=8 and Psi=16 canonical data contracts differ")
    reference = _mapping(
        _at(contracts[0], "reference"), label="P0-4 reviewed data-contract reference"
    )
    contract_sha256 = _nonempty_string(
        _at(reference, "sha256"), label="P0-4 reviewed data-contract SHA-256"
    )
    if HEX64_RE.fullmatch(contract_sha256) is None:
        _fail("P0-4 reviewed data-contract SHA-256 is invalid")

    reviewed_reports = {
        _at(report, "logical_name"): _mapping(
            _at(report, "source_projection"),
            label=f"{_at(report, 'logical_name')} source projection",
        )
        for report in _list(
            _at(tokenizer, "reports"), label="P0-4 tokenizer reload reports"
        )
        if _at(report, "logical_name") in expected_logicals
    }
    if set(reviewed_reports) != set(expected_logicals):
        _fail("P0-4 tokenizer reload projection set is incomplete")
    contract_projection = _mapping(
        _at(contracts[0], "tokenizer_projection"),
        label="P0-4 data-contract tokenizer projection",
    )
    for logical in expected_logicals:
        if reviewed_reports[logical] != contract_projection:
            _fail(
                f"{logical} source tokenizer projection differs from the "
                "P0-4 data contract"
            )

    semantic_logs = _mapping(
        _at(ledger, "semantic_logs"), label="P0-4 command-ledger semantic logs"
    )
    stdout = _mapping(_at(semantic_logs, "p0_4"), label="P0-4 stdout semantics")
    if not set(expected_logicals).issubset(set(stdout)):
        _fail("P0-4 stdout semantic lane set is incomplete or ambiguous")
    for logical in expected_logicals:
        if _at(stdout[logical], "data_contract_sha256") != contract_sha256:
            _fail(f"{logical} stdout data-contract digest differs from its file")

    offline_cache = _mapping(
        _at(semantic_logs, "offline_cache"), label="P0-4 offline-cache semantics"
    )
    offline_checks = _mapping(
        _at(offline_cache, "checks"), label="P0-4 offline-cache checks"
    )
    offline_dataset = _mapping(
        _at(offline_checks, "p0_4_tinystories"),
        label="P0-4 offline TinyStories identity",
    )
    if _at(offline_dataset, "fingerprint_sha256") != _at(
        contracts[0], "dataset_fingerprint_sha256"
    ):
        _fail("P0-4 offline dataset fingerprint differs from the data contract")
    offline_tokenizer = _mapping(
        _at(offline_checks, "p0_4_gpt2_tokenizer"),
        label="P0-4 offline tokenizer identity",
    )
    if _at(offline_tokenizer, "identity_projection") != contract_projection:
        _fail("P0-4 offline tokenizer projection differs from the data contract")
    return {
        "data_contract_sha256": contract_sha256,
        "dataset_fingerprint_sha256": _at(
            contracts[0], "dataset_fingerprint_sha256"
        ),
        "offline_cache_fingerprint_match": True,
        "offline_tokenizer_projection_match": True,
        "stdout_reference_matches": list(expected_logicals),
        "tokenizer_reload_source_matches": list(expected_logicals),
    }


def _review_c3_cross_bindings(
    *, c3_data: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    reviewed_row_manifest = _nonempty_string(
        _at(c3_data, "row_manifest_sha256"),
        label="reviewed C3 data row_manifest_sha256",
    )
    semantic_logs = _mapping(
        _at(ledger, "semantic_logs"), label="C3 command-ledger semantic logs"
    )
    offline_cache = _mapping(
        _at(semantic_logs, "offline_cache"), label="C3 offline-cache semantics"
    )
    offline_c3 = _mapping(
        _at(offline_cache, "checks.p0_5_c3"), label="offline-cache C3 identity"
    )
    offline_row_manifest = _nonempty_string(
        _at(offline_c3, "row_manifest_sha256"),
        label="offline-cache C3 row_manifest_sha256",
    )
    if not HEX64_RE.fullmatch(offline_row_manifest):
        _fail("offline-cache C3 row manifest must be a lowercase SHA-256")
    if offline_row_manifest != reviewed_row_manifest:
        _fail("offline-cache C3 row manifest differs from the reviewed data contract")
    return {
        "offline_cache_row_manifest_match": True,
        "row_manifest_sha256": reviewed_row_manifest,
    }


def review_inputs(
    *,
    p0_3_root: str | os.PathLike[str],
    p0_3_stdout: str | os.PathLike[str],
    p0_4_psi8_root: str | os.PathLike[str],
    p0_4_psi16_root: str | os.PathLike[str],
    p0_4_psi8_review: str | os.PathLike[str],
    c3_data_root: str | os.PathLike[str],
    c3_psi8_operational_root: str | os.PathLike[str],
    c3_psi8_peak_exposure_root: str | os.PathLike[str],
    c3_psi16_operational_root: str | os.PathLike[str],
    c3_psi16_peak_exposure_root: str | os.PathLike[str],
    tokenizer_reports: Mapping[str, str | os.PathLike[str]],
    command_ledger: str | os.PathLike[str],
    tested_commit: str,
) -> dict[str, Any]:
    """Review all Stage 5 raw inputs and return a deterministic report."""

    c3_values = {
        "c3_psi8_operational": c3_psi8_operational_root,
        "c3_psi8_peak_exposure": c3_psi8_peak_exposure_root,
        "c3_psi16_operational": c3_psi16_operational_root,
        "c3_psi16_peak_exposure": c3_psi16_peak_exposure_root,
    }
    expected_options, expected_absent, expected_logs = _full_runner_contract(
        command_ledger=command_ledger,
        tested_commit=tested_commit,
        p0_3_root=p0_3_root,
        p0_3_stdout=p0_3_stdout,
        p0_4_psi8_root=p0_4_psi8_root,
        p0_4_psi16_root=p0_4_psi16_root,
        p0_4_psi8_review=p0_4_psi8_review,
        c3_data_root=c3_data_root,
        c3_roots=c3_values,
        tokenizer_reports=tokenizer_reports,
    )

    hashes: dict[str, str] = {}
    ledger, expected_cache = _review_command_ledger(
        command_ledger,
        required_names=REQUIRED_COMMAND_NAMES,
        required_environment_names=REQUIRED_ENVIRONMENT_NAMES,
        tested_commit=tested_commit,
        expected_options=expected_options,
        expected_absent_paths=expected_absent,
        expected_log_paths=expected_logs,
        bind_ledger_hashes=True,
        allow_extra_records=False,
        hashes=hashes,
    )
    p0_3 = _review_p0_3(p0_3_root, p0_3_stdout, hashes=hashes)
    p0_4_runs = [
        _review_p0_4_lane(
            p0_4_psi8_root,
            psi=8,
            expected_cache=expected_cache,
            hashes=hashes,
        ),
        _review_p0_4_lane(
            p0_4_psi16_root,
            psi=16,
            expected_cache=expected_cache,
            hashes=hashes,
        ),
    ]
    if not _parse_utc(
        p0_4_runs[0]["timestamp_utc"], label="P0-4 Psi=8 timestamp"
    ) < _parse_utc(
        p0_4_runs[1]["timestamp_utc"], label="P0-4 Psi=16 timestamp"
    ):
        _fail("P0-4 runs were not executed in required Psi=8 then Psi=16 order")

    c3_contract, c3_data = _review_c3_data(c3_data_root, hashes=hashes)
    c3_runs = [
        _review_c3_lane(
            c3_values[logical],
            logical=logical,
            psi=psi,
            mode=mode,
            expected_data=c3_contract,
            hashes=hashes,
        )
        for logical, psi, mode in C3_LANES
    ]
    c3_times = [
        _parse_utc(item["timestamp_utc"], label=f"{item['logical_name']} timestamp")
        for item in c3_runs
    ]
    if any(later <= earlier for earlier, later in zip(c3_times, c3_times[1:])):
        _fail("C3 lanes were not executed in required Psi=8 then Psi=16 order")
    c3_cross_bindings = _review_c3_cross_bindings(
        c3_data=c3_data, ledger=ledger
    )

    tokenizer = _review_tokenizer_reports(tokenizer_reports, hashes=hashes)
    for p0_4_run in p0_4_runs:
        p0_4_run["cross_bindings"] = _review_p0_4_cross_bindings(
            p0_4_runs=[p0_4_run],
            tokenizer=tokenizer,
            ledger=ledger,
        )
    p0_4_cross_bindings = _review_p0_4_cross_bindings(
        p0_4_runs=p0_4_runs, tokenizer=tokenizer, ledger=ledger
    )
    p0_3["cross_bindings"] = _review_p0_3_cross_bindings(
        p0_3=p0_3,
        tokenizer=tokenizer,
        ledger=ledger,
    )
    expected_focused_tokenizer = {
        "status": "passed",
        "report_count": 1,
        "reports": [
            next(
                item
                for item in tokenizer["reports"]
                if item["logical_name"] == "p0_4_psi8"
            )
        ],
    }
    focused_review = _review_p0_4_focused_report(
        p0_4_psi8_review,
        tested_commit=tested_commit,
        expected_p0_4=p0_4_runs[0],
        expected_tokenizer_report=expected_focused_tokenizer,
        current_hashes=hashes,
        hashes=hashes,
    )
    raw_event_counts = {
        "p0_3_stdout_step_events": p0_3["stdout_step_event_count"],
        "p0_4_jsonl_events": sum(item["event_count"] for item in p0_4_runs),
        "c3_jsonl_events": sum(item["event_count"] for item in c3_runs),
    }
    raw_event_counts["total"] = sum(raw_event_counts.values())
    aggregate_material = {
        "artifact_hashes": dict(sorted(hashes.items())),
        "raw_event_counts": raw_event_counts,
        "tested_commit": tested_commit,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "tested_commit": tested_commit,
        "p0_3": p0_3,
        "p0_4": {
            "status": "passed",
            "cross_bindings": p0_4_cross_bindings,
            "focused_psi8_review": focused_review,
            "runs": p0_4_runs,
        },
        "p0_5_c3": {
            "status": "passed",
            "cross_bindings": c3_cross_bindings,
            "data": c3_data,
            "runs": c3_runs,
        },
        "tokenizer_reload": tokenizer,
        "command_ledger": ledger,
        "aggregate": {
            **aggregate_material,
            "artifact_count": len(hashes),
            "review_material_sha256": _sha256_bytes(_canonical_bytes(aggregate_material)),
        },
    }
    _validate_finite_tree(report, label="review report")
    return report


def _parse_named_paths(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            _fail("--tokenizer-reload-report must use LOGICAL_NAME=ABSOLUTE_PATH")
        name, path = value.split("=", 1)
        if not name or not path:
            _fail("--tokenizer-reload-report must use non-empty name and path")
        if name in result:
            _fail(f"duplicate tokenizer report logical name: {name}")
        result[name] = path
    return result


def _git_capture(
    repository: Path,
    arguments: Sequence[str],
    *,
    label: str,
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ReviewError(f"{label} could not execute Git: {exc}") from exc
    if result.returncode not in accepted_returncodes:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        _fail(f"{label} failed with return code {result.returncode}: {stderr}")
    return result


def _verify_live_reviewer_checkout(
    tested_commit: str,
    *,
    repository: Path | None = None,
    reviewer_path: Path | None = None,
) -> None:
    """Fail closed if the executing full reviewer checkout is not tested and clean."""

    if COMMIT_RE.fullmatch(tested_commit) is None:
        _fail("tested commit must be a full lowercase 40- or 64-hex identifier")
    expected_repository = (
        Path(__file__).resolve().parents[1] if repository is None else repository
    )
    checkout = _absolute_path(
        os.fspath(expected_repository), label="live reviewer checkout"
    )
    if not checkout.is_dir():
        _fail("live reviewer checkout is not a directory")
    source = _absolute_path(
        os.fspath(
            Path(__file__).resolve()
            if reviewer_path is None
            else reviewer_path
        ),
        label="live reviewer source",
    )
    expected_source = checkout / "scripts/review_level1_requalification.py"
    if source != expected_source or not source.is_file():
        _fail("executing reviewer source is not the fixed tracked repository path")

    top_level = _git_capture(
        checkout, ("rev-parse", "--show-toplevel"), label="live reviewer top-level"
    ).stdout
    if top_level != os.fsencode(checkout) + b"\n":
        _fail("live reviewer checkout differs from Git's canonical top-level")
    head = _git_capture(
        checkout, ("rev-parse", "--verify", "HEAD^{commit}"), label="live reviewer HEAD"
    ).stdout
    if head != tested_commit.encode("ascii") + b"\n":
        _fail("live reviewer checkout HEAD differs from the tested commit")

    porcelain = _git_capture(
        checkout,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        label="live reviewer worktree status",
    ).stdout
    if porcelain != b"":
        _fail("live reviewer checkout is not exactly clean")
    for label, arguments in (
        ("live reviewer worktree diff check", ("diff", "--no-ext-diff", "--check")),
        (
            "live reviewer index diff check",
            ("diff", "--cached", "--no-ext-diff", "--check"),
        ),
    ):
        result = _git_capture(checkout, arguments, label=label)
        if result.stdout != b"" or result.stderr != b"":
            _fail(f"{label} produced unexpected output")

    relative_source = "scripts/review_level1_requalification.py"
    tracked = _git_capture(
        checkout,
        ("ls-files", "--error-unmatch", "--", relative_source),
        label="live reviewer tracked-source check",
    ).stdout
    if tracked != relative_source.encode("ascii") + b"\n":
        _fail("live reviewer source is not tracked at the fixed path")
    committed_source = _git_capture(
        checkout,
        ("cat-file", "blob", f"{tested_commit}:{relative_source}"),
        label="live reviewer committed-source read",
    ).stdout
    actual_source = _stable_read_bytes(source, label="live reviewer source")
    if actual_source != committed_source:
        _fail("executing reviewer source bytes differ from the tested commit blob")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "p0-4-lane"), default="full")
    parser.add_argument("--tested-commit", required=True)
    parser.add_argument("--command-ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--p0-4-root")
    parser.add_argument("--p0-3-root")
    parser.add_argument("--p0-3-stdout")
    parser.add_argument("--p0-4-psi8-root")
    parser.add_argument("--p0-4-psi16-root")
    parser.add_argument("--p0-4-psi8-review")
    parser.add_argument("--c3-data-root")
    parser.add_argument("--c3-psi8-operational-root")
    parser.add_argument("--c3-psi8-peak-exposure-root")
    parser.add_argument("--c3-psi16-operational-root")
    parser.add_argument("--c3-psi16-peak-exposure-root")
    parser.add_argument(
        "--tokenizer-reload-report",
        action="append",
        default=[],
        metavar="LOGICAL_NAME=PATH",
    )
    return parser.parse_args(argv)


def _required_cli_value(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if not isinstance(value, str) or not value:
        _fail(f"--{name.replace('_', '-')} is required in {args.mode} mode")
    return value


def _exclusive_output(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ReviewError(f"refusing to overwrite existing review output: {path}") from exc
    except OSError as exc:
        raise ReviewError(f"could not create review output exclusively: {exc}") from exc
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(raw)
        while view:
            try:
                written = os.write(fd, view)
            except InterruptedError:
                continue
            if written <= 0:
                raise ReviewError("review output write made no progress")
            view = view[written:]
        if os.fstat(fd).st_nlink != 1:
            _fail("review output must not have hard links")
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise ReviewError(f"could not open review output directory safely: {exc}") from exc
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _verify_python_safety_flags(flags: Any | None = None) -> None:
    observed = sys.flags if flags is None else flags
    required = {
        "safe_path": bool(getattr(observed, "safe_path", False)),
        "no_site": bool(getattr(observed, "no_site", False)),
        "dont_write_bytecode": bool(getattr(observed, "dont_write_bytecode", False)),
    }
    missing = sorted(name for name, enabled in required.items() if not enabled)
    if missing:
        _fail(
            "reviewer must run under exact Python safety flags -P -S -B; "
            f"missing={missing}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    _verify_python_safety_flags()
    args = parse_args(argv)
    output = _absolute_path(args.output, label="review output")
    if not output.parent.is_dir():
        _fail(f"review output parent does not exist: {output.parent}")
    tokenizer_reports = _parse_named_paths(args.tokenizer_reload_report)
    _verify_live_reviewer_checkout(args.tested_commit)
    if args.mode == "p0-4-lane":
        report = review_p0_4_lane_inputs(
            p0_4_root=_required_cli_value(args, "p0_4_root"),
            tokenizer_reports=tokenizer_reports,
            command_ledger=args.command_ledger,
            tested_commit=args.tested_commit,
        )
    else:
        report = review_inputs(
            p0_3_root=_required_cli_value(args, "p0_3_root"),
            p0_3_stdout=_required_cli_value(args, "p0_3_stdout"),
            p0_4_psi8_root=_required_cli_value(args, "p0_4_psi8_root"),
            p0_4_psi16_root=_required_cli_value(args, "p0_4_psi16_root"),
            p0_4_psi8_review=_required_cli_value(args, "p0_4_psi8_review"),
            c3_data_root=_required_cli_value(args, "c3_data_root"),
            c3_psi8_operational_root=_required_cli_value(args, "c3_psi8_operational_root"),
            c3_psi8_peak_exposure_root=_required_cli_value(args, "c3_psi8_peak_exposure_root"),
            c3_psi16_operational_root=_required_cli_value(args, "c3_psi16_operational_root"),
            c3_psi16_peak_exposure_root=_required_cli_value(args, "c3_psi16_peak_exposure_root"),
            tokenizer_reports=tokenizer_reports,
            command_ledger=args.command_ledger,
            tested_commit=args.tested_commit,
        )
    _verify_live_reviewer_checkout(args.tested_commit)
    raw = _pretty_canonical_bytes(report)
    _exclusive_output(output, raw)
    sys.stdout.write(
        json.dumps(
            {
                "mode": args.mode,
                "status": "passed",
                "output_sha256": _sha256_bytes(raw),
                "review_material_sha256": report["aggregate"]["review_material_sha256"],
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewError as exc:
        print(f"Level 1 evidence review failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
