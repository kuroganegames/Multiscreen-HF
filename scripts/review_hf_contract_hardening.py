#!/usr/bin/env python3
"""Independently review Stage E HF contract-hardening raw evidence.

This reviewer is a new, standard-library-only profile.  The accepted Level 1
reviewer remains unchanged.  Stage E deliberately reuses its pure parsers for
the unchanged P0-3, tokenizer, recorder, and P0-4 lane details, while enforcing
the prospective P0-4 qualification-v2 predicate and the new fixed matrix here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_LEGACY_PATH = Path(__file__).with_name("review_level1_requalification.py")
_LEGACY_SPEC = importlib.util.spec_from_file_location(
    "_multiscreen_level1_reviewer_for_stage_e", _LEGACY_PATH
)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("could not load the fixed Level 1 reviewer")
_legacy = importlib.util.module_from_spec(_LEGACY_SPEC)
_LEGACY_SPEC.loader.exec_module(_legacy)


ReviewError = _legacy.ReviewError
SCHEMA_VERSION = "multiscreen-hf-contract-hardening-raw-evidence-review-v1"
P0_4_LANE_SCHEMA_VERSION = (
    "multiscreen-hf-contract-hardening-p0-4-lane-review-v1"
)
P0_4_QUALIFICATION_SCHEMA_VERSION = "multiscreen-p0-4-qualification-v2"
OFFLINE_CACHE_SCHEMA_VERSION = (
    "multiscreen-hf-contract-hardening-offline-cache-v1"
)
IMPLEMENTATION_BASELINE = "bf8cc34cb6aa16ffeec1f609166db5efae79e9df"
WORKING_BRANCH = "validation/hf-contract-hardening-requalification"
LEGACY_REVIEWER_PATH = "scripts/review_level1_requalification.py"
LEGACY_REVIEWER_GIT_BLOB = "57f7c2e38ca4674e304436d31e4f4dc77a206a5a"
FULL_REVIEW_RELATIVE = "review/hf-contract-hardening-raw-review.json"
HISTORICAL_PATH_PREFIXES = (
    "docs/validation_results/LEVEL1_CORE_",
    "docs/validation_results/P0_4_",
    "docs/validation_results/P0_5_C3_",
)
HISTORICAL_PATHS = (
    "docs/validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json",
    "docs/validation_results/LEVEL1_CORE_EXACT_VERIFICATION.json",
    "docs/validation_results/LEVEL1_CORE_SANITIZED_VERIFICATION.json",
    "docs/validation_results/LEVEL1_CORE_SUMMARY.json",
    "docs/validation_results/LEVEL1_CORE_SUMMARY.md",
    "docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json",
    "docs/validation_results/P0_4_SUMMARY.json",
    "docs/validation_results/P0_4_SUMMARY.md",
    "docs/validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json",
    "docs/validation_results/P0_5_C3_EVIDENCE_CLOSURE.json",
    "docs/validation_results/P0_5_C3_EXACT_VERIFICATION.json",
    "docs/validation_results/P0_5_C3_SANITIZED_VERIFICATION.json",
    "docs/validation_results/P0_5_C3_SUMMARY.json",
    "docs/validation_results/P0_5_C3_SUMMARY.md",
)
STAGE_E_HERMETIC_FIXED_ENVIRONMENT = (
    "PATH=/usr/bin:/bin",
    "LANG=C.UTF-8",
    "LC_ALL=C.UTF-8",
    "TZ=UTC",
    "HF_DATASETS_DISABLE_PROGRESS_BARS=1",
    "HF_DATASETS_OFFLINE=1",
    "HF_HUB_DISABLE_PROGRESS_BARS=1",
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
P0_4_CONDITIONS = (
    "gpt2_vocab_50257",
    "context_4096",
    "cuda_device",
    "bf16_amp",
    "microbatch_size_1",
    "optimizer_steps_at_least_50",
    "gradient_checkpointing_enabled",
    "gradient_checkpointing_non_reentrant",
)
REQUIRED_ENVIRONMENT_NAMES = ("runtime-tf4576", "runtime-tf5141")

FOCUSED_TESTS = (
    ("hf-output-head", "test_hf_output_head_contract.py", 4, "cpu"),
    ("training-edge", "test_training_edge_contract.py", 10, "cpu"),
    ("gradient-checkpointing", "test_gradient_checkpointing_contract.py", 7, "cuda"),
    ("p0-4-qualification", "test_p0_4_qualification_contract.py", 11, "cpu"),
    ("generation-input", "test_generation_input_contract.py", 14, "cpu"),
    ("packed-text", "test_packed_text_contract.py", 11, "cpu"),
    ("paper-architecture", "test_paper_architecture_contract.py", 5, "cpu"),
    ("paper-initialization", "test_paper_initialization_contract.py", 3, "cpu"),
    ("mipe-position-cache", "test_mipe_position_cache_contract.py", 25, "cuda"),
    ("paper-training-contract", "test_paper_training_contract.py", 27, "cuda"),
)

_PREFIX_COMMANDS = (
    "environment-tf4576",
    "environment-tf5141",
    "environment-cuda0",
    "offline-cache-preflight",
    "repository-hygiene",
    "syntax-hardening",
    "level1-evidence-support-tests",
    "hardening-evidence-support-tests",
    "tokenizer-reload-tests-tf4576",
    "tokenizer-reload-tests-tf5141",
    "validation-evidence-tests",
    "json-validation",
    "workflow-yaml",
    "markdown-links",
)
_FOCUSED_COMMANDS = tuple(
    f"{stem}-{lane}"
    for stem, _filename, _count, _device in FOCUSED_TESTS
    for lane in ("tf4576", "tf5141")
)
_TAIL_COMMANDS = (
    "c1-manifest",
    "formula-units",
    "oracle-selfcheck",
    "oracle-smoke",
    "p0-1-cpu-fp32",
    "p0-1-cuda-bf16",
    "p0-2-cpu-fp32",
    "p0-2-cuda-bf16",
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
REQUIRED_COMMAND_NAMES = (*_PREFIX_COMMANDS, *_FOCUSED_COMMANDS, *_TAIL_COMMANDS)
if len(REQUIRED_COMMAND_NAMES) != 53 or len(set(REQUIRED_COMMAND_NAMES)) != 53:
    raise RuntimeError("internal Stage E command matrix is not exactly 53 commands")
P0_4_PSI8_REVIEW_COMMAND_NAMES = REQUIRED_COMMAND_NAMES[
    : REQUIRED_COMMAND_NAMES.index("p0-4-tokenizer-psi8") + 1
]
if len(P0_4_PSI8_REVIEW_COMMAND_NAMES) != 48:
    raise RuntimeError("internal Stage E focused prefix is not exactly 48 commands")

_FOCUSED_TEST_BY_COMMAND = {
    f"{stem}-{lane}": (lane, filename, count, device)
    for stem, filename, count, device in FOCUSED_TESTS
    for lane in ("tf4576", "tf5141")
}
TF5141_COMMAND_NAMES = frozenset(
    {"environment-tf5141", "tokenizer-reload-tests-tf5141"}
    | {name for name in _FOCUSED_COMMANDS if name.endswith("-tf5141")}
)
CUDA_COMMAND_NAMES = frozenset(
    {
        "environment-cuda0",
        "p0-1-cuda-bf16",
        "p0-2-cuda-bf16",
        "p0-3-checkpointed",
        "p0-4-psi8",
        "p0-4-psi16",
    }
    | {
        name
        for name, (_lane, _filename, _count, device) in _FOCUSED_TEST_BY_COMMAND.items()
        if device == "cuda"
    }
)
CPU_COMMAND_NAMES = frozenset(REQUIRED_COMMAND_NAMES) - CUDA_COMMAND_NAMES
if CPU_COMMAND_NAMES & CUDA_COMMAND_NAMES or (
    CPU_COMMAND_NAMES | CUDA_COMMAND_NAMES
) != frozenset(REQUIRED_COMMAND_NAMES):
    raise RuntimeError("internal Stage E CPU/CUDA classification is incomplete")

_PYTHONPATH_DOT_COMMANDS = frozenset(
    {
        "level1-evidence-support-tests",
        "hardening-evidence-support-tests",
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
        "p0-4-review-psi8",
    }
)
_PYTHONPATH_ORACLE_COMMANDS = frozenset(
    {"formula-units", "oracle-selfcheck", "oracle-smoke"}
)
_PYTHONPATH_FULL_COMMANDS = frozenset(
    set(_FOCUSED_COMMANDS)
    | {
        "p0-1-cpu-fp32",
        "p0-1-cuda-bf16",
        "p0-2-cpu-fp32",
        "p0-2-cuda-bf16",
        "p0-3-checkpointed",
        "p0-4-psi8",
        "p0-4-psi16",
    }
)


def _fail(message: str) -> None:
    raise ReviewError(message)


def _canonical_bytes(value: Any) -> bytes:
    return _legacy._canonical_bytes(value)


def _pretty_canonical_bytes(value: Any) -> bytes:
    return _legacy._pretty_canonical_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_environment(name: str, run_root: Path) -> tuple[str, ...]:
    if name in CPU_COMMAND_NAMES:
        device = ("CUDA_VISIBLE_DEVICES=",)
    elif name in CUDA_COMMAND_NAMES:
        device = ("CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=0")
    else:  # pragma: no cover - protected by constants above
        _fail(f"unknown command environment classification: {name}")
    suffix: tuple[str, ...] = ()
    if name == "syntax-hardening":
        suffix = (
            "PYTHONPATH=.:oracle:third_party/multiscreen-pytorch",
            f"PYTHONPYCACHEPREFIX={run_root / 'pycache/syntax-hardening'}",
        )
    elif name in _PYTHONPATH_DOT_COMMANDS:
        suffix = ("PYTHONPATH=.",)
    elif name in _PYTHONPATH_ORACLE_COMMANDS:
        suffix = ("PYTHONPATH=.:oracle",)
    elif name in _PYTHONPATH_FULL_COMMANDS:
        suffix = ("PYTHONPATH=.:oracle:third_party/multiscreen-pytorch",)
    return (
        f"HOME={run_root}",
        *STAGE_E_HERMETIC_FIXED_ENVIRONMENT,
        *device,
        *suffix,
    )


def _executable_index(argv: Sequence[str], *, name: str, run_root: Path) -> int:
    if len(argv) < 3 or tuple(argv[:2]) != ("/usr/bin/env", "-i"):
        _fail(f"command {name} must start with /usr/bin/env -i")
    index = 2
    while index < len(argv) and _legacy.ENV_ASSIGNMENT_RE.fullmatch(argv[index]):
        index += 1
    if tuple(argv[2:index]) != _expected_environment(name, run_root):
        _fail(f"command {name} has a non-hermetic or incorrectly classified environment")
    if index == len(argv):
        _fail(f"command {name} has no child executable")
    return index


def _unit_test_tail(python: str, filename: str, *, suppress_site: bool = False) -> tuple[str, ...]:
    return (
        python,
        *(("-S",) if suppress_site else ()),
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        filename,
        "-v",
    )


def _expected_command_tails(
    *,
    repository: Path,
    run_root: Path,
    cache: Path,
    tf4576_python: str,
    tf5141_python: str,
    tested_commit: str,
) -> dict[str, tuple[str, ...]]:
    # Reuse the previously pinned tails for commands whose contract is unchanged.
    legacy = _legacy._expected_command_tails(
        required_names=_legacy.REQUIRED_COMMAND_NAMES,
        repository=repository,
        run_root=run_root,
        cache=cache,
        tf4576_python=tf4576_python,
        tf5141_python=tf5141_python,
    )
    tails = {name: legacy[name] for name in REQUIRED_COMMAND_NAMES if name in legacy}
    repo = os.fspath(repository)
    tails["offline-cache-preflight"] = (
        tf4576_python,
        "-P",
        "-B",
        "scripts/check_hf_contract_hardening_offline_cache.py",
        "--repo-root",
        repo,
        "--cache-dir",
        os.fspath(cache),
    )
    tails["syntax-hardening"] = (
        tf4576_python,
        "-m",
        "py_compile",
        *_legacy._tracked_python_files(repository),
    )
    tails["hardening-evidence-support-tests"] = _unit_test_tail(
        tf4576_python, "test_hf_contract_hardening_*.py", suppress_site=True
    )
    for name, (lane, filename, _count, _device) in _FOCUSED_TEST_BY_COMMAND.items():
        tails[name] = _unit_test_tail(
            tf5141_python if lane == "tf5141" else tf4576_python, filename
        )
    p04 = run_root / "artifacts/p0-4/psi8"
    tails["p0-4-review-psi8"] = (
        tf4576_python,
        "-P",
        "-S",
        "-B",
        "scripts/review_hf_contract_hardening.py",
        "--mode",
        "p0-4-lane",
        "--tested-commit",
        tested_commit,
        "--command-ledger",
        os.fspath(run_root / "commands.jsonl"),
        "--p0-4-root",
        os.fspath(p04),
        "--tokenizer-reload-report",
        f"p0_4_psi8={p04 / 'tokenizer-reload.json'}",
        "--output",
        os.fspath(p04 / "raw-review.json"),
    )
    if set(tails) != set(REQUIRED_COMMAND_NAMES):
        _fail("internal Stage E command tail set is incomplete")
    return tails


def _expected_absent_paths(name: str, run_root: Path) -> list[str]:
    values: tuple[Path, ...] = ()
    if name == "syntax-hardening":
        values = (run_root / "pycache/syntax-hardening",)
    elif name == "p0-3-checkpointed":
        values = (run_root / "artifacts/p0-3",)
    elif name in {"p0-3-tokenizer-psi8", "p0-3-tokenizer-psi16"}:
        psi = "8" if name.endswith("psi8") else "16"
        values = (run_root / f"artifacts/p0-3/tokenizer-reload-psi{psi}.json",)
    elif name in {"p0-4-psi8", "p0-4-psi16"}:
        psi = "8" if name.endswith("psi8") else "16"
        values = (run_root / f"artifacts/p0-4/psi{psi}",)
    elif name in {"p0-4-tokenizer-psi8", "p0-4-tokenizer-psi16"}:
        psi = "8" if name.endswith("psi8") else "16"
        values = (run_root / f"artifacts/p0-4/psi{psi}/tokenizer-reload.json",)
    elif name == "p0-4-review-psi8":
        values = (run_root / "artifacts/p0-4/psi8/raw-review.json",)
    return sorted(_legacy._run_relative(run_root, value, label=name) for value in values)


def _review_record_common(record: Mapping[str, Any], *, name: str) -> None:
    _legacy._equals(
        _legacy._at(record, "format_version"),
        "level1-requalification-command-record-v1",
        label=f"record[{name}].format_version",
    )
    _legacy._equals(_legacy._at(record, "name"), name, label=f"record[{name}].name")
    cwd = _legacy._mapping(_legacy._at(record, "cwd"), label=f"record[{name}].cwd")
    if cwd != {"base": "repository_root", "path": "."}:
        _fail(f"record[{name}].cwd must be the repository root")
    started = _legacy._parse_utc(
        _legacy._at(record, "started_at_utc"), label=f"record[{name}].started"
    )
    ended = _legacy._parse_utc(
        _legacy._at(record, "ended_at_utc"), label=f"record[{name}].ended"
    )
    if ended < started:
        _fail(f"record {name} ended before it started")
    duration_ns = _legacy._exact_int(
        _legacy._at(record, "duration_ns"), label=f"record[{name}].duration_ns", minimum=0
    )
    duration_seconds = _legacy._finite_number(
        _legacy._at(record, "duration_seconds"),
        label=f"record[{name}].duration_seconds",
        minimum=0,
    )
    _legacy._close(
        duration_seconds,
        round(duration_ns / 1_000_000_000, 9),
        label=f"record[{name}].duration_seconds",
        atol=5e-10,
    )
    expected_python = "3.12.10" if name == "runtime-tf5141" or name in TF5141_COMMAND_NAMES else "3.12.11"
    _legacy._review_runtime(
        _legacy._at(record, "runtime"),
        label=f"record[{name}].runtime",
        expected_python_version=expected_python,
    )


def _review_focused_unittest_log(raw: bytes, *, name: str, expected: int) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"focused test log {name} is not UTF-8") from exc
    if not raw or not raw.endswith(b"\n") or "\x00" in text or "\r" in text:
        _fail(f"focused test log {name} is not complete canonical LF text")
    matches = re.findall(r"^Ran ([0-9]+) tests? in [0-9.]+s$", text, flags=re.MULTILINE)
    if matches != [str(expected)]:
        _fail(f"focused test {name} did not run exactly {expected} tests")
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty or nonempty[-1] != "OK":
        _fail(f"focused test {name} does not end in an unqualified OK")
    forbidden = re.compile(
        r"^(?:FAILED|ERROR|SKIPPED)|\bskipped=[1-9][0-9]*\b|Traceback \(most recent call last\)",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if forbidden.search(text):
        _fail(f"focused test {name} contains a failure, error, or skip")
    return {"status": "passed", "test_count": expected}


def _review_p0_4_preflight(raw: bytes, *, name: str, repository: Path, psi: int) -> dict[str, Any]:
    value = _legacy._mapping(
        _legacy._decode_json_bytes(raw, label=name), label=name
    )
    if set(value) != {"checks", "config_dir", "psi"}:
        _fail(f"{name} fields are incomplete or ambiguous")
    _legacy._exact_int(value["psi"], label=f"{name}.psi", expected=psi)
    _legacy._equals(
        value["config_dir"],
        os.fspath(repository / f"configs/p0_4_multiscreen_psi{psi}_gpt2_ctx4096"),
        label=f"{name}.config_dir",
    )
    expected_checks = {
        "model_type_multiscreen",
        "vocab_size_50257",
        "max_position_embeddings_4096",
        "hidden_size_is_psi_squared",
        "psi_is_8_or_16",
        "layers_equal_psi",
        "heads_equal_psi",
        "tie_word_embeddings",
        "run_expected_vocab_50257",
        "run_seq_len_4096",
        "run_amp_bf16",
        "run_microbatch_1",
        "run_steps_at_least_50",
        "run_gradient_checkpointing_true",
    }
    checks = _legacy._mapping(value["checks"], label=f"{name}.checks")
    if set(checks) != expected_checks:
        _fail(f"{name} check set is incomplete or ambiguous")
    for check in sorted(expected_checks):
        _legacy._exact_bool(checks[check], label=f"{name}.{check}", expected=True)
    return {"status": "passed", "psi": psi, "check_count": len(checks)}


def _review_offline_cache_log(raw: bytes, *, repository: Path, cache: Path) -> dict[str, Any]:
    report = _legacy._canonical_command_log_object(raw, name="offline-cache-preflight")
    _legacy._review_path_free_offline_tree(
        report, repository=repository, cache=cache, label="offline-cache-preflight"
    )
    if set(report) != {"cache", "checks", "offline_environment", "schema_version", "scope", "status"}:
        _fail("offline-cache-preflight fields are incomplete or ambiguous")
    _legacy._equals(report["schema_version"], OFFLINE_CACHE_SCHEMA_VERSION, label="offline schema")
    _legacy._equals(report["status"], "passed", label="offline status")
    scope = _legacy._mapping(report["scope"], label="offline scope")
    expected_scope = {
        "fresh_p0_3": True,
        "fresh_p0_4": True,
        "fresh_p0_5_c3": False,
    }
    if set(scope) != set(expected_scope):
        _fail("offline-cache-preflight scope differs from Stage E")
    for name, expected in expected_scope.items():
        _legacy._exact_bool(scope[name], label=f"offline.scope.{name}", expected=expected)
    cache_contract = _legacy._mapping(report["cache"], label="offline cache contract")
    expected_cache_contract = {
        "explicit": True,
        "path_recorded": False,
        "single_cache": True,
    }
    if set(cache_contract) != set(expected_cache_contract):
        _fail("offline-cache-preflight cache contract differs")
    for name, expected in expected_cache_contract.items():
        _legacy._exact_bool(
            cache_contract[name], label=f"offline.cache.{name}", expected=expected
        )
    expected_offline = {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if report["offline_environment"] != expected_offline:
        _fail("offline-cache-preflight environment differs")
    checks = _legacy._mapping(report["checks"], label="offline checks")
    if set(checks) != {"p0_3_tinystories", "p0_4_gpt2_tokenizer", "p0_4_tinystories"}:
        _fail("offline-cache-preflight check set differs from Stage E")
    p03 = _legacy._review_offline_tinystories_check(
        checks["p0_3_tinystories"],
        label="offline.p0_3",
        revision=_legacy.P0_3_DATASET_REVISION,
    )
    p04_data = _legacy._review_offline_tinystories_check(
        checks["p0_4_tinystories"], label="offline.p0_4", revision="default"
    )
    tok = _legacy._mapping(checks["p0_4_gpt2_tokenizer"], label="offline tokenizer")
    if set(tok) != {"eos_token_id", "identity_projection", "repository", "revision", "use_fast", "vocab_size"}:
        _fail("offline GPT-2 tokenizer fields are incomplete")
    _legacy._exact_values(
        tok,
        {"eos_token_id": 50256, "repository": "gpt2", "revision": "default", "use_fast": True, "vocab_size": 50257},
        label="offline tokenizer",
    )
    projection = _legacy._review_p0_4_tokenizer_projection(
        tok["identity_projection"], label="offline tokenizer projection"
    )
    return {
        "schema_version": OFFLINE_CACHE_SCHEMA_VERSION,
        "status": "passed",
        "cache": dict(cache_contract),
        "offline_environment": expected_offline,
        "checks": {
            "p0_3_tinystories": p03,
            "p0_4_tinystories": p04_data,
            "p0_4_gpt2_tokenizer": {
                "repository": "gpt2",
                "revision": "default",
                "vocab_size": 50257,
                "identity_projection": projection,
            },
        },
        "scope": dict(scope),
    }


def _review_semantic_logs(
    logs: Mapping[str, bytes], *, tested_commit: str, repository: Path, cache: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    environments = {
        name: _legacy._review_environment_command_log(name, logs[name])
        for name in ("environment-tf4576", "environment-tf5141", "environment-cuda0")
        if name in logs
    }
    repository_checks = {
        name: _legacy._review_repository_command_log(name, logs[name], tested_commit=tested_commit)
        for name in ("repository-hygiene", "json-validation", "workflow-yaml", "markdown-links", "repository-hygiene-final")
        if name in logs
    }
    if (
        "repository-hygiene" in repository_checks
        and "repository-hygiene-final" in repository_checks
    ):
        initial = dict(repository_checks["repository-hygiene"])
        final = dict(repository_checks["repository-hygiene-final"])
        for value in (initial, final):
            value.pop("status", None)
        if initial != final:
            _fail("initial and final repository hygiene identities differ")
    offline = _review_offline_cache_log(logs["offline-cache-preflight"], repository=repository, cache=cache)
    p04 = {}
    for name, logical in (("p0-4-psi8", "p0_4_psi8"), ("p0-4-psi16", "p0_4_psi16")):
        if name in logs:
            p04[logical] = _legacy._review_p0_4_command_stdout(logs[name], name=name)
    preflights = {}
    for name, psi in (("p0-4-psi8-preflight", 8), ("p0-4-psi16-preflight", 16)):
        if name in logs:
            preflights[name] = _review_p0_4_preflight(logs[name], name=name, repository=repository, psi=psi)
    focused: dict[str, Any] = {}
    lane_totals = {"tf4576": 0, "tf5141": 0}
    for name, (lane, _filename, expected, _device) in _FOCUSED_TEST_BY_COMMAND.items():
        if name in logs:
            focused[name] = _review_focused_unittest_log(logs[name], name=name, expected=expected)
            lane_totals[lane] += expected
    for lane, total in lane_totals.items():
        if focused and total != 117:
            _fail(f"focused {lane} lane total must be exactly 117 tests")
    return ({
        "environment": environments,
        "repository": repository_checks,
        "offline_cache": offline,
        "p0_4": p04,
        "p0_4_preflights": preflights,
    }, {
        "status": "passed",
        "expected_per_lane": 117,
        "lanes": {
            lane: {
                "status": "passed",
                "command_count": sum(1 for name in focused if name.endswith("-" + lane)),
                "test_count": lane_totals[lane],
            }
            for lane in ("tf4576", "tf5141")
        },
    })


def _review_command_ledger(
    ledger_value: str | os.PathLike[str], *, tested_commit: str, required_names: Sequence[str], bind_ledgers: bool, hashes: dict[str, str]
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if _legacy.COMMIT_RE.fullmatch(tested_commit) is None:
        _fail("tested commit must be a full lowercase commit identifier")
    ledger_path = _legacy._safe_file(ledger_value, label="command ledger")
    if ledger_path.name != "commands.jsonl":
        _fail("command ledger must be commands.jsonl")
    run_root = _legacy._safe_root(ledger_path.parent, label="recorder run root")
    if stat.S_IMODE(run_root.stat().st_mode) != 0o700:
        _fail("recorder run root must have mode 0700")
    for relative in ("logs", "records"):
        directory = _legacy._child_directory(run_root, relative, label=relative)
        if stat.S_IMODE(directory.stat().st_mode) != 0o700:
            _fail(f"runner {relative} directory must have mode 0700")
    marker = _legacy._mapping(
        _legacy._load_json(
            _legacy._child_file(run_root, ".level1-requalification-run.json", label="run marker"),
            label="runner.run_marker",
            hashes=hashes,
        ),
        label="run marker",
    )
    if set(marker) != {"created_at_utc", "format_version", "repository", "tool_version"}:
        _fail("run marker fields are incomplete")
    _legacy._equals(marker["format_version"], "level1-requalification-run-v1", label="run marker format")
    _legacy._parse_utc(marker["created_at_utc"], label="run marker time")
    _legacy._nonempty_string(marker["tool_version"], label="run marker tool")
    marker_repo = _legacy._mapping(marker["repository"], label="run marker repository")
    if set(marker_repo) != {"head_commit", "worktree_path_sha256"}:
        _fail("run marker repository fields are incomplete")
    _legacy._equals(marker_repo["head_commit"], tested_commit, label="run marker commit")
    repository = Path(__file__).resolve().parents[1]
    _legacy._equals(
        marker_repo["worktree_path_sha256"],
        _sha256_bytes(os.fsencode(repository)),
        label="run marker worktree digest",
    )
    commands, command_lines = _legacy._load_runner_jsonl(
        ledger_path, label="runner.commands_ledger", hashes=hashes, bind_hash=bind_ledgers
    )
    environment_path = _legacy._child_file(run_root, "environment.jsonl", label="environment ledger")
    environments, environment_lines = _legacy._load_runner_jsonl(
        environment_path, label="runner.environment_ledger", hashes=hashes, bind_hash=bind_ledgers
    )
    names = [item.get("name") for item in commands]
    if names != list(required_names):
        _fail("command ledger is missing, duplicated, extra, or out of fixed order")
    environment_names = [item.get("name") for item in environments]
    if environment_names != list(REQUIRED_ENVIRONMENT_NAMES):
        _fail("environment ledger must contain exactly two ordered records")

    first_argv = _legacy._list(commands[0].get("argv"), label="first argv")
    first_index = _executable_index(first_argv, name=required_names[0], run_root=run_root)
    tf4576 = first_argv[first_index]
    tf514_record = commands[1]
    tf514_argv = _legacy._list(tf514_record.get("argv"), label="tf5141 argv")
    tf514_index = _executable_index(tf514_argv, name=required_names[1], run_root=run_root)
    tf5141 = tf514_argv[tf514_index]
    if not Path(tf4576).is_absolute() or not Path(tf5141).is_absolute():
        _fail("exact Transformers lane interpreters must be absolute paths")
    if tf4576 == tf5141:
        _fail("exact Transformers lanes must use distinct interpreters")
    offline_record = next(item for item in commands if item.get("name") == "offline-cache-preflight")
    offline_argv = _legacy._list(offline_record.get("argv"), label="offline argv")
    offline_index = _executable_index(offline_argv, name="offline-cache-preflight", run_root=run_root)
    cache = _legacy._absolute_path(
        _legacy._argv_option(offline_argv[offline_index:], "--cache-dir", label="offline cache"),
        label="offline cache",
    )
    tails = _expected_command_tails(
        repository=repository,
        run_root=run_root,
        cache=cache,
        tf4576_python=tf4576,
        tf5141_python=tf5141,
        tested_commit=tested_commit,
    )
    log_hashes: dict[str, str] = {}
    record_hashes: dict[str, str] = {}
    log_raws: dict[str, bytes] = {}
    times = []
    command_fields = {
        "argv", "cwd", "duration_ns", "duration_seconds", "ended_at_utc", "exit_code",
        "format_version", "log", "name", "preconditions", "record_type", "returncode",
        "runtime", "started_at_utc", "termination_signal",
    }
    for name, record, line in zip(required_names, commands, command_lines, strict=True):
        if set(record) != command_fields:
            _fail(f"command record {name} fields are incomplete")
        _review_record_common(record, name=name)
        _legacy._equals(record["record_type"], "command", label=f"{name}.type")
        argv = _legacy._list(record["argv"], label=f"{name}.argv")
        if any(not isinstance(item, str) or not item for item in argv):
            _fail(f"{name}.argv contains an invalid argument")
        index = _executable_index(argv, name=name, run_root=run_root)
        if tuple(argv[index:]) != tails[name]:
            _fail(f"command {name} child argv differs from the fixed Stage E matrix")
        _legacy._exact_int(record["exit_code"], label=f"{name}.exit", expected=0)
        _legacy._exact_int(record["returncode"], label=f"{name}.returncode", expected=0)
        if record["termination_signal"] is not None:
            _fail(f"command {name} was terminated by a signal")
        preconditions = _legacy._mapping(record["preconditions"], label=f"{name}.preconditions")
        if preconditions != {"absent_paths": _expected_absent_paths(name, run_root)}:
            _fail(f"command {name} fresh-output preconditions differ")
        log = _legacy._mapping(record["log"], label=f"{name}.log")
        if set(log) != {"path", "sha256", "size_bytes"} or log["path"] != f"logs/{name}.log":
            _fail(f"command {name} log identity differs")
        raw = _legacy._read_bytes(
            _legacy._child_file(run_root, log["path"], label=f"{name} log"),
            label=f"runner.log.{name}",
            hashes=hashes,
        )
        _legacy._equals(log["sha256"], _sha256_bytes(raw), label=f"{name}.log.sha256")
        _legacy._exact_int(log["size_bytes"], label=f"{name}.log.size", expected=len(raw))
        record_raw = _legacy._read_bytes(
            _legacy._child_file(run_root, f"records/{name}.json", label=f"{name} record"),
            label=f"runner.record.{name}",
            hashes=hashes,
        )
        if record_raw != line:
            _fail(f"named record {name} differs from commands.jsonl")
        log_hashes[name] = _sha256_bytes(raw)
        record_hashes[name] = _sha256_bytes(record_raw)
        log_raws[name] = raw
        times.append((
            _legacy._parse_utc(record["started_at_utc"], label=f"{name}.started"),
            _legacy._parse_utc(record["ended_at_utc"], label=f"{name}.ended"),
        ))
    if any(earlier[1] >= later[0] for earlier, later in zip(times, times[1:])):
        _fail("Stage E command timestamps overlap or differ from the fixed order")

    environment_fields = {
        "cwd", "duration_ns", "duration_seconds", "ended_at_utc", "format_version",
        "name", "record_type", "repository", "runtime", "started_at_utc",
    }
    env_times = []
    for name, record, line in zip(REQUIRED_ENVIRONMENT_NAMES, environments, environment_lines, strict=True):
        if set(record) != environment_fields:
            _fail(f"environment record {name} fields are incomplete")
        _review_record_common(record, name=name)
        _legacy._equals(record["record_type"], "environment", label=f"{name}.type")
        _legacy._equals(_legacy._at(record, "repository.head_commit"), tested_commit, label=f"{name}.commit")
        record_raw = _legacy._read_bytes(
            _legacy._child_file(run_root, f"records/{name}.json", label=f"{name} record"),
            label=f"runner.record.{name}",
            hashes=hashes,
        )
        if record_raw != line:
            _fail(f"named environment record {name} differs from environment.jsonl")
        record_hashes[name] = _sha256_bytes(record_raw)
        env_times.append((
            _legacy._parse_utc(record["started_at_utc"], label=f"{name}.started"),
            _legacy._parse_utc(record["ended_at_utc"], label=f"{name}.ended"),
        ))
    if env_times[0][1] >= env_times[1][0] or env_times[1][1] >= times[0][0]:
        _fail("environment records must be ordered before all commands")
    semantic, focused_tests = _review_semantic_logs(
        log_raws, tested_commit=tested_commit, repository=repository, cache=cache
    )
    return ({
        "status": "passed",
        "tested_commit": tested_commit,
        "observed_command_count": len(commands),
        "reviewed_command_count": len(required_names),
        "required_command_count": len(required_names),
        "required_commands": sorted(required_names),
        "observed_environment_record_count": len(environments),
        "reviewed_environment_record_count": len(REQUIRED_ENVIRONMENT_NAMES),
        "required_environment_records": sorted(REQUIRED_ENVIRONMENT_NAMES),
        "run_marker_sha256": hashes["runner.run_marker"],
        "record_sha256": dict(sorted(record_hashes.items())),
        "log_sha256": dict(sorted(log_hashes.items())),
        "semantic_logs": semantic,
        "ordering_checks": [
            "runtime-environments-before-commands",
            "all-command-records-nonoverlapping-in-fixed-order",
            (
                "p0-4-psi8-inputs-complete-for-focused-review"
                if tuple(required_names) == P0_4_PSI8_REVIEW_COMMAND_NAMES
                else "p0-4-psi8-focused-review-before-psi16"
            ),
        ],
    }, cache, focused_tests)


def _validate_p0_4_v2(summary: Mapping[str, Any], events: Sequence[Mapping[str, Any]], *, psi: int) -> None:
    qualification = _legacy._mapping(summary.get("qualification"), label=f"P0-4 Psi={psi} qualification")
    if set(qualification) != {"schema_version", "qualified", "conditions"}:
        _fail(f"P0-4 Psi={psi} qualification-v2 fields are incomplete")
    _legacy._equals(qualification["schema_version"], P0_4_QUALIFICATION_SCHEMA_VERSION, label="qualification schema")
    _legacy._exact_bool(qualification["qualified"], label="qualification verdict", expected=True)
    conditions = _legacy._mapping(qualification["conditions"], label="qualification conditions")
    if set(conditions) != set(P0_4_CONDITIONS):
        _fail(f"P0-4 Psi={psi} qualification conditions are missing or extra")
    for name in P0_4_CONDITIONS:
        _legacy._exact_bool(conditions[name], label=f"qualification.{name}", expected=True)
    if tuple(event.get("event") for event in events) != _legacy.P0_4_EVENT_SEQUENCE or len(events) != 57:
        _fail(f"P0-4 Psi={psi} must contain the exact 57-event sequence")
    if events[-1].get("qualification") != qualification:
        _fail(f"P0-4 Psi={psi} run_complete qualification differs from summary")
    _legacy._exact_int(_legacy._at(summary, "settings.batch_size"), label="settings.batch_size", expected=1)
    _legacy._exact_bool(_legacy._at(summary, "settings.gradient_checkpointing"), label="settings.gradient_checkpointing", expected=True)
    _legacy._exact_int(_legacy._at(summary, "training.microbatch_size"), label="training.microbatch_size", expected=1)
    _legacy._exact_int(_legacy._at(summary, "training.optimizer_steps"), label="training.optimizer_steps", expected=50)
    _legacy._exact_bool(_legacy._at(summary, "model.gradient_checkpointing"), label="model.gradient_checkpointing", expected=True)
    if _legacy._at(summary, "model.gradient_checkpointing_kwargs") != {"use_reentrant": False}:
        _fail(f"P0-4 Psi={psi} runtime checkpointing witness is not non-reentrant")


def _review_p0_4_lane_v2(
    root_value: str | os.PathLike[str], *, psi: int, expected_cache: Path, hashes: dict[str, str]
) -> dict[str, Any]:
    logical = f"p0_4_psi{psi}"
    root = _legacy._safe_root(root_value, label=f"P0-4 Psi={psi} root")
    _legacy._failure_artifact_check(root, label=f"P0-4 Psi={psi} root")
    summary_path = _legacy._child_file(root, "summary.json", label=f"{logical} summary")
    metrics_path = _legacy._child_file(root, "metrics.jsonl", label=f"{logical} metrics")
    data_path = _legacy._child_file(root, "data_contract.json", label=f"{logical} data")
    marker_path = _legacy._child_file(root, "P0-4_COMPLETE.md", label=f"{logical} marker")
    summary = _legacy._mapping(
        _legacy._load_json(summary_path, label=f"{logical}.summary", hashes=hashes), label=f"{logical}.summary"
    )
    events = _legacy._load_jsonl(metrics_path, label=f"{logical}.metrics", hashes=hashes)
    data_raw = _legacy._read_bytes(data_path, label=f"{logical}.data_contract", hashes=hashes)
    marker_raw = _legacy._read_bytes(marker_path, label=f"{logical}.completion_marker", hashes=hashes)
    _validate_p0_4_v2(summary, events, psi=psi)
    original_settings = _legacy._mapping(
        _legacy._at(summary, "settings"), label=f"{logical}.settings"
    )
    _legacy._validate_embedded_directory(
        _legacy._at(original_settings, "output_dir"),
        root,
        label=f"{logical}.settings.output_dir",
    )
    if events[0].get("settings") != original_settings:
        _fail(f"P0-4 Psi={psi} run_start settings differ from summary")
    original_reload = _legacy._mapping(
        _legacy._at(summary, "checks.save_reload"),
        label=f"{logical}.checks.save_reload",
    )
    checkpoint = _legacy._child_directory(
        root, "checkpoint", label=f"P0-4 Psi={psi} checkpoint"
    )
    _legacy._validate_embedded_directory(
        _legacy._at(original_reload, "checkpoint_dir"),
        checkpoint,
        label=f"{logical}.checks.save_reload.checkpoint_dir",
    )
    _legacy._validate_embedded_directory(
        _legacy._at(events[53], "checkpoint_dir"),
        checkpoint,
        label=f"{logical}.save_reload_check.checkpoint_dir",
    )
    if events[53].get("checkpoint_dir") != original_reload.get("checkpoint_dir"):
        _fail(f"P0-4 Psi={psi} save_reload checkpoint differs from summary")
    old_qualification = {
        "qualified": True,
        "conditions": {name: True for name in P0_4_CONDITIONS[:4] + ("optimizer_steps_at_least_50",)},
    }
    with tempfile.TemporaryDirectory(prefix=f"multiscreen-stagee-p04-psi{psi}-") as temporary:
        mirror = Path(temporary)
        (mirror / "checkpoint").mkdir()
        (mirror / "data_contract.json").write_bytes(data_raw)
        (mirror / "P0-4_COMPLETE.md").write_bytes(marker_raw)
        projected_summary = copy.deepcopy(dict(summary))
        projected_summary["qualification"] = old_qualification
        projected_summary["settings"]["output_dir"] = os.fspath(mirror)
        projected_summary["checks"]["save_reload"]["checkpoint_dir"] = os.fspath(mirror / "checkpoint")
        projected_events = copy.deepcopy(list(events))
        projected_events[0]["settings"] = projected_summary["settings"]
        projected_events[53]["checkpoint_dir"] = os.fspath(mirror / "checkpoint")
        projected_events[-1]["qualification"] = old_qualification
        (mirror / "summary.json").write_bytes(_pretty_canonical_bytes(projected_summary))
        (mirror / "metrics.jsonl").write_bytes(
            b"".join(_legacy._runner_canonical_bytes(event) for event in projected_events)
        )
        reviewed = _legacy._review_p0_4_lane(
            mirror, psi=psi, expected_cache=expected_cache, hashes={}
        )
    reviewed["qualification_schema_version"] = P0_4_QUALIFICATION_SCHEMA_VERSION
    reviewed["qualification_conditions"] = list(P0_4_CONDITIONS)
    reviewed["microbatch_size"] = 1
    reviewed["gradient_checkpointing_enabled"] = True
    return reviewed


def _historical_tree(repository: Path, commit: str) -> list[dict[str, str]]:
    result = _legacy._git_capture(
        repository,
        (
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit,
            "--",
            "docs/validation_results",
        ),
        label=f"historical tree {commit}",
    )
    records = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, path_raw = raw.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = path_raw.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ReviewError("malformed historical Git tree record") from exc
        if not path.startswith(HISTORICAL_PATH_PREFIXES):
            continue
        if object_type != "blob" or mode != "100644" or not _legacy.COMMIT_RE.fullmatch(object_id):
            _fail("historical evidence tree contains an unsupported object")
        if path not in HISTORICAL_PATHS:
            _fail("historical evidence tree escaped its fixed pathspec")
        records.append({"mode": mode, "object_id": object_id, "path": path, "type": object_type})
    records.sort(key=lambda item: item["path"])
    if len(records) != len(HISTORICAL_PATHS) or {
        item["path"] for item in records
    } != set(HISTORICAL_PATHS):
        _fail("historical evidence tree is incomplete or ambiguous")
    return records


def _review_historical_evidence_immutability(
    *, repository: Path, tested_commit: str, implementation_baseline: str = IMPLEMENTATION_BASELINE
) -> dict[str, Any]:
    for value, label in ((tested_commit, "tested commit"), (implementation_baseline, "implementation baseline")):
        if _legacy.COMMIT_RE.fullmatch(value) is None:
            _fail(f"{label} must be a full lowercase commit identifier")
    ancestor = _legacy._git_capture(
        repository,
        ("merge-base", "--is-ancestor", implementation_baseline, tested_commit),
        label="Stage E implementation ancestry",
        accepted_returncodes=frozenset({0, 1}),
    )
    if ancestor.returncode != 0:
        _fail("tested commit does not descend from the Stage A-D implementation baseline")
    before = _historical_tree(repository, implementation_baseline)
    after = _historical_tree(repository, tested_commit)
    if before != after:
        _fail("accepted historical evidence paths or blobs changed after Stage A-D")
    return {
        "status": "passed",
        "implementation_baseline": implementation_baseline,
        "tested_commit": tested_commit,
        "artifact_count": len(before),
        "tree_material_sha256": _sha256_bytes(_canonical_bytes(before)),
    }


def _bind_layout(run_root: Path, value: str | os.PathLike[str], relative: str, *, label: str) -> Path:
    return _legacy._bind_run_artifact(run_root, value, relative, label=label)


def review_p0_4_lane_inputs(
    *, p0_4_root: str | os.PathLike[str], tokenizer_reports: Mapping[str, str | os.PathLike[str]], command_ledger: str | os.PathLike[str], tested_commit: str
) -> dict[str, Any]:
    if set(tokenizer_reports) != {"p0_4_psi8"}:
        _fail("focused review requires exactly the Psi=8 tokenizer report")
    ledger_path = _legacy._safe_file(command_ledger, label="command ledger")
    run_root = ledger_path.parent
    _bind_layout(run_root, p0_4_root, "artifacts/p0-4/psi8", label="P0-4 Psi=8 root")
    _bind_layout(run_root, tokenizer_reports["p0_4_psi8"], "artifacts/p0-4/psi8/tokenizer-reload.json", label="Psi=8 tokenizer")
    hashes: dict[str, str] = {}
    ledger, cache, focused_tests = _review_command_ledger(
        command_ledger,
        tested_commit=tested_commit,
        required_names=P0_4_PSI8_REVIEW_COMMAND_NAMES,
        bind_ledgers=False,
        hashes=hashes,
    )
    p04 = _review_p0_4_lane_v2(p0_4_root, psi=8, expected_cache=cache, hashes=hashes)
    tokenizer = _legacy._review_tokenizer_reports(
        tokenizer_reports, hashes=hashes, required_names=("p0_4_psi8",)
    )
    p04["cross_bindings"] = _legacy._review_p0_4_cross_bindings(
        p0_4_runs=[p04], tokenizer=tokenizer, ledger=ledger
    )
    artifact_hashes = dict(sorted(hashes.items()))
    material = {
        "artifact_hashes": artifact_hashes,
        "command_ledger": ledger,
        "focused_tests": focused_tests,
        "p0_4": p04,
        "tested_commit": tested_commit,
        "tokenizer_reload": tokenizer,
    }
    report = {
        "schema_version": P0_4_LANE_SCHEMA_VERSION,
        "mode": "p0-4-lane",
        "status": "passed",
        "psi": 8,
        "tested_commit": tested_commit,
        "p0_4": p04,
        "tokenizer_reload": tokenizer,
        "command_ledger": ledger,
        "focused_tests": focused_tests,
        "aggregate": {
            "artifact_count": len(artifact_hashes),
            "artifact_hashes": artifact_hashes,
            "review_material_sha256": _sha256_bytes(_canonical_bytes(material)),
        },
    }
    _legacy._validate_finite_tree(report, label="Stage E focused report")
    return report


def _focused_ledger_projection(ledger: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(ledger))
    projected["observed_command_count"] = len(P0_4_PSI8_REVIEW_COMMAND_NAMES)
    projected["reviewed_command_count"] = len(P0_4_PSI8_REVIEW_COMMAND_NAMES)
    projected["required_command_count"] = len(P0_4_PSI8_REVIEW_COMMAND_NAMES)
    projected["required_commands"] = sorted(P0_4_PSI8_REVIEW_COMMAND_NAMES)
    projected["ordering_checks"] = [
        "runtime-environments-before-commands",
        "all-command-records-nonoverlapping-in-fixed-order",
        "p0-4-psi8-inputs-complete-for-focused-review",
    ]
    record_hashes = _legacy._mapping(
        projected["record_sha256"], label="full ledger record hashes"
    )
    expected_record_names = set(P0_4_PSI8_REVIEW_COMMAND_NAMES) | set(
        REQUIRED_ENVIRONMENT_NAMES
    )
    projected["record_sha256"] = {
        name: record_hashes[name] for name in sorted(expected_record_names)
    }
    log_hashes = _legacy._mapping(
        projected["log_sha256"], label="full ledger log hashes"
    )
    projected["log_sha256"] = {
        name: log_hashes[name] for name in sorted(P0_4_PSI8_REVIEW_COMMAND_NAMES)
    }
    semantic = _legacy._mapping(
        projected["semantic_logs"], label="full ledger semantic logs"
    )
    repository = _legacy._mapping(
        semantic["repository"], label="full ledger repository checks"
    )
    repository.pop("repository-hygiene-final", None)
    p0_4 = _legacy._mapping(semantic["p0_4"], label="full ledger P0-4 logs")
    p0_4.pop("p0_4_psi16", None)
    preflights = _legacy._mapping(
        semantic["p0_4_preflights"], label="full ledger P0-4 preflights"
    )
    preflights.pop("p0-4-psi16-preflight", None)
    return projected


def _review_focused_report(
    report_value: str | os.PathLike[str], *, tested_commit: str, expected_p0_4: Mapping[str, Any], expected_tokenizer: Mapping[str, Any], expected_ledger: Mapping[str, Any], expected_focused_tests: Mapping[str, Any], current_hashes: Mapping[str, str], hashes: dict[str, str]
) -> dict[str, Any]:
    report = _legacy._mapping(
        _legacy._load_json(
            _legacy._safe_file(report_value, label="focused report"),
            label="p0_4_psi8.focused_review",
            hashes=hashes,
        ),
        label="focused report",
    )
    expected_keys = {"aggregate", "command_ledger", "focused_tests", "mode", "p0_4", "psi", "schema_version", "status", "tested_commit", "tokenizer_reload"}
    if set(report) != expected_keys:
        _fail("focused report fields are incomplete or ambiguous")
    _legacy._exact_values(
        report,
        {"schema_version": P0_4_LANE_SCHEMA_VERSION, "mode": "p0-4-lane", "status": "passed", "psi": 8, "tested_commit": tested_commit},
        label="focused report",
    )
    if report["p0_4"] != expected_p0_4 or report["tokenizer_reload"] != expected_tokenizer:
        _fail("focused report projections differ from current raw evidence")
    ledger = _legacy._mapping(report["command_ledger"], label="focused ledger")
    if ledger != _focused_ledger_projection(expected_ledger):
        _fail("focused report ledger differs from the current fixed 48-command prefix")
    focused_tests = _legacy._mapping(
        report["focused_tests"], label="focused test aggregate"
    )
    if focused_tests != expected_focused_tests:
        _fail("focused report test aggregate differs from the current 117-test lanes")
    aggregate = _legacy._mapping(report["aggregate"], label="focused aggregate")
    if set(aggregate) != {"artifact_count", "artifact_hashes", "review_material_sha256"}:
        _fail("focused aggregate fields are incomplete")
    focused_hashes = _legacy._mapping(aggregate["artifact_hashes"], label="focused hashes")
    for label, digest in focused_hashes.items():
        if not isinstance(digest, str) or not _legacy.HEX64_RE.fullmatch(digest) or current_hashes.get(label) != digest:
            _fail(f"focused artifact changed or has an invalid digest: {label}")
    if set(focused_hashes) != set(current_hashes) - {"p0_4_psi8.focused_review", "runner.commands_ledger", "runner.environment_ledger"} - {
        label for label in current_hashes if label.startswith("p0_3") or label.startswith("p0_4_psi16") or label.startswith("tokenizer_reload.p0_3") or label == "tokenizer_reload.p0_4_psi16" or label.startswith("runner.log.p0-4-review-psi8") or label.startswith("runner.record.p0-4-review-psi8") or label.startswith("runner.log.p0-4-psi16") or label.startswith("runner.record.p0-4-psi16") or label.startswith("runner.log.p0-4-tokenizer-psi16") or label.startswith("runner.record.p0-4-tokenizer-psi16") or label.startswith("runner.log.p0-4-psi16-preflight") or label.startswith("runner.record.p0-4-psi16-preflight") or label.startswith("runner.log.repository-hygiene-final") or label.startswith("runner.record.repository-hygiene-final")
    }:
        _fail("focused report artifact inventory differs from the fixed prefix")
    _legacy._exact_int(aggregate["artifact_count"], label="focused artifact count", expected=len(focused_hashes))
    material = {
        "artifact_hashes": dict(focused_hashes),
        "command_ledger": ledger,
        "focused_tests": focused_tests,
        "p0_4": report["p0_4"],
        "tested_commit": tested_commit,
        "tokenizer_reload": report["tokenizer_reload"],
    }
    expected_digest = _sha256_bytes(_canonical_bytes(material))
    _legacy._equals(aggregate["review_material_sha256"], expected_digest, label="focused material hash")
    return {"status": "passed", "psi": 8, "artifact_count": len(focused_hashes), "review_material_sha256": expected_digest}


def review_inputs(
    *, p0_3_root: str | os.PathLike[str], p0_3_stdout: str | os.PathLike[str], p0_4_psi8_root: str | os.PathLike[str], p0_4_psi16_root: str | os.PathLike[str], p0_4_psi8_review: str | os.PathLike[str], tokenizer_reports: Mapping[str, str | os.PathLike[str]], command_ledger: str | os.PathLike[str], tested_commit: str, repository: Path | None = None
) -> dict[str, Any]:
    if set(tokenizer_reports) != set(_legacy.TOKENIZER_NAMES):
        _fail("full review requires exactly four tokenizer reports")
    ledger_path = _legacy._safe_file(command_ledger, label="command ledger")
    run_root = ledger_path.parent
    _bind_layout(run_root, p0_3_root, "artifacts/p0-3", label="P0-3 root")
    _bind_layout(run_root, p0_3_stdout, "logs/p0-3-checkpointed.log", label="P0-3 stdout")
    _bind_layout(run_root, p0_4_psi8_root, "artifacts/p0-4/psi8", label="P0-4 Psi=8 root")
    _bind_layout(run_root, p0_4_psi16_root, "artifacts/p0-4/psi16", label="P0-4 Psi=16 root")
    _bind_layout(run_root, p0_4_psi8_review, "artifacts/p0-4/psi8/raw-review.json", label="focused report")
    expected_tokenizers = {
        "p0_3_psi8": "artifacts/p0-3/tokenizer-reload-psi8.json",
        "p0_3_psi16": "artifacts/p0-3/tokenizer-reload-psi16.json",
        "p0_4_psi8": "artifacts/p0-4/psi8/tokenizer-reload.json",
        "p0_4_psi16": "artifacts/p0-4/psi16/tokenizer-reload.json",
    }
    for name, relative in expected_tokenizers.items():
        _bind_layout(run_root, tokenizer_reports[name], relative, label=f"{name} tokenizer")
    hashes: dict[str, str] = {}
    ledger, cache, focused_tests = _review_command_ledger(
        command_ledger,
        tested_commit=tested_commit,
        required_names=REQUIRED_COMMAND_NAMES,
        bind_ledgers=True,
        hashes=hashes,
    )
    p03 = _legacy._review_p0_3(p0_3_root, p0_3_stdout, hashes=hashes)
    p04_runs = [
        _review_p0_4_lane_v2(p0_4_psi8_root, psi=8, expected_cache=cache, hashes=hashes),
        _review_p0_4_lane_v2(p0_4_psi16_root, psi=16, expected_cache=cache, hashes=hashes),
    ]
    if not _legacy._parse_utc(p04_runs[0]["timestamp_utc"], label="Psi=8 time") < _legacy._parse_utc(p04_runs[1]["timestamp_utc"], label="Psi=16 time"):
        _fail("P0-4 Psi=8 must precede Psi=16")
    tokenizer = _legacy._review_tokenizer_reports(tokenizer_reports, hashes=hashes)
    for run in p04_runs:
        run["cross_bindings"] = _legacy._review_p0_4_cross_bindings(
            p0_4_runs=[run], tokenizer=tokenizer, ledger=ledger
        )
    p04_cross = _legacy._review_p0_4_cross_bindings(
        p0_4_runs=p04_runs, tokenizer=tokenizer, ledger=ledger
    )
    p03["cross_bindings"] = _legacy._review_p0_3_cross_bindings(
        p0_3=p03, tokenizer=tokenizer, ledger=ledger
    )
    expected_focused_tokenizer = {
        "status": "passed",
        "report_count": 1,
        "reports": [next(item for item in tokenizer["reports"] if item["logical_name"] == "p0_4_psi8")],
    }
    focused_review = _review_focused_report(
        p0_4_psi8_review,
        tested_commit=tested_commit,
        expected_p0_4=p04_runs[0],
        expected_tokenizer=expected_focused_tokenizer,
        expected_ledger=ledger,
        expected_focused_tests=focused_tests,
        current_hashes=hashes,
        hashes=hashes,
    )
    historical = _review_historical_evidence_immutability(
        repository=Path(__file__).resolve().parents[1] if repository is None else repository,
        tested_commit=tested_commit,
    )
    raw_counts = {
        "p0_3_stdout_step_events": p03["stdout_step_event_count"],
        "p0_4_jsonl_events": sum(item["event_count"] for item in p04_runs),
    }
    raw_counts["total"] = sum(raw_counts.values())
    if raw_counts != {"p0_3_stdout_step_events": 65, "p0_4_jsonl_events": 114, "total": 179}:
        _fail("Stage E raw-event inventory must be exactly 179")
    artifact_hashes = dict(sorted(hashes.items()))
    aggregate_material = {
        "artifact_hashes": artifact_hashes,
        "raw_event_counts": raw_counts,
        "tested_commit": tested_commit,
        "implementation_baseline": IMPLEMENTATION_BASELINE,
        "historical_evidence_immutability": historical,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "tested_commit": tested_commit,
        "implementation_baseline": IMPLEMENTATION_BASELINE,
        "historical_evidence_immutability": historical,
        "p0_3": p03,
        "p0_4": {"status": "passed", "cross_bindings": p04_cross, "focused_psi8_review": focused_review, "runs": p04_runs},
        "tokenizer_reload": tokenizer,
        "command_ledger": ledger,
        "focused_tests": focused_tests,
        "aggregate": {
            "artifact_count": len(artifact_hashes),
            "artifact_hashes": artifact_hashes,
            "raw_event_counts": raw_counts,
            "review_material_sha256": _sha256_bytes(_canonical_bytes(aggregate_material)),
        },
    }
    _legacy._validate_finite_tree(report, label="Stage E full review")
    return report


def _parse_named_paths(values: Sequence[str]) -> dict[str, str]:
    return _legacy._parse_named_paths(values)


def _verify_live_reviewer_checkout(tested_commit: str) -> None:
    repository = Path(__file__).resolve().parents[1]
    if _legacy.COMMIT_RE.fullmatch(tested_commit) is None:
        _fail("tested commit must be a full lowercase commit identifier")
    top = _legacy._git_capture(repository, ("rev-parse", "--show-toplevel"), label="reviewer top-level").stdout
    if top != os.fsencode(repository) + b"\n":
        _fail("reviewer checkout differs from the Git top-level")
    branch = _legacy._git_capture(
        repository,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        label="reviewer branch",
    ).stdout
    if branch != WORKING_BRANCH.encode("utf-8") + b"\n":
        _fail("reviewer checkout differs from the fixed Stage E branch")
    head = _legacy._git_capture(repository, ("rev-parse", "--verify", "HEAD^{commit}"), label="reviewer HEAD").stdout
    if head != tested_commit.encode("ascii") + b"\n":
        _fail("reviewer checkout HEAD differs from tested commit")
    status = _legacy._git_capture(
        repository,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"),
        label="reviewer status",
    ).stdout
    if status:
        _fail("reviewer checkout is not exactly clean")
    relative = "scripts/review_hf_contract_hardening.py"
    tracked = _legacy._git_capture(repository, ("ls-files", "--error-unmatch", "--", relative), label="reviewer tracked source").stdout
    if tracked != relative.encode("ascii") + b"\n":
        _fail("Stage E reviewer is not tracked at its fixed path")
    committed = _legacy._git_capture(repository, ("cat-file", "blob", f"{tested_commit}:{relative}"), label="reviewer committed source").stdout
    if committed != _legacy._stable_read_bytes(Path(__file__).resolve(), label="reviewer source"):
        _fail("executing Stage E reviewer differs from tested source")
    for revision, label in (
        (IMPLEMENTATION_BASELINE, "implementation baseline"),
        (tested_commit, "tested source"),
    ):
        legacy_blob = _legacy._git_capture(
            repository,
            ("rev-parse", f"{revision}:{LEGACY_REVIEWER_PATH}"),
            label=f"legacy reviewer blob at {label}",
        ).stdout
        if legacy_blob != LEGACY_REVIEWER_GIT_BLOB.encode("ascii") + b"\n":
            _fail(f"legacy Level 1 reviewer changed at {label}")


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
    parser.add_argument("--tokenizer-reload-report", action="append", default=[], metavar="LOGICAL_NAME=PATH")
    return parser.parse_args(argv)


def _required(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if not isinstance(value, str) or not value:
        _fail(f"--{name.replace('_', '-')} is required in {args.mode} mode")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    _legacy._verify_python_safety_flags()
    args = parse_args(argv)
    output = _legacy._absolute_path(args.output, label="review output")
    if not output.parent.is_dir():
        _fail("review output parent does not exist")
    tokenizer_reports = _parse_named_paths(args.tokenizer_reload_report)
    _verify_live_reviewer_checkout(args.tested_commit)
    if args.mode == "p0-4-lane":
        report = review_p0_4_lane_inputs(
            p0_4_root=_required(args, "p0_4_root"),
            tokenizer_reports=tokenizer_reports,
            command_ledger=args.command_ledger,
            tested_commit=args.tested_commit,
        )
    else:
        report = review_inputs(
            p0_3_root=_required(args, "p0_3_root"),
            p0_3_stdout=_required(args, "p0_3_stdout"),
            p0_4_psi8_root=_required(args, "p0_4_psi8_root"),
            p0_4_psi16_root=_required(args, "p0_4_psi16_root"),
            p0_4_psi8_review=_required(args, "p0_4_psi8_review"),
            tokenizer_reports=tokenizer_reports,
            command_ledger=args.command_ledger,
            tested_commit=args.tested_commit,
        )
    _verify_live_reviewer_checkout(args.tested_commit)
    raw = _pretty_canonical_bytes(report)
    _legacy._exclusive_output(output, raw)
    sys.stdout.write(json.dumps({
        "mode": args.mode,
        "status": "passed",
        "output_sha256": _sha256_bytes(raw),
        "review_material_sha256": report["aggregate"]["review_material_sha256"],
    }, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewError as exc:
        sys.stderr.write(f"review failed: {exc}\n")
        raise SystemExit(1)
