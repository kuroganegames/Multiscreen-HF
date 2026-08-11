#!/usr/bin/env python3
"""Build and close the reviewed Level 1 Core evidence descriptor.

This standard-library-only tool never discovers evidence by walking a
directory. Prepare binds the fixed Stage 5 file layout to the complete review
hash inventory. Seal binds verified archives. Close records commit A.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NamedTuple, Sequence

try:
    from validation_evidence_common import (
        PACKAGE_INPUT_VERSION,
        InputValidationError,
        IntegrityError,
        canonical_json_bytes,
        sha256_bytes,
        validate_evidence_document,
        validate_utc,
    )
    from verify_validation_evidence import TOOL_VERSION as VERIFIER_VERSION
    from verify_validation_evidence import verify_archive
except ModuleNotFoundError:  # Support importing as scripts.* in tests.
    from scripts.validation_evidence_common import (  # type: ignore[no-redef]
        PACKAGE_INPUT_VERSION,
        InputValidationError,
        IntegrityError,
        canonical_json_bytes,
        sha256_bytes,
        validate_evidence_document,
        validate_utc,
    )
    from scripts.verify_validation_evidence import (  # type: ignore[no-redef]
        TOOL_VERSION as VERIFIER_VERSION,
    )
    from scripts.verify_validation_evidence import verify_archive  # type: ignore[no-redef]


TOOL_VERSION = "1.0.0"
SUMMARY_VERSION = "multiscreen-level1-core-summary-v1"
REVIEW_VERSION = "multiscreen-level1-raw-evidence-review-v1"
PROVENANCE_VERSION = "validation-provenance-v1"
PACKAGE_REPORT_VERSION = "validation-evidence-package-report-v1"
GATE = "Level 1 Core"
EVIDENCE_GATE = "Stage 5 final requalification"
REPOSITORY_NAME = "kuroganegames/Multiscreen-HF"
IMPLEMENTATION_BASE_COMMIT = "3282eae7cb97ecfe01753460f6bce63d03e3cf88"
SUMMARY_JSON_NAME = "LEVEL1_CORE_SUMMARY.json"
CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RESULTS_ROOT = CANONICAL_REPOSITORY_ROOT / "docs" / "validation_results"
CANONICAL_SCHEMA_PATH = (
    CANONICAL_REPOSITORY_ROOT
    / "schemas"
    / "validation_evidence_v1.schema.json"
)
CANONICAL_SCHEMA_SHA256 = (
    "f4035982599faff8c6dc49e447ebc44594f3988db1303b43ac16534b710c35ab"
)

SUMMARY_MARKDOWN_NAME = "LEVEL1_CORE_SUMMARY.md"
DESCRIPTOR_NAME = "LEVEL1_CORE_EVIDENCE_ARCHIVE.json"
EXACT_VERIFICATION_NAME = "LEVEL1_CORE_EXACT_VERIFICATION.json"
SANITIZED_VERIFICATION_NAME = "LEVEL1_CORE_SANITIZED_VERIFICATION.json"
PACKAGE_INPUT_RELATIVE = "review/level1-package-input.json"
FULL_REVIEW_RELATIVE = "review/level1-core.json"
ACCEPTANCE_PROVENANCE_RELATIVE = "review/acceptance-provenance.json"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
PRIVATE_PATH_RE = re.compile(
    r"(?<![:/A-Za-z0-9_.-])/(?:[^\s\"'<>]+/)*[^\s\"'<>]*"
)
WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]")
CREDENTIAL_URL_RE = re.compile(r"(?i)(?:https?|ssh|git)://[^/@\s]+@")
TOKEN_RE = re.compile(
    r"(?:\bgh[pousr]_[A-Za-z0-9_]{20,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}\b|"
    r"\bhf_[A-Za-z0-9]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"(?i:\bBearer\s+[A-Za-z0-9._~+/-]{12,}={0,2}))"
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
REQUIRED_ENVIRONMENT_NAMES = ("runtime-tf4576", "runtime-tf5141")


class EvidenceBuildError(ValueError):
    """A closure input does not satisfy the fixed Level 1 evidence contract."""


class ArtifactSpec(NamedTuple):
    source_path: str
    classification: str

    @property
    def archive_path(self) -> str:
        return f"artifacts/level1-core/run/{self.source_path}"


def _fail(message: str) -> None:
    raise EvidenceBuildError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON number is forbidden: {value}")


def _walk_no_symlinks(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            status = os.lstat(current)
        except OSError as exc:
            _fail(f"{label} does not exist or cannot be inspected: {exc}")
        if stat.S_ISLNK(status.st_mode):
            _fail(f"{label} contains a symlink component")


def _canonical_path(value: str | os.PathLike[str], *, label: str) -> Path:
    raw = os.fspath(value)
    path = Path(raw)
    if not path.is_absolute():
        _fail(f"{label} must be an absolute path")
    lexical = Path(os.path.abspath(raw))
    if path != lexical:
        _fail(f"{label} must be lexically canonical (no dot or parent components)")
    _walk_no_symlinks(lexical, label=label)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label} cannot be resolved: {exc}")
    if resolved != lexical:
        _fail(f"{label} must be canonical and must not traverse symlinks")
    return lexical


def _canonical_root(value: str | os.PathLike[str], *, label: str) -> Path:
    path = _canonical_path(value, label=label)
    if path == Path(path.anchor):
        _fail(f"{label} must not be a filesystem root")
    status = os.lstat(path)
    if not stat.S_ISDIR(status.st_mode):
        _fail(f"{label} must be a directory")
    return path


def _canonical_file(value: str | os.PathLike[str], *, label: str) -> Path:
    candidate = Path(os.fspath(value))
    if candidate.is_absolute() and candidate.parent == Path(candidate.anchor):
        _fail(
            f"{label} must not be located directly under a filesystem root"
        )
    path = _canonical_path(value, label=label)
    status = os.lstat(path)
    if not stat.S_ISREG(status.st_mode):
        _fail(f"{label} must be a regular file")
    if status.st_nlink != 1:
        _fail(f"{label} must not have hard links")
    return path


def _fixed_child(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        _fail(f"internal fixed path for {label} is invalid")
    path = root.joinpath(*pure.parts)
    try:
        path.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes its fixed root")
    return _canonical_file(path, label=label)


def _stable_read(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail(f"could not open {label} safely: {exc}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(f"{label} must remain a singly linked regular file")
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity):
            _fail(f"{label} changed while it was read")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            _fail(f"{label} changed size while it was read")
        return raw
    finally:
        os.close(descriptor)


def _load_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    raw = _stable_read(path, label=label)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        _fail(f"{label} must be UTF-8 JSON: {exc}")
    except json.JSONDecodeError as exc:
        _fail(f"{label} is invalid JSON: {exc.msg}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object")
    return value, raw


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    return value


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        _fail(f"{label} must be a full lowercase Git object ID")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} must be a UTC timestamp")
    try:
        return validate_utc(value, field=label)
    except InputValidationError as exc:
        _fail(str(exc))


def _timestamp_instant(value: Any, *, label: str) -> dt.datetime:
    timestamp = _timestamp(value, label=label)
    return dt.datetime.fromisoformat(timestamp[:-1] + "+00:00")


def _assert_public_string(value: str, *, label: str) -> None:
    if (
        "file://" in value.casefold()
        or PRIVATE_PATH_RE.search(value)
        or WINDOWS_PATH_RE.search(value)
        or CREDENTIAL_URL_RE.search(value)
        or TOKEN_RE.search(value)
    ):
        _fail(f"{label} contains a private path or credential pattern")


def _assert_public_document(value: Any, *, label: str) -> None:
    pending: list[tuple[str, Any]] = [(label, value)]
    while pending:
        current_label, current = pending.pop()
        if isinstance(current, str):
            _assert_public_string(current, label=current_label)
        elif isinstance(current, Mapping):
            for key, child in current.items():
                _assert_public_string(str(key), label=f"{current_label} key")
                pending.append((f"{current_label}.{key}", child))
        elif isinstance(current, list):
            for index, child in enumerate(current):
                pending.append((f"{current_label}[{index}]", child))


def _validated_storage_locator(value: Any, *, label: str) -> str:
    locator = _string(value, label=label)
    _assert_public_string(locator, label=label)
    if locator.startswith(("/", "\\")) or ".." in PurePosixPath(
        locator.replace("\\", "/")
    ).parts:
        _fail(f"{label} must be privacy-safe and traversal-free")
    return locator


def _fixed_specs() -> dict[str, ArtifactSpec]:
    specs: dict[str, ArtifactSpec] = {}

    def add(label: str, path: str, classification: str) -> None:
        if label in specs:
            _fail(f"duplicate internal artifact label: {label}")
        specs[label] = ArtifactSpec(path, classification)

    add("p0_3.data_contract", "artifacts/p0-3/data_contract.json", "validation_metrics")
    add("p0_3.completion_marker", "artifacts/p0-3/P0-3_COMPLETE.md", "completion_marker")
    add("p0_3.results", "artifacts/p0-3/p0_3_results.json", "validation_summary")
    add("p0_3.stdout", "logs/p0-3-checkpointed.log", "other")
    for psi in (8, 16):
        add(
            f"p0_3.psi{psi}.metrics",
            f"artifacts/p0-3/psi{psi}/p0_3_metrics.json",
            "validation_metrics",
        )
        base = f"artifacts/p0-4/psi{psi}"
        logical = f"p0_4_psi{psi}"
        add(
            f"{logical}.data_contract",
            f"{base}/data_contract.json",
            "validation_metrics",
        )
        add(f"{logical}.completion_marker", f"{base}/P0-4_COMPLETE.md", "completion_marker")
        add(f"{logical}.summary", f"{base}/summary.json", "validation_summary")
        add(f"{logical}.metrics", f"{base}/metrics.jsonl", "validation_metrics")
    add(
        "p0_4_psi8.focused_review",
        "artifacts/p0-4/psi8/raw-review.json",
        "validation_summary",
    )
    add(
        "c3_data.data_contract",
        "artifacts/c3/data/data_contract.json",
        "validation_metrics",
    )
    add(
        "c3_data.completion_marker",
        "artifacts/c3/data/P0_5_C3_DATA_CONTRACT_COMPLETE.json",
        "completion_marker",
    )
    for psi in (8, 16):
        for logical_mode, directory, marker in (
            ("operational", "operational", "P0_5_C3_OPERATIONAL_COMPLETE.json"),
            ("peak_exposure", "peak-exposure", "P0_5_C3_PEAK_EXPOSURE_COMPLETE.json"),
        ):
            logical = f"c3_psi{psi}_{logical_mode}"
            base = f"artifacts/c3/cuda/psi{psi}/{directory}"
            add(f"{logical}.completion_marker", f"{base}/{marker}", "completion_marker")
            add(f"{logical}.summary", f"{base}/summary.json", "validation_summary")
            add(f"{logical}.metrics", f"{base}/metrics.jsonl", "validation_metrics")
    tokenizer_paths = {
        "p0_3_psi8": "artifacts/p0-3/tokenizer-reload-psi8.json",
        "p0_3_psi16": "artifacts/p0-3/tokenizer-reload-psi16.json",
        "p0_4_psi8": "artifacts/p0-4/psi8/tokenizer-reload.json",
        "p0_4_psi16": "artifacts/p0-4/psi16/tokenizer-reload.json",
    }
    for logical, path in tokenizer_paths.items():
        add(f"tokenizer_reload.{logical}", path, "validation_summary")
    add("runner.run_marker", ".level1-requalification-run.json", "provenance")
    add("runner.commands_ledger", "commands.jsonl", "command_record")
    add("runner.environment_ledger", "environment.jsonl", "environment_record")
    for name in REQUIRED_COMMAND_NAMES:
        add(f"runner.log.{name}", f"logs/{name}.log", "other")
        add(f"runner.record.{name}", f"records/{name}.json", "command_record")
    for name in REQUIRED_ENVIRONMENT_NAMES:
        add(f"runner.record.{name}", f"records/{name}.json", "environment_record")
    return specs


def _validate_review(review: Mapping[str, Any]) -> tuple[str, Mapping[str, str]]:
    expected_top = {
        "aggregate",
        "command_ledger",
        "p0_3",
        "p0_4",
        "p0_5_c3",
        "schema_version",
        "status",
        "tested_commit",
        "tokenizer_reload",
    }
    if set(review) != expected_top:
        _fail("full review fields are incomplete or ambiguous")
    if review.get("schema_version") != REVIEW_VERSION or review.get("status") != "passed":
        _fail("full review is not a passed Level 1 raw-evidence review")
    tested_commit = _commit(review.get("tested_commit"), label="review.tested_commit")
    for field in ("p0_3", "p0_4", "p0_5_c3", "tokenizer_reload", "command_ledger"):
        if _mapping(review.get(field), label=f"review.{field}").get("status") != "passed":
            _fail(f"review.{field} is not passed")
    ledger = _mapping(review["command_ledger"], label="review.command_ledger")
    if ledger.get("tested_commit") != tested_commit:
        _fail("review command ledger tested commit differs")
    if set(_list(ledger.get("required_commands"), label="required commands")) != set(
        REQUIRED_COMMAND_NAMES
    ):
        _fail("review command matrix differs from the fixed Stage 5 matrix")
    if set(
        _list(ledger.get("required_environment_records"), label="environment records")
    ) != set(REQUIRED_ENVIRONMENT_NAMES):
        _fail("review environment matrix differs from the fixed Stage 5 matrix")
    aggregate = _mapping(review.get("aggregate"), label="review.aggregate")
    if set(aggregate) != {
        "artifact_count",
        "artifact_hashes",
        "raw_event_counts",
        "review_material_sha256",
        "tested_commit",
    }:
        _fail("review aggregate fields are incomplete or ambiguous")
    if aggregate.get("tested_commit") != tested_commit:
        _fail("review aggregate tested commit differs")
    raw_hashes = _mapping(aggregate.get("artifact_hashes"), label="artifact hashes")
    hashes = {
        str(label): _digest(value, label=f"artifact hash {label}")
        for label, value in raw_hashes.items()
    }
    specs = _fixed_specs()
    if set(hashes) != set(specs):
        missing = sorted(set(specs) - set(hashes))
        extra = sorted(set(hashes) - set(specs))
        _fail(
            "review artifact inventory differs from fixed allowlist: "
            f"missing={missing}, extra={extra}"
        )
    count = aggregate.get("artifact_count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(hashes):
        _fail("review artifact_count differs from its hash inventory")
    raw_counts = _mapping(aggregate.get("raw_event_counts"), label="raw event counts")
    if not raw_counts or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in raw_counts.values()
    ):
        _fail("review raw event counts are invalid")
    material = {
        "artifact_hashes": dict(sorted(hashes.items())),
        "raw_event_counts": raw_counts,
        "tested_commit": tested_commit,
    }
    encoded = (
        json.dumps(
            material,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if _digest(
        aggregate.get("review_material_sha256"),
        label="review material SHA-256",
    ) != sha256_bytes(encoded):
        _fail("review material SHA-256 does not match the complete aggregate")
    return tested_commit, hashes


def _clean_collector_worktree(value: Any, *, label: str) -> Mapping[str, Any]:
    worktree = _mapping(value, label=label)
    if (
        worktree.get("status") != "recorded"
        or worktree.get("clean") is not True
        or worktree.get("staged_changes_present") is not False
        or worktree.get("unstaged_changes_present") is not False
        or worktree.get("untracked_path_count") != 0
        or worktree.get("conflicted_changes_present") is not False
    ):
        _fail(f"{label} is not a recorded clean worktree")
    if worktree.get("staged_change_count") != 0 or worktree.get(
        "unstaged_change_count"
    ) != 0:
        _fail(f"{label} has nonzero change counts")
    porcelain = _mapping(worktree.get("porcelain"), label=f"{label}.porcelain")
    if porcelain.get("byte_count") != 0:
        _fail(f"{label} porcelain byte count is not zero")
    if _digest(
        porcelain.get("sha256"), label=f"{label} porcelain SHA-256"
    ) != hashlib.sha256(b"").hexdigest():
        _fail(f"{label} porcelain SHA-256 is not the empty status hash")
    _timestamp(worktree.get("collected_at_utc"), label=f"{label}.collected_at_utc")
    submodules = _mapping(worktree.get("submodules"), label=f"{label}.submodules")
    if submodules.get("status") != "recorded" or submodules.get("state") not in {
        "none",
        "at_recorded_commit",
    }:
        _fail(f"{label} submodule state is not clean")
    return worktree


def _validate_acceptance(
    provenance: Mapping[str, Any], *, tested_commit: str
) -> tuple[list[dict[str, Any]], Mapping[str, Any], str]:
    if provenance.get("format_version") != PROVENANCE_VERSION:
        _fail("acceptance provenance format_version is invalid")
    if provenance.get("context") != "evidence_handoff":
        _fail("acceptance provenance context is not evidence_handoff")
    repository = _mapping(provenance.get("repository"), label="provenance.repository")
    if repository.get("head_commit") != tested_commit:
        _fail("acceptance provenance HEAD differs from the tested commit")
    branch = _mapping(repository.get("branch"), label="provenance.repository.branch")
    if branch.get("status") != "recorded":
        _fail("acceptance provenance must record a named working branch")
    branch_value = _string(branch.get("value"), label="working branch")
    _assert_public_string(branch_value, label="working branch")
    worktree = _clean_collector_worktree(
        repository.get("worktree"), label="acceptance worktree"
    )
    acceptance = _mapping(
        provenance.get("acceptance_review"), label="acceptance provenance review"
    )
    if acceptance.get("status") != "recorded":
        _fail("acceptance review must be explicitly recorded")
    raw_reviewers = _list(acceptance.get("reviewers"), label="acceptance reviewers")
    if not raw_reviewers:
        _fail("acceptance review must name at least one reviewer")
    reviewers: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_reviewers):
        reviewer = _mapping(raw, label=f"acceptance reviewer {index}")
        required = {
            "identifier",
            "raw_events_reviewed",
            "review_commit",
            "review_method",
            "reviewed_at_utc",
            "role",
        }
        if set(reviewer) != required:
            _fail(f"acceptance reviewer {index} fields are incomplete or ambiguous")
        identifier = _string(reviewer.get("identifier"), label="reviewer identifier")
        method = _string(reviewer.get("review_method"), label="review method")
        _assert_public_string(identifier, label="reviewer identifier")
        _assert_public_string(method, label="review method")
        if identifier in identifiers:
            _fail("acceptance reviewer identifiers must be unique")
        identifiers.add(identifier)
        if reviewer.get("role") != "evidence_reviewer":
            _fail("acceptance provenance reviewer role must be evidence_reviewer")
        if reviewer.get("raw_events_reviewed") is not True:
            _fail("acceptance review must explicitly cover every raw event")
        if reviewer.get("review_commit") != tested_commit:
            _fail("acceptance review commit differs from the tested commit")
        _timestamp(reviewer.get("reviewed_at_utc"), label="reviewed_at_utc")
        reviewers.append(dict(reviewer))
    reviewers.sort(key=lambda item: (item["identifier"].casefold(), item["identifier"]))
    return reviewers, worktree, branch_value


def _worktree_projection(worktree: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clean": True,
        "collected_at_utc": worktree["collected_at_utc"],
        "porcelain_format": "git-status-porcelain-v1",
        "porcelain_sha256": worktree["porcelain"]["sha256"],
        "staged_changes": False,
        "status": "recorded",
        "unstaged_changes": False,
        "untracked_count": 0,
    }




def _registered_worktrees() -> tuple[Path, ...]:
    listing = _git_stdout(
        ("worktree", "list", "--porcelain", "-z"),
        label="registered worktrees",
    )
    roots: list[Path] = []
    for field in listing.split(b"\0"):
        if not field.startswith(b"worktree "):
            continue
        try:
            raw = field[len(b"worktree ") :].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("registered Git worktree path is not UTF-8")
        path = Path(raw)
        if not path.is_absolute():
            _fail("Git returned a non-absolute registered worktree")
        roots.append(path.resolve(strict=False))
    if not roots:
        _fail("Git returned no registered worktrees")
    return tuple(roots)


def _git_stdout(arguments: Sequence[str], *, label: str) -> bytes:
    repository_root = _canonical_root(
        CANONICAL_REPOSITORY_ROOT, label="canonical repository root"
    )
    environment = os.environ.copy()
    for name in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    ):
        environment.pop(name, None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository_root), *arguments],
            check=False,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        _fail(f"could not inspect live Git {label}: {exc}")
    if completed.returncode != 0:
        _fail(f"live Git {label} command failed")
    return completed.stdout


def _validate_live_repository_state(
    *,
    expected_commit: str,
    expected_branch: str,
    collector_worktree: Mapping[str, Any],
    phase: str,
) -> None:
    repository_root = _canonical_root(
        CANONICAL_REPOSITORY_ROOT, label="canonical repository root"
    )
    expected_top = os.fsencode(repository_root) + b"\n"
    live_top = _git_stdout(
        ("rev-parse", "--show-toplevel"), label=f"{phase} top level"
    )
    if live_top != expected_top:
        _fail(f"live Git repository root differs during {phase}")

    expected_commit_value = _commit(expected_commit, label=f"{phase} expected commit")
    head_raw = _git_stdout(
        ("rev-parse", "--verify", "HEAD"), label=f"{phase} HEAD"
    )
    if head_raw != expected_commit_value.encode("ascii") + b"\n":
        _fail(f"live Git HEAD differs during {phase}")

    branch_value = _string(expected_branch, label=f"{phase} expected branch")
    _assert_public_string(branch_value, label=f"{phase} expected branch")
    branch_raw = _git_stdout(
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        label=f"{phase} branch",
    )
    if branch_raw != branch_value.encode("utf-8") + b"\n":
        _fail(f"live Git branch differs during {phase}")

    porcelain_raw = _git_stdout(
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        label=f"{phase} worktree status",
    )
    if porcelain_raw:
        _fail(f"live Git worktree is not clean during {phase}")
    porcelain = _mapping(
        collector_worktree.get("porcelain"), label=f"{phase} collector porcelain"
    )
    if (
        porcelain.get("byte_count") != len(porcelain_raw)
        or porcelain.get("sha256") != sha256_bytes(porcelain_raw)
    ):
        _fail(f"live Git porcelain differs from collector provenance during {phase}")

    for arguments, label in (
        (("diff", "--check"), "worktree diff-check"),
        (("diff", "--cached", "--check"), "index diff-check"),
    ):
        if _git_stdout(arguments, label=f"{phase} {label}"):
            _fail(f"live Git {label} emitted output during {phase}")

    submodule_raw = _git_stdout(
        ("submodule", "status", "--recursive"),
        label=f"{phase} submodule status",
    )
    submodules = _mapping(
        collector_worktree.get("submodules"), label=f"{phase} collector submodules"
    )
    if (
        submodules.get("byte_count") != len(submodule_raw)
        or submodules.get("sha256") != sha256_bytes(submodule_raw)
    ):
        _fail(f"live Git submodules differ from collector provenance during {phase}")


def _validate_commit_evidence_blobs(
    *,
    commit: str,
    tested_commit: str,
    expected_blobs: Mapping[str, bytes],
) -> None:
    commit_value = _commit(commit, label="evidence commit")
    tested_commit_value = _commit(tested_commit, label="tested source commit")
    if not expected_blobs:
        _fail("evidence commit blob set must not be empty")
    results_prefix = "docs/validation_results"
    fixed_paths = {
        f"{results_prefix}/{DESCRIPTOR_NAME}",
        f"{results_prefix}/{EXACT_VERIFICATION_NAME}",
        f"{results_prefix}/{SANITIZED_VERIFICATION_NAME}",
        f"{results_prefix}/{SUMMARY_JSON_NAME}",
        f"{results_prefix}/{SUMMARY_MARKDOWN_NAME}",
    }
    if set(expected_blobs) != fixed_paths:
        _fail("evidence commit blob set is not the fixed five closure files")
    if _git_stdout(
        ("cat-file", "-t", commit_value), label="evidence commit object type"
    ) != b"commit\n":
        _fail("evidence commit is not a commit object")
    parents = _git_stdout(
        ("rev-list", "--parents", "-n", "1", commit_value),
        label="evidence commit parents",
    )
    if parents != f"{commit_value} {tested_commit_value}\n".encode("ascii"):
        _fail("evidence commit must have the tested source as its single parent")

    changed = _git_stdout(
        (
            "diff-tree",
            "--no-commit-id",
            "-r",
            "--name-status",
            "-z",
            "--no-renames",
            tested_commit_value,
            commit_value,
            "--",
        ),
        label="evidence commit exact change set",
    )
    if not changed.endswith(b"\0"):
        _fail("evidence commit change set is malformed")
    fields = changed[:-1].split(b"\0") if changed else []
    if len(fields) % 2:
        _fail("evidence commit change set is malformed")
    actual_paths: list[str] = []
    for index in range(0, len(fields), 2):
        status_raw, path_raw = fields[index : index + 2]
        if status_raw != b"A":
            _fail("evidence commit must add every evidence file without rewriting")
        try:
            path = path_raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("evidence commit change path is not UTF-8")
        actual_paths.append(path)
    expected_paths = sorted(
        expected_blobs, key=lambda path: path.encode("utf-8")
    )
    if actual_paths != expected_paths:
        _fail("evidence commit change set is not exactly the five closure files")
    for repository_path in sorted(expected_blobs):
        pure = PurePosixPath(repository_path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != repository_path
        ):
            _fail("internal evidence commit path is not canonical")
        listing = _git_stdout(
            (
                "ls-tree",
                "-z",
                "--full-tree",
                commit_value,
                "--",
                repository_path,
            ),
            label=f"commit evidence tree entry {repository_path}",
        )
        if not listing.endswith(b"\0") or listing.count(b"\0") != 1:
            _fail(f"evidence commit is missing one exact blob: {repository_path}")
        record = listing[:-1]
        try:
            metadata, listed_path = record.split(b"\t", 1)
            mode, object_type, object_id_raw = metadata.split(b" ", 2)
            object_id = object_id_raw.decode("ascii", errors="strict")
        except (ValueError, UnicodeDecodeError):
            _fail(f"evidence commit tree entry is malformed: {repository_path}")
        if (
            mode != b"100644"
            or object_type != b"blob"
            or listed_path != repository_path.encode("utf-8")
        ):
            _fail(f"evidence commit entry is not the exact regular blob: {repository_path}")
        _commit(object_id, label=f"evidence blob object ID for {repository_path}")
        committed = _git_stdout(
            ("cat-file", "blob", object_id),
            label=f"commit evidence blob {repository_path}",
        )
        if committed != expected_blobs[repository_path]:
            _fail(f"evidence commit blob bytes differ: {repository_path}")


def _validate_implementation_base_commit(value: Any, *, tested_commit: str) -> str:
    base_commit = _commit(value, label="implementation base commit")
    tested_commit_value = _commit(tested_commit, label="tested source commit")
    if base_commit != IMPLEMENTATION_BASE_COMMIT:
        _fail("implementation base commit differs from the fixed Stage 5 base")
    base_type = _git_stdout(
        ("cat-file", "-t", base_commit), label="implementation base commit object type"
    )
    if base_type != b"commit\n":
        _fail("implementation base is not a commit object")
    tested_type = _git_stdout(
        ("cat-file", "-t", tested_commit_value), label="tested source commit object type"
    )
    if tested_type != b"commit\n":
        _fail("tested source is not a commit object")
    try:
        ancestry = _git_stdout(
            ("merge-base", "--is-ancestor", base_commit, tested_commit_value),
            label="implementation base ancestry",
        )
    except EvidenceBuildError:
        _fail("implementation base is not an ancestor of the tested source")
    if ancestry:
        _fail("implementation base ancestry command emitted unexpected output")
    return base_commit


def _validated_roots(
    run_root_value: str | os.PathLike[str],
    results_root_value: str | os.PathLike[str],
) -> tuple[Path, Path]:
    repository_root = _canonical_root(
        CANONICAL_REPOSITORY_ROOT, label="canonical repository root"
    )
    expected_results = _canonical_root(
        CANONICAL_RESULTS_ROOT, label="canonical validation-results root"
    )
    results_root = _canonical_root(results_root_value, label="results root")
    if results_root != expected_results:
        _fail(
            "results root must be the canonical repository validation-results directory"
        )
    run_root = _canonical_root(run_root_value, label="run root")
    if run_root == repository_root or repository_root in run_root.parents:
        _fail("run root must be outside the canonical repository")
    for worktree in _registered_worktrees():
        if run_root == worktree or worktree in run_root.parents:
            _fail("run root must be outside every registered Git worktree")
    return run_root, results_root


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validated_archive_locations(
    *,
    exact_archive_value: str | os.PathLike[str],
    sanitized_archive_value: str | os.PathLike[str],
    sanitized_staging_dir_value: str | os.PathLike[str],
    run_root: Path,
) -> tuple[Path, Path]:
    exact_archive = _canonical_file(exact_archive_value, label="exact archive")
    sanitized_archive = _canonical_file(
        sanitized_archive_value, label="sanitized archive"
    )
    sanitized_staging = _canonical_root(
        sanitized_staging_dir_value, label="sanitized staging directory"
    )
    if exact_archive == sanitized_archive:
        _fail("exact and sanitized archive paths must be distinct")
    if sanitized_archive.parent != sanitized_staging:
        _fail(
            "sanitized archive must be a direct child of its explicit staging directory"
        )
    if _is_within(exact_archive, sanitized_staging):
        _fail("exact archive must be retained separately from sanitized staging")

    repository_root = _canonical_root(
        CANONICAL_REPOSITORY_ROOT, label="canonical repository root"
    )
    managed_roots = (repository_root, *_registered_worktrees(), run_root)
    for path, label in (
        (exact_archive, "exact archive"),
        (sanitized_archive, "sanitized archive"),
        (sanitized_staging, "sanitized staging directory"),
    ):
        for root in managed_roots:
            if _is_within(path, root):
                if root == run_root:
                    _fail(f"{label} must be outside the ephemeral run root")
                _fail(
                    f"{label} must be outside the canonical repository and every Git worktree"
                )
    return exact_archive, sanitized_archive


def _prepare_output(path: Path, *, label: str) -> None:
    if os.path.lexists(path):
        _fail(f"refusing to overwrite existing {label}: {path.name}")
    _canonical_root(path.parent, label=f"{label} parent")


def _temporary_payload(path: Path, raw: bytes, *, label: str) -> Path:
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, raw_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=path.parent
        )
        temporary = Path(raw_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        _fail(f"could not prepare {label} atomically: {exc}")
    assert temporary is not None
    return temporary


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_written_output(path: Path, *, label: str) -> None:
    status = os.lstat(path)
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or stat.S_IMODE(status.st_mode) != 0o600
    ):
        _fail(f"{label} is not a private singly linked regular file")


def _exclusive_write(path: Path, raw: bytes, *, label: str) -> None:
    temporary = _temporary_payload(path, raw, label=label)
    published = False
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
            published = True
        except FileExistsError:
            _fail(f"refusing to overwrite existing {label}: {path.name}")
        except OSError as exc:
            _fail(f"could not publish {label} exclusively: {exc}")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not published:
        _fail(f"could not publish {label}")
    try:
        _validate_written_output(path, label=label)
        _fsync_parent(path)
    except (OSError, EvidenceBuildError):
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _atomic_replace_expected(
    path: Path,
    *,
    expected_raw: bytes,
    replacement_raw: bytes,
    label: str,
) -> None:
    current = _canonical_file(path, label=label)
    if _stable_read(current, label=label) != expected_raw:
        _fail(f"{label} changed before atomic replacement")
    temporary = _temporary_payload(path, replacement_raw, label=label)
    try:
        if _stable_read(current, label=label) != expected_raw:
            _fail(f"{label} changed during atomic replacement")
        try:
            os.replace(temporary, current)
        except OSError as exc:
            _fail(f"could not atomically replace {label}: {exc}")
        temporary = None
        _validate_written_output(current, label=label)
        _fsync_parent(current)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _summary_documents(
    review: Mapping[str, Any], reviewers: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], bytes]:
    aggregate = _mapping(review["aggregate"], label="review aggregate")
    ledger = _mapping(review["command_ledger"], label="review command ledger")
    summary = {
        "acceptance_review": {
            "reviewers": [dict(item) for item in reviewers],
            "status": "recorded",
        },
        "limitations": [
            "This is an unofficial correctness-first Hugging Face implementation.",
            "The dense quadratic screening path is not evidence of paper efficiency.",
            "No paper-scale training, retrieval benchmark, distributed training, or P1 ecosystem capability is validated.",
        ],
        "review": {
            "artifact_count": aggregate["artifact_count"],
            "command_count": ledger["reviewed_command_count"],
            "environment_record_count": ledger["reviewed_environment_record_count"],
            "ordering_checks": list(ledger["ordering_checks"]),
            "raw_event_counts": dict(aggregate["raw_event_counts"]),
            "review_material_sha256": aggregate["review_material_sha256"],
            "status": "passed",
        },
        "schema_version": SUMMARY_VERSION,
        "tested_commit": review["tested_commit"],
        "validation_gate": GATE,
        "validation_status": "passed",
        "validated_components": {
            "c1_architecture_initialization_all_scale": "passed",
            "c2_position_mipe_cache": "passed",
            "c3_paper_training_contract_diagnostics": "passed",
            "gradient_checkpointing_api_matrix": "passed",
            "p0_1_formula_oracle": "passed",
            "p0_2_three_way": "passed",
            "p0_3_checkpointed_smoke": "passed",
            "p0_4_psi8_psi16_qualification": "passed",
        },
    }
    lines = [
        "# Level 1 Core Requalification Summary",
        "",
        "Status: passed",
        f"Tested commit: {review['tested_commit']}",
        f"Reviewed artifacts: {aggregate['artifact_count']}",
        f"Reviewed raw events: {aggregate['raw_event_counts']['total']}",
        f"Reviewed commands: {ledger['reviewed_command_count']}",
        f"Acceptance reviewers: {', '.join(item['identifier'] for item in reviewers)}",
        "",
        "The reviewed matrix passed P0-1, P0-2, checkpointed P0-3, fresh",
        "Psi=8/Psi=16 P0-4 qualification, C1, C2, gradient-checkpointing",
        "compatibility, and the bounded C3 paper-training-contract diagnostics.",
        "",
        "This is an unofficial correctness-first implementation. The dense",
        "quadratic path is not efficiency evidence. This result does not validate",
        "paper-scale training, retrieval benchmarks, distributed training, or any",
        "P1 model/ecosystem capability.",
        "",
        "Archive retention and descriptor closure are recorded separately in the",
        "Level 1 evidence archive descriptor.",
        "",
    ]
    return summary, "\n".join(lines).encode("utf-8")


def prepare_evidence(
    *,
    run_root_value: str | os.PathLike[str],
    results_root_value: str | os.PathLike[str],
) -> dict[str, Any]:
    run_root, results_root = _validated_roots(
        run_root_value, results_root_value
    )
    review_path = _fixed_child(run_root, FULL_REVIEW_RELATIVE, label="full review")
    provenance_path = _fixed_child(
        run_root, ACCEPTANCE_PROVENANCE_RELATIVE, label="acceptance provenance"
    )
    review, review_raw = _load_json(review_path, label="full review")
    provenance, provenance_raw = _load_json(
        provenance_path, label="acceptance provenance"
    )
    tested_commit, expected_hashes = _validate_review(review)
    reviewers, preedit_worktree, branch = _validate_acceptance(
        provenance, tested_commit=tested_commit
    )
    grouped: dict[str, list[tuple[str, ArtifactSpec]]] = {}
    for label, spec in _fixed_specs().items():
        grouped.setdefault(spec.source_path, []).append((label, spec))
    artifacts: list[dict[str, Any]] = []
    for source_path in sorted(grouped):
        aliases = sorted(grouped[source_path], key=lambda item: item[0])
        classifications = {item.classification for _, item in aliases}
        if len(classifications) != 1:
            _fail(f"fixed alias classifications disagree for {source_path}")
        path = _fixed_child(
            run_root, source_path, label=f"reviewed artifact {source_path}"
        )
        raw = _stable_read(path, label=f"reviewed artifact {source_path}")
        actual = sha256_bytes(raw)
        for label, _spec in aliases:
            if expected_hashes[label] != actual:
                _fail(f"reviewed artifact changed after review: {label}")
        primary_label, primary_spec = aliases[0]
        artifacts.append(
            {
                "archive_path": primary_spec.archive_path,
                "classification": primary_spec.classification,
                "logical_name": primary_label,
                "sha256": actual,
                "source_path": source_path,
                "source_root": "run",
            }
        )
    summary, markdown_raw = _summary_documents(review, reviewers)
    summary_raw = canonical_json_bytes(summary)
    summary_json_path = results_root / SUMMARY_JSON_NAME
    summary_markdown_path = results_root / SUMMARY_MARKDOWN_NAME
    package_input_path = run_root / PACKAGE_INPUT_RELATIVE
    for path, label in (
        (summary_json_path, "machine summary"),
        (summary_markdown_path, "human summary"),
        (package_input_path, "package input"),
    ):
        _prepare_output(path, label=label)
    extras = (
        (
            "level1.full_review",
            "run",
            FULL_REVIEW_RELATIVE,
            "artifacts/level1-core/review/level1-core.json",
            review_raw,
            "validation_summary",
        ),
        (
            "level1.acceptance_provenance",
            "run",
            ACCEPTANCE_PROVENANCE_RELATIVE,
            "artifacts/level1-core/review/acceptance-provenance.json",
            provenance_raw,
            "provenance",
        ),
        (
            "level1.machine_summary",
            "results",
            SUMMARY_JSON_NAME,
            "artifacts/level1-core/summary/LEVEL1_CORE_SUMMARY.json",
            summary_raw,
            "validation_summary",
        ),
        (
            "level1.human_summary",
            "results",
            SUMMARY_MARKDOWN_NAME,
            "artifacts/level1-core/summary/LEVEL1_CORE_SUMMARY.md",
            markdown_raw,
            "validation_summary",
        ),
    )
    for logical, root_name, source_path, archive_path, raw, classification in extras:
        artifacts.append(
            {
                "archive_path": archive_path,
                "classification": classification,
                "logical_name": logical,
                "sha256": sha256_bytes(raw),
                "source_path": source_path,
                "source_root": root_name,
            }
        )
    artifacts.sort(key=lambda item: item["archive_path"])
    logical_names = [item["logical_name"] for item in artifacts]
    archive_paths = [item["archive_path"] for item in artifacts]
    source_keys = [(item["source_root"], item["source_path"]) for item in artifacts]
    if len(logical_names) != len(set(logical_names)):
        _fail("package mapping has duplicate logical names")
    if len(archive_paths) != len(set(archive_paths)):
        _fail("package mapping has duplicate archive paths")
    if len(source_keys) != len(set(source_keys)):
        _fail("package mapping has duplicate source aliases")
    package_input = {
        "artifacts": artifacts,
        "format_version": PACKAGE_INPUT_VERSION,
        "gate": GATE,
        "tested_source_commit": tested_commit,
    }
    package_raw = canonical_json_bytes(package_input)
    _validate_live_repository_state(
        expected_commit=tested_commit,
        expected_branch=branch,
        collector_worktree=preedit_worktree,
        phase="prepare",
    )
    _exclusive_write(summary_json_path, summary_raw, label="machine summary")
    _exclusive_write(summary_markdown_path, markdown_raw, label="human summary")
    _exclusive_write(package_input_path, package_raw, label="package input")
    return {
        "artifact_count": len(artifacts),
        "package_input_sha256": sha256_bytes(package_raw),
        "review_material_sha256": review["aggregate"]["review_material_sha256"],
        "status": "prepared",
        "summary_json_sha256": sha256_bytes(summary_raw),
        "summary_markdown_sha256": sha256_bytes(markdown_raw),
        "tested_commit": tested_commit,
    }


def _expected_package_layout() -> dict[str, tuple[str, str, str, str]]:
    grouped: dict[str, list[tuple[str, ArtifactSpec]]] = {}
    for label, spec in _fixed_specs().items():
        grouped.setdefault(spec.source_path, []).append((label, spec))
    expected: dict[str, tuple[str, str, str, str]] = {}
    for source_path, aliases in grouped.items():
        primary_label, spec = sorted(aliases, key=lambda item: item[0])[0]
        expected[primary_label] = (
            "run",
            source_path,
            spec.archive_path,
            spec.classification,
        )
    expected.update(
        {
            "level1.full_review": (
                "run",
                FULL_REVIEW_RELATIVE,
                "artifacts/level1-core/review/level1-core.json",
                "validation_summary",
            ),
            "level1.acceptance_provenance": (
                "run",
                ACCEPTANCE_PROVENANCE_RELATIVE,
                "artifacts/level1-core/review/acceptance-provenance.json",
                "provenance",
            ),
            "level1.machine_summary": (
                "results",
                SUMMARY_JSON_NAME,
                "artifacts/level1-core/summary/LEVEL1_CORE_SUMMARY.json",
                "validation_summary",
            ),
            "level1.human_summary": (
                "results",
                SUMMARY_MARKDOWN_NAME,
                "artifacts/level1-core/summary/LEVEL1_CORE_SUMMARY.md",
                "validation_summary",
            ),
        }
    )
    return expected


def _validated_package_input(
    run_root: Path, results_root: Path, *, tested_commit: str
) -> tuple[Mapping[str, Any], bytes]:
    path = _fixed_child(run_root, PACKAGE_INPUT_RELATIVE, label="package input")
    document, raw = _load_json(path, label="package input")
    if raw != canonical_json_bytes(document):
        _fail("package input is not canonical JSON")
    if set(document) != {"artifacts", "format_version", "gate", "tested_source_commit"}:
        _fail("package input fields are incomplete or ambiguous")
    if (
        document.get("format_version") != PACKAGE_INPUT_VERSION
        or document.get("gate") != GATE
        or document.get("tested_source_commit") != tested_commit
    ):
        _fail("package input identity differs from the reviewed Stage 5 run")
    entries = _list(document.get("artifacts"), label="package input artifacts")
    expected = _expected_package_layout()
    by_name: dict[str, Mapping[str, Any]] = {}
    archive_paths: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    for index, value in enumerate(entries):
        entry = _mapping(value, label=f"package artifact {index}")
        if set(entry) != {
            "archive_path",
            "classification",
            "logical_name",
            "sha256",
            "source_path",
            "source_root",
        }:
            _fail(f"package artifact {index} fields are incomplete or ambiguous")
        logical = _string(entry.get("logical_name"), label="package logical name")
        if logical in by_name:
            _fail(f"duplicate package logical name: {logical}")
        archive_path = _string(entry.get("archive_path"), label="package archive path")
        source_root = _string(entry.get("source_root"), label="package source root")
        source_path = _string(entry.get("source_path"), label="package source path")
        if archive_path in archive_paths:
            _fail(f"duplicate package archive path: {archive_path}")
        source_key = (source_root, source_path)
        if source_key in source_keys:
            _fail(f"duplicate package source alias: {source_root}={source_path}")
        archive_paths.add(archive_path)
        source_keys.add(source_key)
        by_name[logical] = entry
    if set(by_name) != set(expected):
        _fail("package source set differs from the fixed reviewed artifact set")
    for logical, fields in expected.items():
        entry = by_name[logical]
        actual_fields = (
            entry.get("source_root"),
            entry.get("source_path"),
            entry.get("archive_path"),
            entry.get("classification"),
        )
        if actual_fields != fields:
            _fail(f"package mapping differs for {logical}")
        root_name, source_path, _archive_path, _classification = fields
        root = run_root if root_name == "run" else results_root
        source = _fixed_child(root, source_path, label=f"package source {logical}")
        if _digest(entry.get("sha256"), label=f"package SHA-256 {logical}") != sha256_bytes(
            _stable_read(source, label=f"package source {logical}")
        ):
            _fail(f"package source changed after prepare: {logical}")
    return document, raw


def _validate_package_report(
    report: Mapping[str, Any],
    *,
    tested_commit: str,
    exact_archive: Path,
    sanitized_archive: Path,
) -> dict[str, Mapping[str, Any]]:
    if set(report) != {
        "archives",
        "dry_run",
        "format_version",
        "gate",
        "status",
        "tested_source_commit",
    }:
        _fail("package report fields are incomplete or ambiguous")
    if (
        report.get("format_version") != PACKAGE_REPORT_VERSION
        or report.get("status") != "created"
        or report.get("dry_run") is not False
        or report.get("gate") != GATE
        or report.get("tested_source_commit") != tested_commit
    ):
        _fail("package report is not a written Stage 5 archive pair")
    entries = _list(report.get("archives"), label="package report archives")
    by_kind: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(entries):
        entry = _mapping(value, label=f"package archive {index}")
        if set(entry) != {
            "archive_filename",
            "archive_kind",
            "created_at_utc",
            "manifest_sha256",
            "sha256",
            "size_bytes",
            "written",
        }:
            _fail(f"package archive {index} fields are incomplete or ambiguous")
        kind = entry.get("archive_kind")
        if kind not in {"exact_private", "sanitized_shareable"} or kind in by_kind:
            _fail("package report archive kinds are incomplete or duplicated")
        if entry.get("written") is not True:
            _fail(f"package report says {kind} was not written")
        _timestamp(entry.get("created_at_utc"), label=f"{kind} created_at_utc")
        _digest(entry.get("sha256"), label=f"{kind} SHA-256")
        _digest(entry.get("manifest_sha256"), label=f"{kind} manifest SHA-256")
        if not isinstance(entry.get("size_bytes"), int) or isinstance(
            entry.get("size_bytes"), bool
        ) or entry["size_bytes"] < 0:
            _fail(f"{kind} archive size is invalid")
        by_kind[str(kind)] = entry
    if set(by_kind) != {"exact_private", "sanitized_shareable"}:
        _fail("package report must contain exactly both archive kinds")
    if by_kind["exact_private"]["archive_filename"] != exact_archive.name:
        _fail("exact archive filename differs from the package report")
    if by_kind["sanitized_shareable"]["archive_filename"] != sanitized_archive.name:
        _fail("sanitized archive filename differs from the package report")
    return by_kind


def _validate_primary_report(
    path: Path,
    *,
    archive: Path,
    expected_sha256: str,
    expected_kind: str,
    timestamp: str,
) -> tuple[Mapping[str, Any], bytes]:
    report, raw = _load_json(path, label=f"{expected_kind} primary report")
    if raw != canonical_json_bytes(report):
        _fail(f"{expected_kind} primary report is not canonical JSON")
    generated = verify_archive(
        archive,
        expected_sha256=expected_sha256,
        verification_timestamp_utc=timestamp,
    )
    generated_raw = canonical_json_bytes(generated)
    if raw != generated_raw:
        _fail(f"{expected_kind} primary report differs from independent verification")
    if (
        report.get("status") != "verified"
        or report.get("verified_at_utc") != timestamp
        or _mapping(report.get("archive"), label="primary archive").get(
            "archive_kind"
        )
        != expected_kind
        or _mapping(report.get("checks"), label="primary checks").get(
            "evidence_document"
        )
        != "not_requested"
    ):
        _fail(f"{expected_kind} primary report is not a descriptor-free pass")
    return report, raw


def _load_schema(path_value: str | os.PathLike[str]) -> Mapping[str, Any]:
    path = _canonical_file(path_value, label="evidence schema")
    canonical = _canonical_file(
        CANONICAL_SCHEMA_PATH, label="canonical evidence schema"
    )
    if path != canonical:
        _fail("evidence schema must be the canonical repository v1 schema")
    schema, raw = _load_json(path, label="evidence schema")
    if sha256_bytes(raw) != CANONICAL_SCHEMA_SHA256:
        _fail("canonical evidence schema bytes differ from the pinned v1 schema")
    return schema


def _unknown_worktree(status: str) -> dict[str, Any]:
    return {
        "clean": None,
        "collected_at_utc": None,
        "porcelain_format": None,
        "porcelain_sha256": None,
        "staged_changes": None,
        "status": status,
        "unstaged_changes": None,
        "untracked_count": None,
    }


def _run_worktree(run_root: Path, *, name: str) -> dict[str, Any]:
    record = _fixed_child(
        run_root, f"records/{name}.json", label=f"{name} command record"
    )
    value, _raw = _load_json(record, label=f"{name} command record")
    if (
        value.get("name") != name
        or value.get("record_type") != "command"
        or value.get("returncode") != 0
        or value.get("exit_code") != 0
    ):
        _fail(f"{name} command record does not prove a passed hygiene check")
    collected = _timestamp(value.get("ended_at_utc"), label=f"{name}.ended_at_utc")
    return {
        "clean": True,
        "collected_at_utc": collected,
        "porcelain_format": "git-status-porcelain-v1",
        "porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "staged_changes": False,
        "status": "recorded",
        "unstaged_changes": False,
        "untracked_count": 0,
    }


def _source_descriptors(
    package_input: Mapping[str, Any],
    *,
    run_root: Path,
    results_root: Path,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in _list(package_input.get("artifacts"), label="package artifacts"):
        item = _mapping(entry, label="package artifact")
        root = run_root if item["source_root"] == "run" else results_root
        source = _fixed_child(root, item["source_path"], label="descriptor source")
        raw = _stable_read(source, label=f"descriptor source {item['logical_name']}")
        if sha256_bytes(raw) != item["sha256"]:
            _fail(f"descriptor source hash drift: {item['logical_name']}")
        result.append(
            {
                "archive_path": item["archive_path"],
                "classification": item["classification"],
                "exact_bytes_retained": True,
                "logical_name": item["logical_name"],
                "sanitized_copy_status": "verified",
                "sha256": item["sha256"],
                "size_bytes": len(raw),
            }
        )
    result.sort(key=lambda item: item["archive_path"])
    return result


def _archive_descriptor(
    package: Mapping[str, Any],
    *,
    storage_class: str,
    storage_locator: str,
    verified_at_utc: str,
    verification_report_sha256: str,
) -> dict[str, Any]:
    return {
        "archive_filename": package["archive_filename"],
        "created_at_utc": package["created_at_utc"],
        "manifest_sha256": package["manifest_sha256"],
        "public": False,
        "sha256": package["sha256"],
        "size_bytes": package["size_bytes"],
        "status": "verified",
        "storage_class": storage_class,
        "storage_locator": storage_locator,
        "verification_report_sha256": verification_report_sha256,
        "verified_at_utc": verified_at_utc,
    }


def _verification_result(
    report: Mapping[str, Any], *, archive_kind: str
) -> dict[str, Any]:
    return {
        "archive_kind": archive_kind,
        "archive_sha256": report["archive"]["sha256"],
        "errors": [],
        "status": "verified",
        "verified_at_utc": report["verified_at_utc"],
    }


def _validate_descriptor(
    descriptor: Mapping[str, Any], schema: Mapping[str, Any], *, state: str
) -> None:
    errors = validate_evidence_document(descriptor, schema)
    if errors:
        _fail(f"{state} descriptor is not schema-valid: {'; '.join(errors[:12])}")


def _descriptor_base(
    *,
    package_input: Mapping[str, Any],
    package_archives: Mapping[str, Mapping[str, Any]],
    primary_exact: Mapping[str, Any],
    primary_sanitized: Mapping[str, Any],
    run_root: Path,
    results_root: Path,
    provenance: Mapping[str, Any],
    reviewers: Sequence[Mapping[str, Any]],
    preedit_worktree: Mapping[str, Any],
    branch: str,
    implementation_base_commit: str,
    exact_storage_locator: str,
    sanitized_storage_locator: str,
    verification_timestamp: str,
    exact_report_sha256: str,
    sanitized_report_sha256: str,
) -> dict[str, Any]:
    checks = _mapping(primary_sanitized.get("checks"), label="sanitized checks")
    sanitization_check = _mapping(
        checks.get("sanitization"), label="sanitized primary check"
    )
    descriptor_values = _mapping(
        sanitization_check.get("descriptor_values"),
        label="sanitization descriptor projection",
    )
    if set(descriptor_values) != {
        "files_scanned",
        "replacement_count",
        "report_sha256",
        "rules_applied",
    }:
        _fail("sanitization descriptor projection is incomplete or ambiguous")
    created_values = {
        package_archives[kind]["created_at_utc"]
        for kind in ("exact_private", "sanitized_shareable")
    }
    if len(created_values) != 1:
        _fail("exact and sanitized archives must share one package timestamp")
    created = next(iter(created_values))
    if _timestamp_instant(
        created, label="archive creation timestamp"
    ) > _timestamp_instant(
        verification_timestamp,
        label="archive verification timestamp",
    ):
        _fail("archive creation timestamp is after archive verification")
    return {
        "acceptance_review": {
            "reviewers": [dict(item) for item in reviewers],
            "status": "recorded",
        },
        "archives": {
            "exact_private": _archive_descriptor(
                package_archives["exact_private"],
                storage_class="private_external",
                storage_locator=exact_storage_locator,
                verified_at_utc=verification_timestamp,
                verification_report_sha256=exact_report_sha256,
            ),
            "sanitized_shareable": _archive_descriptor(
                package_archives["sanitized_shareable"],
                storage_class="sanitized_staging",
                storage_locator=sanitized_storage_locator,
                verified_at_utc=verification_timestamp,
                verification_report_sha256=sanitized_report_sha256,
            ),
        },
        "evidence_gate": EVIDENCE_GATE,
        "evidence_handoff_provenance": {
            "archive_created_at_utc": {"status": "recorded", "value": created},
            "archive_verified_at_utc": {
                "status": "recorded",
                "value": verification_timestamp,
            },
            "final_commit": {"status": "pending", "value": None},
            "implementation_base_commit": implementation_base_commit,
            "working_branch": branch,
            "worktree_after_commit": _unknown_worktree("pending"),
            "worktree_before_edits": _worktree_projection(preedit_worktree),
        },
        "evidence_status": "partial",
        "limitations": [
            "Final descriptor self-reference fields remain pending until clean commit A is observed.",
            "This is an unofficial correctness-first Hugging Face implementation.",
            "The dense quadratic screening path is not evidence of paper efficiency.",
            "No paper-scale training, retrieval benchmark, distributed training, or P1 ecosystem capability is validated.",
            "Neither archive is published by default; the exact archive must remain private.",
        ],
        "original_run_provenance": {
            "original_run_review": {"reviewers": [], "status": "not_applicable"},
            "run_worktree_at_end": _run_worktree(
                run_root, name="repository-hygiene-final"
            ),
            "run_worktree_at_start": _run_worktree(
                run_root, name="repository-hygiene"
            ),
        },
        "retention": {
            "descriptor_updated_at_utc": verification_timestamp,
            "exact_private_retained": True,
            "public_asset": None,
            "public_asset_published": False,
            "sanitized_archive_verified": True,
            "status": "verified",
        },
        "sanitization": {
            "files_scanned": descriptor_values["files_scanned"],
            "replacement_count": descriptor_values["replacement_count"],
            "report_sha256": descriptor_values["report_sha256"],
            "rules_applied": descriptor_values["rules_applied"],
            "status": "verified",
            "unresolved_findings": [],
        },
        "schema_version": "1.0.0",
        "source_artifacts": _source_descriptors(
            package_input, run_root=run_root, results_root=results_root
        ),
        "tested_source": {
            "branch": branch,
            "commit": package_input["tested_source_commit"],
            "repository": REPOSITORY_NAME,
        },
        "validation_gate": GATE,
        "validation_status": "passed",
        "verification": {
            "reports": [
                _verification_result(primary_exact, archive_kind="exact_private"),
                _verification_result(
                    primary_sanitized, archive_kind="sanitized_shareable"
                ),
            ],
            "status": "verified",
            "verifier_version": VERIFIER_VERSION,
        },
    }


def seal_evidence(
    *,
    run_root_value: str | os.PathLike[str],
    results_root_value: str | os.PathLike[str],
    schema_value: str | os.PathLike[str],
    package_report_value: str | os.PathLike[str],
    exact_archive_value: str | os.PathLike[str],
    sanitized_archive_value: str | os.PathLike[str],
    sanitized_staging_dir_value: str | os.PathLike[str],
    exact_primary_report_value: str | os.PathLike[str],
    sanitized_primary_report_value: str | os.PathLike[str],
    implementation_base_commit: str,
    exact_storage_locator: str,
    sanitized_storage_locator: str,
    verification_timestamp_utc: str,
) -> dict[str, Any]:
    run_root, results_root = _validated_roots(
        run_root_value, results_root_value
    )
    schema = _load_schema(schema_value)
    exact_archive, sanitized_archive = _validated_archive_locations(
        exact_archive_value=exact_archive_value,
        sanitized_archive_value=sanitized_archive_value,
        sanitized_staging_dir_value=sanitized_staging_dir_value,
        run_root=run_root,
    )
    timestamp = _timestamp(
        verification_timestamp_utc, label="verification timestamp"
    )
    exact_locator = _validated_storage_locator(
        exact_storage_locator, label="exact storage locator"
    )
    sanitized_locator = _validated_storage_locator(
        sanitized_storage_locator, label="sanitized storage locator"
    )
    review, _review_raw = _load_json(
        _fixed_child(run_root, FULL_REVIEW_RELATIVE, label="full review"),
        label="full review",
    )
    tested_commit, _hashes = _validate_review(review)
    base_commit = _validate_implementation_base_commit(
        implementation_base_commit, tested_commit=tested_commit
    )
    provenance, _provenance_raw = _load_json(
        _fixed_child(
            run_root,
            ACCEPTANCE_PROVENANCE_RELATIVE,
            label="acceptance provenance",
        ),
        label="acceptance provenance",
    )
    reviewers, preedit_worktree, branch = _validate_acceptance(
        provenance, tested_commit=tested_commit
    )
    package_input, _package_input_raw = _validated_package_input(
        run_root, results_root, tested_commit=tested_commit
    )
    package_report_path = _canonical_file(
        package_report_value, label="package report"
    )
    package_report, package_report_raw = _load_json(
        package_report_path, label="package report"
    )
    if package_report_raw != canonical_json_bytes(package_report):
        _fail("package report is not canonical JSON")
    package_archives = _validate_package_report(
        package_report,
        tested_commit=tested_commit,
        exact_archive=exact_archive,
        sanitized_archive=sanitized_archive,
    )
    exact_primary_path = _canonical_file(
        exact_primary_report_value, label="exact primary report"
    )
    sanitized_primary_path = _canonical_file(
        sanitized_primary_report_value, label="sanitized primary report"
    )
    exact_primary, _exact_primary_raw = _validate_primary_report(
        exact_primary_path,
        archive=exact_archive,
        expected_sha256=package_archives["exact_private"]["sha256"],
        expected_kind="exact_private",
        timestamp=timestamp,
    )
    sanitized_primary, _sanitized_primary_raw = _validate_primary_report(
        sanitized_primary_path,
        archive=sanitized_archive,
        expected_sha256=package_archives["sanitized_shareable"]["sha256"],
        expected_kind="sanitized_shareable",
        timestamp=timestamp,
    )
    seed = _descriptor_base(
        package_input=package_input,
        package_archives=package_archives,
        primary_exact=exact_primary,
        primary_sanitized=sanitized_primary,
        run_root=run_root,
        results_root=results_root,
        provenance=provenance,
        reviewers=reviewers,
        preedit_worktree=preedit_worktree,
        branch=branch,
        implementation_base_commit=base_commit,
        exact_storage_locator=exact_locator,
        sanitized_storage_locator=sanitized_locator,
        verification_timestamp=timestamp,
        exact_report_sha256="0" * 64,
        sanitized_report_sha256="0" * 64,
    )
    _validate_descriptor(seed, schema, state="seed partial")
    exact_bound = verify_archive(
        exact_archive,
        expected_sha256=package_archives["exact_private"]["sha256"],
        evidence_document=seed,
        verification_timestamp_utc=timestamp,
    )
    sanitized_bound = verify_archive(
        sanitized_archive,
        expected_sha256=package_archives["sanitized_shareable"]["sha256"],
        evidence_document=seed,
        verification_timestamp_utc=timestamp,
    )
    exact_bound_raw = canonical_json_bytes(exact_bound)
    sanitized_bound_raw = canonical_json_bytes(sanitized_bound)
    _assert_public_document(exact_bound, label="exact descriptor-aware report")
    _assert_public_document(sanitized_bound, label="sanitized descriptor-aware report")
    descriptor = copy.deepcopy(seed)
    descriptor["archives"]["exact_private"]["verification_report_sha256"] = (
        sha256_bytes(exact_bound_raw)
    )
    descriptor["archives"]["sanitized_shareable"][
        "verification_report_sha256"
    ] = sha256_bytes(sanitized_bound_raw)
    _validate_descriptor(descriptor, schema, state="sealed partial")
    _assert_public_document(descriptor, label="partial descriptor")
    exact_second = canonical_json_bytes(
        verify_archive(
            exact_archive,
            expected_sha256=package_archives["exact_private"]["sha256"],
            evidence_document=descriptor,
            verification_timestamp_utc=timestamp,
        )
    )
    sanitized_second = canonical_json_bytes(
        verify_archive(
            sanitized_archive,
            expected_sha256=package_archives["sanitized_shareable"]["sha256"],
            evidence_document=descriptor,
            verification_timestamp_utc=timestamp,
        )
    )
    if exact_second != exact_bound_raw or sanitized_second != sanitized_bound_raw:
        _fail("descriptor-aware verification reports are not byte-stable")
    descriptor_raw = canonical_json_bytes(descriptor)
    descriptor_path = results_root / DESCRIPTOR_NAME
    exact_bound_path = results_root / EXACT_VERIFICATION_NAME
    sanitized_bound_path = results_root / SANITIZED_VERIFICATION_NAME
    for path, label in (
        (descriptor_path, "partial descriptor"),
        (exact_bound_path, "exact descriptor-aware report"),
        (sanitized_bound_path, "sanitized descriptor-aware report"),
    ):
        _prepare_output(path, label=label)
    _exclusive_write(
        exact_bound_path, exact_bound_raw, label="exact descriptor-aware report"
    )
    _exclusive_write(
        sanitized_bound_path,
        sanitized_bound_raw,
        label="sanitized descriptor-aware report",
    )
    _exclusive_write(descriptor_path, descriptor_raw, label="partial descriptor")
    return {
        "descriptor_sha256": sha256_bytes(descriptor_raw),
        "exact_verification_report_sha256": sha256_bytes(exact_bound_raw),
        "sanitized_verification_report_sha256": sha256_bytes(sanitized_bound_raw),
        "source_artifact_count": len(descriptor["source_artifacts"]),
        "status": "sealed_partial",
        "tested_commit": tested_commit,
    }


def _validate_commit_provenance(
    provenance: Mapping[str, Any],
    *,
    commit_a: str,
    expected_branch: str,
) -> Mapping[str, Any]:
    if (
        provenance.get("format_version") != PROVENANCE_VERSION
        or provenance.get("context") != "evidence_handoff"
    ):
        _fail("commit-A provenance is not a validation provenance v1 handoff")
    repository = _mapping(
        provenance.get("repository"), label="commit-A provenance repository"
    )
    if repository.get("head_commit") != commit_a:
        _fail("commit-A provenance HEAD differs from --commit-a")
    branch = _mapping(repository.get("branch"), label="commit-A branch")
    if branch.get("status") != "recorded" or branch.get("value") != expected_branch:
        _fail("commit-A provenance branch differs from the partial descriptor")
    return _clean_collector_worktree(
        repository.get("worktree"), label="commit-A worktree"
    )


def _bound_report(
    path: Path,
    *,
    kind: str,
    descriptor: Mapping[str, Any],
    archive: Path,
    timestamp: str,
) -> tuple[Mapping[str, Any], bytes]:
    report, raw = _load_json(path, label=f"{kind} descriptor-aware report")
    if raw != canonical_json_bytes(report):
        _fail(f"{kind} descriptor-aware report is not canonical JSON")
    _assert_public_document(report, label=f"{kind} descriptor-aware report")
    archive_descriptor = descriptor["archives"][kind]
    if sha256_bytes(raw) != archive_descriptor["verification_report_sha256"]:
        _fail(f"{kind} descriptor-aware report hash drift")
    generated = verify_archive(
        archive,
        expected_sha256=archive_descriptor["sha256"],
        evidence_document=descriptor,
        verification_timestamp_utc=timestamp,
    )
    if raw != canonical_json_bytes(generated):
        _fail(f"{kind} descriptor-aware report differs from fresh verification")
    return report, raw


def close_evidence(
    *,
    run_root_value: str | os.PathLike[str],
    results_root_value: str | os.PathLike[str],
    schema_value: str | os.PathLike[str],
    commit_provenance_value: str | os.PathLike[str],
    package_report_value: str | os.PathLike[str],
    implementation_base_commit: str,
    exact_storage_locator: str,
    sanitized_storage_locator: str,
    commit_a: str,
    exact_archive_value: str | os.PathLike[str],
    sanitized_archive_value: str | os.PathLike[str],
    sanitized_staging_dir_value: str | os.PathLike[str],
    verification_timestamp_utc: str,
) -> dict[str, Any]:
    run_root, results_root = _validated_roots(
        run_root_value, results_root_value
    )
    schema = _load_schema(schema_value)
    timestamp = _timestamp(
        verification_timestamp_utc, label="verification timestamp"
    )
    commit_a_value = _commit(commit_a, label="commit A")
    partial_path = _fixed_child(
        results_root, DESCRIPTOR_NAME, label="partial descriptor"
    )
    partial, partial_raw = _load_json(partial_path, label="partial descriptor")
    if partial_raw != canonical_json_bytes(partial):
        _fail("partial descriptor is not canonical JSON")
    _validate_descriptor(partial, schema, state="partial")
    handoff = _mapping(
        partial.get("evidence_handoff_provenance"), label="partial handoff"
    )
    if (
        partial.get("evidence_status") != "partial"
        or _mapping(handoff.get("final_commit"), label="partial final commit").get(
            "status"
        )
        != "pending"
        or _mapping(
            handoff.get("worktree_after_commit"), label="partial post-commit worktree"
        ).get("status")
        != "pending"
    ):
        _fail("partial descriptor is not at the seal closure boundary")
    for kind in ("exact_private", "sanitized_shareable"):
        archive_descriptor = _mapping(
            _mapping(partial.get("archives"), label="partial archives").get(kind),
            label=f"partial {kind} archive",
        )
        if (
            archive_descriptor.get("status") != "verified"
            or archive_descriptor.get("verified_at_utc") != timestamp
        ):
            _fail(f"partial {kind} archive timestamp/status differs")

    exact_archive, sanitized_archive = _validated_archive_locations(
        exact_archive_value=exact_archive_value,
        sanitized_archive_value=sanitized_archive_value,
        sanitized_staging_dir_value=sanitized_staging_dir_value,
        run_root=run_root,
    )
    review, _review_raw = _load_json(
        _fixed_child(run_root, FULL_REVIEW_RELATIVE, label="full review"),
        label="full review",
    )
    tested_commit, _hashes = _validate_review(review)
    acceptance_provenance, _acceptance_provenance_raw = _load_json(
        _fixed_child(
            run_root,
            ACCEPTANCE_PROVENANCE_RELATIVE,
            label="acceptance provenance",
        ),
        label="acceptance provenance",
    )
    reviewers, preedit_worktree, branch = _validate_acceptance(
        acceptance_provenance, tested_commit=tested_commit
    )
    base_commit = _validate_implementation_base_commit(
        implementation_base_commit, tested_commit=tested_commit
    )
    exact_locator = _validated_storage_locator(
        exact_storage_locator, label="exact storage locator"
    )
    sanitized_locator = _validated_storage_locator(
        sanitized_storage_locator, label="sanitized storage locator"
    )
    package_input, _package_input_raw = _validated_package_input(
        run_root, results_root, tested_commit=tested_commit
    )
    package_report_path = _canonical_file(
        package_report_value, label="package report"
    )
    package_report, package_report_raw = _load_json(
        package_report_path, label="package report"
    )
    if package_report_raw != canonical_json_bytes(package_report):
        _fail("package report is not canonical JSON")
    package_archives = _validate_package_report(
        package_report,
        tested_commit=tested_commit,
        exact_archive=exact_archive,
        sanitized_archive=sanitized_archive,
    )
    exact_report_path = _fixed_child(
        results_root, EXACT_VERIFICATION_NAME, label="exact descriptor-aware report"
    )
    sanitized_report_path = _fixed_child(
        results_root,
        SANITIZED_VERIFICATION_NAME,
        label="sanitized descriptor-aware report",
    )
    exact_report, exact_report_raw = _bound_report(
        exact_report_path,
        kind="exact_private",
        descriptor=partial,
        archive=exact_archive,
        timestamp=timestamp,
    )
    sanitized_report, sanitized_report_raw = _bound_report(
        sanitized_report_path,
        kind="sanitized_shareable",
        descriptor=partial,
        archive=sanitized_archive,
        timestamp=timestamp,
    )
    reconstructed = _descriptor_base(
        package_input=package_input,
        package_archives=package_archives,
        primary_exact=exact_report,
        primary_sanitized=sanitized_report,
        run_root=run_root,
        results_root=results_root,
        provenance=acceptance_provenance,
        reviewers=reviewers,
        preedit_worktree=preedit_worktree,
        branch=branch,
        implementation_base_commit=base_commit,
        exact_storage_locator=exact_locator,
        sanitized_storage_locator=sanitized_locator,
        verification_timestamp=timestamp,
        exact_report_sha256=sha256_bytes(exact_report_raw),
        sanitized_report_sha256=sha256_bytes(sanitized_report_raw),
    )
    _validate_descriptor(reconstructed, schema, state="reconstructed sealed partial")
    _assert_public_document(reconstructed, label="reconstructed sealed partial")
    reconstructed_raw = canonical_json_bytes(reconstructed)
    if reconstructed_raw != partial_raw:
        _fail("partial descriptor differs from the reconstructed seal boundary")
    handoff = reconstructed["evidence_handoff_provenance"]
    provenance_path = _canonical_file(
        commit_provenance_value, label="commit-A provenance"
    )
    commit_provenance, _commit_provenance_raw = _load_json(
        provenance_path, label="commit-A provenance"
    )
    commit_worktree = _validate_commit_provenance(
        commit_provenance,
        commit_a=commit_a_value,
        expected_branch=branch,
    )
    complete = copy.deepcopy(partial)
    complete["evidence_status"] = "complete"
    complete["evidence_handoff_provenance"]["final_commit"] = {
        "status": "recorded",
        "value": commit_a_value,
    }
    complete["evidence_handoff_provenance"]["worktree_after_commit"] = (
        _worktree_projection(commit_worktree)
    )
    complete["retention"]["descriptor_updated_at_utc"] = commit_worktree[
        "collected_at_utc"
    ]
    complete["limitations"] = [
        "This is an unofficial correctness-first Hugging Face implementation.",
        "The dense quadratic screening path is not evidence of paper efficiency.",
        "No paper-scale training, retrieval benchmark, distributed training, or P1 ecosystem capability is validated.",
        "Neither archive is published by default; the exact archive must remain private.",
    ]
    _validate_descriptor(complete, schema, state="complete")
    exact_after = canonical_json_bytes(
        verify_archive(
            exact_archive,
            expected_sha256=complete["archives"]["exact_private"]["sha256"],
            evidence_document=complete,
            verification_timestamp_utc=timestamp,
        )
    )
    sanitized_after = canonical_json_bytes(
        verify_archive(
            sanitized_archive,
            expected_sha256=complete["archives"]["sanitized_shareable"]["sha256"],
            evidence_document=complete,
            verification_timestamp_utc=timestamp,
        )
    )
    if exact_after != exact_report_raw or sanitized_after != sanitized_report_raw:
        _fail("verification report bytes changed during descriptor closure")
    complete_raw = canonical_json_bytes(complete)
    summary_json_raw = _stable_read(
        _fixed_child(results_root, SUMMARY_JSON_NAME, label="machine summary"),
        label="machine summary",
    )
    summary_markdown_raw = _stable_read(
        _fixed_child(results_root, SUMMARY_MARKDOWN_NAME, label="human summary"),
        label="human summary",
    )
    results_prefix = "docs/validation_results"
    _validate_commit_evidence_blobs(
        commit=commit_a_value,
        tested_commit=tested_commit,
        expected_blobs={
            f"{results_prefix}/{DESCRIPTOR_NAME}": partial_raw,
            f"{results_prefix}/{EXACT_VERIFICATION_NAME}": exact_report_raw,
            f"{results_prefix}/{SANITIZED_VERIFICATION_NAME}": sanitized_report_raw,
            f"{results_prefix}/{SUMMARY_JSON_NAME}": summary_json_raw,
            f"{results_prefix}/{SUMMARY_MARKDOWN_NAME}": summary_markdown_raw,
        },
    )
    _validate_live_repository_state(
        expected_commit=commit_a_value,
        expected_branch=branch,
        collector_worktree=commit_worktree,
        phase="close",
    )
    _atomic_replace_expected(
        partial_path,
        expected_raw=partial_raw,
        replacement_raw=complete_raw,
        label="canonical evidence descriptor",
    )
    return {
        "commit_a": commit_a_value,
        "descriptor_sha256": sha256_bytes(complete_raw),
        "exact_verification_report_sha256": sha256_bytes(exact_report_raw),
        "sanitized_verification_report_sha256": sha256_bytes(
            sanitized_report_raw
        ),
        "status": "closed_complete",
        "tested_commit": complete["tested_source"]["commit"],
    }


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="validate reviewed raw evidence and write summaries/package input",
    )
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--results-root", required=True)

    seal = subparsers.add_parser(
        "seal",
        help="bind verified archives into a schema-valid partial descriptor",
    )
    seal.add_argument("--run-root", required=True)
    seal.add_argument("--results-root", required=True)
    seal.add_argument(
        "--schema",
        default=os.fspath(
            repository_root / "schemas" / "validation_evidence_v1.schema.json"
        ),
    )
    seal.add_argument("--package-report", required=True)
    seal.add_argument("--exact-archive", required=True)
    seal.add_argument("--sanitized-archive", required=True)
    seal.add_argument("--sanitized-staging-dir", required=True)
    seal.add_argument("--exact-primary-report", required=True)
    seal.add_argument("--sanitized-primary-report", required=True)
    seal.add_argument("--implementation-base-commit", required=True)
    seal.add_argument("--exact-storage-locator", required=True)
    seal.add_argument("--sanitized-storage-locator", required=True)
    seal.add_argument("--verification-timestamp-utc", required=True)

    close = subparsers.add_parser(
        "close",
        help="record clean commit-A provenance and write a complete descriptor",
    )
    close.add_argument("--run-root", required=True)
    close.add_argument("--results-root", required=True)
    close.add_argument(
        "--schema",
        default=os.fspath(
            repository_root / "schemas" / "validation_evidence_v1.schema.json"
        ),
    )
    close.add_argument("--commit-provenance", required=True)
    close.add_argument("--package-report", required=True)
    close.add_argument("--implementation-base-commit", required=True)
    close.add_argument("--exact-storage-locator", required=True)
    close.add_argument("--sanitized-storage-locator", required=True)
    close.add_argument("--commit-a", required=True)
    close.add_argument("--exact-archive", required=True)
    close.add_argument("--sanitized-archive", required=True)
    close.add_argument("--sanitized-staging-dir", required=True)
    close.add_argument("--verification-timestamp-utc", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_evidence(
                run_root_value=args.run_root,
                results_root_value=args.results_root,
            )
        elif args.command == "seal":
            result = seal_evidence(
                run_root_value=args.run_root,
                results_root_value=args.results_root,
                schema_value=args.schema,
                package_report_value=args.package_report,
                exact_archive_value=args.exact_archive,
                sanitized_archive_value=args.sanitized_archive,
                sanitized_staging_dir_value=args.sanitized_staging_dir,
                exact_primary_report_value=args.exact_primary_report,
                sanitized_primary_report_value=args.sanitized_primary_report,
                implementation_base_commit=args.implementation_base_commit,
                exact_storage_locator=args.exact_storage_locator,
                sanitized_storage_locator=args.sanitized_storage_locator,
                verification_timestamp_utc=args.verification_timestamp_utc,
            )
        else:
            result = close_evidence(
                run_root_value=args.run_root,
                results_root_value=args.results_root,
                schema_value=args.schema,
                commit_provenance_value=args.commit_provenance,
                package_report_value=args.package_report,
                implementation_base_commit=args.implementation_base_commit,
                exact_storage_locator=args.exact_storage_locator,
                sanitized_storage_locator=args.sanitized_storage_locator,
                commit_a=args.commit_a,
                exact_archive_value=args.exact_archive,
                sanitized_archive_value=args.sanitized_archive,
                sanitized_staging_dir_value=args.sanitized_staging_dir,
                verification_timestamp_utc=args.verification_timestamp_utc,
            )
    except EvidenceBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except InputValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except IntegrityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"error: evidence I/O failed: {exc}", file=sys.stderr)
        return 4
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
