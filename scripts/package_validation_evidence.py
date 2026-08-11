#!/usr/bin/env python3
"""Create deterministic exact/private and sanitized validation archives.

Packaging is allowlist-only.  Every source file must be named in a v1 package
input, remain below an explicitly mapped source root, and match its expected
SHA-256.  The exact archive preserves those bytes unchanged.  The sanitized
archive accepts UTF-8 evidence only, applies the shared fail-closed sanitizer,
and records a deterministic sanitization report.
"""

from __future__ import annotations

import argparse
import getpass
import gzip
import io
import json
import os
import re
import socket
import stat
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

try:
    from validation_evidence_common import (
        ARCHIVE_FORMAT_VERSION,
        MAX_ARCHIVE_CONTROL_MEMBER_COUNT,
        MAX_ARCHIVE_MEMBER_COUNT,
        MAX_ARCHIVE_MEMBER_SIZE_BYTES,
        MAX_SOURCE_ARTIFACT_COUNT,
        MAX_TOTAL_ARCHIVE_MEMBER_BYTES,
        MAX_TOTAL_CONTROL_BYTES,
        MAX_TOTAL_SOURCE_BYTES,
        PACKAGE_INPUT_VERSION,
        SANITIZATION_FORMAT_VERSION,
        SANITIZATION_REPORT,
        SANITIZATION_RULES,
        SANITIZATION_RULESET_VERSION,
        SOURCE_ARTIFACT_CLASSIFICATIONS,
        InputValidationError,
        IntegrityError,
        build_sha256sums,
        canonical_json_bytes,
        ensure_payload_archive_path,
        safe_archive_path,
        sanitized_member,
        sha256_bytes,
        utc_now,
    )
except ModuleNotFoundError:  # Support importing this file as scripts.* in tests.
    from scripts.validation_evidence_common import (  # type: ignore[no-redef]
        ARCHIVE_FORMAT_VERSION,
        MAX_ARCHIVE_CONTROL_MEMBER_COUNT,
        MAX_ARCHIVE_MEMBER_COUNT,
        MAX_ARCHIVE_MEMBER_SIZE_BYTES,
        MAX_SOURCE_ARTIFACT_COUNT,
        MAX_TOTAL_ARCHIVE_MEMBER_BYTES,
        MAX_TOTAL_CONTROL_BYTES,
        MAX_TOTAL_SOURCE_BYTES,
        PACKAGE_INPUT_VERSION,
        SANITIZATION_FORMAT_VERSION,
        SANITIZATION_REPORT,
        SANITIZATION_RULES,
        SANITIZATION_RULESET_VERSION,
        SOURCE_ARTIFACT_CLASSIFICATIONS,
        InputValidationError,
        IntegrityError,
        build_sha256sums,
        canonical_json_bytes,
        ensure_payload_archive_path,
        safe_archive_path,
        sanitized_member,
        sha256_bytes,
        utc_now,
    )


PACKAGE_REPORT_VERSION = "validation-evidence-package-report-v1"
ARCHIVE_KINDS = frozenset({"exact_private", "sanitized_shareable"})
ROOT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
LOGICAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")
BANNED_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".gguf",
)
BANNED_DIRECTORY_NAMES = frozenset(
    {
        "checkpoint",
        "checkpoints",
        "model-weights",
        "model_weights",
        "optimizer-state",
        "optimizer_state",
    }
)


class PackagingIOError(Exception):
    """An explicitly requested file operation could not be completed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise InputValidationError(f"non-finite JSON number is forbidden: {value}")


def _load_package_input(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PackagingIOError(
            f"cannot read package input {path.name!r}: {exc.strerror or exc.__class__.__name__}"
        ) from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise InputValidationError("package input must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"invalid package input JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise InputValidationError("package input must contain a JSON object")
    return value


def _required_text(
    value: Any,
    *,
    field: str,
    pattern: re.Pattern[str] | None = None,
    maximum: int = 1024,
) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{field} must be a non-empty string")
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        raise InputValidationError(f"{field} contains control characters or is too long")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise InputValidationError(f"{field} has an invalid format: {value!r}")
    return value


def _strict_relative_path(value: Any, *, field: str) -> str:
    path = safe_archive_path(value, field=field)
    if (
        WINDOWS_DRIVE_RE.match(path)
        or path.startswith("//")
        or any(":" in part for part in PurePosixPath(path).parts)
    ):
        raise InputValidationError(f"drive-qualified or UNC {field} is forbidden: {path!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise InputValidationError(f"control character in {field}: {path!r}")
    return path


def _check_banned_path(value: str, *, field: str) -> None:
    parts = PurePosixPath(value).parts
    for part in parts:
        lowered = part.casefold()
        if lowered in BANNED_DIRECTORY_NAMES or lowered.startswith("checkpoint-"):
            raise InputValidationError(f"checkpoint directory is forbidden in {field}: {value!r}")
    lowered_name = parts[-1].casefold()
    if lowered_name.endswith(BANNED_SUFFIXES):
        raise InputValidationError(f"model/checkpoint suffix is forbidden in {field}: {value!r}")


def _parse_root_mapping(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise InputValidationError("--root must use NAME=PATH")
        name, raw_path = raw.split("=", 1)
        _required_text(name, field="--root name", pattern=ROOT_NAME_RE, maximum=64)
        if name in roots:
            raise InputValidationError(f"duplicate --root mapping: {name!r}")
        if not raw_path:
            raise InputValidationError(f"--root {name!r} has an empty path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        lexical = Path(os.path.abspath(os.fspath(path)))
        _validate_root_path(lexical, name=name)
        roots[name] = lexical.resolve(strict=True)
    return roots


def _validate_root_path(path: Path, *, name: str) -> None:
    """Reject a source root reached through any symlink component."""

    anchor = Path(path.anchor)
    current = anchor
    parts = path.parts[1:] if path.anchor else path.parts
    try:
        for part in parts:
            current = current / part
            status = os.lstat(current)
            if stat.S_ISLNK(status.st_mode):
                raise InputValidationError(
                    f"--root {name!r} contains a symlink component"
                )
    except FileNotFoundError as exc:
        raise InputValidationError(f"--root {name!r} does not exist") from exc
    except PermissionError as exc:
        raise PackagingIOError(f"cannot inspect --root {name!r}: permission denied") from exc
    try:
        root_status = os.lstat(path)
    except OSError as exc:
        raise PackagingIOError(
            f"cannot inspect --root {name!r}: {exc.strerror or exc.__class__.__name__}"
        ) from exc
    if not stat.S_ISDIR(root_status.st_mode):
        raise InputValidationError(f"--root {name!r} is not a directory")


def _source_component_snapshots(root: Path, relative_path: str) -> tuple[Path, list[tuple[Path, tuple[int, int, int]]]]:
    current = root
    snapshots: list[tuple[Path, tuple[int, int, int]]] = []
    parts = PurePosixPath(relative_path).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError as exc:
            raise InputValidationError(f"allowlisted source file does not exist: {relative_path!r}") from exc
        except PermissionError as exc:
            raise PackagingIOError(
                f"cannot inspect allowlisted source {relative_path!r}: permission denied"
            ) from exc
        if stat.S_ISLNK(status.st_mode):
            raise InputValidationError(f"symlink source component is forbidden: {relative_path!r}")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(status.st_mode):
                raise InputValidationError(
                    f"non-directory source component in {relative_path!r}"
                )
        elif not stat.S_ISREG(status.st_mode):
            raise InputValidationError(f"source must be a regular file: {relative_path!r}")
        elif status.st_nlink != 1:
            raise InputValidationError(f"hard-linked source leaf is forbidden: {relative_path!r}")
        snapshots.append(
            (current, (status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode)))
        )
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"source escapes its explicit root: {relative_path!r}") from exc
    return current, snapshots


def _read_source(root: Path, relative_path: str) -> bytes:
    source, snapshots = _source_component_snapshots(root, relative_path)
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY  # type: ignore[attr-defined]
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW  # type: ignore[attr-defined]
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise PackagingIOError(
            f"cannot open allowlisted source {relative_path!r}: "
            f"{exc.strerror or exc.__class__.__name__}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InputValidationError(f"source must remain a regular file: {relative_path!r}")
        if before.st_nlink != 1:
            raise InputValidationError(f"hard-linked source leaf is forbidden: {relative_path!r}")
        if before.st_size > MAX_ARCHIVE_MEMBER_SIZE_BYTES:
            raise InputValidationError(
                f"source exceeds {MAX_ARCHIVE_MEMBER_SIZE_BYTES}-byte limit: {relative_path!r}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise IntegrityError(f"source changed or truncated while reading: {relative_path!r}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise IntegrityError(f"source grew while reading: {relative_path!r}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise IntegrityError(f"source changed while packaging: {relative_path!r}")
    for component, expected in snapshots:
        try:
            observed = os.lstat(component)
        except OSError as exc:
            raise IntegrityError(f"source path changed while packaging: {relative_path!r}") from exc
        actual = (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode))
        if actual != expected or stat.S_ISLNK(observed.st_mode):
            raise IntegrityError(f"source path changed while packaging: {relative_path!r}")
    return b"".join(chunks)


def _validate_package_input(
    document: Mapping[str, Any],
    roots: Mapping[str, Path],
) -> tuple[str, str, list[dict[str, Any]]]:
    expected_keys = {"format_version", "gate", "tested_source_commit", "artifacts"}
    unexpected = sorted(set(document) - expected_keys)
    missing = sorted(expected_keys - set(document))
    if missing:
        raise InputValidationError(f"package input is missing fields: {missing!r}")
    if unexpected:
        raise InputValidationError(f"package input has unexpected fields: {unexpected!r}")
    if document.get("format_version") != PACKAGE_INPUT_VERSION:
        raise InputValidationError(
            f"format_version must be {PACKAGE_INPUT_VERSION!r}"
        )
    gate = _required_text(document.get("gate"), field="gate", maximum=128)
    commit = _required_text(
        document.get("tested_source_commit"),
        field="tested_source_commit",
        pattern=COMMIT_RE,
        maximum=64,
    )
    raw_artifacts = document.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise InputValidationError("artifacts must be a non-empty array")
    if len(raw_artifacts) > MAX_SOURCE_ARTIFACT_COUNT:
        raise InputValidationError(
            f"artifacts exceeds the {MAX_SOURCE_ARTIFACT_COUNT}-entry limit"
        )

    required_artifact_keys = {
        "logical_name",
        "source_root",
        "source_path",
        "archive_path",
        "sha256",
        "classification",
    }
    logical_names: set[str] = set()
    archive_paths: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    artifacts: list[dict[str, Any]] = []
    total_size = 0
    for index, raw_entry in enumerate(raw_artifacts):
        context = f"artifacts[{index}]"
        if not isinstance(raw_entry, dict):
            raise InputValidationError(f"{context} must be an object")
        missing_entry = sorted(required_artifact_keys - set(raw_entry))
        unexpected_entry = sorted(set(raw_entry) - required_artifact_keys)
        if missing_entry:
            raise InputValidationError(f"{context} is missing fields: {missing_entry!r}")
        if unexpected_entry:
            raise InputValidationError(
                f"{context} has unexpected fields: {unexpected_entry!r}"
            )
        logical_name = _required_text(
            raw_entry.get("logical_name"),
            field=f"{context}.logical_name",
            pattern=LOGICAL_NAME_RE,
            maximum=128,
        )
        source_root = _required_text(
            raw_entry.get("source_root"),
            field=f"{context}.source_root",
            pattern=ROOT_NAME_RE,
            maximum=64,
        )
        if source_root not in roots:
            raise InputValidationError(
                f"{context}.source_root has no matching --root: {source_root!r}"
            )
        source_path = _strict_relative_path(
            raw_entry.get("source_path"), field=f"{context}.source_path"
        )
        archive_path = ensure_payload_archive_path(raw_entry.get("archive_path"))
        if (
            WINDOWS_DRIVE_RE.match(archive_path)
            or any(":" in part for part in PurePosixPath(archive_path).parts)
        ):
            raise InputValidationError(f"{context}.archive_path is drive-qualified")
        if any(ord(character) < 32 or ord(character) == 127 for character in archive_path):
            raise InputValidationError(f"control character in {context}.archive_path")
        _check_banned_path(source_path, field=f"{context}.source_path")
        _check_banned_path(archive_path, field=f"{context}.archive_path")
        expected_sha256 = raw_entry.get("sha256")
        if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise InputValidationError(
                f"{context}.sha256 must be a lowercase SHA-256 digest"
            )
        classification = _required_text(
            raw_entry.get("classification"),
            field=f"{context}.classification",
            maximum=128,
        )
        if classification not in SOURCE_ARTIFACT_CLASSIFICATIONS:
            raise InputValidationError(
                f"{context}.classification must be one of "
                f"{SOURCE_ARTIFACT_CLASSIFICATIONS!r}"
            )
        if logical_name in logical_names:
            raise InputValidationError(f"duplicate logical_name: {logical_name!r}")
        if archive_path in archive_paths:
            raise InputValidationError(f"duplicate archive_path: {archive_path!r}")
        source_key = (source_root, source_path)
        if source_key in source_keys:
            raise InputValidationError(
                f"duplicate source selection: {source_root}={source_path}"
            )
        raw = _read_source(roots[source_root], source_path)
        actual_sha256 = sha256_bytes(raw)
        if actual_sha256 != expected_sha256:
            raise IntegrityError(
                f"source SHA-256 mismatch for logical artifact {logical_name!r}"
            )
        total_size += len(raw)
        if total_size > MAX_TOTAL_SOURCE_BYTES:
            raise InputValidationError(
                f"source artifacts exceed the {MAX_TOTAL_SOURCE_BYTES}-byte limit"
            )
        logical_names.add(logical_name)
        archive_paths.add(archive_path)
        source_keys.add(source_key)
        artifacts.append(
            {
                "archive_path": archive_path,
                "classification": classification,
                "logical_name": logical_name,
                "raw": raw,
                "source_sha256": actual_sha256,
                "source_size_bytes": len(raw),
            }
        )
    artifacts.sort(key=lambda item: item["archive_path"])
    return gate, commit, artifacts


def _media_type(path: str) -> str:
    lowered = path.casefold()
    if lowered.endswith(".json"):
        return "application/json"
    if lowered.endswith(".jsonl") or lowered.endswith(".ndjson"):
        return "application/x-ndjson"
    if lowered.endswith(".md") or lowered.endswith(".markdown"):
        return "text/markdown"
    return "text/plain"


def _source_member_entry(
    artifact: Mapping[str, Any],
    payload: bytes,
    *,
    transformation: str,
) -> dict[str, Any]:
    return {
        "classification": artifact["classification"],
        "kind": "source_artifact",
        "logical_name": artifact["logical_name"],
        "media_type": _media_type(artifact["archive_path"]),
        "path": artifact["archive_path"],
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "source_sha256": artifact["source_sha256"],
        "source_size_bytes": artifact["source_size_bytes"],
        "transformation": transformation,
    }


def _report_member_entry(payload: bytes) -> dict[str, Any]:
    return {
        "classification": "generated_report",
        "kind": "sanitization_report",
        "logical_name": "sanitization_report",
        "media_type": "application/json",
        "path": SANITIZATION_REPORT,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "transformation": "generated",
    }


def _assert_clean_sanitized_bytes(
    payload: bytes,
    *,
    archive_path: str,
    sensitive_values: Iterable[str],
) -> None:
    rescanned, report = sanitized_member(
        payload,
        archive_path=archive_path,
        sensitive_values=sensitive_values,
    )
    if rescanned != payload or report.get("replacements") or report.get("unresolved_findings"):
        raise IntegrityError(
            f"sanitized member {archive_path!r} retains sensitive or non-canonical private data"
        )


def _build_sanitized_payloads(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    sensitive_values: Sequence[str],
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    payloads: dict[str, bytes] = {}
    file_reports: list[dict[str, Any]] = []
    replacement_totals: dict[str, int] = {}
    for artifact in artifacts:
        path = artifact["archive_path"]
        sanitized, report = sanitized_member(
            artifact["raw"],
            archive_path=path,
            sensitive_values=sensitive_values,
        )
        if report.get("unresolved_findings"):
            raise IntegrityError(
                f"sanitization left unresolved findings in logical artifact "
                f"{artifact['logical_name']!r}"
            )
        _assert_clean_sanitized_bytes(
            sanitized,
            archive_path=path,
            sensitive_values=sensitive_values,
        )
        replacements = report.get("replacements", {})
        if not isinstance(replacements, dict):
            raise IntegrityError("sanitizer returned an invalid replacements report")
        for rule, count in replacements.items():
            if not isinstance(rule, str) or not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise IntegrityError("sanitizer returned an invalid replacement count")
            replacement_totals[rule] = replacement_totals.get(rule, 0) + count
        payloads[path] = sanitized
        file_reports.append(dict(report))

    sanitization_report = {
        "files_scanned": sorted(file_reports, key=lambda item: item["path"]),
        "final_scan": {
            "high_confidence_findings": 0,
            "status": "passed",
        },
        "format_version": SANITIZATION_FORMAT_VERSION,
        "replacement_totals": [
            {"count": count, "rule_id": rule}
            for rule, count in sorted(replacement_totals.items())
        ],
        "rules_applied": list(SANITIZATION_RULES),
        "ruleset_version": SANITIZATION_RULESET_VERSION,
        "status": "passed",
        "unresolved_findings": [],
    }
    report_bytes = canonical_json_bytes(sanitization_report)
    _assert_clean_sanitized_bytes(
        report_bytes,
        archive_path=SANITIZATION_REPORT,
        sensitive_values=sensitive_values,
    )
    payloads[SANITIZATION_REPORT] = report_bytes
    return payloads, file_reports


def _tar_info(path: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(path)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    info.linkname = ""
    info.pax_headers = {}
    try:
        info.tobuf(format=tarfile.USTAR_FORMAT, encoding="utf-8", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise InputValidationError(
            f"archive path cannot be represented safely in normalized USTAR: {path!r}"
        ) from exc
    return info


def _normalized_tar_gz(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=output,
        mtime=0,
    ) as compressed:
        with tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive:
            for path in sorted(members):
                safe_archive_path(path, field="archive member path")
                payload = members[path]
                archive.addfile(_tar_info(path, len(payload)), io.BytesIO(payload))
    return output.getvalue()


def _validate_archive_layout(
    member_sizes: Mapping[str, int],
    *,
    source_paths: Iterable[str],
) -> None:
    source_set = set(source_paths)
    member_set = set(member_sizes)
    if not source_set.issubset(member_set):
        raise InputValidationError("archive layout is missing one or more source members")
    if len(source_set) > MAX_SOURCE_ARTIFACT_COUNT:
        raise InputValidationError(
            f"archive source members exceed the {MAX_SOURCE_ARTIFACT_COUNT}-entry limit"
        )
    if len(member_set) > MAX_ARCHIVE_MEMBER_COUNT:
        raise InputValidationError(
            f"archive members exceed the {MAX_ARCHIVE_MEMBER_COUNT}-entry limit"
        )
    control_count = len(member_set - source_set)
    if control_count > MAX_ARCHIVE_CONTROL_MEMBER_COUNT:
        raise InputValidationError(
            f"archive controls exceed the {MAX_ARCHIVE_CONTROL_MEMBER_COUNT}-entry limit"
        )

    for path, size in member_sizes.items():
        safe_archive_path(path, field="archive member path")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise InputValidationError(f"archive member {path!r} has an invalid size")
        if size > MAX_ARCHIVE_MEMBER_SIZE_BYTES:
            raise InputValidationError(
                f"archive member {path!r} exceeds the {MAX_ARCHIVE_MEMBER_SIZE_BYTES}-byte limit"
            )

    source_total = sum(member_sizes[path] for path in source_set)
    control_total = sum(member_sizes[path] for path in member_set - source_set)
    total = source_total + control_total
    if source_total > MAX_TOTAL_SOURCE_BYTES:
        raise InputValidationError(
            f"archive source members exceed the {MAX_TOTAL_SOURCE_BYTES}-byte limit"
        )
    if control_total > MAX_TOTAL_CONTROL_BYTES:
        raise InputValidationError(
            f"archive control members exceed the {MAX_TOTAL_CONTROL_BYTES}-byte limit"
        )
    if total > MAX_TOTAL_ARCHIVE_MEMBER_BYTES:
        raise InputValidationError(
            f"archive members exceed the {MAX_TOTAL_ARCHIVE_MEMBER_BYTES}-byte limit"
        )


def _build_archive(
    *,
    archive_kind: str,
    gate: str,
    tested_source_commit: str,
    artifacts: Sequence[Mapping[str, Any]],
    sensitive_values: Sequence[str],
    manifest_sensitive_values: Sequence[str],
) -> tuple[bytes, str]:
    if archive_kind not in ARCHIVE_KINDS:
        raise InputValidationError(f"unsupported archive kind: {archive_kind!r}")
    if archive_kind == "exact_private":
        payloads = {item["archive_path"]: item["raw"] for item in artifacts}
        entries = [
            _source_member_entry(item, item["raw"], transformation="none")
            for item in artifacts
        ]
    else:
        payloads, _file_reports = _build_sanitized_payloads(
            artifacts, sensitive_values=sensitive_values
        )
        entries = [
            _source_member_entry(
                item,
                payloads[item["archive_path"]],
                transformation="sanitized-v1",
            )
            for item in artifacts
        ]
        report_payload = payloads[SANITIZATION_REPORT]
        entries.append(_report_member_entry(report_payload))
    entries.sort(key=lambda item: item["path"])
    manifest = {
        "archive_kind": archive_kind,
        "format_version": ARCHIVE_FORMAT_VERSION,
        "gate": gate,
        "members": entries,
        "tested_source_commit": tested_source_commit,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    if archive_kind == "sanitized_shareable":
        _assert_clean_sanitized_bytes(
            manifest_bytes,
            archive_path="MANIFEST.json",
            sensitive_values=manifest_sensitive_values,
        )
    manifest_sha256 = sha256_bytes(manifest_bytes)
    checksummed = [
        {"path": entry["path"], "sha256": entry["sha256"]}
        for entry in entries
    ]
    checksummed.append({"path": "MANIFEST.json", "sha256": manifest_sha256})
    sha256sums = build_sha256sums(checksummed)
    archive_members = {
        **payloads,
        "MANIFEST.json": manifest_bytes,
        "SHA256SUMS": sha256sums,
    }
    _validate_archive_layout(
        {path: len(payload) for path, payload in archive_members.items()},
        source_paths=(item["archive_path"] for item in artifacts),
    )
    return _normalized_tar_gz(archive_members), manifest_sha256


def _resolved_output_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    lexical = Path(os.path.abspath(os.fspath(expanded)))
    if os.path.lexists(lexical):
        raise InputValidationError(f"output already exists: {lexical.name!r}")
    try:
        resolved = lexical.resolve(strict=False)
    except OSError as exc:
        raise PackagingIOError(
            f"cannot resolve output parent for {lexical.name!r}: "
            f"{exc.strerror or exc.__class__.__name__}"
        ) from exc
    if os.path.lexists(resolved):
        raise InputValidationError(f"resolved output already exists: {resolved.name!r}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _enclosing_git_worktree(path: Path) -> Path | None:
    """Discover a containing worktree from the resolved output path itself."""

    candidate = path.parent
    while not os.path.lexists(candidate):
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    if not candidate.is_dir():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if os.path.lexists(directory / ".git"):
            return directory
    return None


def _atomic_create(path: Path, data: bytes, *, mode: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise PackagingIOError(
            f"cannot prepare output directory for {path.name!r}: "
            f"{exc.strerror or exc.__class__.__name__}"
        ) from exc
    target = resolved_parent / path.name
    if os.path.lexists(target):
        raise InputValidationError(f"output already exists: {target.name!r}")
    descriptor: int | None = None
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=resolved_parent
        )
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise InputValidationError(f"output already exists: {target.name!r}") from exc
        except OSError as exc:
            raise PackagingIOError(
                f"cannot publish output {target.name!r}: "
                f"{exc.strerror or exc.__class__.__name__}"
            ) from exc
        try:
            directory_descriptor = os.open(resolved_parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except (InputValidationError, PackagingIOError):
        raise
    except OSError as exc:
        raise PackagingIOError(
            f"cannot write output {path.name!r}: {exc.strerror or exc.__class__.__name__}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _archive_result(
    *,
    archive_kind: str,
    output: Path,
    archive_bytes: bytes,
    manifest_sha256: str,
    created_at_utc: str,
    written: bool,
) -> dict[str, Any]:
    return {
        "archive_filename": output.name,
        "archive_kind": archive_kind,
        "created_at_utc": created_at_utc,
        "manifest_sha256": manifest_sha256,
        "sha256": sha256_bytes(archive_bytes),
        "size_bytes": len(archive_bytes),
        "written": written,
    }


def package_evidence(
    package_input: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    mode: str,
    exact_output: Path | None,
    sanitized_output: Path | None,
    repository_root: Path,
    sensitive_values: Sequence[str] = (),
    dry_run: bool = False,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate, package, and optionally write the requested evidence archives."""

    if mode not in {"both", "exact-only", "sanitized-only"}:
        raise InputValidationError(f"unsupported package mode: {mode!r}")
    wants_exact = mode in {"both", "exact-only"}
    wants_sanitized = mode in {"both", "sanitized-only"}
    if wants_exact and exact_output is None:
        raise InputValidationError(f"--exact-output is required for --mode {mode}")
    if not wants_exact and exact_output is not None:
        raise InputValidationError("--exact-output is not valid with --mode sanitized-only")
    if wants_sanitized and sanitized_output is None:
        raise InputValidationError(f"--sanitized-output is required for --mode {mode}")
    if not wants_sanitized and sanitized_output is not None:
        raise InputValidationError("--sanitized-output is not valid with --mode exact-only")

    try:
        repo = repository_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise InputValidationError("--repository-root does not exist") from exc
    if not repo.is_dir():
        raise InputValidationError("--repository-root must be a directory")

    resolved_outputs: dict[str, Path] = {}
    if exact_output is not None:
        resolved_exact = _resolved_output_path(exact_output)
        if _is_within(resolved_exact, repo):
            raise InputValidationError("exact/private archive output must be outside the repository")
        if _enclosing_git_worktree(resolved_exact) is not None:
            raise InputValidationError(
                "exact/private archive output must be outside every Git worktree"
            )
        resolved_outputs["exact_private"] = resolved_exact
    if sanitized_output is not None:
        resolved_outputs["sanitized_shareable"] = _resolved_output_path(sanitized_output)
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise InputValidationError("exact and sanitized outputs must be different files")

    shared_automatic_sensitive_values = (
        socket.gethostname(),
        os.fspath(Path.home().expanduser().resolve(strict=False)),
        os.fspath(repo),
        *(os.fspath(root) for root in roots.values()),
        *(os.fspath(output.parent) for output in resolved_outputs.values()),
    )
    # An ambient account name may also be a canonical evidence namespace such
    # as ``runner.*``.  Keep it private in source payloads, but only treat it as
    # sensitive control metadata when the caller supplies it explicitly.
    effective_sensitive_values = tuple(
        sorted(
            {
                value
                for value in (
                    *sensitive_values,
                    getpass.getuser(),
                    *shared_automatic_sensitive_values,
                )
                if isinstance(value, str) and len(value) >= 3
            },
            key=lambda value: (-len(value), value),
        )
    )
    manifest_sensitive_values = tuple(
        sorted(
            {
                value
                for value in (
                    *sensitive_values,
                    *shared_automatic_sensitive_values,
                )
                if isinstance(value, str) and len(value) >= 3
            },
            key=lambda value: (-len(value), value),
        )
    )

    gate, commit, artifacts = _validate_package_input(package_input, roots)
    created = created_at_utc or utc_now()
    archives_to_write: list[tuple[str, Path, bytes, str]] = []
    if wants_exact:
        exact_bytes, exact_manifest = _build_archive(
            archive_kind="exact_private",
            gate=gate,
            tested_source_commit=commit,
            artifacts=artifacts,
            sensitive_values=effective_sensitive_values,
            manifest_sensitive_values=manifest_sensitive_values,
        )
        archives_to_write.append(
            (
                "exact_private",
                resolved_outputs["exact_private"],
                exact_bytes,
                exact_manifest,
            )
        )
    if wants_sanitized:
        sanitized_bytes, sanitized_manifest = _build_archive(
            archive_kind="sanitized_shareable",
            gate=gate,
            tested_source_commit=commit,
            artifacts=artifacts,
            sensitive_values=effective_sensitive_values,
            manifest_sensitive_values=manifest_sensitive_values,
        )
        archives_to_write.append(
            (
                "sanitized_shareable",
                resolved_outputs["sanitized_shareable"],
                sanitized_bytes,
                sanitized_manifest,
            )
        )

    if not dry_run:
        for archive_kind, path, archive_bytes, _manifest_sha256 in archives_to_write:
            _atomic_create(
                path,
                archive_bytes,
                mode=0o600 if archive_kind == "exact_private" else 0o644,
            )
    return {
        "archives": [
            _archive_result(
                archive_kind=archive_kind,
                output=path,
                archive_bytes=archive_bytes,
                manifest_sha256=manifest_sha256,
                created_at_utc=created,
                written=not dry_run,
            )
            for archive_kind, path, archive_bytes, manifest_sha256 in archives_to_write
        ],
        "dry_run": dry_run,
        "format_version": PACKAGE_REPORT_VERSION,
        "gate": gate,
        "status": "dry_run" if dry_run else "created",
        "tested_source_commit": commit,
    }


def _resolve_input_argument(positional: Path | None, optional: Path | None) -> Path:
    if positional is not None and optional is not None:
        raise InputValidationError("provide the package input either positionally or with --input, not both")
    selected = optional or positional
    if selected is None:
        raise InputValidationError("a package input JSON file is required")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_input", nargs="?", type=Path, help="v1 package input JSON")
    parser.add_argument("--input", "--manifest", dest="input_option", type=Path)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Map a manifest source_root name to an explicit filesystem root (repeatable)",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "exact-only", "sanitized-only"],
        default="both",
    )
    parser.add_argument("--exact-output", type=Path)
    parser.add_argument("--sanitized-output", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--sensitive-value",
        action="append",
        default=[],
        help="Explicit sensitive literal to remove from sanitized payloads (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and hash without writing archives")
    parser.add_argument("--output", type=Path, help="Write the package result as canonical JSON")
    parser.add_argument("--json", action="store_true", help="Emit the package result as JSON")
    return parser


def _emit_report(report: Mapping[str, Any], *, json_mode: bool, output: Path | None) -> None:
    encoded = canonical_json_bytes(report)
    if output is not None:
        resolved = _resolved_output_path(output)
        _atomic_create(resolved, encoded, mode=0o600)
    if json_mode:
        sys.stdout.buffer.write(encoded)
        return
    print(f"status: {report.get('status')}")
    for archive in report.get("archives", []):
        print(f"archive kind: {archive.get('archive_kind')}")
        print(f"archive: {archive.get('archive_filename')}")
        print(f"archive sha256: {archive.get('sha256')}")
        print(f"manifest sha256: {archive.get('manifest_sha256')}")
        print(f"size bytes: {archive.get('size_bytes')}")
        print(f"created at UTC: {archive.get('created_at_utc')}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        package_input_path = _resolve_input_argument(args.package_input, args.input_option)
        roots = _parse_root_mapping(args.root)
        if not roots:
            raise InputValidationError("at least one --root NAME=PATH mapping is required")
        if args.output is not None:
            candidate = _resolved_output_path(args.output)
            for archive_output in (args.exact_output, args.sanitized_output):
                if archive_output is not None and candidate == _resolved_output_path(archive_output):
                    raise InputValidationError("--output must differ from archive outputs")
        document = _load_package_input(package_input_path)
        report = package_evidence(
            document,
            roots=roots,
            mode=args.mode,
            exact_output=args.exact_output,
            sanitized_output=args.sanitized_output,
            repository_root=args.repository_root,
            sensitive_values=args.sensitive_value,
            dry_run=args.dry_run,
        )
        _emit_report(report, json_mode=args.json, output=args.output)
        return 0
    except InputValidationError as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except IntegrityError as exc:
        print(f"integrity or sanitization failure: {exc}", file=sys.stderr)
        return 3
    except PackagingIOError as exc:
        print(f"I/O or runtime failure: {exc}", file=sys.stderr)
        return 4
    except OSError as exc:
        print(
            f"I/O or runtime failure: {exc.strerror or exc.__class__.__name__}",
            file=sys.stderr,
        )
        return 4
    except Exception as exc:  # Stable, traceback-free CLI by default.
        print(f"I/O or runtime failure: unexpected {exc.__class__.__name__}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
