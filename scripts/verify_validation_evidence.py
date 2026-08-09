#!/usr/bin/env python3
"""Verify a Multiscreen validation-evidence archive fully offline.

The verifier never extracts archive members and never resolves network
resources.  It validates the archive member set, member types and paths,
manifest, SHA256SUMS, optional evidence descriptor, and the independent
sanitization contract for shareable archives.
"""

from __future__ import annotations

import argparse
import gzip
import re
import struct
import sys
import tarfile
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from validation_evidence_common import (
        ARCHIVE_FORMAT_VERSION,
        MAX_ARCHIVE_MEMBER_COUNT,
        MAX_ARCHIVE_MEMBER_SIZE_BYTES,
        MAX_TOTAL_ARCHIVE_MEMBER_BYTES,
        SANITIZATION_REPORT,
        SANITIZATION_FORMAT_VERSION,
        SANITIZATION_RULES,
        SANITIZATION_RULESET_VERSION,
        SOURCE_ARTIFACT_CLASSIFICATIONS,
        TOOL_VERSION,
        InputValidationError,
        IntegrityError,
        build_sha256sums,
        canonical_json_bytes,
        ensure_payload_archive_path,
        parse_json_bytes,
        parse_sha256sums,
        require_sha256,
        safe_archive_path,
        safe_write_bytes,
        sanitize_text,
        sha256_bytes,
        sha256_file,
        utc_now,
        validate_evidence_document,
        validate_utc,
    )
except ModuleNotFoundError:  # Support importing this file as scripts.* in tests.
    from scripts.validation_evidence_common import (  # type: ignore[no-redef]
        ARCHIVE_FORMAT_VERSION,
        MAX_ARCHIVE_MEMBER_COUNT,
        MAX_ARCHIVE_MEMBER_SIZE_BYTES,
        MAX_TOTAL_ARCHIVE_MEMBER_BYTES,
        SANITIZATION_REPORT,
        SANITIZATION_FORMAT_VERSION,
        SANITIZATION_RULES,
        SANITIZATION_RULESET_VERSION,
        SOURCE_ARTIFACT_CLASSIFICATIONS,
        TOOL_VERSION,
        InputValidationError,
        IntegrityError,
        build_sha256sums,
        canonical_json_bytes,
        ensure_payload_archive_path,
        parse_json_bytes,
        parse_sha256sums,
        require_sha256,
        safe_archive_path,
        safe_write_bytes,
        sanitize_text,
        sha256_bytes,
        sha256_file,
        utc_now,
        validate_evidence_document,
        validate_utc,
    )


REPORT_FORMAT_VERSION = "validation-evidence-verification-v1"
ARCHIVE_KINDS = frozenset({"exact_private", "sanitized_shareable"})
_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
_STREAM_CHUNK_SIZE = 1024 * 1024
_MAX_DECOMPRESSED_TAR_BYTES = (
    MAX_TOTAL_ARCHIVE_MEMBER_BYTES
    + ((MAX_ARCHIVE_MEMBER_COUNT + 4) * (tarfile.BLOCKSIZE * 2))
    + tarfile.RECORDSIZE
)
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")
_TESTED_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MANIFEST_KEYS = frozenset(
    {"format_version", "archive_kind", "gate", "tested_source_commit", "members"}
)
_MEMBER_BASE_KEYS = frozenset(
    {
        "path",
        "kind",
        "logical_name",
        "classification",
        "media_type",
        "size_bytes",
        "sha256",
        "transformation",
    }
)
_SOURCE_MEMBER_KEYS = _MEMBER_BASE_KEYS | frozenset(
    {"source_size_bytes", "source_sha256"}
)
_SANITIZATION_REPORT_MEMBER_KEYS = _MEMBER_BASE_KEYS
_SANITIZATION_TOP_KEYS = frozenset(
    {
        "files_scanned",
        "final_scan",
        "format_version",
        "replacement_totals",
        "rules_applied",
        "ruleset_version",
        "status",
        "unresolved_findings",
    }
)
_SANITIZATION_FILE_KEYS = frozenset(
    {
        "path",
        "replacements",
        "sanitized_sha256",
        "sanitized_size_bytes",
        "source_sha256",
        "source_size_bytes",
        "unresolved_findings",
    }
)
_SANITIZATION_FINAL_SCAN_KEYS = frozenset(
    {"high_confidence_findings", "status"}
)
_SANITIZATION_REPLACEMENT_KEYS = frozenset({"count", "rule_id"})


class VerificationIOError(Exception):
    """An archive or report could not be read or written."""


def _as_integrity_error(exc: Exception) -> IntegrityError:
    return IntegrityError(str(exc))


def _safe_tar_member_name(value: Any) -> str:
    try:
        path = safe_archive_path(value, field="tar member path")
    except InputValidationError as exc:
        raise _as_integrity_error(exc) from exc
    if _WINDOWS_DRIVE_RE.match(path):
        raise IntegrityError(f"Windows drive-qualified tar member path is forbidden: {path!r}")
    return path


def _parse_json_bytes(data: bytes, *, member_name: str) -> Any:
    try:
        return parse_json_bytes(data, field=member_name)
    except InputValidationError as exc:
        raise IntegrityError(str(exc)) from exc


def _read_json_input(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VerificationIOError(
            f"cannot read {label} {path.name!r}: {exc.strerror or exc.__class__.__name__}"
        ) from exc
    value = parse_json_bytes(data, field=f"{label} {path.name!r}")
    if not isinstance(value, dict):
        raise InputValidationError(f"{label} must contain a JSON object")
    return value


def _prevalidate_canonical_gzip(archive_path: Path) -> int:
    """Validate one normalized gzip member and return its output byte count."""

    try:
        stream = archive_path.open("rb")
    except OSError as exc:
        raise VerificationIOError(
            f"cannot read archive {archive_path.name!r}: "
            f"{exc.strerror or exc.__class__.__name__}"
        ) from exc

    with stream:
        try:
            header = stream.read(len(_GZIP_HEADER))
            if header != _GZIP_HEADER:
                raise IntegrityError(
                    "archive does not have the canonical single-member gzip header"
                )

            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            checksum = 0
            decompressed_size = 0
            after_deflate = b""
            while not decompressor.eof:
                compressed = stream.read(_STREAM_CHUNK_SIZE)
                if not compressed:
                    raise IntegrityError("gzip deflate stream is truncated")
                while compressed and not decompressor.eof:
                    output_limit = min(
                        _STREAM_CHUNK_SIZE,
                        _MAX_DECOMPRESSED_TAR_BYTES - decompressed_size + 1,
                    )
                    try:
                        payload = decompressor.decompress(compressed, output_limit)
                    except zlib.error as exc:
                        raise IntegrityError("gzip deflate stream is malformed") from exc
                    decompressed_size += len(payload)
                    if decompressed_size > _MAX_DECOMPRESSED_TAR_BYTES:
                        raise IntegrityError(
                            "gzip output exceeds the bounded tar-stream verification limit"
                        )
                    checksum = zlib.crc32(payload, checksum)
                    compressed = decompressor.unconsumed_tail
                    if decompressor.eof:
                        after_deflate = decompressor.unused_data

            while len(after_deflate) < 8:
                chunk = stream.read(8 - len(after_deflate))
                if not chunk:
                    raise IntegrityError("gzip trailer is truncated")
                after_deflate += chunk
            expected_crc32, expected_size = struct.unpack("<II", after_deflate[:8])
            if expected_crc32 != checksum & 0xFFFFFFFF:
                raise IntegrityError("gzip trailer CRC32 does not match decompressed bytes")
            if expected_size != decompressed_size & 0xFFFFFFFF:
                raise IntegrityError("gzip trailer size does not match decompressed bytes")
            if after_deflate[8:] or stream.read(1):
                raise IntegrityError(
                    "archive contains trailing bytes or concatenated gzip members"
                )
        except OSError as exc:
            raise VerificationIOError(
                f"cannot read archive {archive_path.name!r}: "
                f"{exc.strerror or exc.__class__.__name__}"
            ) from exc
    return decompressed_size


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_SIZE_BYTES:
        raise IntegrityError(
            f"archive member {member.name!r} has forbidden size {member.size}"
        )
    try:
        stream = archive.extractfile(member)
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise IntegrityError(f"cannot read archive member {member.name!r}") from exc
    if stream is None:
        raise IntegrityError(f"archive member {member.name!r} has no regular-file payload")
    chunks: list[bytes] = []
    remaining = member.size
    try:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise IntegrityError(f"archive member {member.name!r} is truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        extra = stream.read(1)
    except (gzip.BadGzipFile, tarfile.TarError, OSError, EOFError) as exc:
        raise IntegrityError(f"archive member {member.name!r} is unreadable") from exc
    finally:
        stream.close()
    if extra:
        raise IntegrityError(f"archive member {member.name!r} exceeds its declared size")
    data = b"".join(chunks)
    if len(data) != member.size:
        raise IntegrityError(f"archive member {member.name!r} size changed while reading")
    return data


def _inspect_tar_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    raw_members: list[tarfile.TarInfo] = []
    try:
        while True:
            member = archive.next()
            if member is None:
                break
            if len(raw_members) >= MAX_ARCHIVE_MEMBER_COUNT:
                raise IntegrityError(
                    f"archive contains more than {MAX_ARCHIVE_MEMBER_COUNT} members"
                )
            raw_members.append(member)
    except (gzip.BadGzipFile, tarfile.TarError, OSError, EOFError) as exc:
        raise IntegrityError("archive member table is malformed or truncated") from exc

    members: dict[str, tarfile.TarInfo] = {}
    ordered_paths: list[str] = []
    total_size = 0
    for member in raw_members:
        path = _safe_tar_member_name(member.name)
        if path in members:
            raise IntegrityError(f"duplicate archive member path: {path!r}")
        if not member.isreg() or member.type != tarfile.REGTYPE:
            raise IntegrityError(
                f"archive member {path!r} is not a regular file (type={member.type!r})"
            )
        if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_SIZE_BYTES:
            raise IntegrityError(f"archive member {path!r} has forbidden size {member.size}")
        total_size += member.size
        if (
            member.mtime != 0
            or member.uid != 0
            or member.gid != 0
            or member.uname
            or member.gname
            or member.linkname
            or member.mode != 0o644
            or member.pax_headers
            or member.devmajor != 0
            or member.devminor != 0
        ):
            raise IntegrityError(f"archive member {path!r} has non-normalized metadata")
        if total_size > MAX_TOTAL_ARCHIVE_MEMBER_BYTES:
            raise IntegrityError(
                "archive members exceed the "
                f"{MAX_TOTAL_ARCHIVE_MEMBER_BYTES}-byte verification limit"
            )
        members[path] = member
        ordered_paths.append(path)
    if ordered_paths != sorted(ordered_paths):
        raise IntegrityError("tar members are not sorted by canonical path")
    return members


def _read_exact(stream: Any, size: int, *, context: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise IntegrityError(f"canonical tar stream is truncated while reading {context}")
    return data


def _validate_canonical_tar_stream(
    archive_path: Path,
    tar_members: Mapping[str, tarfile.TarInfo],
    member_data: Mapping[str, bytes],
    *,
    decompressed_size: int,
) -> None:
    """Require the complete uncompressed tar stream to match normalized USTAR."""

    expected_offset = 0
    for path, member in tar_members.items():
        if member.offset != expected_offset or member.offset_data != expected_offset + tarfile.BLOCKSIZE:
            raise IntegrityError(f"archive member {path!r} has a non-canonical tar offset")
        padded_size = (
            (member.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
        ) * tarfile.BLOCKSIZE
        expected_offset = member.offset_data + padded_size

    expected_tar_size = (
        (
            expected_offset
            + (tarfile.BLOCKSIZE * 2)
            + tarfile.RECORDSIZE
            - 1
        )
        // tarfile.RECORDSIZE
    ) * tarfile.RECORDSIZE
    if decompressed_size != expected_tar_size:
        raise IntegrityError(
            "gzip payload contains bytes outside the canonical tar member stream"
        )

    try:
        stream = gzip.open(archive_path, mode="rb")
    except OSError as exc:
        raise VerificationIOError(
            f"cannot read archive {archive_path.name!r}: "
            f"{exc.strerror or exc.__class__.__name__}"
        ) from exc

    try:
        with stream:
            for path, member in tar_members.items():
                actual_header = _read_exact(
                    stream, tarfile.BLOCKSIZE, context=f"header for {path!r}"
                )
                try:
                    expected_header = member.tobuf(
                        format=tarfile.USTAR_FORMAT,
                        encoding="utf-8",
                        errors="strict",
                    )
                except (UnicodeError, ValueError) as exc:
                    raise IntegrityError(
                        f"archive member {path!r} cannot be represented as normalized USTAR"
                    ) from exc
                if actual_header != expected_header:
                    raise IntegrityError(
                        f"archive member {path!r} has a non-canonical USTAR header"
                    )

                expected_payload = member_data[path]
                payload_offset = 0
                while payload_offset < len(expected_payload):
                    chunk_size = min(
                        _STREAM_CHUNK_SIZE, len(expected_payload) - payload_offset
                    )
                    actual = _read_exact(
                        stream, chunk_size, context=f"payload for {path!r}"
                    )
                    expected = expected_payload[
                        payload_offset : payload_offset + chunk_size
                    ]
                    if actual != expected:
                        raise IntegrityError(
                            f"archive member {path!r} differs from its canonical payload"
                        )
                    payload_offset += chunk_size

                padding_size = (-member.size) % tarfile.BLOCKSIZE
                if padding_size:
                    padding = _read_exact(
                        stream, padding_size, context=f"padding for {path!r}"
                    )
                    if any(padding):
                        raise IntegrityError(
                            f"archive member {path!r} has nonzero tar padding"
                        )

            terminal_size = expected_tar_size - expected_offset
            terminal = _read_exact(stream, terminal_size, context="terminal tar padding")
            if any(terminal):
                raise IntegrityError("archive has nonzero terminal tar padding")
            if stream.read(1):
                raise IntegrityError("archive has data after canonical terminal tar padding")
    except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
        raise IntegrityError("archive gzip stream changed or became unreadable") from exc
    except OSError as exc:
        raise VerificationIOError(
            f"cannot read archive {archive_path.name!r}: "
            f"{exc.strerror or exc.__class__.__name__}"
        ) from exc


def _required_text(entry: Mapping[str, Any], field: str, *, context: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{context}.{field} must be a non-empty string")
    return value


def _required_size(entry: Mapping[str, Any], field: str, *, context: str) -> int:
    value = entry.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IntegrityError(f"{context}.{field} must be a non-negative integer")
    if value > MAX_ARCHIVE_MEMBER_SIZE_BYTES:
        raise IntegrityError(f"{context}.{field} exceeds the verification size limit")
    return value


def _required_digest(entry: Mapping[str, Any], field: str, *, context: str) -> str:
    try:
        return require_sha256(entry.get(field), field=f"{context}.{field}")
    except InputValidationError as exc:
        raise _as_integrity_error(exc) from exc


def _validate_manifest(
    manifest: Any,
) -> tuple[str, str, str, dict[str, dict[str, Any]]]:
    if not isinstance(manifest, dict):
        raise IntegrityError("MANIFEST.json must contain a JSON object")
    if manifest.get("format_version") != ARCHIVE_FORMAT_VERSION:
        raise IntegrityError(
            f"MANIFEST.json format_version must be {ARCHIVE_FORMAT_VERSION!r}"
        )
    archive_kind = manifest.get("archive_kind")
    manifest_keys = set(manifest)
    if manifest_keys != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS - manifest_keys)
        unexpected = sorted(manifest_keys - _MANIFEST_KEYS)
        raise IntegrityError(
            f"MANIFEST.json keys mismatch (missing={missing!r}, unexpected={unexpected!r})"
        )
    if archive_kind not in ARCHIVE_KINDS:
        raise IntegrityError(
            "MANIFEST.json archive_kind must be exact_private or sanitized_shareable"
        )
    gate = manifest.get("gate")
    if not isinstance(gate, str) or not gate.strip():
        raise IntegrityError("MANIFEST.json gate must be a non-empty string")
    tested_commit = manifest.get("tested_source_commit")
    if not isinstance(tested_commit, str) or _TESTED_COMMIT_RE.fullmatch(tested_commit) is None:
        raise IntegrityError("MANIFEST.json tested_source_commit must be 40 or 64 lowercase hex characters")
    raw_entries = manifest.get("members")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise IntegrityError("MANIFEST.json members must be a non-empty array")

    entries: dict[str, dict[str, Any]] = {}
    logical_names: set[str] = set()
    ordered_paths: list[str] = []
    source_artifact_count = 0
    for index, raw_entry in enumerate(raw_entries):
        context = f"MANIFEST.json.members[{index}]"
        if not isinstance(raw_entry, dict):
            raise IntegrityError(f"{context} must be an object")
        try:
            path = safe_archive_path(raw_entry.get("path"), field=f"{context}.path")
        except InputValidationError as exc:
            raise _as_integrity_error(exc) from exc
        if _WINDOWS_DRIVE_RE.match(path):
            raise IntegrityError(f"{context}.path is drive-qualified")
        if path in {"MANIFEST.json", "SHA256SUMS"}:
            raise IntegrityError(f"{context}.path collides with an archive control member")
        if path in entries:
            raise IntegrityError(f"duplicate MANIFEST.json member path: {path!r}")

        member_kind = _required_text(raw_entry, "kind", context=context)
        logical_name = _required_text(raw_entry, "logical_name", context=context)
        if logical_name in logical_names:
            raise IntegrityError(f"duplicate MANIFEST.json logical_name: {logical_name!r}")
        logical_names.add(logical_name)
        classification = _required_text(raw_entry, "classification", context=context)
        _required_text(raw_entry, "media_type", context=context)
        size = _required_size(raw_entry, "size_bytes", context=context)
        digest = _required_digest(raw_entry, "sha256", context=context)
        transformation = _required_text(raw_entry, "transformation", context=context)

        if path == SANITIZATION_REPORT:
            if archive_kind != "sanitized_shareable":
                raise IntegrityError("exact_private archives must not contain a sanitization report")
            if member_kind != "sanitization_report" or transformation != "generated":
                raise IntegrityError(
                    "SANITIZATION_REPORT.json must be a generated sanitization_report member"
                )
            expected_entry_keys = _SANITIZATION_REPORT_MEMBER_KEYS
        else:
            try:
                ensure_payload_archive_path(path)
            except InputValidationError as exc:
                raise _as_integrity_error(exc) from exc
            if member_kind != "source_artifact":
                raise IntegrityError(f"payload member {path!r} must have kind=source_artifact")
            if classification not in SOURCE_ARTIFACT_CLASSIFICATIONS:
                raise IntegrityError(
                    f"payload member {path!r} has unsupported source-artifact classification"
                )
            source_artifact_count += 1
            source_size = _required_size(raw_entry, "source_size_bytes", context=context)
            source_digest = _required_digest(raw_entry, "source_sha256", context=context)
            expected_entry_keys = _SOURCE_MEMBER_KEYS
            if archive_kind == "exact_private":
                if transformation != "none":
                    raise IntegrityError(
                        f"exact member {path!r} must have transformation=none"
                    )
                if source_size != size or source_digest != digest:
                    raise IntegrityError(
                        f"exact member {path!r} does not preserve its source size/hash"
                    )
            elif transformation != "sanitized-v1":
                raise IntegrityError(
                    f"sanitized member {path!r} must have transformation=sanitized-v1"
                )

        entry_keys = set(raw_entry)
        if entry_keys != expected_entry_keys:
            missing = sorted(expected_entry_keys - entry_keys)
            unexpected = sorted(entry_keys - expected_entry_keys)
            raise IntegrityError(
                f"{context} keys mismatch (missing={missing!r}, unexpected={unexpected!r})"
            )

        normalized = dict(raw_entry)
        normalized["path"] = path
        normalized["size_bytes"] = size
        normalized["sha256"] = digest
        entries[path] = normalized
        ordered_paths.append(path)

    if source_artifact_count == 0:
        raise IntegrityError("MANIFEST.json must list at least one source artifact")
    if ordered_paths != sorted(ordered_paths):
        raise IntegrityError("MANIFEST.json members are not sorted by path")
    if archive_kind == "sanitized_shareable" and SANITIZATION_REPORT not in entries:
        raise IntegrityError("sanitized archive is missing SANITIZATION_REPORT.json")
    return archive_kind, gate, tested_commit, entries


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise IntegrityError(
            f"{context} keys mismatch (missing={missing!r}, unexpected={unexpected!r})"
        )


def _require_count(value: Any, *, context: str, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise IntegrityError(f"{context} must be a {qualifier} integer")
    return value


def _validate_final_scan(value: Any) -> None:
    if not isinstance(value, dict):
        raise IntegrityError("sanitization final_scan must be an object")
    _require_exact_keys(
        value, _SANITIZATION_FINAL_SCAN_KEYS, context="sanitization final_scan"
    )
    if value.get("status") != "passed":
        raise IntegrityError("sanitization final_scan status is not passed")
    finding_count = _require_count(
        value.get("high_confidence_findings"),
        context="sanitization final_scan.high_confidence_findings",
    )
    if finding_count != 0:
        raise IntegrityError("sanitization final_scan reports high-confidence findings")


def _verify_sanitized_archive(
    member_data: Mapping[str, bytes],
    manifest_entries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_data = member_data.get(SANITIZATION_REPORT)
    if report_data is None:
        raise IntegrityError("sanitized archive has no sanitization report payload")
    report = _parse_json_bytes(report_data, member_name=SANITIZATION_REPORT)
    if not isinstance(report, dict):
        raise IntegrityError("SANITIZATION_REPORT.json must contain a JSON object")
    if canonical_json_bytes(report) != report_data:
        raise IntegrityError("SANITIZATION_REPORT.json is not canonical JSON")
    _require_exact_keys(
        report, _SANITIZATION_TOP_KEYS, context="SANITIZATION_REPORT.json"
    )
    if report.get("format_version") != SANITIZATION_FORMAT_VERSION:
        raise IntegrityError(
            f"SANITIZATION_REPORT.json format_version must be {SANITIZATION_FORMAT_VERSION!r}"
        )
    if report.get("ruleset_version") != SANITIZATION_RULESET_VERSION:
        raise IntegrityError(
            "SANITIZATION_REPORT.json ruleset_version does not match the verifier"
        )
    rules_applied = report.get("rules_applied")
    if not isinstance(rules_applied, list):
        raise IntegrityError("sanitization report rules_applied must be an array")
    if not all(isinstance(rule_id, str) for rule_id in rules_applied):
        raise IntegrityError("sanitization report rules_applied entries must be strings")
    if rules_applied != sorted(set(rules_applied)):
        raise IntegrityError("sanitization report rules_applied must be sorted and unique")
    if rules_applied != list(SANITIZATION_RULES):
        raise IntegrityError("sanitization report rules_applied does not match the verifier")
    if report.get("status") != "passed":
        raise IntegrityError("sanitization report status is not passed")
    unresolved = report.get("unresolved_findings")
    if not isinstance(unresolved, list) or unresolved:
        raise IntegrityError("sanitization report contains unresolved findings")
    _validate_final_scan(report.get("final_scan"))

    source_paths = {
        path
        for path, entry in manifest_entries.items()
        if entry.get("kind") == "source_artifact"
    }
    scanned = report.get("files_scanned")
    if not isinstance(scanned, list):
        raise IntegrityError("sanitization report files_scanned must be an array")
    scanned_paths: set[str] = set()
    ordered_scanned_paths: list[str] = []
    aggregated_replacements: dict[str, int] = {}
    for index, item in enumerate(scanned):
        context = f"SANITIZATION_REPORT.json.files_scanned[{index}]"
        if not isinstance(item, dict):
            raise IntegrityError(f"{context} must be an object")
        _require_exact_keys(item, _SANITIZATION_FILE_KEYS, context=context)
        try:
            path = ensure_payload_archive_path(item.get("path"))
        except InputValidationError as exc:
            raise _as_integrity_error(exc) from exc
        if path in scanned_paths:
            raise IntegrityError(f"duplicate sanitization files_scanned path: {path!r}")
        scanned_paths.add(path)
        ordered_scanned_paths.append(path)
        item_unresolved = item.get("unresolved_findings")
        if not isinstance(item_unresolved, list) or item_unresolved:
            raise IntegrityError(f"{context} contains unresolved findings")
        entry = manifest_entries.get(path)
        if entry is None or entry.get("kind") != "source_artifact":
            raise IntegrityError(f"{context} does not identify a manifest source artifact")

        source_digest = _required_digest(item, "source_sha256", context=context)
        sanitized_digest = _required_digest(item, "sanitized_sha256", context=context)
        source_size = _required_size(item, "source_size_bytes", context=context)
        sanitized_size = _required_size(
            item, "sanitized_size_bytes", context=context
        )
        actual_payload = member_data[path]
        if source_digest != entry.get("source_sha256"):
            raise IntegrityError(f"{context}.source_sha256 does not match MANIFEST.json")
        if source_size != entry.get("source_size_bytes"):
            raise IntegrityError(f"{context}.source_size_bytes does not match MANIFEST.json")
        if sanitized_digest != entry.get("sha256"):
            raise IntegrityError(f"{context}.sanitized_sha256 does not match MANIFEST.json")
        if sanitized_size != entry.get("size_bytes"):
            raise IntegrityError(
                f"{context}.sanitized_size_bytes does not match MANIFEST.json"
            )
        if sanitized_digest != sha256_bytes(actual_payload):
            raise IntegrityError(f"{context}.sanitized_sha256 does not match payload")
        if sanitized_size != len(actual_payload):
            raise IntegrityError(f"{context}.sanitized_size_bytes does not match payload")

        replacements = item.get("replacements")
        if not isinstance(replacements, dict):
            raise IntegrityError(f"{context}.replacements must be an object")
        replacement_rules = list(replacements)
        if replacement_rules != sorted(replacement_rules):
            raise IntegrityError(f"{context}.replacements rules must be sorted")
        for rule_id, raw_count in replacements.items():
            if rule_id not in SANITIZATION_RULES:
                raise IntegrityError(
                    f"{context}.replacements contains unknown rule {rule_id!r}"
                )
            count = _require_count(
                raw_count,
                context=f"{context}.replacements[{rule_id!r}]",
                positive=True,
            )
            aggregated_replacements[rule_id] = (
                aggregated_replacements.get(rule_id, 0) + count
            )
    if scanned_paths != source_paths:
        raise IntegrityError("sanitization files_scanned set does not match source artifacts")
    if ordered_scanned_paths != sorted(ordered_scanned_paths):
        raise IntegrityError("sanitization files_scanned paths must be sorted")

    raw_totals = report.get("replacement_totals")
    if not isinstance(raw_totals, list):
        raise IntegrityError("sanitization report replacement_totals must be an array")
    normalized_totals: list[dict[str, Any]] = []
    seen_total_rules: set[str] = set()
    for index, item in enumerate(raw_totals):
        context = f"SANITIZATION_REPORT.json.replacement_totals[{index}]"
        if not isinstance(item, dict):
            raise IntegrityError(f"{context} must be an object")
        _require_exact_keys(item, _SANITIZATION_REPLACEMENT_KEYS, context=context)
        rule_id = item.get("rule_id")
        if not isinstance(rule_id, str) or rule_id not in SANITIZATION_RULES:
            raise IntegrityError(f"{context}.rule_id is not a known sanitization rule")
        if rule_id in seen_total_rules:
            raise IntegrityError(f"duplicate replacement_totals rule: {rule_id!r}")
        seen_total_rules.add(rule_id)
        count = _require_count(item.get("count"), context=f"{context}.count", positive=True)
        normalized_totals.append({"count": count, "rule_id": rule_id})
    if [item["rule_id"] for item in normalized_totals] != sorted(seen_total_rules):
        raise IntegrityError("sanitization replacement_totals rules must be sorted")
    expected_totals = [
        {"count": count, "rule_id": rule_id}
        for rule_id, count in sorted(aggregated_replacements.items())
    ]
    if normalized_totals != expected_totals:
        raise IntegrityError(
            "sanitization replacement_totals do not equal per-file replacement counts"
        )

    independently_scanned = 0
    for path, raw in sorted(member_data.items()):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IntegrityError(f"sanitized member {path!r} is not UTF-8") from exc
        rescanned, replacements, findings = sanitize_text(text)
        if rescanned != text or replacements or findings:
            rule_names = sorted(
                set(replacements)
                | {item.get("rule", "unknown") for item in findings}
            )
            raise IntegrityError(
                f"sanitized member {path!r} has residual sensitive content "
                f"detected by rules: {', '.join(rule_names)}"
            )
        independently_scanned += 1
    return (
        {
            "files_scanned": len(source_paths),
            "independent_members_scanned": independently_scanned,
            "status": "verified",
        },
        {
            "files_scanned": len(source_paths),
            "replacement_count": sum(aggregated_replacements.values()),
            "report_sha256": sha256_bytes(report_data),
            "rules_applied": list(rules_applied),
        },
    )


def _validate_evidence_input(document_path: Path, schema_path: Path) -> Mapping[str, Any]:
    document = _read_json_input(document_path, label="evidence document")
    schema = _read_json_input(schema_path, label="JSON schema")
    try:
        errors = validate_evidence_document(document, schema)
    except InputValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"cannot validate evidence document: {exc}") from exc
    if errors:
        detail = "; ".join(errors[:20])
        if len(errors) > 20:
            detail += f"; and {len(errors) - 20} more errors"
        raise InputValidationError(f"evidence document does not conform to schema: {detail}")
    return document


def _cross_check_descriptor(
    document: Mapping[str, Any],
    *,
    archive_kind: str,
    archive_filename: str,
    archive_sha256: str,
    archive_size: int,
    manifest_gate: str,
    manifest_sha256: str,
    manifest_source_entries: Mapping[str, Mapping[str, Any]],
    manifest_tested_commit: str,
    sanitization_descriptor_values: Mapping[str, Any] | None,
) -> None:
    archives = document.get("archives")
    if not isinstance(archives, dict):
        raise InputValidationError("evidence document archives must be an object")
    descriptor = archives.get(archive_kind)
    if not isinstance(descriptor, dict):
        raise InputValidationError(
            f"evidence document has no {archive_kind!r} archive descriptor"
        )
    if descriptor.get("status") != "verified":
        raise InputValidationError(
            f"evidence document {archive_kind} archive descriptor is not verified"
        )
    described_kind = descriptor.get("archive_kind")
    if described_kind is not None and described_kind != archive_kind:
        raise IntegrityError("evidence document archive_kind does not match MANIFEST.json")
    described_filename = descriptor.get("archive_filename", descriptor.get("filename"))
    required_values = {
        "archive filename": (described_filename, archive_filename),
        "archive SHA-256": (descriptor.get("sha256"), archive_sha256),
        "archive size": (descriptor.get("size_bytes"), archive_size),
        "manifest SHA-256": (descriptor.get("manifest_sha256"), manifest_sha256),
    }
    for label, (described, actual) in required_values.items():
        if described is None:
            raise InputValidationError(f"evidence document has no verified {label}")
        if described != actual:
            raise IntegrityError(f"evidence document {label} does not match the archive")

    if document.get("validation_gate") != manifest_gate:
        raise IntegrityError(
            "evidence document validation gate does not match MANIFEST.json"
        )
    tested_source = document.get("tested_source")
    if (
        not isinstance(tested_source, dict)
        or tested_source.get("commit") != manifest_tested_commit
    ):
        raise IntegrityError(
            "evidence document tested-source commit does not match MANIFEST.json"
        )

    raw_artifacts = document.get("source_artifacts")
    if not isinstance(raw_artifacts, list):
        raise InputValidationError("evidence document source_artifacts must be an array")
    described_artifacts: dict[str, Mapping[str, Any]] = {}
    for artifact in raw_artifacts:
        if not isinstance(artifact, dict):
            raise InputValidationError(
                "evidence document source_artifacts entries must be objects"
            )
        path = artifact.get("archive_path")
        if not isinstance(path, str):
            raise InputValidationError(
                "evidence document source artifact has no archive_path"
            )
        if path in described_artifacts:
            raise InputValidationError(
                f"evidence document has duplicate source artifact path {path!r}"
            )
        described_artifacts[path] = artifact

    manifest_artifacts = {
        path: entry
        for path, entry in manifest_source_entries.items()
        if entry.get("kind") == "source_artifact"
    }
    if set(described_artifacts) != set(manifest_artifacts):
        missing = sorted(set(manifest_artifacts) - set(described_artifacts))
        extra = sorted(set(described_artifacts) - set(manifest_artifacts))
        raise IntegrityError(
            "evidence document source-artifact set does not match MANIFEST.json "
            f"(missing={missing!r}, extra={extra!r})"
        )

    field_pairs = (
        ("logical_name", "logical_name"),
        ("classification", "classification"),
        ("size_bytes", "source_size_bytes"),
        ("sha256", "source_sha256"),
    )
    for path in sorted(manifest_artifacts):
        described = described_artifacts[path]
        manifested = manifest_artifacts[path]
        for descriptor_field, manifest_field in field_pairs:
            if described.get(descriptor_field) != manifested.get(manifest_field):
                raise IntegrityError(
                    f"evidence document source artifact {path!r} field "
                    f"{descriptor_field!r} does not match MANIFEST.json"
                )

    if archive_kind == "sanitized_shareable":
        if sanitization_descriptor_values is None:
            raise IntegrityError("verified sanitized archive has no sanitization metadata")
        described_sanitization = document.get("sanitization")
        if not isinstance(described_sanitization, dict):
            raise InputValidationError(
                "evidence document sanitization must be an object for sanitized archives"
            )
        for field in (
            "report_sha256",
            "rules_applied",
            "files_scanned",
            "replacement_count",
        ):
            if field not in described_sanitization:
                raise InputValidationError(
                    f"evidence document sanitization has no verified {field}"
                )
            if described_sanitization[field] != sanitization_descriptor_values[field]:
                raise IntegrityError(
                    f"evidence document sanitization {field} does not match "
                    "SANITIZATION_REPORT.json"
                )


def verify_archive(
    archive_path: str | Path,
    *,
    expected_sha256: str | None = None,
    evidence_document: Mapping[str, Any] | None = None,
    verification_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Verify one archive and return a deterministic structured report."""

    archive_file = Path(archive_path)
    timestamp = validate_utc(
        verification_timestamp_utc or utc_now(), field="verification timestamp"
    )
    if expected_sha256 is not None:
        require_sha256(expected_sha256, field="expected archive SHA-256")
    try:
        initial_stat = archive_file.stat()
    except OSError as exc:
        raise VerificationIOError(
            f"cannot stat archive {archive_file.name!r}: {exc.strerror or exc.__class__.__name__}"
        ) from exc
    if not archive_file.is_file():
        raise VerificationIOError(f"archive {archive_file.name!r} is not a regular file")
    archive_size = initial_stat.st_size
    try:
        archive_sha256 = sha256_file(archive_file)
    except OSError as exc:
        raise VerificationIOError(
            f"cannot hash archive {archive_file.name!r}: {exc.strerror or exc.__class__.__name__}"
        ) from exc
    if expected_sha256 is not None and archive_sha256 != expected_sha256:
        raise IntegrityError("archive SHA-256 does not match --expected-sha256")
    decompressed_size = _prevalidate_canonical_gzip(archive_file)

    try:
        with tarfile.open(archive_file, mode="r:gz") as archive:
            tar_members = _inspect_tar_members(archive)
            if "MANIFEST.json" not in tar_members or "SHA256SUMS" not in tar_members:
                raise IntegrityError("archive must contain MANIFEST.json and SHA256SUMS")
            manifest_data = _read_member(archive, tar_members["MANIFEST.json"])
            manifest = _parse_json_bytes(manifest_data, member_name="MANIFEST.json")
            if canonical_json_bytes(manifest) != manifest_data:
                raise IntegrityError("MANIFEST.json is not canonical JSON")
            (
                archive_kind,
                manifest_gate,
                manifest_tested_commit,
                manifest_entries,
            ) = _validate_manifest(manifest)
            expected_members = {"MANIFEST.json", "SHA256SUMS", *manifest_entries}
            actual_members = set(tar_members)
            missing = sorted(expected_members - actual_members)
            unexpected = sorted(actual_members - expected_members)
            if missing:
                raise IntegrityError(f"archive is missing manifest members: {missing!r}")
            if unexpected:
                raise IntegrityError(f"archive contains unexpected members: {unexpected!r}")

            member_data: dict[str, bytes] = {"MANIFEST.json": manifest_data}
            for path in sorted(manifest_entries):
                member_data[path] = _read_member(archive, tar_members[path])
            sums_data = _read_member(archive, tar_members["SHA256SUMS"])
    except (gzip.BadGzipFile, tarfile.TarError, EOFError) as exc:
        raise IntegrityError("archive is not a readable gzip-compressed tar file") from exc
    except OSError as exc:
        raise VerificationIOError(
            f"cannot read archive {archive_file.name!r}: {exc.strerror or exc.__class__.__name__}"
        ) from exc

    member_data["SHA256SUMS"] = sums_data
    _validate_canonical_tar_stream(
        archive_file,
        tar_members,
        member_data,
        decompressed_size=decompressed_size,
    )

    manifest_sha256 = sha256_bytes(manifest_data)
    for path, entry in manifest_entries.items():
        data = member_data[path]
        if len(data) != entry["size_bytes"]:
            raise IntegrityError(f"manifest size mismatch for member {path!r}")
        if sha256_bytes(data) != entry["sha256"]:
            raise IntegrityError(f"manifest SHA-256 mismatch for member {path!r}")

    try:
        sums = parse_sha256sums(sums_data)
    except (InputValidationError, IntegrityError) as exc:
        raise _as_integrity_error(exc) from exc
    expected_sum_paths = {"MANIFEST.json", *manifest_entries}
    if set(sums) != expected_sum_paths:
        missing = sorted(expected_sum_paths - set(sums))
        extra = sorted(set(sums) - expected_sum_paths)
        raise IntegrityError(
            f"SHA256SUMS member set mismatch (missing={missing!r}, extra={extra!r})"
        )
    canonical_sums = build_sha256sums(
        [{"path": path, "sha256": digest} for path, digest in sums.items()]
    )
    if sums_data != canonical_sums:
        raise IntegrityError("SHA256SUMS is not canonical and path-sorted")
    actual_digests = {
        "MANIFEST.json": manifest_sha256,
        **{path: sha256_bytes(data) for path, data in member_data.items() if path != "MANIFEST.json"},
    }
    for path, digest in sums.items():
        if actual_digests.get(path) != digest:
            raise IntegrityError(f"SHA256SUMS digest mismatch for member {path!r}")

    if archive_kind == "sanitized_shareable":
        sanitization, sanitization_descriptor_values = _verify_sanitized_archive(
            member_data, manifest_entries
        )
        sanitization = {
            **sanitization,
            "descriptor_values": sanitization_descriptor_values,
        }
    else:
        sanitization = {"status": "not_applicable"}
        sanitization_descriptor_values = None

    try:
        final_stat = archive_file.stat()
        final_sha256 = sha256_file(archive_file)
    except OSError as exc:
        raise VerificationIOError(
            f"cannot recheck archive {archive_file.name!r}: {exc.strerror or exc.__class__.__name__}"
        ) from exc
    if final_stat.st_size != archive_size or final_sha256 != archive_sha256:
        raise IntegrityError("archive changed during verification")

    if evidence_document is not None:
        _cross_check_descriptor(
            evidence_document,
            archive_kind=archive_kind,
            archive_filename=archive_file.name,
            archive_sha256=archive_sha256,
            archive_size=archive_size,
            manifest_gate=manifest_gate,
            manifest_sha256=manifest_sha256,
            manifest_source_entries=manifest_entries,
            manifest_tested_commit=manifest_tested_commit,
            sanitization_descriptor_values=sanitization_descriptor_values,
        )

    return {
        "archive": {
            "archive_kind": archive_kind,
            "filename": archive_file.name,
            "manifest_sha256": manifest_sha256,
            "member_count": len(tar_members),
            "sha256": archive_sha256,
            "size_bytes": archive_size,
        },
        "checks": {
            "archive_sha256": "verified" if expected_sha256 is not None else "computed",
            "canonical_gzip_and_tar": "verified",
            "evidence_document": "verified" if evidence_document is not None else "not_requested",
            "manifest": "verified",
            "member_hashes": "verified",
            "member_paths_and_types": "verified",
            "sanitization": sanitization,
            "sha256sums": "verified",
        },
        "errors": [],
        "format_version": REPORT_FORMAT_VERSION,
        "status": "verified",
        "verified_at_utc": timestamp,
        "verifier_version": TOOL_VERSION,
    }


def _failure_report(
    *,
    archive_filename: str,
    timestamp: str,
    category: str,
    message: str,
) -> dict[str, Any]:
    return {
        "archive": {"filename": archive_filename},
        "errors": [{"category": category, "message": message}],
        "format_version": REPORT_FORMAT_VERSION,
        "status": "failed",
        "verified_at_utc": timestamp,
        "verifier_version": TOOL_VERSION,
    }


def _emit_report(report: Mapping[str, Any], *, json_mode: bool, output: Path | None) -> None:
    encoded = canonical_json_bytes(report)
    if output is not None:
        try:
            safe_write_bytes(output, encoded)
        except OSError as exc:
            raise VerificationIOError(
                f"cannot write verification report {output.name!r}: {exc.strerror or exc.__class__.__name__}"
            ) from exc
    if json_mode:
        sys.stdout.buffer.write(encoded)
        return
    archive = report.get("archive", {})
    print(f"status: {report.get('status')}")
    print(f"archive: {archive.get('filename', 'not_available')}")
    if report.get("status") == "verified":
        print(f"archive kind: {archive.get('archive_kind')}")
        print(f"archive sha256: {archive.get('sha256')}")
        print(f"manifest sha256: {archive.get('manifest_sha256')}")
        print(f"members verified: {archive.get('member_count')}")
    else:
        for error in report.get("errors", []):
            print(f"error [{error.get('category')}]: {error.get('message')}")
    print(f"verified at UTC: {report.get('verified_at_utc')}")


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Archive to verify")
    parser.add_argument(
        "--expected-sha256",
        help="Optional expected SHA-256 for the complete .tar.gz archive",
    )
    parser.add_argument(
        "--evidence-document",
        type=Path,
        help="Optional v1 evidence descriptor to validate and cross-check",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=repository_root / "schemas" / "validation_evidence_v1.schema.json",
        help="Local JSON schema used with --evidence-document",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON verification report")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    parser.add_argument(
        "--timestamp-utc",
        help="Inject the report verification timestamp for deterministic testing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        timestamp = validate_utc(args.timestamp_utc or utc_now(), field="--timestamp-utc")
    except InputValidationError as exc:
        timestamp = utc_now()
        report = _failure_report(
            archive_filename=args.archive.name,
            timestamp=timestamp,
            category="invalid_input",
            message=str(exc),
        )
        try:
            _emit_report(report, json_mode=args.json, output=args.output)
        except VerificationIOError as output_exc:
            print(f"verification report output failed: {output_exc}", file=sys.stderr)
            return 4
        return 2

    code = 0
    try:
        if args.output is not None:
            try:
                if args.output.resolve() == args.archive.resolve():
                    raise InputValidationError("--output must not overwrite --archive")
            except OSError as exc:
                raise VerificationIOError("cannot resolve archive/report output paths") from exc
        if args.expected_sha256 is not None:
            require_sha256(args.expected_sha256, field="--expected-sha256")
        evidence_document = None
        if args.evidence_document is not None:
            evidence_document = _validate_evidence_input(args.evidence_document, args.schema)
        report = verify_archive(
            args.archive,
            expected_sha256=args.expected_sha256,
            evidence_document=evidence_document,
            verification_timestamp_utc=timestamp,
        )
    except InputValidationError as exc:
        code = 2
        report = _failure_report(
            archive_filename=args.archive.name,
            timestamp=timestamp,
            category="invalid_input",
            message=str(exc),
        )
    except IntegrityError as exc:
        code = 3
        report = _failure_report(
            archive_filename=args.archive.name,
            timestamp=timestamp,
            category="integrity_or_security",
            message=str(exc),
        )
    except VerificationIOError as exc:
        code = 4
        report = _failure_report(
            archive_filename=args.archive.name,
            timestamp=timestamp,
            category="io_or_runtime",
            message=str(exc),
        )
    except Exception as exc:  # Keep the CLI stable and traceback-free by default.
        code = 4
        report = _failure_report(
            archive_filename=args.archive.name,
            timestamp=timestamp,
            category="io_or_runtime",
            message=f"unexpected {exc.__class__.__name__}",
        )

    try:
        _emit_report(report, json_mode=args.json, output=args.output)
    except VerificationIOError as exc:
        print(f"verification report output failed: {exc}", file=sys.stderr)
        return 4
    return code


if __name__ == "__main__":
    raise SystemExit(main())
