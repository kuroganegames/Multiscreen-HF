#!/usr/bin/env python3
"""Shared standard-library helpers for validation evidence tooling.

This module deliberately implements only the JSON Schema keywords used by the
checked-in v1 schema.  It never resolves network references.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


TOOL_VERSION = "1.0.0"
ARCHIVE_FORMAT_VERSION = "validation-evidence-archive-v1"
PACKAGE_INPUT_VERSION = "validation-evidence-package-input-v1"
SANITIZATION_FORMAT_VERSION = "validation-evidence-sanitization-v1"
SANITIZATION_RULESET_VERSION = "multiscreen-sanitization-v1"
CONTROL_MEMBERS = frozenset({"MANIFEST.json", "SHA256SUMS"})
SANITIZATION_REPORT = "SANITIZATION_REPORT.json"
SANITIZATION_RULES = (
    "bearer_token",
    "cache_path",
    "credential_url",
    "file_uri",
    "github_token",
    "huggingface_token",
    "json_secret_assignment",
    "openai_token",
    "python_interpreter",
    "secret_assignment",
    "sensitive_literal",
    "unix_absolute_path",
    "windows_absolute_path",
)
SOURCE_ARTIFACT_CLASSIFICATIONS = (
    "validation_summary",
    "validation_metrics",
    "completion_marker",
    "provenance",
    "command_record",
    "environment_record",
    "other",
)

# Source payload limits are distinct from full archive limits. A sanitized
# archive can contain all source artifacts plus MANIFEST.json, SHA256SUMS, and
# SANITIZATION_REPORT.json, and bounded control bytes sit above source bytes.
MAX_SOURCE_ARTIFACT_COUNT = 10_000
MAX_ARCHIVE_CONTROL_MEMBER_COUNT = 3
MAX_ARCHIVE_MEMBER_COUNT = MAX_SOURCE_ARTIFACT_COUNT + MAX_ARCHIVE_CONTROL_MEMBER_COUNT
MAX_ARCHIVE_MEMBER_SIZE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_CONTROL_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ARCHIVE_MEMBER_BYTES = MAX_TOTAL_SOURCE_BYTES + MAX_TOTAL_CONTROL_BYTES
MAX_TOTAL_MEMBER_BYTES = MAX_TOTAL_ARCHIVE_MEMBER_BYTES
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)

SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "title",
        "description",
        "type",
        "const",
        "enum",
        "oneOf",
        "anyOf",
        "allOf",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
    }
)


class EvidenceError(Exception):
    """Base class for expected evidence-tooling failures."""


class InputValidationError(EvidenceError):
    """The CLI input, manifest, descriptor, or filesystem selection is invalid."""


class IntegrityError(EvidenceError):
    """Evidence integrity or sanitization verification failed."""


def utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def validate_utc(value: str, *, field: str = "timestamp") -> str:
    if not isinstance(value, str) or UTC_RE.fullmatch(value) is None:
        raise InputValidationError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        _datetime.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise InputValidationError(f"{field} is not a real UTC timestamp: {value!r}") from exc
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON with a trailing newline."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise InputValidationError(f"non-finite JSON number is forbidden: {value}")


def parse_json_bytes(data: bytes, *, field: str = "JSON input") -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"invalid UTF-8 {field}: {exc}") from exc


def load_json(path: Path) -> Any:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InputValidationError(f"cannot read JSON file {path}: {exc}") from exc
    return parse_json_bytes(data, field=f"JSON file {path}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise InputValidationError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def safe_archive_path(value: Any, *, field: str = "archive path") -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{field} must be a non-empty POSIX relative path")
    if not value.isascii() or "\\" in value or "\x00" in value or any(ord(char) < 32 for char in value):
        raise InputValidationError(f"unsafe {field}: {value!r}")
    if re.match(r"^[A-Za-z]:/", value):
        raise InputValidationError(f"drive-qualified {field} is forbidden: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise InputValidationError(f"absolute {field} is forbidden: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise InputValidationError(f"traversal or ambiguous component in {field}: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise InputValidationError(f"non-canonical {field}: {value!r}")
    return normalized


def ensure_payload_archive_path(value: Any) -> str:
    path = safe_archive_path(value)
    if not path.startswith("artifacts/"):
        raise InputValidationError(f"payload archive path must be below artifacts/: {path!r}")
    if path in CONTROL_MEMBERS or path == SANITIZATION_REPORT:
        raise InputValidationError(f"payload path collides with a control member: {path!r}")
    return path


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise InputValidationError(f"expected a boolean value, got {value!r}")


def _json_type_matches(instance: Any, type_name: str) -> bool:
    if type_name == "null":
        return instance is None
    if type_name == "boolean":
        return isinstance(instance, bool)
    if type_name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if type_name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if type_name == "string":
        return isinstance(instance, str)
    if type_name == "array":
        return isinstance(instance, list)
    if type_name == "object":
        return isinstance(instance, dict)
    return False


def assert_supported_schema(schema: Any, *, path: str = "$") -> None:
    """Reject schema keywords that the offline subset validator cannot enforce."""

    if isinstance(schema, dict):
        for key, value in schema.items():
            if (
                key not in SUPPORTED_SCHEMA_KEYWORDS
                and not path.endswith(".$defs")
                and not path.endswith(".properties")
            ):
                raise InputValidationError(f"unsupported JSON Schema keyword {key!r} at {path}")
            child_path = f"{path}.{key}"
            if key in {"properties", "$defs"}:
                if not isinstance(value, dict):
                    raise InputValidationError(f"{child_path} must be an object")
                for child_name, child_schema in value.items():
                    assert_supported_schema(child_schema, path=f"{child_path}.{child_name}")
            elif key in {"items", "additionalProperties"} and isinstance(value, dict):
                assert_supported_schema(value, path=child_path)
            elif key in {"oneOf", "anyOf", "allOf"}:
                if not isinstance(value, list):
                    raise InputValidationError(f"{child_path} must be an array")
                for index, child_schema in enumerate(value):
                    assert_supported_schema(child_schema, path=f"{child_path}[{index}]")


def _resolve_local_ref(root_schema: Mapping[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise InputValidationError(f"only local JSON Schema references are supported: {ref!r}")
    current: Any = root_schema
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise InputValidationError(f"unresolvable local JSON Schema reference: {ref!r}")
        current = current[part]
    return current


def validate_json_schema_instance(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Validate an instance with the deliberately limited offline schema subset."""

    assert_supported_schema(schema)
    errors: list[str] = []

    def check(value: Any, rule: Any, path: str) -> None:
        if not isinstance(rule, dict):
            errors.append(f"{path}: schema node is not an object")
            return
        if "$ref" in rule:
            try:
                resolved = _resolve_local_ref(schema, rule["$ref"])
            except InputValidationError as exc:
                errors.append(f"{path}: {exc}")
                return
            check(value, resolved, path)
            return
        if "allOf" in rule:
            for candidate in rule["allOf"]:
                check(value, candidate, path)
        if "anyOf" in rule:
            candidate_errors = []
            for candidate in rule["anyOf"]:
                local: list[str] = []
                before = len(errors)
                check(value, candidate, path)
                local.extend(errors[before:])
                del errors[before:]
                candidate_errors.append(local)
            if all(candidate_errors):
                errors.append(f"{path}: does not satisfy anyOf")
                return
        if "oneOf" in rule:
            matches = 0
            for candidate in rule["oneOf"]:
                before = len(errors)
                check(value, candidate, path)
                if len(errors) == before:
                    matches += 1
                else:
                    del errors[before:]
            if matches != 1:
                errors.append(f"{path}: must satisfy exactly one oneOf branch (matched {matches})")
                return
        expected_type = rule.get("type")
        if expected_type is not None:
            type_names = expected_type if isinstance(expected_type, list) else [expected_type]
            if not all(isinstance(item, str) for item in type_names):
                errors.append(f"{path}: schema type is invalid")
                return
            if not any(_json_type_matches(value, item) for item in type_names):
                errors.append(f"{path}: expected type {type_names}, got {type(value).__name__}")
                return
        if "const" in rule and value != rule["const"]:
            errors.append(f"{path}: expected constant {rule['const']!r}")
        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"{path}: value {value!r} is not in enum")
        if isinstance(value, dict):
            required = rule.get("required", [])
            for key in required:
                if key not in value:
                    errors.append(f"{path}: missing required property {key!r}")
            properties = rule.get("properties", {})
            if not isinstance(properties, dict):
                errors.append(f"{path}: schema properties is invalid")
                return
            for key, child in value.items():
                if key in properties:
                    check(child, properties[key], f"{path}.{key}")
                elif rule.get("additionalProperties") is False:
                    errors.append(f"{path}: unexpected property {key!r}")
                elif isinstance(rule.get("additionalProperties"), dict):
                    check(child, rule["additionalProperties"], f"{path}.{key}")
        if isinstance(value, list):
            if "minItems" in rule and len(value) < rule["minItems"]:
                errors.append(f"{path}: has fewer than {rule['minItems']} items")
            if "maxItems" in rule and len(value) > rule["maxItems"]:
                errors.append(f"{path}: has more than {rule['maxItems']} items")
            if rule.get("uniqueItems"):
                encoded = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
                if len(encoded) != len(set(encoded)):
                    errors.append(f"{path}: items are not unique")
            if isinstance(rule.get("items"), dict):
                for index, child in enumerate(value):
                    check(child, rule["items"], f"{path}[{index}]")
        if isinstance(value, str):
            if "minLength" in rule and len(value) < rule["minLength"]:
                errors.append(f"{path}: string is shorter than {rule['minLength']}")
            if "maxLength" in rule and len(value) > rule["maxLength"]:
                errors.append(f"{path}: string is longer than {rule['maxLength']}")
            if "pattern" in rule:
                try:
                    matched = re.search(rule["pattern"], value)
                except re.error as exc:
                    errors.append(f"{path}: invalid schema regex: {exc}")
                else:
                    if matched is None:
                        errors.append(f"{path}: string does not match required pattern")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                errors.append(f"{path}: value is below minimum {rule['minimum']}")
            if "maximum" in rule and value > rule["maximum"]:
                errors.append(f"{path}: value is above maximum {rule['maximum']}")

    check(instance, schema, "$")
    return errors


def validate_evidence_document(document: Any, schema: Mapping[str, Any]) -> list[str]:
    """Run schema checks plus v1 cross-field integrity rules."""

    errors = validate_json_schema_instance(document, schema)
    if errors or not isinstance(document, dict):
        return errors

    def validate_timestamps(value: Any, path: str, inherited: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                is_timestamp = key.endswith("_at_utc") or (inherited and key == "value")
                if isinstance(child, str) and is_timestamp:
                    try:
                        validate_utc(child, field=child_path)
                    except InputValidationError as exc:
                        errors.append(str(exc))
                else:
                    validate_timestamps(child, child_path, is_timestamp)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                validate_timestamps(child, f"{path}[{index}]", inherited)

    validate_timestamps(document, "$")

    artifacts = document.get("source_artifacts", [])
    logical_names = [item.get("logical_name") for item in artifacts if isinstance(item, dict)]
    archive_paths = [item.get("archive_path") for item in artifacts if isinstance(item, dict)]
    if len(logical_names) != len(set(logical_names)):
        errors.append("$.source_artifacts: logical_name values must be unique")
    if len(archive_paths) != len(set(archive_paths)):
        errors.append("$.source_artifacts: archive_path values must be unique")
    for index, path in enumerate(archive_paths):
        try:
            ensure_payload_archive_path(path)
        except InputValidationError as exc:
            errors.append(f"$.source_artifacts[{index}].archive_path: {exc}")

    worktrees = (
        (
            "$.original_run_provenance.run_worktree_at_start",
            document["original_run_provenance"]["run_worktree_at_start"],
        ),
        (
            "$.original_run_provenance.run_worktree_at_end",
            document["original_run_provenance"]["run_worktree_at_end"],
        ),
        (
            "$.evidence_handoff_provenance.worktree_before_edits",
            document["evidence_handoff_provenance"]["worktree_before_edits"],
        ),
        (
            "$.evidence_handoff_provenance.worktree_after_commit",
            document["evidence_handoff_provenance"]["worktree_after_commit"],
        ),
    )
    for path, observation in worktrees:
        if observation.get("status") != "recorded":
            continue
        derived_clean = (
            not observation["staged_changes"]
            and not observation["unstaged_changes"]
            and observation["untracked_count"] == 0
        )
        if observation["clean"] != derived_clean:
            errors.append(f"{path}.clean: contradicts staged/unstaged/untracked fields")

    archives = document.get("archives", {})
    required_archive_fields = (
        "archive_filename",
        "storage_class",
        "storage_locator",
        "sha256",
        "size_bytes",
        "manifest_sha256",
        "created_at_utc",
        "verified_at_utc",
        "public",
        "verification_report_sha256",
    )
    for kind in ("exact_private", "sanitized_shareable"):
        descriptor = archives.get(kind, {}) if isinstance(archives, dict) else {}
        if descriptor.get("status") == "verified":
            for field in required_archive_fields:
                if descriptor.get(field) is None:
                    errors.append(f"$.archives.{kind}.{field}: verified archive field cannot be null")
        locator = descriptor.get("storage_locator")
        if isinstance(locator, str):
            if locator.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", locator):
                errors.append(f"$.archives.{kind}.storage_locator: private absolute paths are forbidden")
            if ".." in PurePosixPath(locator.replace("\\", "/")).parts:
                errors.append(f"$.archives.{kind}.storage_locator: traversal is forbidden")
    exact = archives.get("exact_private", {}) if isinstance(archives, dict) else {}
    if exact.get("public") is not False:
        errors.append(
            "$.archives.exact_private.public: exact evidence must be explicitly non-public"
        )
    if exact.get("storage_class") == "public_release":
        errors.append(
            "$.archives.exact_private.storage_class: exact evidence must never use public_release"
        )
    if exact.get("status") == "verified" and exact.get("storage_class") != "private_external":
        errors.append(
            "$.archives.exact_private.storage_class: verified exact evidence requires private_external"
        )

    sanitization = document.get("sanitization", {})
    if sanitization.get("status") == "verified":
        if sanitization.get("unresolved_findings"):
            errors.append("$.sanitization.unresolved_findings: verified sanitization must have no findings")
        if sanitization.get("report_sha256") is None:
            errors.append("$.sanitization.report_sha256: verified sanitization requires a report hash")

    sanitized = archives.get("sanitized_shareable", {}) if isinstance(archives, dict) else {}
    exact_verified = exact.get("status") == "verified"
    sanitized_verified = sanitized.get("status") == "verified"
    sanitization_verified = sanitization.get("status") == "verified"
    if sanitized_verified and sanitized.get("storage_class") not in {
        "sanitized_staging",
        "public_release",
    }:
        errors.append(
            "$.archives.sanitized_shareable.storage_class: verified sanitized evidence "
            "requires sanitized_staging or public_release"
        )

    exact_flags = [item.get("exact_bytes_retained") for item in artifacts]
    if any(flag is not exact_verified for flag in exact_flags):
        errors.append(
            "$.source_artifacts: every exact_bytes_retained flag must agree with "
            "the exact_private archive verified status"
        )
    sanitized_statuses = [item.get("sanitized_copy_status") for item in artifacts]
    if sanitized_verified and any(status != "verified" for status in sanitized_statuses):
        errors.append(
            "$.source_artifacts: a verified sanitized archive requires every sanitized copy verified"
        )
    if not sanitized_verified and any(status == "verified" for status in sanitized_statuses):
        errors.append(
            "$.source_artifacts: a non-verified sanitized archive cannot claim a verified copy"
        )
    if sanitized_verified != sanitization_verified:
        errors.append(
            "$.sanitization.status: must agree with the sanitized_shareable archive verified status"
        )

    verification = document.get("verification", {})
    raw_reports = verification.get("reports", [])
    reports: dict[str, Mapping[str, Any]] = {}
    for index, report in enumerate(raw_reports):
        kind = report.get("archive_kind")
        if kind in reports:
            errors.append(
                f"$.verification.reports[{index}].archive_kind: duplicate report for {kind!r}"
            )
        else:
            reports[kind] = report

    for kind, descriptor in (("exact_private", exact), ("sanitized_shareable", sanitized)):
        report = reports.get(kind)
        descriptor_verified = descriptor.get("status") == "verified"
        report_verified = report is not None and report.get("status") == "verified"
        if descriptor_verified and report is None:
            errors.append(f"$.verification.reports: verified {kind} archive requires a report")
            continue
        if descriptor_verified and not report_verified:
            errors.append(
                f"$.verification.reports: verified {kind} archive requires a verified report"
            )
        if report_verified and not descriptor_verified:
            errors.append(
                f"$.verification.reports: verified {kind} report requires a verified archive"
            )
        if report_verified:
            if report.get("archive_sha256") != descriptor.get("sha256"):
                errors.append(
                    f"$.verification.reports: {kind} report hash does not match archive descriptor"
                )
            if report.get("verified_at_utc") != descriptor.get("verified_at_utc"):
                errors.append(
                    f"$.verification.reports: {kind} verification timestamp does not match archive descriptor"
                )
            if report.get("errors"):
                errors.append(f"$.verification.reports: verified {kind} report must have no errors")

    required_report_kinds = {"exact_private", "sanitized_shareable"}
    both_reports_verified = (
        set(reports) == required_report_kinds
        and all(report.get("status") == "verified" for report in reports.values())
    )
    both_archives_verified = exact_verified and sanitized_verified
    if verification.get("status") == "verified":
        if not both_reports_verified or not both_archives_verified:
            errors.append(
                "$.verification.status: verified requires both archive descriptors and reports verified"
            )
    elif both_reports_verified and both_archives_verified:
        errors.append(
            "$.verification.status: both verified archives and reports require status=verified"
        )

    retention = document.get("retention", {})
    exact_retained = retention.get("exact_private_retained")
    sanitized_retained = retention.get("sanitized_archive_verified")
    expected_exact_retained = exact_verified and all(flag is True for flag in exact_flags)
    sanitized_report = reports.get("sanitized_shareable")
    expected_sanitized_retained = (
        sanitized_verified
        and sanitization_verified
        and sanitized_report is not None
        and sanitized_report.get("status") == "verified"
        and all(status == "verified" for status in sanitized_statuses)
    )
    if exact_retained is not expected_exact_retained:
        errors.append("$.retention.exact_private_retained: contradicts exact archive/artifact state")
    if sanitized_retained is not expected_sanitized_retained:
        errors.append(
            "$.retention.sanitized_archive_verified: contradicts sanitized archive/artifact state"
        )

    published = retention.get("public_asset_published") is True
    sanitized_public = sanitized.get("public") is True
    if published != sanitized_public:
        errors.append("$.retention.public_asset_published: contradicts sanitized archive public state")
    if published:
        if not isinstance(retention.get("public_asset"), str):
            errors.append("$.retention.public_asset: published public evidence requires an asset locator")
        if sanitized.get("storage_class") != "public_release" or not sanitized_verified:
            errors.append(
                "$.archives.sanitized_shareable: public publication requires a verified public_release archive"
            )
    else:
        if retention.get("public_asset") is not None:
            errors.append("$.retention.public_asset: unpublished evidence must not name a public asset")
        if sanitized.get("storage_class") == "public_release":
            errors.append(
                "$.archives.sanitized_shareable.storage_class: public_release requires publication"
            )

    if retention.get("status") == "verified":
        if not retention.get("exact_private_retained"):
            errors.append("$.retention: verified retention requires exact_private_retained=true")
        if not retention.get("sanitized_archive_verified"):
            errors.append("$.retention: verified retention requires sanitized_archive_verified=true")
        if exact.get("status") != "verified" or archives.get("sanitized_shareable", {}).get("status") != "verified":
            errors.append("$.retention: verified retention requires both archive descriptors verified")
    elif exact_retained is True and sanitized_retained is True:
        errors.append("$.retention.status: both retained archives require status=verified")

    if document.get("evidence_status") == "complete":
        handoff = document.get("evidence_handoff_provenance", {})
        worktree_before = handoff.get("worktree_before_edits", {})
        worktree_after = handoff.get("worktree_after_commit", {})
        if retention.get("status") != "verified":
            errors.append("$.evidence_status: complete requires verified retention")
        if sanitization.get("status") != "verified":
            errors.append("$.evidence_status: complete requires verified sanitization")
        if verification.get("status") != "verified":
            errors.append("$.evidence_status: complete requires verified archive verification")
        if not both_archives_verified:
            errors.append("$.evidence_status: complete requires both archives verified")
        if not expected_exact_retained or not expected_sanitized_retained:
            errors.append("$.evidence_status: complete requires every source artifact verified")
        if document.get("acceptance_review", {}).get("status") != "recorded":
            errors.append("$.evidence_status: complete requires a recorded acceptance review")
        if handoff.get("final_commit", {}).get("status") != "recorded":
            errors.append("$.evidence_status: complete requires a recorded final commit")
        if worktree_before.get("status") != "recorded":
            errors.append("$.evidence_status: complete requires recorded pre-edit worktree state")
        elif worktree_before.get("clean") is not True:
            errors.append("$.evidence_status: complete requires a clean pre-edit worktree")
        if worktree_after.get("status") != "recorded":
            errors.append("$.evidence_status: complete requires recorded post-commit worktree state")
        elif worktree_after.get("clean") is not True:
            errors.append("$.evidence_status: complete requires a clean post-commit worktree")
        if handoff.get("archive_created_at_utc", {}).get("status") != "recorded":
            errors.append("$.evidence_status: complete requires a recorded archive creation timestamp")
        if handoff.get("archive_verified_at_utc", {}).get("status") != "recorded":
            errors.append("$.evidence_status: complete requires a recorded archive verification timestamp")

    def walk_strings(value: Any, path: str) -> Iterable[tuple[str, str]]:
        if isinstance(value, str):
            yield path, value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield from walk_strings(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk_strings(child, f"{path}[{index}]")

    for path, value in walk_strings(document, "$"):
        if _CREDENTIAL_URL_RE.search(value):
            errors.append(f"{path}: credential-bearing URL is forbidden")
        if _FILE_URI_RE.search(value):
            errors.append(f"{path}: private file URI is forbidden")
        if _WINDOWS_PATH_RE.search(value) or _UNIX_PATH_RE.search(value):
            errors.append(f"{path}: private absolute path is forbidden")
        if any(
            _is_unredacted_secret_assignment(match)
            for pattern in (_JSON_SECRET_RE, _ASSIGNMENT_SECRET_RE)
            for match in pattern.finditer(value)
        ):
            errors.append(f"{path}: secret assignment is forbidden")
        for name, pattern in _TOKEN_PATTERNS:
            if pattern.search(value):
                errors.append(f"{path}: {name} pattern is forbidden")
    return errors


_INTERPRETER_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/[A-Za-z0-9_.+@-]+)+/python(?:3(?:\.[0-9]+)?)?(?=\s|[\"'])"
)
_CREDENTIAL_URL_RE = re.compile(
    r"(?P<scheme>(?:https?|ssh|git)://)(?!\[REDACTED\]@)[^/@\s]+@",
    re.IGNORECASE,
)
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("openai_token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}={0,2}")),
)
_SECRET_KEY_PATTERN = (
    r"(?:password|api[-_]?key|access[-_]?token|auth[-_]?token|token|secret|"
    r"client[-_]?secret|private[-_]?key|aws_access_key_id|aws_secret_access_key|"
    r"aws_session_token)"
)
_SECRET_VALUE_PATTERN = (
    r'(?:"(?P<double_value>(?:\\.|[^"\\\r\n])*)"|'
    r"'(?P<single_value>(?:\\.|[^'\\\r\n])*)'|"
    r"(?P<bare_value>[^\s,;#}\]]+))"
)
_JSON_SECRET_RE = re.compile(
    (
        rf"(?P<prefix>(?<![A-Za-z0-9_-])(?P<key_quote>[\"']?)"
        rf"(?P<key>{_SECRET_KEY_PATTERN})(?P=key_quote)(?![A-Za-z0-9_-])[ \t]*:[ \t]*)"
        + _SECRET_VALUE_PATTERN
    ),
    re.IGNORECASE | re.MULTILINE,
)
_ASSIGNMENT_SECRET_RE = re.compile(
    (
        rf"(?P<prefix>(?<![A-Za-z0-9_-])(?P<key_quote>[\"']?)"
        rf"(?P<key>{_SECRET_KEY_PATTERN})(?P=key_quote)(?![A-Za-z0-9_-])[ \t]*=[ \t]*)"
        + _SECRET_VALUE_PATTERN
    ),
    re.IGNORECASE | re.MULTILINE,
)
_FILE_URI_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])file://(?:localhost)?/(?!<REDACTED)[^\s\"'<>]+"
)
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:[\\/](?:[^\s\"'<>]+[\\/]?)+")
_UNIX_PATH_RE = re.compile(r"(?<![:/A-Za-z0-9_.-])/(?:[^\s\"'<>]+/)*[^\s\"'<>]*")
_CACHE_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])"
    r"(?:\.cache|hf_cache|huggingface_cache|datasets_cache|tokenizer_cache|torch_cache)"
    r"(?:[/\\][^\s\"'<>]*)?"
)


def _substitute_with_count(pattern: re.Pattern[str], replacement: Any, text: str) -> tuple[str, int]:
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        value = replacement(match) if callable(replacement) else match.expand(replacement)
        if value != match.group(0):
            changed += 1
        return value

    return pattern.sub(replace, text), changed


def _secret_assignment_value(match: re.Match[str]) -> str:
    for group in ("double_value", "single_value", "bare_value"):
        value = match.group(group)
        if value is not None:
            return value
    return ""


def _redact_secret_assignment(match: re.Match[str]) -> str:
    value = _secret_assignment_value(match)
    if value == "<REDACTED:SECRET>":
        return match.group(0)
    if match.group("double_value") is not None:
        quote = '"'
    elif match.group("single_value") is not None:
        quote = "'"
    else:
        quote = ""
    return match.group("prefix") + quote + "<REDACTED:SECRET>" + quote


def _is_unredacted_secret_assignment(match: re.Match[str]) -> bool:
    return _secret_assignment_value(match) != "<REDACTED:SECRET>"


def sanitize_text(
    text: str,
    *,
    sensitive_values: Iterable[str] = (),
) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
    """Redact high-confidence local/private values and fail closed on leftovers."""

    counts: dict[str, int] = {}

    text, counts["python_interpreter"] = _substitute_with_count(_INTERPRETER_RE, "python", text)
    text, counts["credential_url"] = _substitute_with_count(
        _CREDENTIAL_URL_RE, lambda match: match.group("scheme") + "[REDACTED]@", text
    )
    for name, pattern in _TOKEN_PATTERNS:
        text, counts[name] = _substitute_with_count(pattern, f"<REDACTED:{name.upper()}>", text)
    text, counts["json_secret_assignment"] = _substitute_with_count(
        _JSON_SECRET_RE, _redact_secret_assignment, text
    )
    text, counts["secret_assignment"] = _substitute_with_count(
        _ASSIGNMENT_SECRET_RE, _redact_secret_assignment, text
    )
    text, counts["file_uri"] = _substitute_with_count(
        _FILE_URI_RE,
        "<REDACTED:FILE_URI>",
        text,
    )

    literal_count = 0
    normalized_values = sorted(
        {value for value in sensitive_values if isinstance(value, str) and len(value) >= 3},
        key=len,
        reverse=True,
    )
    for value in normalized_values:
        occurrences = text.count(value)
        if occurrences:
            text = text.replace(value, "<REDACTED:SENSITIVE_VALUE>")
            literal_count += occurrences
    counts["sensitive_literal"] = literal_count

    text, counts["windows_absolute_path"] = _substitute_with_count(
        _WINDOWS_PATH_RE, "<REDACTED:ABSOLUTE_PATH>", text
    )
    text, counts["unix_absolute_path"] = _substitute_with_count(
        _UNIX_PATH_RE, "<REDACTED:ABSOLUTE_PATH>", text
    )
    text, counts["cache_path"] = _substitute_with_count(_CACHE_PATH_RE, "<REDACTED:CACHE_PATH>", text)

    findings: list[dict[str, Any]] = []
    scans: Sequence[tuple[str, re.Pattern[str]]] = (
        ("credential_url", _CREDENTIAL_URL_RE),
        *_TOKEN_PATTERNS,
        ("file_uri", _FILE_URI_RE),
        ("windows_absolute_path", _WINDOWS_PATH_RE),
        ("unix_absolute_path", _UNIX_PATH_RE),
        ("cache_path", _CACHE_PATH_RE),
    )
    for name, pattern in scans:
        matches = list(pattern.finditer(text))
        if matches:
            findings.append({"rule": name, "count": len(matches), "severity": "high"})
    residual_json_secrets = [
        match
        for match in _JSON_SECRET_RE.finditer(text)
        if _is_unredacted_secret_assignment(match)
    ]
    if residual_json_secrets:
        findings.append(
            {
                "rule": "json_secret_assignment",
                "count": len(residual_json_secrets),
                "severity": "high",
            }
        )
    residual_assignment_secrets = [
        match
        for match in _ASSIGNMENT_SECRET_RE.finditer(text)
        if _is_unredacted_secret_assignment(match)
    ]
    if residual_assignment_secrets:
        findings.append(
            {
                "rule": "secret_assignment",
                "count": len(residual_assignment_secrets),
                "severity": "high",
            }
        )
    for value in normalized_values:
        remaining = text.count(value)
        if remaining:
            findings.append({"rule": "sensitive_literal", "count": remaining, "severity": "high"})
    counts = {key: value for key, value in sorted(counts.items()) if value}
    return text, counts, findings


def sanitized_member(
    raw: bytes,
    *,
    archive_path: str,
    sensitive_values: Iterable[str] = (),
) -> tuple[bytes, dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError(
            f"sanitized archive requires UTF-8 text; {archive_path!r} is not UTF-8"
        ) from exc
    sanitized, replacements, findings = sanitize_text(text, sensitive_values=sensitive_values)
    encoded = sanitized.encode("utf-8")
    report = {
        "path": archive_path,
        "source_sha256": sha256_bytes(raw),
        "sanitized_sha256": sha256_bytes(encoded),
        "source_size_bytes": len(raw),
        "sanitized_size_bytes": len(encoded),
        "replacements": replacements,
        "unresolved_findings": findings,
    }
    return encoded, report


def build_sha256sums(entries: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [f"{entry['sha256']}  {entry['path']}\n" for entry in sorted(entries, key=lambda item: item["path"])]
    return "".join(lines).encode("utf-8")


def parse_sha256sums(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError("SHA256SUMS is not UTF-8") from exc
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise IntegrityError(f"invalid SHA256SUMS line {line_number}")
        digest, raw_path = match.groups()
        path = safe_archive_path(raw_path, field=f"SHA256SUMS path on line {line_number}")
        if path in result:
            raise IntegrityError(f"duplicate SHA256SUMS path: {path!r}")
        result[path] = digest
    return result


def safe_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
