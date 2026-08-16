#!/usr/bin/env python3
"""Build and close reviewed Stage E Hugging Face contract evidence.

The accepted Level 1 builder and artifacts are immutable historical records.
This standard-library-only entry point loads an isolated copy of the proven
closure primitives, then installs a separate, fixed Stage E profile.  It never
discovers run artifacts by walking a directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence


_BOOTSTRAP_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_BOOTSTRAP_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_BOOTSTRAP_REPOSITORY_ROOT))


def _load_isolated_core() -> Any:
    path = Path(__file__).with_name("build_level1_evidence.py")
    name = "_multiscreen_hf_contract_hardening_builder_core"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the evidence closure core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_CORE = _load_isolated_core()

TOOL_VERSION = "1.0.0"
SUMMARY_VERSION = "multiscreen-hf-contract-hardening-summary-v1"
REVIEW_VERSION = "multiscreen-hf-contract-hardening-raw-evidence-review-v1"
PROVENANCE_VERSION = "validation-provenance-v1"
PACKAGE_REPORT_VERSION = "validation-evidence-package-report-v1"
GATE = "HF Contract Hardening"
EVIDENCE_GATE = "Stage E final requalification"
REPOSITORY_NAME = "kuroganegames/Multiscreen-HF"
IMPLEMENTATION_BASE_COMMIT = "bf8cc34cb6aa16ffeec1f609166db5efae79e9df"
WORKING_BRANCH = "validation/hf-contract-hardening-requalification"

SUMMARY_JSON_NAME = "HF_CONTRACT_HARDENING_SUMMARY.json"
SUMMARY_MARKDOWN_NAME = "HF_CONTRACT_HARDENING_SUMMARY.md"
DESCRIPTOR_NAME = "HF_CONTRACT_HARDENING_EVIDENCE_ARCHIVE.json"
EXACT_VERIFICATION_NAME = "HF_CONTRACT_HARDENING_EXACT_VERIFICATION.json"
SANITIZED_VERIFICATION_NAME = (
    "HF_CONTRACT_HARDENING_SANITIZED_VERIFICATION.json"
)
PACKAGE_INPUT_RELATIVE = "review/hf-contract-hardening-package-input.json"
FULL_REVIEW_RELATIVE = "review/hf-contract-hardening-raw-review.json"
ACCEPTANCE_PROVENANCE_RELATIVE = "review/acceptance-provenance.json"

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

PRECHECK_COMMAND_NAMES = (
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
FOCUSED_COMMAND_STEMS = (
    "hf-output-head",
    "training-edge",
    "gradient-checkpointing",
    "p0-4-qualification",
    "generation-input",
    "packed-text",
    "paper-architecture",
    "paper-initialization",
    "mipe-position-cache",
    "paper-training-contract",
)
FOCUSED_COMMAND_NAMES = tuple(
    f"{stem}-{lane}"
    for stem in FOCUSED_COMMAND_STEMS
    for lane in ("tf4576", "tf5141")
)
REQUIRED_COMMAND_NAMES = (
    *PRECHECK_COMMAND_NAMES,
    *FOCUSED_COMMAND_NAMES,
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
REQUIRED_ENVIRONMENT_NAMES = ("runtime-tf4576", "runtime-tf5141")

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

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

EvidenceBuildError = _CORE.EvidenceBuildError
InputValidationError = _CORE.InputValidationError
IntegrityError = _CORE.IntegrityError
canonical_json_bytes = _CORE.canonical_json_bytes
sha256_bytes = _CORE.sha256_bytes
validate_evidence_document = _CORE.validate_evidence_document
verify_archive = _CORE.verify_archive
PACKAGE_INPUT_VERSION = _CORE.PACKAGE_INPUT_VERSION
VERIFIER_VERSION = _CORE.VERIFIER_VERSION

_fail = _CORE._fail
_mapping = _CORE._mapping
_list = _CORE._list
_string = _CORE._string
_digest = _CORE._digest
_commit = _CORE._commit
_fixed_child = _CORE._fixed_child
_stable_read = _CORE._stable_read
_load_json = _CORE._load_json
_assert_public_document = _CORE._assert_public_document
_exclusive_write = _CORE._exclusive_write
_prepare_output = _CORE._prepare_output
_validated_roots = _CORE._validated_roots
_validate_live_repository_state = _CORE._validate_live_repository_state
_validate_commit_evidence_blobs = _CORE._validate_commit_evidence_blobs
_registered_worktrees = _CORE._registered_worktrees
_git_stdout = _CORE._git_stdout
_legacy_validate_acceptance = _CORE._validate_acceptance


class ArtifactSpec(NamedTuple):
    source_path: str
    classification: str

    @property
    def archive_path(self) -> str:
        return f"artifacts/hf-contract-hardening/run/{self.source_path}"


def _review_canonical_bytes(value: Any) -> bytes:
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
        add(f"{logical}.data_contract", f"{base}/data_contract.json", "validation_metrics")
        add(f"{logical}.completion_marker", f"{base}/P0-4_COMPLETE.md", "completion_marker")
        add(f"{logical}.summary", f"{base}/summary.json", "validation_summary")
        add(f"{logical}.metrics", f"{base}/metrics.jsonl", "validation_metrics")
    add(
        "p0_4_psi8.focused_review",
        "artifacts/p0-4/psi8/raw-review.json",
        "validation_summary",
    )
    for logical, path in {
        "p0_3_psi8": "artifacts/p0-3/tokenizer-reload-psi8.json",
        "p0_3_psi16": "artifacts/p0-3/tokenizer-reload-psi16.json",
        "p0_4_psi8": "artifacts/p0-4/psi8/tokenizer-reload.json",
        "p0_4_psi16": "artifacts/p0-4/psi16/tokenizer-reload.json",
    }.items():
        add(f"tokenizer_reload.{logical}", path, "validation_summary")
    add("runner.run_marker", ".level1-requalification-run.json", "provenance")
    add("runner.commands_ledger", "commands.jsonl", "command_record")
    add("runner.environment_ledger", "environment.jsonl", "environment_record")
    for name in REQUIRED_COMMAND_NAMES:
        add(f"runner.log.{name}", f"logs/{name}.log", "other")
        add(f"runner.record.{name}", f"records/{name}.json", "command_record")
    for name in REQUIRED_ENVIRONMENT_NAMES:
        add(f"runner.record.{name}", f"records/{name}.json", "environment_record")
    if len(REQUIRED_COMMAND_NAMES) != 53 or len(set(REQUIRED_COMMAND_NAMES)) != 53:
        _fail("internal Stage E command allowlist is not exactly 53 unique names")
    return specs


def _historical_tree(commit: str) -> list[dict[str, str]]:
    commit_value = _commit(commit, label="historical tree commit")
    raw = _git_stdout(
        (
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit_value,
            "--",
            "docs/validation_results",
        ),
        label=f"historical validation results at {commit_value}",
    )
    if raw and not raw.endswith(b"\0"):
        _fail("historical validation-results tree listing is malformed")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in raw[:-1].split(b"\0") if raw else ():
        try:
            metadata, path_raw = record.split(b"\t", 1)
            mode_raw, type_raw, object_raw = metadata.split(b" ", 2)
            path = path_raw.decode("utf-8", errors="strict")
            mode = mode_raw.decode("ascii", errors="strict")
            object_type = type_raw.decode("ascii", errors="strict")
            object_id = object_raw.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError):
            _fail("historical validation-results tree entry is malformed")
        if not path.startswith(HISTORICAL_PATH_PREFIXES):
            continue
        if path in seen:
            _fail("historical validation-results tree contains a duplicate path")
        seen.add(path)
        if mode != "100644" or object_type != "blob":
            _fail("historical evidence is not a regular blob")
        if GIT_OBJECT_RE.fullmatch(object_id) is None:
            _fail("historical evidence blob object ID is invalid")
        entries.append(
            {
                "mode": mode,
                "object_id": object_id,
                "path": path,
                "type": object_type,
            }
        )
    entries.sort(key=lambda item: item["path"].encode("utf-8"))
    if tuple(item["path"] for item in entries) != HISTORICAL_PATHS:
        _fail("historical evidence path set differs from the fixed accepted set")
    return entries


def _historical_evidence_projection(tested_commit: str) -> dict[str, Any]:
    tested = _commit(tested_commit, label="tested source commit")
    baseline_tree = _historical_tree(IMPLEMENTATION_BASE_COMMIT)
    tested_tree = _historical_tree(tested)
    if tested_tree != baseline_tree:
        _fail("accepted historical evidence paths or blobs changed after the baseline")
    return {
        "artifact_count": len(baseline_tree),
        "implementation_baseline": IMPLEMENTATION_BASE_COMMIT,
        "status": "passed",
        "tested_commit": tested,
        "tree_material_sha256": sha256_bytes(_review_canonical_bytes(baseline_tree)),
    }


def _validate_implementation_base_commit(value: Any, *, tested_commit: str) -> str:
    base = _commit(value, label="implementation base commit")
    tested = _commit(tested_commit, label="tested source commit")
    if base != IMPLEMENTATION_BASE_COMMIT:
        _fail("implementation base commit differs from the fixed Stage E baseline")
    if tested == base:
        _fail("Stage E tested source must be a post-baseline evidence-support commit")
    if _git_stdout(("cat-file", "-t", base), label="implementation baseline type") != b"commit\n":
        _fail("implementation baseline is not a commit object")
    if _git_stdout(("cat-file", "-t", tested), label="tested source type") != b"commit\n":
        _fail("tested source is not a commit object")
    try:
        ancestry = _git_stdout(
            ("merge-base", "--is-ancestor", base, tested),
            label="implementation baseline ancestry",
        )
    except EvidenceBuildError:
        _fail("implementation baseline is not an ancestor of the tested source")
    if ancestry:
        _fail("implementation baseline ancestry command emitted unexpected output")
    _historical_evidence_projection(tested)
    return base


def _validate_acceptance(
    provenance: Mapping[str, Any], *, tested_commit: str
) -> tuple[list[dict[str, Any]], Mapping[str, Any], str]:
    reviewers, worktree, branch = _legacy_validate_acceptance(
        provenance, tested_commit=tested_commit
    )
    if branch != WORKING_BRANCH:
        _fail("acceptance provenance branch differs from the fixed Stage E branch")
    return reviewers, worktree, branch


def _validate_status_section(review: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    section = _mapping(review.get(field), label=f"review.{field}")
    if section.get("status") != "passed":
        _fail(f"review.{field} is not passed")
    return section


def _validate_reviewer_contract_alignment() -> None:
    try:
        from scripts import review_hf_contract_hardening as reviewer
    except (ImportError, OSError) as exc:
        _fail(f"Stage E reviewer contract cannot be loaded: {exc}")
    expected = {
        "SCHEMA_VERSION": REVIEW_VERSION,
        "IMPLEMENTATION_BASELINE": IMPLEMENTATION_BASE_COMMIT,
        "WORKING_BRANCH": WORKING_BRANCH,
        "FULL_REVIEW_RELATIVE": FULL_REVIEW_RELATIVE,
        "HISTORICAL_PATHS": HISTORICAL_PATHS,
        "REQUIRED_COMMAND_NAMES": REQUIRED_COMMAND_NAMES,
        "REQUIRED_ENVIRONMENT_NAMES": REQUIRED_ENVIRONMENT_NAMES,
    }
    for name, value in expected.items():
        if getattr(reviewer, name, None) != value:
            _fail(f"Stage E reviewer contract differs for {name}")


def _validate_review(review: Mapping[str, Any]) -> tuple[str, Mapping[str, str]]:
    _validate_reviewer_contract_alignment()
    expected_top = {
        "aggregate",
        "command_ledger",
        "focused_tests",
        "historical_evidence_immutability",
        "implementation_baseline",
        "p0_3",
        "p0_4",
        "schema_version",
        "status",
        "tested_commit",
        "tokenizer_reload",
    }
    if set(review) != expected_top:
        _fail("full Stage E review fields are incomplete or ambiguous")
    if review.get("schema_version") != REVIEW_VERSION or review.get("status") != "passed":
        _fail("full review is not a passed Stage E raw-evidence review")
    tested = _commit(review.get("tested_commit"), label="review.tested_commit")
    if review.get("implementation_baseline") != IMPLEMENTATION_BASE_COMMIT:
        _fail("full review implementation baseline differs")
    for field in ("p0_3", "p0_4", "tokenizer_reload"):
        _validate_status_section(review, field)

    historical = _mapping(
        review.get("historical_evidence_immutability"),
        label="review.historical_evidence_immutability",
    )
    if set(historical) != {
        "artifact_count",
        "implementation_baseline",
        "status",
        "tested_commit",
        "tree_material_sha256",
    }:
        _fail("historical evidence projection fields are incomplete or ambiguous")
    if (
        historical.get("status") != "passed"
        or historical.get("implementation_baseline") != IMPLEMENTATION_BASE_COMMIT
        or historical.get("tested_commit") != tested
        or historical.get("artifact_count") != len(HISTORICAL_PATHS)
    ):
        _fail("historical evidence projection identity differs")
    _digest(
        historical.get("tree_material_sha256"),
        label="historical evidence tree material SHA-256",
    )

    focused = _validate_status_section(review, "focused_tests")
    if set(focused) != {"expected_per_lane", "lanes", "status"} or focused.get(
        "expected_per_lane"
    ) != 117:
        _fail("focused test review does not require 117 tests per exact lane")
    lanes = _mapping(focused.get("lanes"), label="review.focused_tests.lanes")
    if set(lanes) != {"tf4576", "tf5141"}:
        _fail("focused test review exact lanes are incomplete or ambiguous")
    for name in sorted(lanes):
        lane = _mapping(lanes[name], label=f"focused lane {name}")
        if set(lane) != {"command_count", "status", "test_count"} or lane != {
            "command_count": 10,
            "status": "passed",
            "test_count": 117,
        }:
            _fail(f"focused lane {name} is not passed")

    ledger = _validate_status_section(review, "command_ledger")
    expected_ledger_fields = {
        "log_sha256",
        "observed_command_count",
        "observed_environment_record_count",
        "ordering_checks",
        "record_sha256",
        "required_command_count",
        "required_commands",
        "required_environment_records",
        "reviewed_command_count",
        "reviewed_environment_record_count",
        "run_marker_sha256",
        "semantic_logs",
        "status",
        "tested_commit",
    }
    if set(ledger) != expected_ledger_fields:
        _fail("Stage E command ledger fields are incomplete or ambiguous")
    if ledger.get("tested_commit") != tested:
        _fail("Stage E command ledger tested commit differs")
    for field in (
        "observed_command_count",
        "required_command_count",
        "reviewed_command_count",
    ):
        if ledger.get(field) != 53:
            _fail(f"Stage E command ledger {field} is not 53")
    for field in (
        "observed_environment_record_count",
        "reviewed_environment_record_count",
    ):
        if ledger.get(field) != 2:
            _fail(f"Stage E command ledger {field} is not 2")
    if _list(ledger.get("required_commands"), label="required commands") != sorted(
        REQUIRED_COMMAND_NAMES
    ):
        _fail("review command matrix differs from the fixed 53-command matrix")
    if _list(
        ledger.get("required_environment_records"), label="environment records"
    ) != sorted(REQUIRED_ENVIRONMENT_NAMES):
        _fail("review environment matrix differs from the fixed Stage E matrix")
    record_hashes = _mapping(ledger.get("record_sha256"), label="record hashes")
    if set(record_hashes) != set(REQUIRED_COMMAND_NAMES) | set(REQUIRED_ENVIRONMENT_NAMES):
        _fail("review record hash set differs from the fixed Stage E records")
    log_hashes = _mapping(ledger.get("log_sha256"), label="log hashes")
    if set(log_hashes) != set(REQUIRED_COMMAND_NAMES):
        _fail("review log hash set differs from the fixed Stage E logs")
    for label, value in (*record_hashes.items(), *log_hashes.items()):
        _digest(value, label=f"ledger hash {label}")
    _digest(ledger.get("run_marker_sha256"), label="run marker SHA-256")
    ordering = _list(ledger.get("ordering_checks"), label="ordering checks")
    if (
        not ordering
        or any(not isinstance(value, str) or not value for value in ordering)
        or len(ordering) != len(set(ordering))
        or "p0-4-psi8-focused-review-before-psi16" not in ordering
    ):
        _fail("Stage E ordering checks do not prove the Psi=8 review boundary")
    _mapping(ledger.get("semantic_logs"), label="semantic logs")

    aggregate = _mapping(review.get("aggregate"), label="review.aggregate")
    if set(aggregate) != {
        "artifact_count",
        "artifact_hashes",
        "raw_event_counts",
        "review_material_sha256",
    }:
        _fail("review aggregate fields are incomplete or ambiguous")
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
            "review artifact inventory differs from fixed Stage E allowlist: "
            f"missing={missing}, extra={extra}"
        )
    if aggregate.get("artifact_count") != len(hashes):
        _fail("review artifact_count differs from its hash inventory")
    for name, digest in record_hashes.items():
        if digest != hashes[f"runner.record.{name}"]:
            _fail(f"review record hash is not bound to the artifact inventory: {name}")
    for name, digest in log_hashes.items():
        if digest != hashes[f"runner.log.{name}"]:
            _fail(f"review log hash is not bound to the artifact inventory: {name}")
    if ledger["run_marker_sha256"] != hashes["runner.run_marker"]:
        _fail("review run-marker hash is not bound to the artifact inventory")
    raw_counts = _mapping(aggregate.get("raw_event_counts"), label="raw event counts")
    expected_counts = {
        "p0_3_stdout_step_events": 65,
        "p0_4_jsonl_events": 114,
        "total": 179,
    }
    if raw_counts != expected_counts:
        _fail("review raw event counts differ from the fixed Stage E totals")
    material = {
        "artifact_hashes": dict(sorted(hashes.items())),
        "historical_evidence_immutability": dict(historical),
        "implementation_baseline": IMPLEMENTATION_BASE_COMMIT,
        "raw_event_counts": expected_counts,
        "tested_commit": tested,
    }
    if _digest(
        aggregate.get("review_material_sha256"),
        label="review material SHA-256",
    ) != sha256_bytes(_review_canonical_bytes(material)):
        _fail("review material SHA-256 does not match the complete Stage E aggregate")
    return tested, hashes


def _summary_documents(
    review: Mapping[str, Any], reviewers: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], bytes]:
    aggregate = _mapping(review["aggregate"], label="review aggregate")
    ledger = _mapping(review["command_ledger"], label="review command ledger")
    historical = _mapping(
        review["historical_evidence_immutability"], label="historical evidence"
    )
    summary = {
        "acceptance_review": {
            "reviewers": [dict(item) for item in reviewers],
            "status": "recorded",
        },
        "implementation_baseline": IMPLEMENTATION_BASE_COMMIT,
        "limitations": [
            "This is an unofficial correctness-first Hugging Face implementation.",
            "The dense quadratic screening path is not evidence of paper efficiency.",
            "No paper-scale training, retrieval benchmark, distributed training, broad generation compatibility, or P1 ecosystem capability is validated.",
        ],
        "review": {
            "artifact_count": aggregate["artifact_count"],
            "command_count": ledger["reviewed_command_count"],
            "environment_record_count": ledger["reviewed_environment_record_count"],
            "focused_tests_per_exact_lane": 117,
            "historical_evidence": {
                "artifact_count": historical["artifact_count"],
                "status": historical["status"],
                "tree_material_sha256": historical["tree_material_sha256"],
            },
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
            "generation_input_contract": "passed",
            "gradient_checkpointing_api_matrix": "passed",
            "hf_output_head_and_deepcopy_lifecycle": "passed",
            "historical_evidence_immutability": "passed",
            "p0_1_formula_oracle": "passed",
            "p0_2_three_way": "passed",
            "p0_3_checkpointed_smoke": "passed",
            "p0_4_hardened_psi8_psi16_qualification": "passed",
            "packed_text_eos_contract": "passed",
            "paper_training_contract": "passed",
            "training_edge_contracts": "passed",
        },
    }
    lines = [
        "# Hugging Face Contract Hardening Requalification Summary",
        "",
        "Status: passed",
        f"Tested commit: {review['tested_commit']}",
        f"Implementation baseline: {IMPLEMENTATION_BASE_COMMIT}",
        f"Reviewed artifacts: {aggregate['artifact_count']}",
        f"Reviewed raw events: {aggregate['raw_event_counts']['total']}",
        f"Reviewed commands: {ledger['reviewed_command_count']}",
        "Focused tests per exact Transformers lane: 117",
        f"Acceptance reviewers: {', '.join(item['identifier'] for item in reviewers)}",
        "",
        "The reviewed Stage E matrix passed the seven post-Level-1 hardening",
        "resolutions, the hardened P0-4 predicate, both exact Transformers lanes,",
        "P0-1/P0-2, fresh checkpointed P0-3, and fresh strict Psi=8/Psi=16 P0-4.",
        "Accepted Level 1, P0-4, and P0.5-C3 record blobs remained unchanged.",
        "",
        "This is an unofficial correctness-first implementation. The dense",
        "quadratic path is not efficiency evidence. This result does not validate",
        "paper-scale training, retrieval, distributed training, broad generation",
        "compatibility, or any P1 model/ecosystem capability.",
        "",
        "Archive retention and descriptor closure are recorded separately in the",
        "HF contract hardening evidence archive descriptor.",
        "",
    ]
    return summary, "\n".join(lines).encode("utf-8")


def _package_extras(
    *, review_raw: bytes, provenance_raw: bytes, summary_raw: bytes, markdown_raw: bytes
) -> tuple[tuple[str, str, str, str, bytes, str], ...]:
    root = "artifacts/hf-contract-hardening"
    return (
        (
            "hardening.full_review",
            "run",
            FULL_REVIEW_RELATIVE,
            f"{root}/review/hf-contract-hardening-raw-review.json",
            review_raw,
            "validation_summary",
        ),
        (
            "hardening.acceptance_provenance",
            "run",
            ACCEPTANCE_PROVENANCE_RELATIVE,
            f"{root}/review/acceptance-provenance.json",
            provenance_raw,
            "provenance",
        ),
        (
            "hardening.machine_summary",
            "results",
            SUMMARY_JSON_NAME,
            f"{root}/summary/{SUMMARY_JSON_NAME}",
            summary_raw,
            "validation_summary",
        ),
        (
            "hardening.human_summary",
            "results",
            SUMMARY_MARKDOWN_NAME,
            f"{root}/summary/{SUMMARY_MARKDOWN_NAME}",
            markdown_raw,
            "validation_summary",
        ),
    )


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
    empty = b""
    for logical, root_name, source_path, archive_path, _raw, classification in _package_extras(
        review_raw=empty,
        provenance_raw=empty,
        summary_raw=empty,
        markdown_raw=empty,
    ):
        expected[logical] = (root_name, source_path, archive_path, classification)
    return expected


def _sync_profile() -> None:
    for name in (
        "TOOL_VERSION",
        "SUMMARY_VERSION",
        "REVIEW_VERSION",
        "PROVENANCE_VERSION",
        "PACKAGE_REPORT_VERSION",
        "GATE",
        "EVIDENCE_GATE",
        "REPOSITORY_NAME",
        "IMPLEMENTATION_BASE_COMMIT",
        "SUMMARY_JSON_NAME",
        "SUMMARY_MARKDOWN_NAME",
        "DESCRIPTOR_NAME",
        "EXACT_VERIFICATION_NAME",
        "SANITIZED_VERIFICATION_NAME",
        "PACKAGE_INPUT_RELATIVE",
        "FULL_REVIEW_RELATIVE",
        "ACCEPTANCE_PROVENANCE_RELATIVE",
        "CANONICAL_REPOSITORY_ROOT",
        "CANONICAL_RESULTS_ROOT",
        "CANONICAL_SCHEMA_PATH",
        "CANONICAL_SCHEMA_SHA256",
        "REQUIRED_COMMAND_NAMES",
        "REQUIRED_ENVIRONMENT_NAMES",
    ):
        setattr(_CORE, name, globals()[name])
    _CORE.ArtifactSpec = ArtifactSpec
    _CORE._fixed_specs = _fixed_specs
    _CORE._validate_review = _validate_review
    _CORE._validate_acceptance = _validate_acceptance
    _CORE._summary_documents = _summary_documents
    _CORE._expected_package_layout = _expected_package_layout
    _CORE._validate_implementation_base_commit = _validate_implementation_base_commit
    _CORE._validate_live_repository_state = _validate_live_repository_state
    _CORE._validate_commit_evidence_blobs = _validate_commit_evidence_blobs
    _CORE._registered_worktrees = _registered_worktrees
    _CORE._git_stdout = _git_stdout


def prepare_evidence(
    *,
    run_root_value: str | os.PathLike[str],
    results_root_value: str | os.PathLike[str],
) -> dict[str, Any]:
    _sync_profile()
    run_root, results_root = _validated_roots(run_root_value, results_root_value)
    review_path = _fixed_child(run_root, FULL_REVIEW_RELATIVE, label="full review")
    provenance_path = _fixed_child(
        run_root, ACCEPTANCE_PROVENANCE_RELATIVE, label="acceptance provenance"
    )
    review, review_raw = _load_json(review_path, label="full review")
    provenance, provenance_raw = _load_json(provenance_path, label="acceptance provenance")
    tested_commit, expected_hashes = _validate_review(review)
    _validate_implementation_base_commit(
        IMPLEMENTATION_BASE_COMMIT, tested_commit=tested_commit
    )
    live_historical = _historical_evidence_projection(tested_commit)
    if review["historical_evidence_immutability"] != live_historical:
        _fail("reviewed historical evidence projection differs from live Git")
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
        path = _fixed_child(run_root, source_path, label=f"reviewed artifact {source_path}")
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
    for logical, root_name, source_path, archive_path, raw, classification in _package_extras(
        review_raw=review_raw,
        provenance_raw=provenance_raw,
        summary_raw=summary_raw,
        markdown_raw=markdown_raw,
    ):
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
        "historical_evidence_tree_material_sha256": live_historical[
            "tree_material_sha256"
        ],
        "package_input_sha256": sha256_bytes(package_raw),
        "review_material_sha256": review["aggregate"]["review_material_sha256"],
        "status": "prepared",
        "summary_json_sha256": sha256_bytes(summary_raw),
        "summary_markdown_sha256": sha256_bytes(markdown_raw),
        "tested_commit": tested_commit,
    }


def seal_evidence(**kwargs: Any) -> dict[str, Any]:
    _sync_profile()
    return _CORE.seal_evidence(**kwargs)


def close_evidence(**kwargs: Any) -> dict[str, Any]:
    _sync_profile()
    return _CORE.close_evidence(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--results-root", required=True)
    seal = subparsers.add_parser("seal")
    close = subparsers.add_parser("close")
    for target in (seal, close):
        target.add_argument("--run-root", required=True)
        target.add_argument("--results-root", required=True)
        target.add_argument("--schema", default=os.fspath(CANONICAL_SCHEMA_PATH))
        target.add_argument("--package-report", required=True)
        target.add_argument("--implementation-base-commit", required=True)
        target.add_argument("--exact-storage-locator", required=True)
        target.add_argument("--sanitized-storage-locator", required=True)
        target.add_argument("--exact-archive", required=True)
        target.add_argument("--sanitized-archive", required=True)
        target.add_argument("--sanitized-staging-dir", required=True)
        target.add_argument("--verification-timestamp-utc", required=True)
    seal.add_argument("--exact-primary-report", required=True)
    seal.add_argument("--sanitized-primary-report", required=True)
    close.add_argument("--commit-provenance", required=True)
    close.add_argument("--commit-a", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_evidence(
                run_root_value=args.run_root, results_root_value=args.results_root
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
