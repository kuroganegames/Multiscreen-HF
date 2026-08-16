from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_hf_contract_hardening_evidence as BUILDER
from scripts import build_level1_evidence as LEGACY_BUILDER
from scripts.package_validation_evidence import package_evidence
from scripts import review_hf_contract_hardening as REVIEWER
from scripts.validation_evidence_common import canonical_json_bytes
from scripts.verify_validation_evidence import verify_archive


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "validation_evidence_v1.schema.json"
TESTED_COMMIT = "d" * 40
COMMIT_A = "b" * 40
TIMESTAMP = "2026-08-16T10:00:00Z"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)


def write_json(path: Path, value: object) -> None:
    write_bytes(path, canonical_json_bytes(value))


def clean_worktree(timestamp: str = TIMESTAMP) -> dict[str, object]:
    return {
        "clean": True,
        "collected_at_utc": timestamp,
        "conflicted_changes_present": False,
        "porcelain": {
            "byte_count": 0,
            "command": [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            "sha256": EMPTY_SHA256,
        },
        "staged_change_count": 0,
        "staged_changes_present": False,
        "status": "recorded",
        "submodules": {
            "byte_count": 0,
            "collected_at_utc": timestamp,
            "command": ["git", "submodule", "status", "--recursive"],
            "count": 0,
            "sha256": EMPTY_SHA256,
            "state": "none",
            "state_counts": {
                "at_recorded_commit": 0,
                "commit_mismatch": 0,
                "conflicted": 0,
                "uninitialized": 0,
            },
            "status": "recorded",
        },
        "unstaged_change_count": 0,
        "unstaged_changes_present": False,
        "untracked_path_count": 0,
    }


def historical_entries(*, changed: str | None = None, extra: bool = False):
    entries = []
    for path in BUILDER.HISTORICAL_PATHS:
        object_id = hashlib.sha1(path.encode("utf-8"), usedforsecurity=False).hexdigest()
        if path == changed:
            object_id = "e" * 40
        entries.append(
            {"mode": "100644", "object_id": object_id, "path": path, "type": "blob"}
        )
    if extra:
        entries.append(
            {
                "mode": "100644",
                "object_id": "f" * 40,
                "path": "docs/validation_results/LEVEL1_CORE_UNEXPECTED.json",
                "type": "blob",
            }
        )
    return sorted(entries, key=lambda item: item["path"].encode("utf-8"))


def historical_listing(entries: list[dict[str, str]]) -> bytes:
    chunks = []
    for entry in entries:
        chunks.append(
            (
                f"{entry['mode']} {entry['type']} {entry['object_id']}\t{entry['path']}"
            ).encode("utf-8")
            + b"\0"
        )
    return b"".join(chunks)


def historical_projection() -> dict[str, object]:
    material = historical_entries()
    return {
        "artifact_count": len(material),
        "implementation_baseline": BUILDER.IMPLEMENTATION_BASE_COMMIT,
        "status": "passed",
        "tested_commit": TESTED_COMMIT,
        "tree_material_sha256": hashlib.sha256(
            BUILDER._review_canonical_bytes(material)
        ).hexdigest(),
    }


def acceptance_provenance(*, head: str, timestamp: str = TIMESTAMP) -> dict[str, object]:
    return {
        "acceptance_review": {
            "reviewers": [
                {
                    "identifier": "explicit-reviewer",
                    "raw_events_reviewed": True,
                    "review_commit": TESTED_COMMIT,
                    "review_method": "manual review of 179 raw events and 53 logs",
                    "reviewed_at_utc": TIMESTAMP,
                    "role": "evidence_reviewer",
                }
            ],
            "status": "recorded",
        },
        "collected_at_utc": timestamp,
        "context": "evidence_handoff",
        "format_version": "validation-provenance-v1",
        "repository": {
            "branch": {"status": "recorded", "value": BUILDER.WORKING_BRANCH},
            "detached_head": False,
            "head_commit": head,
            "name": "Multiscreen-HF",
            "remotes": [],
            "root_kind": "git_worktree",
            "worktree": clean_worktree(timestamp),
        },
    }


class BuilderCase:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run = root / "run"
        self.results = root / "results"
        self.private = root / "private"
        self.sanitized_staging = root / "sanitized-staging"
        for directory in (self.run, self.results, self.private, self.sanitized_staging):
            directory.mkdir(parents=True, mode=0o700)
        self._write_fixed_artifacts()
        self.review = self._review()
        self.review_path = self.run / BUILDER.FULL_REVIEW_RELATIVE
        write_json(self.review_path, self.review)
        self.provenance_path = self.run / BUILDER.ACCEPTANCE_PROVENANCE_RELATIVE
        write_json(self.provenance_path, acceptance_provenance(head=TESTED_COMMIT))
        self.exact_archive = self.private / "hf-contract-hardening-exact.tar.gz"
        self.sanitized_archive = (
            self.sanitized_staging / "hf-contract-hardening-sanitized.tar.gz"
        )
        self.package_report_path = root / "package-report.json"
        self.exact_primary_path = root / "exact-primary.json"
        self.sanitized_primary_path = root / "sanitized-primary.json"

    def _write_fixed_artifacts(self) -> None:
        for source_path in sorted({spec.source_path for spec in BUILDER._fixed_specs().values()}):
            path = self.run / source_path
            if source_path in {
                "records/repository-hygiene.json",
                "records/repository-hygiene-final.json",
            }:
                name = Path(source_path).stem
                write_json(
                    path,
                    {
                        "ended_at_utc": TIMESTAMP,
                        "exit_code": 0,
                        "name": name,
                        "record_type": "command",
                        "returncode": 0,
                    },
                )
            else:
                write_bytes(path, f"fixture:{source_path}\n".encode("utf-8"))

    def _review(self) -> dict[str, object]:
        specs = BUILDER._fixed_specs()
        hashes = {
            label: hashlib.sha256((self.run / spec.source_path).read_bytes()).hexdigest()
            for label, spec in specs.items()
        }
        record_hashes = {
            name: hashes[f"runner.record.{name}"]
            for name in (*BUILDER.REQUIRED_COMMAND_NAMES, *BUILDER.REQUIRED_ENVIRONMENT_NAMES)
        }
        log_hashes = {
            name: hashes[f"runner.log.{name}"] for name in BUILDER.REQUIRED_COMMAND_NAMES
        }
        historical = historical_projection()
        raw_counts = {
            "p0_3_stdout_step_events": 65,
            "p0_4_jsonl_events": 114,
            "total": 179,
        }
        material = {
            "artifact_hashes": dict(sorted(hashes.items())),
            "historical_evidence_immutability": historical,
            "implementation_baseline": BUILDER.IMPLEMENTATION_BASE_COMMIT,
            "raw_event_counts": raw_counts,
            "tested_commit": TESTED_COMMIT,
        }
        return {
            "aggregate": {
                "artifact_count": len(hashes),
                "artifact_hashes": hashes,
                "raw_event_counts": raw_counts,
                "review_material_sha256": hashlib.sha256(
                    BUILDER._review_canonical_bytes(material)
                ).hexdigest(),
            },
            "command_ledger": {
                "log_sha256": log_hashes,
                "observed_command_count": 53,
                "observed_environment_record_count": 2,
                "ordering_checks": [
                    "runtime-environment-records-before-environment-commands",
                    "all-matrix-commands-inside-hygiene-bracket",
                    "p0-3-run-and-tokenizer-reloads-ordered",
                    "p0-4-psi8-focused-review-before-psi16",
                ],
                "record_sha256": record_hashes,
                "required_command_count": 53,
                "required_commands": sorted(BUILDER.REQUIRED_COMMAND_NAMES),
                "required_environment_records": sorted(
                    BUILDER.REQUIRED_ENVIRONMENT_NAMES
                ),
                "reviewed_command_count": 53,
                "reviewed_environment_record_count": 2,
                "run_marker_sha256": hashes["runner.run_marker"],
                "semantic_logs": {"status": "passed"},
                "status": "passed",
                "tested_commit": TESTED_COMMIT,
            },
            "focused_tests": {
                "expected_per_lane": 117,
                "lanes": {
                    "tf4576": {
                        "command_count": 10,
                        "status": "passed",
                        "test_count": 117,
                    },
                    "tf5141": {
                        "command_count": 10,
                        "status": "passed",
                        "test_count": 117,
                    },
                },
                "status": "passed",
            },
            "historical_evidence_immutability": historical,
            "implementation_baseline": BUILDER.IMPLEMENTATION_BASE_COMMIT,
            "p0_3": {"status": "passed"},
            "p0_4": {"status": "passed"},
            "schema_version": BUILDER.REVIEW_VERSION,
            "status": "passed",
            "tested_commit": TESTED_COMMIT,
            "tokenizer_reload": {"status": "passed"},
        }

    def patches(self):
        return (
            mock.patch.object(BUILDER, "CANONICAL_RESULTS_ROOT", self.results),
            mock.patch.object(
                BUILDER, "_registered_worktrees", return_value=(REPOSITORY_ROOT,)
            ),
            mock.patch.object(BUILDER, "_validate_live_repository_state"),
            mock.patch.object(
                BUILDER,
                "_validate_implementation_base_commit",
                return_value=BUILDER.IMPLEMENTATION_BASE_COMMIT,
            ),
            mock.patch.object(
                BUILDER,
                "_historical_evidence_projection",
                return_value=historical_projection(),
            ),
        )

    def prepare(self) -> dict[str, object]:
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return BUILDER.prepare_evidence(
                run_root_value=self.run, results_root_value=self.results
            )

    def package(self) -> dict[str, object]:
        package_input = json.loads(
            (self.run / BUILDER.PACKAGE_INPUT_RELATIVE).read_text(encoding="utf-8")
        )
        report = package_evidence(
            package_input,
            roots={"results": self.results, "run": self.run},
            mode="both",
            exact_output=self.exact_archive,
            sanitized_output=self.sanitized_archive,
            repository_root=REPOSITORY_ROOT,
            sensitive_values=("synthetic-user",),
            created_at_utc=TIMESTAMP,
        )
        write_json(self.package_report_path, report)
        by_kind = {item["archive_kind"]: item for item in report["archives"]}
        exact = verify_archive(
            self.exact_archive,
            expected_sha256=by_kind["exact_private"]["sha256"],
            verification_timestamp_utc=TIMESTAMP,
        )
        sanitized = verify_archive(
            self.sanitized_archive,
            expected_sha256=by_kind["sanitized_shareable"]["sha256"],
            verification_timestamp_utc=TIMESTAMP,
        )
        write_json(self.exact_primary_path, exact)
        write_json(self.sanitized_primary_path, sanitized)
        return report

    def seal(self) -> dict[str, object]:
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return BUILDER.seal_evidence(
                run_root_value=self.run,
                results_root_value=self.results,
                schema_value=SCHEMA_PATH,
                package_report_value=self.package_report_path,
                exact_archive_value=self.exact_archive,
                sanitized_archive_value=self.sanitized_archive,
                sanitized_staging_dir_value=self.sanitized_staging,
                exact_primary_report_value=self.exact_primary_path,
                sanitized_primary_report_value=self.sanitized_primary_path,
                implementation_base_commit=BUILDER.IMPLEMENTATION_BASE_COMMIT,
                exact_storage_locator="private-external:hf-hardening/test",
                sanitized_storage_locator="sanitized-staging:hf-hardening/test",
                verification_timestamp_utc=TIMESTAMP,
            )

    def commit_provenance(self) -> Path:
        value = acceptance_provenance(head=COMMIT_A, timestamp="2026-08-16T11:00:00Z")
        value["acceptance_review"]["reviewers"][0]["review_commit"] = TESTED_COMMIT
        path = self.root / "commit-a-provenance.json"
        write_json(path, value)
        return path

    def close(self) -> dict[str, object]:
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], mock.patch.object(
            BUILDER, "_validate_commit_evidence_blobs"
        ):
            return BUILDER.close_evidence(
                run_root_value=self.run,
                results_root_value=self.results,
                schema_value=SCHEMA_PATH,
                commit_provenance_value=self.commit_provenance(),
                package_report_value=self.package_report_path,
                implementation_base_commit=BUILDER.IMPLEMENTATION_BASE_COMMIT,
                exact_storage_locator="private-external:hf-hardening/test",
                sanitized_storage_locator="sanitized-staging:hf-hardening/test",
                commit_a=COMMIT_A,
                exact_archive_value=self.exact_archive,
                sanitized_archive_value=self.sanitized_archive,
                sanitized_staging_dir_value=self.sanitized_staging,
                verification_timestamp_utc=TIMESTAMP,
            )


class HFContractHardeningEvidenceBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = 0

    def new_case(self) -> BuilderCase:
        self.index += 1
        return BuilderCase(self.root / f"case-{self.index}")

    def test_profile_is_fixed_separate_and_has_no_fresh_c3(self) -> None:
        self.assertEqual(len(BUILDER.REQUIRED_COMMAND_NAMES), 53)
        self.assertEqual(len(set(BUILDER.REQUIRED_COMMAND_NAMES)), 53)
        self.assertEqual(len(BUILDER.REQUIRED_ENVIRONMENT_NAMES), 2)
        self.assertFalse(any(name.startswith("c3-") for name in BUILDER.REQUIRED_COMMAND_NAMES))
        specs = BUILDER._fixed_specs()
        self.assertEqual(len(specs), 130)
        self.assertEqual(len({item.source_path for item in specs.values()}), 129)
        self.assertFalse(any("artifacts/c3/" in item.source_path for item in specs.values()))
        self.assertFalse(
            any(
                item.source_path.startswith("docs/validation_results/")
                for item in specs.values()
            )
        )
        self.assertEqual(len(BUILDER._expected_package_layout()), 133)
        serialized = json.dumps(BUILDER._expected_package_layout(), sort_keys=True)
        self.assertNotIn("artifacts/level1-core/", serialized)
        for name in (
            BUILDER.SUMMARY_JSON_NAME,
            BUILDER.SUMMARY_MARKDOWN_NAME,
            BUILDER.DESCRIPTOR_NAME,
            BUILDER.EXACT_VERIFICATION_NAME,
            BUILDER.SANITIZED_VERIFICATION_NAME,
        ):
            self.assertTrue(name.startswith("HF_CONTRACT_HARDENING_"))
        self.assertEqual(LEGACY_BUILDER.GATE, "Level 1 Core")
        self.assertEqual(LEGACY_BUILDER.SUMMARY_JSON_NAME, "LEVEL1_CORE_SUMMARY.json")
        self.assertEqual(len(LEGACY_BUILDER.REQUIRED_COMMAND_NAMES), 46)

    def test_prepare_seal_close_cycle_is_schema_valid_and_canonical(self) -> None:
        case = self.new_case()
        prepared = case.prepare()
        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(prepared["artifact_count"], 133)
        package = json.loads(
            (case.run / BUILDER.PACKAGE_INPUT_RELATIVE).read_text(encoding="utf-8")
        )
        self.assertEqual(package["gate"], BUILDER.GATE)
        self.assertEqual(len(package["artifacts"]), 133)
        self.assertEqual(
            len({item["archive_path"] for item in package["artifacts"]}), 133
        )
        serialized = canonical_json_bytes(package).decode("utf-8")
        for forbidden in (
            "artifacts/c3/",
            "artifacts/level1-core/",
            BUILDER.DESCRIPTOR_NAME,
            BUILDER.EXACT_VERIFICATION_NAME,
            BUILDER.SANITIZED_VERIFICATION_NAME,
            ".tar.gz",
            ".safetensors",
            ".bin",
        ):
            self.assertNotIn(forbidden, serialized)
        for path in (
            case.results / BUILDER.SUMMARY_JSON_NAME,
            case.results / BUILDER.SUMMARY_MARKDOWN_NAME,
            case.run / BUILDER.PACKAGE_INPUT_RELATIVE,
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        case.package()
        sealed = case.seal()
        self.assertEqual(sealed["status"], "sealed_partial")
        descriptor_path = case.results / BUILDER.DESCRIPTOR_NAME
        partial = json.loads(descriptor_path.read_text(encoding="utf-8"))
        self.assertEqual(partial["evidence_status"], "partial")
        self.assertEqual(partial["validation_gate"], BUILDER.GATE)
        self.assertEqual(partial["evidence_gate"], BUILDER.EVIDENCE_GATE)
        self.assertEqual(
            partial["evidence_handoff_provenance"]["implementation_base_commit"],
            BUILDER.IMPLEMENTATION_BASE_COMMIT,
        )
        self.assertEqual(
            BUILDER.validate_evidence_document(
                partial, json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            ),
            [],
        )
        closed = case.close()
        self.assertEqual(closed["status"], "closed_complete")
        complete_raw = descriptor_path.read_bytes()
        complete = json.loads(complete_raw)
        self.assertEqual(complete_raw, canonical_json_bytes(complete))
        self.assertEqual(complete["evidence_status"], "complete")
        self.assertEqual(
            complete["evidence_handoff_provenance"]["final_commit"],
            {"status": "recorded", "value": COMMIT_A},
        )

    def test_prepare_rejects_review_identity_inventory_and_event_tampering(self) -> None:
        mutations = {
            "legacy schema": lambda value: value.update(
                {"schema_version": "multiscreen-level1-raw-evidence-review-v1"}
            ),
            "wrong baseline": lambda value: value.update(
                {"implementation_baseline": "a" * 40}
            ),
            "missing command": lambda value: value["command_ledger"][
                "required_commands"
            ].pop(),
            "extra environment": lambda value: value["command_ledger"][
                "required_environment_records"
            ].append("runtime-extra"),
            "C3 raw events": lambda value: value["aggregate"]["raw_event_counts"].update(
                {"c3_jsonl_events": 8}
            ),
            "wrong total": lambda value: value["aggregate"]["raw_event_counts"].update(
                {"total": 178}
            ),
            "missing artifact": lambda value: value["aggregate"][
                "artifact_hashes"
            ].pop("p0_3.data_contract"),
            "failed focused lane": lambda value: value["focused_tests"]["lanes"][
                "tf5141"
            ].update({"status": "failed"}),
            "historical drift": lambda value: value[
                "historical_evidence_immutability"
            ].update({"tree_material_sha256": "e" * 64}),
            "review material drift": lambda value: value["aggregate"].update(
                {"review_material_sha256": "e" * 64}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                case = self.new_case()
                review = copy.deepcopy(case.review)
                mutate(review)
                write_json(case.review_path, review)
                with self.assertRaises(BUILDER.EvidenceBuildError):
                    case.prepare()

    def test_prepare_rejects_live_historical_projection_mismatch(self) -> None:
        case = self.new_case()
        patches = case.patches()
        changed = historical_projection()
        changed["tree_material_sha256"] = "e" * 64
        with patches[0], patches[1], patches[2], patches[3], mock.patch.object(
            BUILDER, "_historical_evidence_projection", return_value=changed
        ), self.assertRaises(BUILDER.EvidenceBuildError):
            BUILDER.prepare_evidence(
                run_root_value=case.run, results_root_value=case.results
            )

    def test_actual_full_reviewer_report_is_accepted_by_builder_contract(self) -> None:
        case = self.new_case()

        def review_ledger(_value, *, hashes, **_kwargs):
            hashes.update(case.review["aggregate"]["artifact_hashes"])
            return (
                copy.deepcopy(case.review["command_ledger"]),
                case.root / "offline-cache",
                copy.deepcopy(case.review["focused_tests"]),
            )

        def review_p04(_root, *, psi, **_kwargs):
            return {
                "event_count": 57,
                "psi": psi,
                "status": "passed",
                "timestamp_utc": (
                    "2026-08-16T10:10:00Z"
                    if psi == 8
                    else "2026-08-16T10:20:00Z"
                ),
            }

        tokenizer = {
            "report_count": 4,
            "reports": [
                {"logical_name": name, "status": "passed"}
                for name in (
                    "p0_3_psi8",
                    "p0_3_psi16",
                    "p0_4_psi8",
                    "p0_4_psi16",
                )
            ],
            "status": "passed",
        }
        tokenizer_paths = {
            "p0_3_psi8": case.run / "artifacts/p0-3/tokenizer-reload-psi8.json",
            "p0_3_psi16": case.run / "artifacts/p0-3/tokenizer-reload-psi16.json",
            "p0_4_psi8": case.run / "artifacts/p0-4/psi8/tokenizer-reload.json",
            "p0_4_psi16": case.run / "artifacts/p0-4/psi16/tokenizer-reload.json",
        }
        with mock.patch.object(
            REVIEWER, "_review_command_ledger", side_effect=review_ledger
        ), mock.patch.object(
            REVIEWER._legacy,
            "_review_p0_3",
            return_value={"status": "passed", "stdout_step_event_count": 65},
        ), mock.patch.object(
            REVIEWER, "_review_p0_4_lane_v2", side_effect=review_p04
        ), mock.patch.object(
            REVIEWER._legacy,
            "_review_tokenizer_reports",
            return_value=tokenizer,
        ), mock.patch.object(
            REVIEWER._legacy,
            "_review_p0_4_cross_bindings",
            return_value={"status": "passed"},
        ), mock.patch.object(
            REVIEWER._legacy,
            "_review_p0_3_cross_bindings",
            return_value={"status": "passed"},
        ), mock.patch.object(
            REVIEWER,
            "_review_focused_report",
            return_value={"artifact_count": 1, "psi": 8, "status": "passed"},
        ), mock.patch.object(
            REVIEWER,
            "_review_historical_evidence_immutability",
            return_value=historical_projection(),
        ):
            report = REVIEWER.review_inputs(
                p0_3_root=case.run / "artifacts/p0-3",
                p0_3_stdout=case.run / "logs/p0-3-checkpointed.log",
                p0_4_psi8_root=case.run / "artifacts/p0-4/psi8",
                p0_4_psi16_root=case.run / "artifacts/p0-4/psi16",
                p0_4_psi8_review=case.run / "artifacts/p0-4/psi8/raw-review.json",
                tokenizer_reports=tokenizer_paths,
                command_ledger=case.run / "commands.jsonl",
                tested_commit=TESTED_COMMIT,
                repository=REPOSITORY_ROOT,
            )
        tested, hashes = BUILDER._validate_review(report)
        self.assertEqual(tested, TESTED_COMMIT)
        self.assertEqual(set(hashes), set(BUILDER._fixed_specs()))
        self.assertEqual(REVIEWER.REQUIRED_COMMAND_NAMES, BUILDER.REQUIRED_COMMAND_NAMES)
        self.assertEqual(
            REVIEWER.REQUIRED_ENVIRONMENT_NAMES,
            BUILDER.REQUIRED_ENVIRONMENT_NAMES,
        )

    def test_historical_path_mode_and_blob_set_is_compared_at_live_git(self) -> None:
        baseline = historical_listing(historical_entries())

        def reader(arguments, *, label):
            del label
            command = tuple(arguments)
            if command[:2] == ("cat-file", "-t"):
                return b"commit\n"
            if command[:2] == ("merge-base", "--is-ancestor"):
                return b""
            if command[0] == "ls-tree":
                return baseline
            raise AssertionError(command)

        with mock.patch.object(BUILDER, "_git_stdout", side_effect=reader):
            result = BUILDER._historical_evidence_projection(TESTED_COMMIT)
            self.assertEqual(result, historical_projection())
            self.assertEqual(
                BUILDER._validate_implementation_base_commit(
                    BUILDER.IMPLEMENTATION_BASE_COMMIT,
                    tested_commit=TESTED_COMMIT,
                ),
                BUILDER.IMPLEMENTATION_BASE_COMMIT,
            )

        for label, tested in (
            (
                "blob drift",
                historical_listing(
                    historical_entries(changed=BUILDER.HISTORICAL_PATHS[-1])
                ),
            ),
            ("extra prefixed path", historical_listing(historical_entries(extra=True))),
            ("malformed listing", baseline[:-1]),
        ):
            calls = 0

            def drift_reader(arguments, *, label):
                nonlocal calls
                del label
                if arguments[0] != "ls-tree":
                    raise AssertionError(arguments)
                calls += 1
                return baseline if calls == 1 else tested

            with self.subTest(label=label), mock.patch.object(
                BUILDER, "_git_stdout", side_effect=drift_reader
            ), self.assertRaises(BUILDER.EvidenceBuildError):
                BUILDER._historical_evidence_projection(TESTED_COMMIT)

    def test_baseline_identity_and_post_baseline_source_are_fail_closed(self) -> None:
        with self.assertRaises(BUILDER.EvidenceBuildError):
            BUILDER._validate_implementation_base_commit(
                "a" * 40, tested_commit=TESTED_COMMIT
            )
        with self.assertRaises(BUILDER.EvidenceBuildError):
            BUILDER._validate_implementation_base_commit(
                BUILDER.IMPLEMENTATION_BASE_COMMIT,
                tested_commit=BUILDER.IMPLEMENTATION_BASE_COMMIT,
            )

    def test_safe_path_standard_library_cli_starts_without_pythonpath(self) -> None:
        environment = {
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        completed = subprocess.run(
            [
                os.fspath(Path(os.sys.executable).resolve()),
                "-P",
                "-S",
                "-B",
                os.fspath(
                    REPOSITORY_ROOT
                    / "scripts"
                    / "build_hf_contract_hardening_evidence.py"
                ),
                "--version",
            ],
            cwd=self.root,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), BUILDER.TOOL_VERSION)


if __name__ == "__main__":
    unittest.main()
