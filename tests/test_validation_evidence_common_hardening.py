"""Adversarial offline tests for shared validation-evidence invariants."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import package_validation_evidence as packager
from scripts.validation_evidence_common import (
    MAX_ARCHIVE_CONTROL_MEMBER_COUNT,
    MAX_ARCHIVE_MEMBER_COUNT,
    MAX_ARCHIVE_MEMBER_SIZE_BYTES,
    MAX_SOURCE_ARTIFACT_COUNT,
    MAX_TOTAL_ARCHIVE_MEMBER_BYTES,
    MAX_TOTAL_CONTROL_BYTES,
    MAX_TOTAL_SOURCE_BYTES,
    SANITIZATION_REPORT,
    SANITIZATION_RULES,
    SOURCE_ARTIFACT_CLASSIFICATIONS,
    InputValidationError,
    safe_write_bytes,
    sanitize_text,
    sha256_bytes,
    validate_evidence_document,
)
from tests.test_validation_evidence import COMMIT, TIMESTAMP, _recorded_clean_worktree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "validation_evidence_v1.schema.json"
DESCRIPTOR_PATH = (
    REPOSITORY_ROOT / "docs" / "validation_results" / "P0_4_EVIDENCE_ARCHIVE.json"
)


def _current_partial_descriptor() -> dict[str, object]:
    return json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))


def _complete_descriptor() -> dict[str, object]:
    descriptor = _current_partial_descriptor()
    exact_sha256 = "1" * 64
    exact_verified_at = TIMESTAMP
    descriptor["archives"]["exact_private"] = {
        "archive_filename": "validation-evidence-exact-p0-4-v1.tar.gz",
        "created_at_utc": TIMESTAMP,
        "manifest_sha256": "2" * 64,
        "public": False,
        "sha256": exact_sha256,
        "size_bytes": 1234,
        "status": "verified",
        "storage_class": "private_external",
        "storage_locator": "private-external:p0-4",
        "verification_report_sha256": "3" * 64,
        "verified_at_utc": exact_verified_at,
    }
    for artifact in descriptor["source_artifacts"]:
        artifact["exact_bytes_retained"] = True

    exact_report = descriptor["verification"]["reports"][0]
    exact_report.update(
        {
            "archive_sha256": exact_sha256,
            "errors": [],
            "status": "verified",
            "verified_at_utc": exact_verified_at,
        }
    )
    descriptor["verification"]["status"] = "verified"
    descriptor["retention"].update(
        {
            "exact_private_retained": True,
            "sanitized_archive_verified": True,
            "status": "verified",
        }
    )
    descriptor["acceptance_review"] = {
        "reviewers": [
            {
                "identifier": "explicit-reviewer",
                "raw_events_reviewed": True,
                "review_commit": COMMIT,
                "review_method": "manual raw-event and compact-summary review",
                "reviewed_at_utc": TIMESTAMP,
                "role": "acceptance_reviewer",
            }
        ],
        "status": "recorded",
    }
    handoff = descriptor["evidence_handoff_provenance"]
    handoff["final_commit"] = {"status": "recorded", "value": COMMIT}
    handoff["worktree_after_commit"] = _recorded_clean_worktree()
    descriptor["evidence_status"] = "complete"
    return descriptor


class SanitizerHardeningTests(unittest.TestCase):
    def test_secret_assignments_urls_and_file_uris_are_independently_redacted(self) -> None:
        secrets = (
            "password-value",
            "api-value",
            "token-value",
            "secret-value",
            "client-value",
            "private-value",
            "access-value",
            "aws-secret-value",
            "session-value",
        )
        text = "\n".join(
            (
                'password = "password-value"',
                "'api-key': 'api-value'",
                "TOKEN: token-value",
                "secret=secret-value",
                'client_secret: "client-value"',
                "private-key='private-value'",
                "AWS_ACCESS_KEY_ID=access-value",
                "AWS_SECRET_ACCESS_KEY: aws-secret-value",
                'AWS_SESSION_TOKEN = "session-value"',
                "ssh://user:pass@example.invalid/repo",
                "git://user:pass@example.invalid/repo",
                "file:///private/path/report.json",
                "file://localhost/private/path/report.json",
            )
        )

        redacted, counts, findings = sanitize_text(text)

        for secret in secrets:
            self.assertNotIn(secret, redacted)
        self.assertNotIn("user:pass", redacted)
        self.assertNotIn("file://", redacted)
        self.assertEqual(counts["json_secret_assignment"], 4)
        self.assertEqual(counts["secret_assignment"], 5)
        self.assertEqual(counts["credential_url"], 2)
        self.assertEqual(counts["file_uri"], 2)
        self.assertEqual(findings, [])

        second, second_counts, second_findings = sanitize_text(redacted)
        self.assertEqual(second, redacted)
        self.assertEqual(second_counts, {})
        self.assertEqual(second_findings, [])

    def test_shared_rules_include_file_uri_and_are_used_by_packager(self) -> None:
        self.assertIn("file_uri", SANITIZATION_RULES)
        self.assertEqual(packager.SANITIZATION_RULES, SANITIZATION_RULES)

    def test_modern_github_pat_is_redacted_by_the_shared_rule(self) -> None:
        token = "github_pat_11ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghij"

        redacted, counts, findings = sanitize_text(f"credential: {token}\n")

        self.assertNotIn(token, redacted)
        self.assertEqual(counts["github_token"], 1)
        self.assertEqual(findings, [])


class SafeWriteHardeningTests(unittest.TestCase):
    def test_existing_report_is_not_clobbered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            output.write_bytes(b"existing\n")

            with self.assertRaises(FileExistsError):
                safe_write_bytes(output, b"replacement\n")

            self.assertEqual(output.read_bytes(), b"existing\n")
            self.assertEqual(sorted(path.name for path in output.parent.iterdir()), ["report.json"])

    def test_preexisting_foreign_predictable_temp_is_never_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            foreign = output.with_name(output.name + f".tmp-{os.getpid()}")
            foreign.write_bytes(b"foreign\n")

            safe_write_bytes(output, b"new\n")

            self.assertEqual(output.read_bytes(), b"new\n")
            self.assertEqual(foreign.read_bytes(), b"foreign\n")


class DescriptorSemanticHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def errors(self, descriptor: dict[str, object]) -> list[str]:
        return validate_evidence_document(descriptor, self.schema)

    def test_current_partial_and_fully_complete_states_are_valid(self) -> None:
        self.assertEqual(self.errors(_current_partial_descriptor()), [])
        self.assertEqual(self.errors(_complete_descriptor()), [])

    def test_source_artifact_classifications_match_the_schema(self) -> None:
        schema_values = self.schema["$defs"]["sourceArtifact"]["properties"][
            "classification"
        ]["enum"]
        self.assertEqual(tuple(schema_values), SOURCE_ARTIFACT_CLASSIFICATIONS)
        self.assertEqual(
            packager.SOURCE_ARTIFACT_CLASSIFICATIONS,
            SOURCE_ARTIFACT_CLASSIFICATIONS,
        )

    def test_verified_archive_requires_report_hash(self) -> None:
        descriptor = _current_partial_descriptor()
        descriptor["archives"]["sanitized_shareable"]["verification_report_sha256"] = None
        self.assertTrue(any("verification_report_sha256" in error for error in self.errors(descriptor)))

    def test_duplicate_and_mismatched_verification_reports_are_rejected(self) -> None:
        descriptor = _current_partial_descriptor()
        descriptor["verification"]["reports"].append(
            copy.deepcopy(descriptor["verification"]["reports"][1])
        )
        self.assertTrue(any("duplicate report" in error for error in self.errors(descriptor)))

        descriptor = _current_partial_descriptor()
        descriptor["verification"]["reports"][1]["archive_sha256"] = "0" * 64
        self.assertTrue(any("report hash does not match" in error for error in self.errors(descriptor)))

    def test_archive_artifact_sanitization_and_retention_contradictions_are_rejected(self) -> None:
        mutations = (
            ("artifact", lambda value: value["source_artifacts"][0].update(
                {"sanitized_copy_status": "pending"}
            )),
            ("sanitization", lambda value: value["sanitization"].update({"status": "pending"})),
            ("retention", lambda value: value["retention"].update(
                {"sanitized_archive_verified": False}
            )),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                descriptor = _current_partial_descriptor()
                mutate(descriptor)
                self.assertTrue(self.errors(descriptor))

    def test_publication_state_must_be_consistent(self) -> None:
        descriptor = _current_partial_descriptor()
        descriptor["retention"].update(
            {
                "public_asset": "release:p0-4-sanitized",
                "public_asset_published": True,
            }
        )
        errors = self.errors(descriptor)
        self.assertTrue(any("public_asset_published" in error for error in errors))

    def test_descriptor_strings_reject_file_uris_secrets_and_public_exact_storage(self) -> None:
        descriptor = _current_partial_descriptor()
        descriptor["limitations"][0] = "password: exposed-value"
        self.assertTrue(any("secret assignment" in error for error in self.errors(descriptor)))

        descriptor = _current_partial_descriptor()
        descriptor["limitations"][0] = "See file://localhost/private/report.json"
        self.assertTrue(any("private file URI" in error for error in self.errors(descriptor)))

        descriptor = _current_partial_descriptor()
        descriptor["archives"]["exact_private"]["storage_class"] = "public_release"
        self.assertTrue(any("must never use public_release" in error for error in self.errors(descriptor)))

        descriptor = _current_partial_descriptor()
        descriptor["archives"]["exact_private"]["public"] = None
        self.assertTrue(any("explicitly non-public" in error for error in self.errors(descriptor)))

        descriptor = _current_partial_descriptor()
        descriptor["archives"]["sanitized_shareable"]["storage_class"] = "private_external"
        self.assertTrue(any("verified sanitized evidence" in error for error in self.errors(descriptor)))

    def test_impossible_dates_are_rejected_in_every_timestamp_shape(self) -> None:
        timestamp_paths = (
            ("retention", "descriptor_updated_at_utc"),
            ("archives", "sanitized_shareable", "created_at_utc"),
            ("verification", "reports", 1, "verified_at_utc"),
            ("acceptance_review", "reviewers", 0, "reviewed_at_utc"),
            ("evidence_handoff_provenance", "archive_created_at_utc", "value"),
        )
        for timestamp_path in timestamp_paths:
            with self.subTest(path=timestamp_path):
                descriptor = _complete_descriptor()
                target = descriptor
                for part in timestamp_path[:-1]:
                    target = target[part]
                target[timestamp_path[-1]] = "2026-02-30T00:00:00Z"
                self.assertTrue(any("not a real UTC timestamp" in error for error in self.errors(descriptor)))

    def test_complete_status_rejects_partial_evidence(self) -> None:
        descriptor = _current_partial_descriptor()
        descriptor["evidence_status"] = "complete"
        errors = self.errors(descriptor)
        required_messages = (
            "verified retention",
            "verified archive verification",
            "both archives verified",
            "recorded acceptance review",
            "recorded final commit",
            "recorded post-commit worktree state",
        )
        for message in required_messages:
            self.assertTrue(any(message in error for error in errors), message)

    def test_complete_status_requires_clean_handoff_and_recorded_archive_times(self) -> None:
        pending_worktree = copy.deepcopy(
            _current_partial_descriptor()["evidence_handoff_provenance"][
                "worktree_after_commit"
            ]
        )

        def dirty_worktree() -> dict[str, object]:
            observation = _recorded_clean_worktree()
            observation.update({"clean": False, "untracked_count": 1})
            return observation

        mutations = (
            (
                "before-not-recorded",
                lambda handoff: handoff.update({"worktree_before_edits": pending_worktree}),
                "recorded pre-edit worktree state",
            ),
            (
                "before-dirty",
                lambda handoff: handoff.update({"worktree_before_edits": dirty_worktree()}),
                "clean pre-edit worktree",
            ),
            (
                "after-not-recorded",
                lambda handoff: handoff.update({"worktree_after_commit": pending_worktree}),
                "recorded post-commit worktree state",
            ),
            (
                "after-dirty",
                lambda handoff: handoff.update({"worktree_after_commit": dirty_worktree()}),
                "clean post-commit worktree",
            ),
            (
                "archive-created-not-recorded",
                lambda handoff: handoff.update(
                    {"archive_created_at_utc": {"status": "pending", "value": None}}
                ),
                "recorded archive creation timestamp",
            ),
            (
                "archive-verified-not-recorded",
                lambda handoff: handoff.update(
                    {"archive_verified_at_utc": {"status": "pending", "value": None}}
                ),
                "recorded archive verification timestamp",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label):
                descriptor = _complete_descriptor()
                mutate(descriptor["evidence_handoff_provenance"])
                self.assertTrue(
                    any(expected in error for error in self.errors(descriptor)),
                    expected,
                )


class PackagerBoundaryHardeningTests(unittest.TestCase):
    def test_package_input_classification_uses_the_shared_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "summary.txt"
            raw = b"summary\n"
            source.write_bytes(raw)

            def package_input(classification: str) -> dict[str, object]:
                return {
                    "artifacts": [
                        {
                            "archive_path": "artifacts/summary.txt",
                            "classification": classification,
                            "logical_name": "summary",
                            "sha256": sha256_bytes(raw),
                            "source_path": source.name,
                            "source_root": "raw",
                        }
                    ],
                    "format_version": packager.PACKAGE_INPUT_VERSION,
                    "gate": "P0-4",
                    "tested_source_commit": COMMIT,
                }

            for classification in SOURCE_ARTIFACT_CLASSIFICATIONS:
                with self.subTest(classification=classification):
                    _, _, artifacts = packager._validate_package_input(
                        package_input(classification),
                        {"raw": root},
                    )
                    self.assertEqual(artifacts[0]["classification"], classification)

            with self.assertRaisesRegex(InputValidationError, "classification must be one of"):
                packager._validate_package_input(
                    package_input("unreviewed_custom_class"),
                    {"raw": root},
                )

    def test_hard_linked_source_leaf_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            source.write_text("evidence\n", encoding="utf-8")
            os.link(source, root / "alias.txt")

            with self.assertRaisesRegex(InputValidationError, "hard-linked source leaf"):
                packager._read_source(root, "source.txt")

    def test_member_count_boundary_includes_three_control_members(self) -> None:
        self.assertEqual(
            MAX_ARCHIVE_MEMBER_COUNT,
            MAX_SOURCE_ARTIFACT_COUNT + MAX_ARCHIVE_CONTROL_MEMBER_COUNT,
        )
        source_paths = {f"artifacts/item-{index}.txt" for index in range(MAX_SOURCE_ARTIFACT_COUNT)}
        member_sizes = {path: 0 for path in source_paths}
        member_sizes.update(
            {
                "MANIFEST.json": 0,
                "SHA256SUMS": 0,
                SANITIZATION_REPORT: 0,
            }
        )

        packager._validate_archive_layout(member_sizes, source_paths=source_paths)

        member_sizes["EXTRA-CONTROL.json"] = 0
        with self.assertRaises(InputValidationError):
            packager._validate_archive_layout(member_sizes, source_paths=source_paths)

    def test_byte_boundaries_are_numeric_and_require_no_large_allocations(self) -> None:
        self.assertEqual(
            MAX_TOTAL_ARCHIVE_MEMBER_BYTES,
            MAX_TOTAL_SOURCE_BYTES + MAX_TOTAL_CONTROL_BYTES,
        )
        source_paths = {f"artifacts/{index}.txt" for index in range(4)}
        valid_sizes = {path: MAX_ARCHIVE_MEMBER_SIZE_BYTES for path in source_paths}
        valid_sizes["MANIFEST.json"] = MAX_TOTAL_CONTROL_BYTES
        packager._validate_archive_layout(valid_sizes, source_paths=source_paths)

        oversized_member = {"artifacts/a.txt": MAX_ARCHIVE_MEMBER_SIZE_BYTES + 1}
        with self.assertRaises(InputValidationError):
            packager._validate_archive_layout(
                oversized_member,
                source_paths={"artifacts/a.txt"},
            )

        source_sizes = {path: MAX_ARCHIVE_MEMBER_SIZE_BYTES for path in source_paths}
        source_sizes["artifacts/overflow.txt"] = 1
        with self.assertRaisesRegex(InputValidationError, "source members exceed"):
            packager._validate_archive_layout(source_sizes, source_paths=set(source_sizes))

        control_overflow = {
            "artifacts/a.txt": 0,
            "MANIFEST.json": MAX_TOTAL_CONTROL_BYTES + 1,
        }
        with self.assertRaisesRegex(InputValidationError, "control members exceed"):
            packager._validate_archive_layout(
                control_overflow,
                source_paths={"artifacts/a.txt"},
            )

    def test_sanitized_build_checks_layout_before_tar_emission(self) -> None:
        raw = b"password=private-value\n"
        artifacts = [
            {
                "archive_path": "artifacts/summary.txt",
                "classification": "validation_summary",
                "logical_name": "summary",
                "raw": raw,
                "source_sha256": sha256_bytes(raw),
                "source_size_bytes": len(raw),
            }
        ]
        with mock.patch.object(
            packager,
            "_validate_archive_layout",
            wraps=packager._validate_archive_layout,
        ) as validate_layout:
            packager._build_archive(
                archive_kind="sanitized_shareable",
                gate="P0-4",
                tested_source_commit=COMMIT,
                artifacts=artifacts,
                sensitive_values=(),
                manifest_sensitive_values=(),
            )
        validate_layout.assert_called_once()


if __name__ == "__main__":
    unittest.main()
