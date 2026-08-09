"""Adversarial coverage for validation-evidence verifier boundaries."""

from __future__ import annotations

import copy
import contextlib
import gzip
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.package_validation_evidence import package_evidence  # noqa: E402
from scripts.validation_evidence_common import (  # noqa: E402
    PACKAGE_INPUT_VERSION,
    SANITIZATION_REPORT,
    InputValidationError,
    IntegrityError,
    build_sha256sums,
    canonical_json_bytes,
    sha256_bytes,
)
from scripts.verify_validation_evidence import (  # noqa: E402
    main as verify_main,
    verify_archive,
)


COMMIT = "a" * 40
TIMESTAMP = "2026-08-05T00:00:00Z"


def _read_members(archive_path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isreg():
                raise AssertionError(f"unexpected non-regular fixture member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise AssertionError(f"missing fixture payload: {member.name}")
            with stream:
                result[member.name] = stream.read()
    return result


def _write_normalized_archive(output: Path, members: dict[str, bytes]) -> None:
    with output.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_output,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                for path, payload in sorted(members.items()):
                    info = tarfile.TarInfo(path)
                    info.size = len(payload)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.type = tarfile.REGTYPE
                    info.linkname = ""
                    info.pax_headers = {}
                    archive.addfile(info, io.BytesIO(payload))


def _refresh_integrity(members: dict[str, bytes]) -> None:
    manifest = json.loads(members["MANIFEST.json"].decode("utf-8"))
    for entry in manifest["members"]:
        payload = members[entry["path"]]
        entry["sha256"] = sha256_bytes(payload)
        entry["size_bytes"] = len(payload)
    manifest_bytes = canonical_json_bytes(manifest)
    members["MANIFEST.json"] = manifest_bytes
    members["SHA256SUMS"] = build_sha256sums(
        [
            {"path": "MANIFEST.json", "sha256": sha256_bytes(manifest_bytes)},
            *[
                {"path": entry["path"], "sha256": entry["sha256"]}
                for entry in manifest["members"]
            ],
        ]
    )


class VerifierHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.repository = cls.root / "repository"
        cls.repository.mkdir()
        cls.source_root = cls.root / "source"
        cls.source_root.mkdir()
        cls.source_bytes = (
            b"loss=8.25\n"
            b"private_path=/home/synthetic-user/private/results.json\n"
        )
        (cls.source_root / "summary.txt").write_bytes(cls.source_bytes)
        cls.package_input = {
            "artifacts": [
                {
                    "archive_path": "artifacts/summary.txt",
                    "classification": "validation_summary",
                    "logical_name": "summary",
                    "sha256": sha256_bytes(cls.source_bytes),
                    "source_path": "summary.txt",
                    "source_root": "raw",
                }
            ],
            "format_version": PACKAGE_INPUT_VERSION,
            "gate": "P0-4",
            "tested_source_commit": COMMIT,
        }
        cls.archive = cls.root / "sanitized.tar.gz"
        cls.package_report = package_evidence(
            cls.package_input,
            roots={"raw": cls.source_root},
            mode="sanitized-only",
            exact_output=None,
            sanitized_output=cls.archive,
            repository_root=cls.repository,
            sensitive_values=("synthetic-user",),
            created_at_utc=TIMESTAMP,
        )
        cls.members = _read_members(cls.archive)
        cls.manifest = json.loads(cls.members["MANIFEST.json"].decode("utf-8"))
        cls.sanitization_report = json.loads(
            cls.members[SANITIZATION_REPORT].decode("utf-8")
        )
        cls.archive_result = cls.package_report["archives"][0]
        source_entry = next(
            entry
            for entry in cls.manifest["members"]
            if entry["kind"] == "source_artifact"
        )
        cls.descriptor = {
            "archives": {
                "sanitized_shareable": {
                    "archive_filename": cls.archive.name,
                    "manifest_sha256": cls.archive_result["manifest_sha256"],
                    "sha256": cls.archive_result["sha256"],
                    "size_bytes": cls.archive_result["size_bytes"],
                    "status": "verified",
                }
            },
            "source_artifacts": [
                {
                    "archive_path": source_entry["path"],
                    "classification": source_entry["classification"],
                    "logical_name": source_entry["logical_name"],
                    "sha256": source_entry["source_sha256"],
                    "size_bytes": source_entry["source_size_bytes"],
                }
            ],
            "sanitization": {
                "files_scanned": len(cls.sanitization_report["files_scanned"]),
                "replacement_count": sum(
                    item["count"]
                    for item in cls.sanitization_report["replacement_totals"]
                ),
                "report_sha256": sha256_bytes(cls.members[SANITIZATION_REPORT]),
                "rules_applied": cls.sanitization_report["rules_applied"],
            },
            "tested_source": {"commit": COMMIT},
            "validation_gate": "P0-4",
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _mutate_report(
        self,
        label: str,
        mutate: Callable[[dict[str, Any]], None],
    ) -> Path:
        members = dict(self.members)
        report = json.loads(members[SANITIZATION_REPORT].decode("utf-8"))
        mutate(report)
        members[SANITIZATION_REPORT] = canonical_json_bytes(report)
        _refresh_integrity(members)
        output = self.root / f"report-{label}.tar.gz"
        _write_normalized_archive(output, members)
        return output

    def test_baseline_fixture_verifies(self) -> None:
        report = verify_archive(
            self.archive,
            expected_sha256=self.archive_result["sha256"],
            evidence_document=self.descriptor,
            verification_timestamp_utc=TIMESTAMP,
        )
        self.assertEqual(report["status"], "verified")
        self.assertEqual(
            report["checks"]["sanitization"]["independent_members_scanned"],
            len(self.members),
        )
        self.assertEqual(
            report["checks"]["sanitization"]["descriptor_values"],
            self.descriptor["sanitization"],
        )

    def test_matching_hash_does_not_allow_raw_or_concatenated_gzip_trailers(self) -> None:
        cases = {
            "raw": self.archive.read_bytes() + b"PASSWORD=\"raw-secret\"\n",
            "concatenated": self.archive.read_bytes()
            + gzip.compress(b"PASSWORD=\"gzip-secret\"\n", mtime=0),
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                archive = self.root / f"trailer-{label}.tar.gz"
                archive.write_bytes(payload)
                with self.assertRaisesRegex(
                    IntegrityError, "trailing bytes or concatenated gzip"
                ):
                    verify_archive(
                        archive,
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        verification_timestamp_utc=TIMESTAMP,
                    )
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    code = verify_main(
                        [
                            "--archive",
                            str(archive),
                            "--expected-sha256",
                            hashlib.sha256(payload).hexdigest(),
                            "--timestamp-utc",
                            TIMESTAMP,
                        ]
                    )
                self.assertEqual(code, 3)
                self.assertIn("integrity_or_security", stdout.getvalue())
                self.assertEqual(stderr.getvalue(), "")

    def test_noncanonical_gzip_header_is_rejected(self) -> None:
        tar_payload = gzip.decompress(self.archive.read_bytes())
        output = self.root / "noncanonical-header.tar.gz"
        with output.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="private-hostname",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=1,
            ) as compressed:
                compressed.write(tar_payload)
        with self.assertRaisesRegex(IntegrityError, "canonical single-member gzip"):
            verify_archive(
                output,
                expected_sha256=sha256_bytes(output.read_bytes()),
                verification_timestamp_utc=TIMESTAMP,
            )

    def test_hidden_tar_padding_bytes_are_rejected(self) -> None:
        tar_payload = bytearray(gzip.decompress(self.archive.read_bytes()))
        with tarfile.open(self.archive, mode="r:gz") as archive:
            member = next(item for item in archive.getmembers() if item.size % 512)
        tar_payload[member.offset_data + member.size] = ord("S")
        output = self.root / "hidden-padding-secret.tar.gz"
        with output.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="", mode="wb", compresslevel=9, fileobj=raw_output, mtime=0
            ) as compressed:
                compressed.write(tar_payload)
        with self.assertRaisesRegex(IntegrityError, "nonzero tar padding"):
            verify_archive(
                output,
                expected_sha256=sha256_bytes(output.read_bytes()),
                verification_timestamp_utc=TIMESTAMP,
            )

    def test_sensitive_manifest_control_metadata_is_rescanned(self) -> None:
        secrets = {
            "legacy": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "fine-grained": (
                "github_pat_11AAAAAAAAAAAAAAAAAAAA_BBBBBBBBBBBBBBBBBBBB"
            ),
        }
        for label, secret in secrets.items():
            with self.subTest(label=label):
                members = dict(self.members)
                manifest = json.loads(members["MANIFEST.json"].decode("utf-8"))
                manifest["gate"] = secret
                members["MANIFEST.json"] = canonical_json_bytes(manifest)
                _refresh_integrity(members)
                output = self.root / f"manifest-secret-{label}.tar.gz"
                _write_normalized_archive(output, members)
                with self.assertRaisesRegex(
                    IntegrityError, "MANIFEST.json.*residual sensitive"
                ):
                    verify_archive(output, verification_timestamp_utc=TIMESTAMP)

    def test_descriptor_is_bound_to_gate_commit_and_source_artifacts(self) -> None:
        cases: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
            "gate": (
                lambda value: value.__setitem__("validation_gate", "P0-X"),
                "validation gate",
            ),
            "commit": (
                lambda value: value["tested_source"].__setitem__(
                    "commit", "f" * 40
                ),
                "tested-source commit",
            ),
            "artifact_set": (
                lambda value: value.__setitem__("source_artifacts", []),
                "source-artifact set",
            ),
            "logical_name": (
                lambda value: value["source_artifacts"][0].__setitem__(
                    "logical_name", "unrelated"
                ),
                "field 'logical_name'",
            ),
            "classification": (
                lambda value: value["source_artifacts"][0].__setitem__(
                    "classification", "other"
                ),
                "field 'classification'",
            ),
            "source_size": (
                lambda value: value["source_artifacts"][0].__setitem__(
                    "size_bytes", value["source_artifacts"][0]["size_bytes"] + 1
                ),
                "field 'size_bytes'",
            ),
            "source_hash": (
                lambda value: value["source_artifacts"][0].__setitem__(
                    "sha256", "0" * 64
                ),
                "field 'sha256'",
            ),
        }
        for label, (mutate, expected_error) in cases.items():
            with self.subTest(label=label):
                descriptor = copy.deepcopy(self.descriptor)
                mutate(descriptor)
                with self.assertRaisesRegex(IntegrityError, expected_error):
                    verify_archive(
                        self.archive,
                        expected_sha256=self.archive_result["sha256"],
                        evidence_document=descriptor,
                        verification_timestamp_utc=TIMESTAMP,
                    )

    def test_descriptor_sanitization_metadata_is_bound_to_embedded_report(self) -> None:
        cases: dict[str, Callable[[dict[str, Any]], None]] = {
            "report_sha256": lambda value: value["sanitization"].__setitem__(
                "report_sha256", "0" * 64
            ),
            "rules_applied": lambda value: value["sanitization"].__setitem__(
                "rules_applied", []
            ),
            "files_scanned": lambda value: value["sanitization"].__setitem__(
                "files_scanned", value["sanitization"]["files_scanned"] + 1
            ),
            "replacement_count": lambda value: value["sanitization"].__setitem__(
                "replacement_count",
                value["sanitization"]["replacement_count"] + 1,
            ),
        }
        for field, mutate in cases.items():
            with self.subTest(field=field):
                descriptor = copy.deepcopy(self.descriptor)
                mutate(descriptor)
                with self.assertRaisesRegex(
                    IntegrityError, rf"sanitization {field} does not match"
                ):
                    verify_archive(
                        self.archive,
                        expected_sha256=self.archive_result["sha256"],
                        evidence_document=descriptor,
                        verification_timestamp_utc=TIMESTAMP,
                    )

    def test_manifest_source_classification_uses_shared_domain(self) -> None:
        members = dict(self.members)
        manifest = json.loads(members["MANIFEST.json"].decode("utf-8"))
        source_entry = next(
            entry for entry in manifest["members"] if entry["kind"] == "source_artifact"
        )
        source_entry["classification"] = "custom-unregistered-classification"
        members["MANIFEST.json"] = canonical_json_bytes(manifest)
        _refresh_integrity(members)
        output = self.root / "unsupported-source-classification.tar.gz"
        _write_normalized_archive(output, members)
        with self.assertRaisesRegex(
            IntegrityError, "unsupported source-artifact classification"
        ):
            verify_archive(output, verification_timestamp_utc=TIMESTAMP)

    def test_spoofed_repository_root_cannot_allow_exact_output_in_checkout(self) -> None:
        exact_output = REPOSITORY_ROOT / f".private-{self.root.name}.tar.gz"
        with self.assertRaisesRegex(InputValidationError, "outside every Git worktree"):
            package_evidence(
                self.package_input,
                roots={"raw": self.source_root},
                mode="exact-only",
                exact_output=exact_output,
                sanitized_output=None,
                repository_root=self.repository,
                dry_run=True,
                created_at_utc=TIMESTAMP,
            )

    def test_malformed_and_lying_sanitization_reports_are_rejected(self) -> None:
        def missing_top(report: dict[str, Any]) -> None:
            del report["ruleset_version"]

        def missing_file_hash(report: dict[str, Any]) -> None:
            del report["files_scanned"][0]["source_sha256"]

        def final_scan_extra_key(report: dict[str, Any]) -> None:
            report["final_scan"]["untrusted"] = True

        def wrong_sanitized_size(report: dict[str, Any]) -> None:
            report["files_scanned"][0]["sanitized_size_bytes"] += 1

        def lying_replacement_total(report: dict[str, Any]) -> None:
            report["replacement_totals"][0]["count"] += 1

        def unsorted_rules(report: dict[str, Any]) -> None:
            report["rules_applied"] = list(reversed(report["rules_applied"]))

        cases = {
            "missing-top": missing_top,
            "missing-file-hash": missing_file_hash,
            "final-extra": final_scan_extra_key,
            "wrong-size": wrong_sanitized_size,
            "lying-total": lying_replacement_total,
            "unsorted-rules": unsorted_rules,
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                archive = self._mutate_report(label, mutate)
                with self.assertRaises(IntegrityError):
                    verify_archive(archive, verification_timestamp_utc=TIMESTAMP)


if __name__ == "__main__":
    unittest.main()
