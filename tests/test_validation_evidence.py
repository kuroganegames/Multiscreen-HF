"""Offline tests for validation provenance and evidence retention v1."""

from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.collect_validation_provenance import (  # noqa: E402
    collect_provenance,
    collect_worktree,
    main as provenance_main,
    parse_reviewers,
    redact_remote_url,
)
from scripts.package_validation_evidence import (  # noqa: E402
    main as package_main,
    package_evidence,
)
from scripts.validation_evidence_common import (  # noqa: E402
    PACKAGE_INPUT_VERSION,
    InputValidationError,
    IntegrityError,
    build_sha256sums,
    canonical_json_bytes,
    sha256_bytes,
    validate_evidence_document,
)
from scripts.verify_validation_evidence import (  # noqa: E402
    VerificationIOError,
    main as verify_main,
    verify_archive,
)


COMMIT = "a" * 40
TIMESTAMP = "2026-08-04T00:00:00Z"
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "validation_evidence_v1.schema.json"


def _git(repo: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _initialize_git_repository(root: Path) -> Path:
    repo = root / "git-fixture"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Synthetic Reviewer")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    (repo / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "initial fixture")
    return repo


def _unknown_worktree(status: str = "not_recorded_in_original_run") -> dict[str, object]:
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


def _recorded_clean_worktree() -> dict[str, object]:
    return {
        "clean": True,
        "collected_at_utc": TIMESTAMP,
        "porcelain_format": "git-status-porcelain-v1",
        "porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "staged_changes": False,
        "status": "recorded",
        "unstaged_changes": False,
        "untracked_count": 0,
    }


def _pending_archive(*, exact: bool) -> dict[str, object]:
    return {
        "archive_filename": None,
        "created_at_utc": None,
        "manifest_sha256": None,
        "public": False if exact else None,
        "sha256": None,
        "size_bytes": None,
        "status": "pending",
        "storage_class": None,
        "storage_locator": None,
        "verification_report_sha256": None,
        "verified_at_utc": None,
    }


def _truthful_partial_descriptor() -> dict[str, object]:
    artifact = b'{"qualification":"passed"}\n'
    return {
        "acceptance_review": {
            "reviewers": [
                {
                    "identifier": "evidence-reviewer",
                    "raw_events_reviewed": True,
                    "review_commit": COMMIT,
                    "review_method": "manual raw-event and compact-summary review",
                    "reviewed_at_utc": TIMESTAMP,
                    "role": "evidence_reviewer",
                }
            ],
            "status": "recorded",
        },
        "archives": {
            "exact_private": _pending_archive(exact=True),
            "sanitized_shareable": _pending_archive(exact=False),
        },
        "evidence_gate": "P1-preflight A",
        "evidence_handoff_provenance": {
            "archive_created_at_utc": {"status": "pending", "value": None},
            "archive_verified_at_utc": {"status": "pending", "value": None},
            "final_commit": {"status": "pending", "value": None},
            "implementation_base_commit": COMMIT,
            "working_branch": "infra/p1-preflight-a-evidence-v1",
            "worktree_after_commit": _unknown_worktree("pending"),
            "worktree_before_edits": _recorded_clean_worktree(),
        },
        "evidence_status": "partial",
        "limitations": ["Durable private retention and final review remain pending."],
        "original_run_provenance": {
            "original_run_review": {
                "reviewers": [],
                "status": "not_recorded_in_original_run",
            },
            "run_worktree_at_end": _unknown_worktree(),
            "run_worktree_at_start": _unknown_worktree(),
        },
        "retention": {
            "descriptor_updated_at_utc": TIMESTAMP,
            "exact_private_retained": False,
            "public_asset": None,
            "public_asset_published": False,
            "sanitized_archive_verified": False,
            "status": "partial",
        },
        "sanitization": {
            "files_scanned": 0,
            "replacement_count": 0,
            "report_sha256": None,
            "rules_applied": [],
            "status": "pending",
            "unresolved_findings": [],
        },
        "schema_version": "1.0.0",
        "source_artifacts": [
            {
                "archive_path": "artifacts/summary.json",
                "classification": "validation_summary",
                "exact_bytes_retained": False,
                "logical_name": "summary",
                "sanitized_copy_status": "pending",
                "sha256": sha256_bytes(artifact),
                "size_bytes": len(artifact),
            }
        ],
        "tested_source": {
            "branch": "main",
            "commit": COMMIT,
            "repository": "kuroganegames/Multiscreen-HF",
        },
        "validation_gate": "P0-4",
        "validation_status": "passed",
        "verification": {
            "reports": [],
            "status": "pending",
            "verifier_version": "1.0.0",
        },
    }


def _read_regular_members(archive_path: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isreg():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise AssertionError(f"missing payload for {member.name}")
            with stream:
                members[member.name] = stream.read()
    return members


def _write_tar_entries(
    output: Path,
    entries: list[tuple[str, bytes, bytes, str]],
) -> None:
    with output.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_output,
            mode="wb",
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                for name, payload, member_type, linkname in entries:
                    info = tarfile.TarInfo(name)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.type = member_type
                    info.linkname = linkname
                    info.size = len(payload) if member_type == tarfile.REGTYPE else 0
                    archive.addfile(
                        info,
                        io.BytesIO(payload) if member_type == tarfile.REGTYPE else None,
                    )


def _write_regular_tar(output: Path, members: dict[str, bytes]) -> None:
    entries = [
        (name, payload, tarfile.REGTYPE, "")
        for name, payload in sorted(members.items())
    ]
    _write_tar_entries(output, entries)


class EvidenceSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def assert_valid(self, descriptor: dict[str, object]) -> None:
        self.assertEqual(validate_evidence_document(descriptor, self.schema), [])

    def test_truthful_partial_descriptor_is_valid(self) -> None:
        self.assert_valid(_truthful_partial_descriptor())

    def test_missing_provenance_is_rejected(self) -> None:
        descriptor = _truthful_partial_descriptor()
        del descriptor["original_run_provenance"]
        errors = validate_evidence_document(descriptor, self.schema)
        self.assertTrue(errors)
        self.assertTrue(any("original_run_provenance" in error for error in errors))

    def test_not_recorded_requires_null_instead_of_fabricated_boolean(self) -> None:
        descriptor = _truthful_partial_descriptor()
        self.assert_valid(descriptor)
        descriptor["original_run_provenance"]["run_worktree_at_start"]["clean"] = False
        errors = validate_evidence_document(descriptor, self.schema)
        self.assertTrue(errors)

    def test_contradictory_recorded_clean_state_is_rejected(self) -> None:
        descriptor = _truthful_partial_descriptor()
        observation = descriptor["evidence_handoff_provenance"]["worktree_before_edits"]
        observation["clean"] = True
        observation["staged_changes"] = True
        errors = validate_evidence_document(descriptor, self.schema)
        self.assertTrue(any("contradicts" in error for error in errors))

    def test_private_absolute_descriptor_path_is_rejected(self) -> None:
        descriptor = _truthful_partial_descriptor()
        descriptor["archives"]["exact_private"]["storage_locator"] = (
            "/private/evidence/p0-4-exact.tar.gz"
        )
        errors = validate_evidence_document(descriptor, self.schema)
        self.assertTrue(any("absolute path" in error for error in errors))


class ReviewerAndCollectorTests(unittest.TestCase):
    def _new_repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary, _initialize_git_repository(Path(temporary.name))

    def test_reviewer_one_multiple_and_missing(self) -> None:
        self.assertEqual(parse_reviewers(["@alice"]), ["alice"])
        self.assertEqual(
            parse_reviewers(["bob", "alice,bob"], "carol"),
            ["alice", "bob", "carol"],
        )
        with self.assertRaises(ValueError):
            parse_reviewers(None, None)
        with self.assertRaises(ValueError):
            parse_reviewers(["../not-a-reviewer"])

    def test_missing_reviewer_is_not_inferred_from_ambient_identity(self) -> None:
        _temporary, repo = self._new_repo()
        _git(repo, "remote", "add", "origin", "https://github.com/remote-owner/repo.git")
        stderr = io.StringIO()
        stdout = io.StringIO()
        ambient = {
            "GITHUB_ACTOR": "implicit-actor",
            "MULTISCREEN_EVIDENCE_REVIEWERS": "",
            "USER": "implicit-user",
        }
        with mock.patch.dict(os.environ, ambient, clear=False):
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                code = provenance_main(["--repo", os.fspath(repo)])
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")

    def test_clean_staged_unstaged_and_untracked_states(self) -> None:
        for state in ("clean", "staged", "unstaged", "untracked"):
            with self.subTest(state=state):
                _temporary, repo = self._new_repo()
                if state == "staged":
                    (repo / "tracked.txt").write_text("staged\n", encoding="utf-8")
                    _git(repo, "add", "tracked.txt")
                elif state == "unstaged":
                    (repo / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
                elif state == "untracked":
                    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

                observed = collect_worktree(repo)
                exact_status = _git(
                    repo,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                )
                self.assertEqual(
                    observed["porcelain"]["sha256"],
                    hashlib.sha256(exact_status).hexdigest(),
                )
                self.assertEqual(observed["porcelain"]["byte_count"], len(exact_status))
                self.assertEqual(observed["clean"], state == "clean")
                self.assertEqual(observed["staged_changes_present"], state == "staged")
                self.assertEqual(observed["unstaged_changes_present"], state == "unstaged")
                self.assertEqual(observed["untracked_path_count"], int(state == "untracked"))

    def test_remote_credentials_are_redacted(self) -> None:
        _temporary, repo = self._new_repo()
        secret = "fake-user:fake-password"
        _git(
            repo,
            "remote",
            "add",
            "origin",
            f"https://{secret}@example.invalid/team/repo.git",
        )
        head = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
        provenance = collect_provenance(
            repo,
            reviewers=["reviewer"],
            review_method="manual",
            review_commit=head,
            raw_events_reviewed=True,
            timestamp_utc=TIMESTAMP,
        )
        encoded = json.dumps(provenance, sort_keys=True)
        self.assertNotIn(secret, encoded)
        self.assertIn("https://example.invalid/team/repo.git", encoded)
        self.assertEqual(
            redact_remote_url("ssh://token:password@example.invalid/team/repo.git"),
            "ssh://example.invalid/team/repo.git",
        )
        self.assertEqual(
            redact_remote_url("token@example.invalid:team/repo.git"),
            "example.invalid:team/repo.git",
        )


class EvidenceArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.repository = cls.root / "repository"
        cls.repository.mkdir()
        cls.source_root = cls.root / "raw-source"
        cls.source_root.mkdir()
        cls.source_bytes = (
            "transformers_version=5.9.0\n"
            "loss=8.25\n"
            "python=/home/fake-user/miniforge/envs/evidence/bin/python\n"
            "cache=/home/fake-user/.cache/huggingface/hub\n"
            "windows=C:\\Users\\fake-user\\AppData\\Local\\hf_cache\\model\n"
            "username=fake-user\n"
            "hostname=private-host.invalid\n"
            "remote=https://fake-user:fake-password@example.invalid/team/repo.git\n"
            "github=ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
            "huggingface=hf_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n"
            "openai=sk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456\n"
            "API_KEY=fake-secret-assignment\n"
        ).encode("utf-8")
        (cls.source_root / "summary.txt").write_bytes(cls.source_bytes)
        cls.package_input = cls._package_document_for("summary.txt", cls.source_bytes)
        cls.exact_one = cls.root / "exact-one.tar.gz"
        cls.sanitized_one = cls.root / "sanitized-one.tar.gz"
        cls.exact_two = cls.root / "exact-two.tar.gz"
        cls.sanitized_two = cls.root / "sanitized-two.tar.gz"
        cls.report_one = cls._package_pair(cls.exact_one, cls.sanitized_one)
        cls.report_two = cls._package_pair(cls.exact_two, cls.sanitized_two)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def _package_document_for(
        cls,
        source_path: str,
        source_bytes: bytes,
        *,
        archive_path: str = "artifacts/summary.txt",
    ) -> dict[str, object]:
        return {
            "artifacts": [
                {
                    "archive_path": archive_path,
                    "classification": "validation_summary",
                    "logical_name": "summary",
                    "sha256": sha256_bytes(source_bytes),
                    "source_path": source_path,
                    "source_root": "raw",
                }
            ],
            "format_version": PACKAGE_INPUT_VERSION,
            "gate": "P0-4",
            "tested_source_commit": COMMIT,
        }

    @classmethod
    def _package_pair(cls, exact: Path, sanitized: Path) -> dict[str, object]:
        return package_evidence(
            cls.package_input,
            roots={"raw": cls.source_root},
            mode="both",
            exact_output=exact,
            sanitized_output=sanitized,
            repository_root=cls.repository,
            sensitive_values=("fake-user", "private-host.invalid"),
            created_at_utc=TIMESTAMP,
        )

    def test_deterministic_exact_and_sanitized_packaging(self) -> None:
        self.assertEqual(self.exact_one.read_bytes(), self.exact_two.read_bytes())
        self.assertEqual(
            self.sanitized_one.read_bytes(), self.sanitized_two.read_bytes()
        )
        first = {
            entry["archive_kind"]: entry["sha256"]
            for entry in self.report_one["archives"]
        }
        second = {
            entry["archive_kind"]: entry["sha256"]
            for entry in self.report_two["archives"]
        }
        self.assertEqual(first, second)

    def test_exact_bytes_retained_and_sanitized_content_is_safe(self) -> None:
        exact_members = _read_regular_members(self.exact_one)
        sanitized_members = _read_regular_members(self.sanitized_one)
        self.assertEqual(exact_members["artifacts/summary.txt"], self.source_bytes)

        sanitized = sanitized_members["artifacts/summary.txt"].decode("utf-8")
        for sensitive in (
            "/home/fake-user",
            r"C:\Users\fake-user",
            ".cache/huggingface",
            "fake-user",
            "private-host.invalid",
            "fake-password",
            "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "hf_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            "sk-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "abcdefghijklmnopqrstuvwxyz123456",
            "fake-secret-assignment",
        ):
            self.assertNotIn(sensitive, sanitized)
        self.assertIn("transformers_version=5.9.0", sanitized)
        self.assertIn("loss=8.25", sanitized)
        self.assertIn("SANITIZATION_REPORT.json", sanitized_members)

    def test_valid_exact_and_sanitized_archives_verify(self) -> None:
        for archive, kind in (
            (self.exact_one, "exact_private"),
            (self.sanitized_one, "sanitized_shareable"),
        ):
            with self.subTest(kind=kind):
                expected = hashlib.sha256(archive.read_bytes()).hexdigest()
                report = verify_archive(
                    archive,
                    expected_sha256=expected,
                    verification_timestamp_utc=TIMESTAMP,
                )
                self.assertEqual(report["status"], "verified")
                self.assertEqual(report["archive"]["archive_kind"], kind)

    def _assert_package_rejected(
        self,
        document: dict[str, object],
        exception: type[Exception],
        label: str,
    ) -> None:
        with self.assertRaises(exception):
            package_evidence(
                document,
                roots={"raw": self.source_root},
                mode="exact-only",
                exact_output=self.root / f"rejected-{label}.tar.gz",
                sanitized_output=None,
                repository_root=self.repository,
                created_at_utc=TIMESTAMP,
            )

    def test_source_hash_mismatch_is_rejected(self) -> None:
        document = copy.deepcopy(self.package_input)
        document["artifacts"][0]["sha256"] = "0" * 64
        self._assert_package_rejected(document, IntegrityError, "hash")

    def test_absolute_traversal_and_banned_checkpoint_paths_are_rejected(self) -> None:
        cases = {
            "absolute": ("source_path", "/private/summary.txt"),
            "traversal": ("source_path", "../summary.txt"),
            "archive_traversal": ("archive_path", "artifacts/../summary.txt"),
            "checkpoint": ("source_path", "checkpoint-7/summary.txt"),
        }
        for label, (field, value) in cases.items():
            with self.subTest(label=label):
                document = copy.deepcopy(self.package_input)
                document["artifacts"][0][field] = value
                self._assert_package_rejected(document, InputValidationError, label)

    def test_leaf_and_parent_symlinks_are_rejected(self) -> None:
        leaf = self.source_root / "leaf-link.txt"
        parent = self.source_root / "parent-link"
        real_parent = self.source_root / "real-parent"
        real_parent.mkdir(exist_ok=True)
        parent_target = real_parent / "inside.txt"
        parent_target.write_bytes(self.source_bytes)
        try:
            leaf.symlink_to(self.source_root / "summary.txt")
            parent.symlink_to(real_parent, target_is_directory=True)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        for label, path in (
            ("leaf", "leaf-link.txt"),
            ("parent", "parent-link/inside.txt"),
        ):
            with self.subTest(label=label):
                document = self._package_document_for(path, self.source_bytes)
                self._assert_package_rejected(document, InputValidationError, f"symlink-{label}")

    def test_exact_archive_output_inside_repository_is_rejected(self) -> None:
        with self.assertRaises(InputValidationError):
            package_evidence(
                self.package_input,
                roots={"raw": self.source_root},
                mode="exact-only",
                exact_output=self.repository / "private-exact.tar.gz",
                sanitized_output=None,
                repository_root=self.repository,
                created_at_utc=TIMESTAMP,
            )

    def test_wrong_expected_hash_and_byte_level_tampering_are_detected(self) -> None:
        with self.assertRaises(IntegrityError):
            verify_archive(
                self.exact_one,
                expected_sha256="0" * 64,
                verification_timestamp_utc=TIMESTAMP,
            )

        original = self.exact_one.read_bytes()
        expected_original = hashlib.sha256(original).hexdigest()
        one_byte = self.root / "one-byte-tampered.tar.gz"
        modified = bytearray(original)
        modified[-1] ^= 1
        one_byte.write_bytes(modified)
        truncated = self.root / "truncated.tar.gz"
        truncated.write_bytes(original[:-16])
        for archive in (one_byte, truncated):
            with self.subTest(archive=archive.name):
                with self.assertRaises((IntegrityError, VerificationIOError)):
                    verify_archive(
                        archive,
                        expected_sha256=expected_original,
                        verification_timestamp_utc=TIMESTAMP,
                    )

    def test_malicious_tar_paths_types_duplicates_and_unexpected_members(self) -> None:
        attacks: dict[str, list[tuple[str, bytes, bytes, str]]] = {
            "traversal": [("../escape", b"x", tarfile.REGTYPE, "")],
            "duplicate": [
                ("MANIFEST.json", b"{}", tarfile.REGTYPE, ""),
                ("MANIFEST.json", b"{}", tarfile.REGTYPE, ""),
            ],
            "symlink": [("artifacts/link", b"", tarfile.SYMTYPE, "target")],
        }
        for label, entries in attacks.items():
            with self.subTest(label=label):
                archive = self.root / f"malicious-{label}.tar.gz"
                _write_tar_entries(archive, entries)
                with self.assertRaises(IntegrityError):
                    verify_archive(archive, verification_timestamp_utc=TIMESTAMP)

        members = _read_regular_members(self.exact_one)
        members["artifacts/unexpected.txt"] = b"not allowlisted\n"
        unexpected = self.root / "malicious-unexpected.tar.gz"
        _write_regular_tar(unexpected, members)
        with self.assertRaises(IntegrityError):
            verify_archive(unexpected, verification_timestamp_utc=TIMESTAMP)

    def test_sanitization_report_lie_is_independently_detected(self) -> None:
        members = _read_regular_members(self.sanitized_one)
        payload_path = "artifacts/summary.txt"
        injected = members[payload_path] + (
            b"\nAPI_KEY=ghp_ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\n"
        )
        members[payload_path] = injected

        report = json.loads(members["SANITIZATION_REPORT.json"].decode("utf-8"))
        for scanned in report["files_scanned"]:
            if scanned["path"] == payload_path:
                scanned["sanitized_sha256"] = sha256_bytes(injected)
                scanned["sanitized_size_bytes"] = len(injected)
                scanned["replacements"] = {}
                scanned["unresolved_findings"] = []
        report["replacement_totals"] = []
        report["status"] = "passed"
        report["unresolved_findings"] = []
        report["final_scan"] = {
            "high_confidence_findings": 0,
            "status": "passed",
        }
        report_bytes = canonical_json_bytes(report)
        members["SANITIZATION_REPORT.json"] = report_bytes

        manifest = json.loads(members["MANIFEST.json"].decode("utf-8"))
        for entry in manifest["members"]:
            if entry["path"] == payload_path:
                entry["sha256"] = sha256_bytes(injected)
                entry["size_bytes"] = len(injected)
            elif entry["path"] == "SANITIZATION_REPORT.json":
                entry["sha256"] = sha256_bytes(report_bytes)
                entry["size_bytes"] = len(report_bytes)
        manifest_bytes = canonical_json_bytes(manifest)
        members["MANIFEST.json"] = manifest_bytes
        sums_entries = [
            {"path": "MANIFEST.json", "sha256": sha256_bytes(manifest_bytes)},
            *[
                {"path": entry["path"], "sha256": entry["sha256"]}
                for entry in manifest["members"]
            ],
        ]
        members["SHA256SUMS"] = build_sha256sums(sums_entries)

        lying_archive = self.root / "lying-sanitization-report.tar.gz"
        _write_regular_tar(lying_archive, members)
        with self.assertRaises(IntegrityError):
            verify_archive(lying_archive, verification_timestamp_utc=TIMESTAMP)

    def test_stable_main_exit_codes(self) -> None:
        valid_input = self.root / "package-input.json"
        valid_input.write_bytes(canonical_json_bytes(self.package_input))
        invalid_input = self.root / "invalid-package-input.json"
        invalid_input.write_text("{}\n", encoding="utf-8")
        mismatch_input = self.root / "mismatch-package-input.json"
        mismatch = copy.deepcopy(self.package_input)
        mismatch["artifacts"][0]["sha256"] = "0" * 64
        mismatch_input.write_bytes(canonical_json_bytes(mismatch))

        sink_out = io.StringIO()
        sink_err = io.StringIO()
        with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
            package_success = package_main(
                [
                    os.fspath(valid_input),
                    "--root",
                    f"raw={self.source_root}",
                    "--mode",
                    "both",
                    "--exact-output",
                    os.fspath(self.root / "cli-dry-exact.tar.gz"),
                    "--sanitized-output",
                    os.fspath(self.root / "cli-dry-sanitized.tar.gz"),
                    "--repository-root",
                    os.fspath(self.repository),
                    "--dry-run",
                ]
            )
            package_invalid = package_main(
                [
                    os.fspath(invalid_input),
                    "--root",
                    f"raw={self.source_root}",
                    "--mode",
                    "exact-only",
                    "--exact-output",
                    os.fspath(self.root / "cli-invalid.tar.gz"),
                    "--repository-root",
                    os.fspath(self.repository),
                ]
            )
            package_integrity = package_main(
                [
                    os.fspath(mismatch_input),
                    "--root",
                    f"raw={self.source_root}",
                    "--mode",
                    "exact-only",
                    "--exact-output",
                    os.fspath(self.root / "cli-mismatch.tar.gz"),
                    "--repository-root",
                    os.fspath(self.repository),
                ]
            )
            package_io = package_main(
                [
                    os.fspath(self.root / "missing-input.json"),
                    "--root",
                    f"raw={self.source_root}",
                    "--mode",
                    "exact-only",
                    "--exact-output",
                    os.fspath(self.root / "cli-io.tar.gz"),
                    "--repository-root",
                    os.fspath(self.repository),
                ]
            )
            verify_success = verify_main(
                [
                    "--archive",
                    os.fspath(self.exact_one),
                    "--timestamp-utc",
                    TIMESTAMP,
                ]
            )
            verify_invalid = verify_main(
                [
                    "--archive",
                    os.fspath(self.exact_one),
                    "--expected-sha256",
                    "invalid",
                    "--timestamp-utc",
                    TIMESTAMP,
                ]
            )
            verify_integrity = verify_main(
                [
                    "--archive",
                    os.fspath(self.exact_one),
                    "--expected-sha256",
                    "0" * 64,
                    "--timestamp-utc",
                    TIMESTAMP,
                ]
            )
            verify_io = verify_main(
                [
                    "--archive",
                    os.fspath(self.root / "missing.tar.gz"),
                    "--timestamp-utc",
                    TIMESTAMP,
                ]
            )
            with mock.patch.dict(
                os.environ,
                {"MULTISCREEN_EVIDENCE_REVIEWERS": ""},
                clear=False,
            ):
                provenance_invalid = provenance_main(
                    ["--repo", os.fspath(self.repository)]
                )
            provenance_runtime = provenance_main(
                [
                    "--repo",
                    os.fspath(self.root / "not-a-repository"),
                    "--reviewer",
                    "reviewer",
                ]
            )

        self.assertEqual(package_success, 0)
        self.assertEqual(package_invalid, 2)
        self.assertEqual(package_integrity, 3)
        self.assertEqual(package_io, 4)
        self.assertEqual(verify_success, 0)
        self.assertEqual(verify_invalid, 2)
        self.assertEqual(verify_integrity, 3)
        self.assertEqual(verify_io, 4)
        self.assertEqual(provenance_invalid, 2)
        self.assertEqual(provenance_runtime, 4)


if __name__ == "__main__":
    unittest.main()
