from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(REPOSITORY_ROOT))


from scripts import build_level1_evidence as BUILDER
from scripts.package_validation_evidence import package_evidence
from scripts.validation_evidence_common import (
    canonical_json_bytes,
    IntegrityError,
    validate_evidence_document,
)
from scripts.verify_validation_evidence import verify_archive

TESTED_COMMIT = "d" * 40

SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "validation_evidence_v1.schema.json"
TIMESTAMP = "2026-08-09T10:00:00Z"
COMMIT_A = "b" * 40

REVIEW_SPEC = importlib.util.spec_from_file_location(
    "level1_builder_review_fixture",
    REPOSITORY_ROOT / "tests" / "test_level1_evidence_review.py",
)
assert REVIEW_SPEC is not None and REVIEW_SPEC.loader is not None
REVIEW_FIXTURE = importlib.util.module_from_spec(REVIEW_SPEC)
sys.modules[REVIEW_SPEC.name] = REVIEW_FIXTURE
REVIEW_SPEC.loader.exec_module(REVIEW_FIXTURE)


def write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def clean_worktree(timestamp: str = TIMESTAMP) -> dict[str, object]:
    empty = hashlib.sha256(b"").hexdigest()
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
            "sha256": empty,
        },
        "staged_change_count": 0,
        "staged_changes_present": False,
        "status": "recorded",
        "submodules": {
            "byte_count": 0,
            "collected_at_utc": timestamp,
            "command": ["git", "submodule", "status", "--recursive"],
            "count": 0,
            "sha256": empty,
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


def live_git_reader(
    *,
    commit: str,
    branch: str = "validation/level1-core-requalification",
    porcelain: bytes = b"",
    submodules: bytes = b"",
):
    outputs = {
        ("rev-parse", "--show-toplevel"): os.fsencode(REPOSITORY_ROOT) + b"\n",
        ("rev-parse", "--verify", "HEAD"): commit.encode("ascii") + b"\n",
        ("symbolic-ref", "--quiet", "--short", "HEAD"): branch.encode("utf-8")
        + b"\n",
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ): porcelain,
        ("diff", "--check"): b"",
        ("diff", "--cached", "--check"): b"",
        ("submodule", "status", "--recursive"): submodules,
    }

    def read(arguments, *, label):
        del label
        return outputs[tuple(arguments)]

    return read


def commit_blob_reader(
    expected: dict[str, bytes],
    *,
    missing: str | None = None,
    different: str | None = None,
    nonregular: str | None = None,
    tested_commit: str = TESTED_COMMIT,
    parents: tuple[str, ...] | None = None,
    extra: str | None = None,
    rewritten: str | None = None,
    commit_type: bytes = b"commit\n",
):
    object_ids = {
        path: hashlib.sha1(path.encode("utf-8"), usedforsecurity=False).hexdigest()
        for path in expected
    }
    by_object = {object_ids[path]: raw for path, raw in expected.items()}

    def read(arguments, *, label):
        del label
        command = tuple(arguments)
        if command == ("cat-file", "-t", COMMIT_A):
            return commit_type
        if command == ("rev-list", "--parents", "-n", "1", COMMIT_A):
            actual_parents = parents if parents is not None else (tested_commit,)
            fields = (COMMIT_A, *actual_parents)
            return (" ".join(fields) + "\n").encode("ascii")
        if command[0] == "diff-tree":
            paths = list(expected)
            if extra is not None:
                paths.append(extra)
            paths.sort(key=lambda path: path.encode("utf-8"))
            chunks: list[bytes] = []
            for path in paths:
                status = b"M" if path == rewritten else b"A"
                chunks.extend(
                    (
                        status,
                        b"\0",
                        path.encode("utf-8"),
                        b"\0",
                    )
                )
            return b"".join(chunks)
        if command[0] == "ls-tree":
            path = command[-1]
            if path == missing:
                return b""
            mode = "120000" if path == nonregular else "100644"
            return (
                f"{mode} blob {object_ids[path]}\t{path}".encode("utf-8")
                + b"\0"
            )
        if command[:2] == ("cat-file", "blob"):
            object_id = command[2]
            raw = by_object[object_id]
            if different is not None and object_id == object_ids[different]:
                return raw + b"different"
            return raw
        raise AssertionError(f"unexpected Git command: {command!r}")

    return read


class BuilderCase:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run = root / "run"
        self.results = root / "results"
        self.results.mkdir(parents=True)
        self.fixture = REVIEW_FIXTURE.EvidenceFixture(self.run)
        with mock.patch.object(
            REVIEW_FIXTURE.REVIEW,
            "C3_ROW_MANIFEST_SHA256",
            self.fixture.c3_row_manifest_sha256,
        ):
            self.review = REVIEW_FIXTURE.REVIEW.review_inputs(
                **self.fixture.kwargs()
            )
        (self.run / "review").mkdir(mode=0o700)
        self.review_path = self.run / BUILDER.FULL_REVIEW_RELATIVE
        self.review_path.write_bytes(
            REVIEW_FIXTURE.REVIEW._pretty_canonical_bytes(self.review)
        )
        self.provenance = self._provenance(
            head=self.fixture.tested_commit, timestamp=TIMESTAMP
        )
        self.provenance_path = self.run / BUILDER.ACCEPTANCE_PROVENANCE_RELATIVE
        write_canonical(self.provenance_path, self.provenance)
        self.private_dir = root / "private"
        self.sanitized_staging = root / "sanitized-staging"
        self.private_dir.mkdir()
        self.sanitized_staging.mkdir()
        self.exact_archive = self.private_dir / "level1-exact.tar.gz"
        self.sanitized_archive = self.sanitized_staging / "level1-sanitized.tar.gz"
        self.package_report_path = root / "package-report.json"
        self.exact_primary_path = root / "exact-primary.json"
        self.sanitized_primary_path = root / "sanitized-primary.json"

    def _provenance(self, *, head: str, timestamp: str) -> dict[str, object]:
        reviewer = {
            "identifier": "explicit-reviewer",
            "raw_events_reviewed": True,
            "review_commit": self.fixture.tested_commit,
            "review_method": "manual raw-event and lossless-log review",
            "reviewed_at_utc": TIMESTAMP,
            "role": "evidence_reviewer",
        }
        return {
            "acceptance_review": {
                "reviewers": [reviewer],
                "status": "recorded",
            },
            "collected_at_utc": timestamp,
            "context": "evidence_handoff",
            "format_version": "validation-provenance-v1",
            "repository": {
                "branch": {
                    "status": "recorded",
                    "value": "validation/level1-core-requalification",
                },
                "detached_head": False,
                "head_commit": head,
                "name": "Multiscreen-HF",
                "remotes": [],
                "root_kind": "git_worktree",
                "worktree": clean_worktree(timestamp),
            },
        }

    def prepare(self) -> dict[str, object]:
        with mock.patch.object(
            BUILDER, "CANONICAL_RESULTS_ROOT", self.results
        ), mock.patch.object(BUILDER, "_validate_live_repository_state"):
            return BUILDER.prepare_evidence(
                run_root_value=self.run,
                results_root_value=self.results,
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
        write_canonical(self.package_report_path, report)
        by_kind = {item["archive_kind"]: item for item in report["archives"]}
        exact_primary = verify_archive(
            self.exact_archive,
            expected_sha256=by_kind["exact_private"]["sha256"],
            verification_timestamp_utc=TIMESTAMP,
        )
        sanitized_primary = verify_archive(
            self.sanitized_archive,
            expected_sha256=by_kind["sanitized_shareable"]["sha256"],
            verification_timestamp_utc=TIMESTAMP,
        )
        write_canonical(self.exact_primary_path, exact_primary)
        write_canonical(self.sanitized_primary_path, sanitized_primary)
        return report

    def seal(self, *, staging: Path | None = None) -> dict[str, object]:
        with mock.patch.object(
            BUILDER, "CANONICAL_RESULTS_ROOT", self.results
        ), mock.patch.object(
            BUILDER, "_validate_implementation_base_commit",
            return_value=BUILDER.IMPLEMENTATION_BASE_COMMIT,
        ):
            return BUILDER.seal_evidence(
                run_root_value=self.run,
                results_root_value=self.results,
                schema_value=SCHEMA_PATH,
                package_report_value=self.package_report_path,
                exact_archive_value=self.exact_archive,
                sanitized_archive_value=self.sanitized_archive,
                sanitized_staging_dir_value=staging or self.sanitized_staging,
                exact_primary_report_value=self.exact_primary_path,
                sanitized_primary_report_value=self.sanitized_primary_path,
                implementation_base_commit=BUILDER.IMPLEMENTATION_BASE_COMMIT,
                exact_storage_locator="private-external:level1/test",
                sanitized_storage_locator="sanitized-staging:level1/test",
                verification_timestamp_utc=TIMESTAMP,
            )

    def commit_provenance(self, *, dirty: bool = False) -> Path:
        value = self._provenance(
            head=COMMIT_A, timestamp="2026-08-09T11:00:00Z"
        )
        if dirty:
            value["repository"]["worktree"].update(
                {"clean": False, "untracked_path_count": 1}
            )
        path = self.root / "commit-a-provenance.json"
        write_canonical(path, value)
        return path

    def close(self, provenance: Path | None = None) -> dict[str, object]:
        with mock.patch.object(
            BUILDER, "CANONICAL_RESULTS_ROOT", self.results
        ), mock.patch.object(
            BUILDER, "_validate_commit_evidence_blobs"
        ), mock.patch.object(
            BUILDER, "_validate_live_repository_state"
        ), mock.patch.object(
            BUILDER, "_validate_implementation_base_commit",
            return_value=BUILDER.IMPLEMENTATION_BASE_COMMIT,
        ):
            return BUILDER.close_evidence(
                run_root_value=self.run,
                results_root_value=self.results,
                schema_value=SCHEMA_PATH,
                commit_provenance_value=provenance or self.commit_provenance(),
                package_report_value=self.package_report_path,
                implementation_base_commit=BUILDER.IMPLEMENTATION_BASE_COMMIT,
                exact_storage_locator="private-external:level1/test",
                sanitized_storage_locator="sanitized-staging:level1/test",
                commit_a=COMMIT_A,
                exact_archive_value=self.exact_archive,
                sanitized_archive_value=self.sanitized_archive,
                sanitized_staging_dir_value=self.sanitized_staging,
                verification_timestamp_utc=TIMESTAMP,
            )


class Level1EvidenceBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case_index = 0

    def new_case(self) -> BuilderCase:
        self.case_index += 1
        return BuilderCase(self.root / f"case-{self.case_index}")

    def test_prepare_has_complete_deduplicated_fixed_allowlist(self) -> None:
        case = self.new_case()
        result = case.prepare()
        self.assertEqual(len(case.review["aggregate"]["artifact_hashes"]), 130)
        self.assertEqual(
            BUILDER.REQUIRED_COMMAND_NAMES,
            REVIEW_FIXTURE.REVIEW.REQUIRED_COMMAND_NAMES,
        )
        self.assertEqual(set(case.review["aggregate"]["artifact_hashes"]), set(BUILDER._fixed_specs()))
        self.assertEqual(result["artifact_count"], 133)
        package = json.loads(
            (case.run / BUILDER.PACKAGE_INPUT_RELATIVE).read_text(encoding="utf-8")
        )
        entries = package["artifacts"]
        self.assertEqual(len(entries), 133)
        self.assertEqual(len({item["logical_name"] for item in entries}), 133)
        self.assertEqual(len({item["archive_path"] for item in entries}), 133)
        self.assertEqual(
            len({(item["source_root"], item["source_path"]) for item in entries}),
            133,
        )
        self.assertEqual(
            sum(item["source_path"] == "logs/p0-3-checkpointed.log" for item in entries),
            1,
        )
        serialized = canonical_json_bytes(package).decode("utf-8")
        for forbidden in (
            BUILDER.DESCRIPTOR_NAME,
            BUILDER.EXACT_VERIFICATION_NAME,
            BUILDER.SANITIZED_VERIFICATION_NAME,
            "package-report.json",
            "descriptor-verification",
            ".tar.gz",
            ".safetensors",
            ".bin",
        ):
            self.assertNotIn(forbidden, serialized)
        for item in entries:
            self.assertNotIn("checkpoint", Path(item["source_path"]).parts)
            self.assertFalse(any(part.startswith("checkpoint-") for part in Path(item["source_path"]).parts))
        for path in (
            case.results / BUILDER.SUMMARY_JSON_NAME,
            case.results / BUILDER.SUMMARY_MARKDOWN_NAME,
            case.run / BUILDER.PACKAGE_INPUT_RELATIVE,
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        summary_text = (case.results / BUILDER.SUMMARY_JSON_NAME).read_text()
        markdown_text = (case.results / BUILDER.SUMMARY_MARKDOWN_NAME).read_text()
        self.assertNotIn(os.fspath(case.root), summary_text + markdown_text)
        first, first_markdown = BUILDER._summary_documents(
            case.review,
            case.provenance["acceptance_review"]["reviewers"],
        )
        second, second_markdown = BUILDER._summary_documents(
            case.review,
            case.provenance["acceptance_review"]["reviewers"],
        )
        self.assertEqual(first, second)
        self.assertEqual(first_markdown, second_markdown)

    def test_full_prepare_seal_close_cycle_is_schema_valid_and_stable(self) -> None:
        case = self.new_case()
        case.prepare()
        case.package()
        seal = case.seal()
        partial_path = case.results / BUILDER.DESCRIPTOR_NAME
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(validate_evidence_document(partial, schema), [])
        self.assertEqual(partial["evidence_status"], "partial")
        self.assertEqual(len(partial["source_artifacts"]), 133)
        exact_report = case.results / BUILDER.EXACT_VERIFICATION_NAME
        sanitized_report = case.results / BUILDER.SANITIZED_VERIFICATION_NAME
        self.assertEqual(
            seal["exact_verification_report_sha256"],
            hashlib.sha256(exact_report.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            seal["sanitized_verification_report_sha256"],
            hashlib.sha256(sanitized_report.read_bytes()).hexdigest(),
        )
        close = case.close()
        complete_path = partial_path
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        self.assertEqual(validate_evidence_document(complete, schema), [])
        self.assertEqual(complete["evidence_status"], "complete")
        self.assertEqual(
            complete["evidence_handoff_provenance"]["final_commit"]["value"],
            COMMIT_A,
        )
        self.assertEqual(
            complete["original_run_provenance"]["original_run_review"]["status"],
            "not_applicable",
        )
        self.assertEqual(
            close["exact_verification_report_sha256"],
            seal["exact_verification_report_sha256"],
        )
        self.assertEqual(
            close["sanitized_verification_report_sha256"],
            seal["sanitized_verification_report_sha256"],
        )
        for report_path in (exact_report, sanitized_report):
            self.assertNotIn(os.fspath(case.root), report_path.read_text())
        self.assertEqual(len(list(case.results.glob("*.complete.json"))), 0)
        for path in (complete_path, exact_report, sanitized_report):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_close_reconstructs_and_rejects_all_seal_invariant_tampering(self) -> None:
        case = self.new_case()
        case.prepare()
        case.package()
        case.seal()
        descriptor_path = case.results / BUILDER.DESCRIPTOR_NAME
        partial_raw = descriptor_path.read_bytes()
        original = json.loads(partial_raw)
        mutations = (
            ("reviewer identifier", ("acceptance_review", "reviewers", 0, "identifier"), "forged-reviewer"),
            ("raw event review", ("acceptance_review", "reviewers", 0, "raw_events_reviewed"), False),
            ("review commit", ("acceptance_review", "reviewers", 0, "review_commit"), "c" * 40),
            ("implementation base", ("evidence_handoff_provenance", "implementation_base_commit"), "c" * 40),
            ("tested branch", ("tested_source", "branch"), "forged/branch"),
            ("storage locator", ("archives", "exact_private", "storage_locator"), "private-external:forged"),
            ("validation status", ("validation_status",), "failed"),
        )
        for name, path, value in mutations:
            with self.subTest(name=name):
                descriptor = json.loads(json.dumps(original))
                cursor = descriptor
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                write_canonical(descriptor_path, descriptor)
                tampered_raw = descriptor_path.read_bytes()
                with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "reconstructed seal boundary"):
                    case.close()
                self.assertEqual(descriptor_path.read_bytes(), tampered_raw)
                descriptor_path.write_bytes(partial_raw)

    def test_prepare_rejects_artifact_tampering_and_failed_review(self) -> None:
        tampered = self.new_case()
        path = tampered.run / "logs" / "formula-units.log"
        path.write_bytes(path.read_bytes() + b"tampered\n")
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "changed after review"):
            tampered.prepare()

        failed = self.new_case()
        failed.review["status"] = "failed"
        write_canonical(failed.review_path, failed.review)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "not a passed"):
            failed.prepare()

    def test_prepare_rejects_unreviewed_or_dirty_acceptance_provenance(self) -> None:
        unreviewed = self.new_case()
        unreviewed.provenance["acceptance_review"]["reviewers"][0][
            "raw_events_reviewed"
        ] = False
        write_canonical(unreviewed.provenance_path, unreviewed.provenance)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "every raw event"):
            unreviewed.prepare()

        dirty = self.new_case()
        dirty.provenance["repository"]["worktree"].update(
            {"clean": False, "untracked_path_count": 1}
        )
        write_canonical(dirty.provenance_path, dirty.provenance)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "not a recorded clean"):
            dirty.prepare()

    def test_paths_symlinks_hardlinks_and_overwrites_are_rejected(self) -> None:
        relative = self.new_case()
        with mock.patch.object(BUILDER, "CANONICAL_RESULTS_ROOT", relative.results):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "absolute path"):
                BUILDER.prepare_evidence(
                    run_root_value=Path("relative-run"),
                    results_root_value=relative.results,
                )

        linked_root = self.new_case()
        symlink = linked_root.root / "run-link"
        symlink.symlink_to(linked_root.run, target_is_directory=True)
        with mock.patch.object(BUILDER, "CANONICAL_RESULTS_ROOT", linked_root.results):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "symlink"):
                BUILDER.prepare_evidence(
                    run_root_value=symlink,
                    results_root_value=linked_root.results,
                )

        hardlinked = self.new_case()
        os.link(hardlinked.review_path, hardlinked.root / "review-alias.json")
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "hard links"):
            hardlinked.prepare()

        overwrite = self.new_case()
        (overwrite.results / BUILDER.SUMMARY_JSON_NAME).write_text("existing\n")
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "overwrite"):
            overwrite.prepare()

    def test_canonical_repository_results_and_schema_are_fixed(self) -> None:
        case = self.new_case()
        with self.assertRaisesRegex(
            BUILDER.EvidenceBuildError, "canonical repository validation-results"
        ):
            BUILDER._validated_roots(case.run, case.results)

        alternate_schema = case.root / "alternate-schema.json"
        alternate_schema.write_bytes(SCHEMA_PATH.read_bytes())
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "canonical repository"):
            BUILDER._load_schema(alternate_schema)

        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "outside the canonical"):
            BUILDER._validated_roots(
                REPOSITORY_ROOT,
                BUILDER.CANONICAL_RESULTS_ROOT,
            )

        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "filesystem root"):
            BUILDER._canonical_root(Path("/"), label="run root")
        with self.assertRaisesRegex(
            BUILDER.EvidenceBuildError, "directly under a filesystem root"
        ):
            BUILDER._canonical_file(
                Path("/level1-exact-private.tar.gz"),
                label="exact archive",
            )

        fake_worktree = case.root / "registered-worktree"
        nested_run = fake_worktree / "run"
        nested_run.mkdir(parents=True)
        with mock.patch.object(
            BUILDER, "CANONICAL_RESULTS_ROOT", case.results
        ), mock.patch.object(
            BUILDER, "_registered_worktrees", return_value=(fake_worktree,)
        ):
            with self.assertRaisesRegex(
                BUILDER.EvidenceBuildError, "outside every registered Git worktree"
            ):
                BUILDER._validated_roots(nested_run, case.results)

    def test_live_git_state_rejects_forged_clean_provenance_and_identity_drift(self) -> None:
        collector = clean_worktree()
        expected_branch = "validation/level1-core-requalification"
        with mock.patch.object(
            BUILDER,
            "_git_stdout",
            side_effect=live_git_reader(commit=COMMIT_A),
        ):
            BUILDER._validate_live_repository_state(
                expected_commit=COMMIT_A,
                expected_branch=expected_branch,
                collector_worktree=collector,
                phase="test",
            )

        with mock.patch.object(
            BUILDER,
            "_git_stdout",
            side_effect=live_git_reader(
                commit=COMMIT_A,
                porcelain=b"?? unrecorded-output\n",
            ),
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "not clean"):
                BUILDER._validate_live_repository_state(
                    expected_commit=COMMIT_A,
                    expected_branch=expected_branch,
                    collector_worktree=collector,
                    phase="test",
                )

        with mock.patch.object(
            BUILDER,
            "_git_stdout",
            side_effect=live_git_reader(commit="c" * 40),
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "HEAD differs"):
                BUILDER._validate_live_repository_state(
                    expected_commit=COMMIT_A,
                    expected_branch=expected_branch,
                    collector_worktree=collector,
                    phase="test",
                )

        with mock.patch.object(
            BUILDER,
            "_git_stdout",
            side_effect=live_git_reader(commit=COMMIT_A, branch="wrong-branch"),
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "branch differs"):
                BUILDER._validate_live_repository_state(
                    expected_commit=COMMIT_A,
                    expected_branch=expected_branch,
                    collector_worktree=collector,
                    phase="test",
                )


    def test_implementation_base_is_fixed_commit_object_and_tested_ancestor(self) -> None:
        outputs = {
            ("cat-file", "-t", BUILDER.IMPLEMENTATION_BASE_COMMIT): b"commit\n",
            ("cat-file", "-t", TESTED_COMMIT): b"commit\n",
            (
                "merge-base", "--is-ancestor", BUILDER.IMPLEMENTATION_BASE_COMMIT,
                TESTED_COMMIT,
            ): b"",
        }

        def read(arguments, *, label):
            del label
            return outputs[tuple(arguments)]

        with mock.patch.object(BUILDER, "_git_stdout", side_effect=read):
            self.assertEqual(
                BUILDER._validate_implementation_base_commit(
                    BUILDER.IMPLEMENTATION_BASE_COMMIT, tested_commit=TESTED_COMMIT
                ),
                BUILDER.IMPLEMENTATION_BASE_COMMIT,
            )

        with mock.patch.object(BUILDER, "_git_stdout") as git:
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "fixed Stage 5 base"):
                BUILDER._validate_implementation_base_commit(
                    "c" * 40, tested_commit=TESTED_COMMIT
                )
            git.assert_not_called()

        with mock.patch.object(BUILDER, "_git_stdout", return_value=b"blob\n"):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "not a commit object"):
                BUILDER._validate_implementation_base_commit(
                    BUILDER.IMPLEMENTATION_BASE_COMMIT, tested_commit=TESTED_COMMIT
                )

        def no_ancestor(arguments, *, label):
            if tuple(arguments)[0] == "merge-base":
                raise BUILDER.EvidenceBuildError("not an ancestor")
            return read(arguments, label=label)

        with mock.patch.object(BUILDER, "_git_stdout", side_effect=no_ancestor):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "not an ancestor"):
                BUILDER._validate_implementation_base_commit(
                    BUILDER.IMPLEMENTATION_BASE_COMMIT, tested_commit=TESTED_COMMIT
                )

    def test_commit_a_must_contain_exact_regular_evidence_blobs(self) -> None:
        prefix = "docs/validation_results"
        expected = {
            f"{prefix}/{BUILDER.DESCRIPTOR_NAME}": b"partial descriptor\n",
            f"{prefix}/{BUILDER.EXACT_VERIFICATION_NAME}": b"exact report\n",
            f"{prefix}/{BUILDER.SANITIZED_VERIFICATION_NAME}": b"sanitized report\n",
            f"{prefix}/{BUILDER.SUMMARY_JSON_NAME}": b"summary json\n",
            f"{prefix}/{BUILDER.SUMMARY_MARKDOWN_NAME}": b"summary markdown\n",
        }
        with mock.patch.object(
            BUILDER,
            "_git_stdout",
            side_effect=commit_blob_reader(expected),
        ):
            BUILDER._validate_commit_evidence_blobs(
                commit=COMMIT_A,
                tested_commit=TESTED_COMMIT,
                expected_blobs=expected,
            )

        wrong_expected = dict(expected)
        wrong_expected.pop(next(iter(wrong_expected)))
        with mock.patch.object(BUILDER, "_git_stdout") as git:
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "fixed five"):
                BUILDER._validate_commit_evidence_blobs(
                    commit=COMMIT_A,
                    tested_commit=TESTED_COMMIT,
                    expected_blobs=wrong_expected,
                )
            git.assert_not_called()

        target = f"{prefix}/{BUILDER.DESCRIPTOR_NAME}"
        with mock.patch.object(
            BUILDER,
            "_git_stdout",
            side_effect=commit_blob_reader(expected, missing=target),
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "missing"):
                BUILDER._validate_commit_evidence_blobs(
                    commit=COMMIT_A,
                    tested_commit=TESTED_COMMIT,
                    expected_blobs=expected,
                )

        with mock.patch.object(
            BUILDER,
            "_git_stdout",
            side_effect=commit_blob_reader(expected, different=target),
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "bytes differ"):
                BUILDER._validate_commit_evidence_blobs(
                    commit=COMMIT_A,
                    tested_commit=TESTED_COMMIT,
                    expected_blobs=expected,
                )

        with mock.patch.object(
            BUILDER,
            "_git_stdout",
            side_effect=commit_blob_reader(expected, nonregular=target),
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "regular blob"):
                BUILDER._validate_commit_evidence_blobs(
                    commit=COMMIT_A,
                    tested_commit=TESTED_COMMIT,
                    expected_blobs=expected,
                )

        boundary_cases = (
            ("extra private file", {"extra": "private/raw-evidence.txt"}, "exactly the five"),
            ("rewritten file", {"rewritten": target}, "without rewriting"),
            ("wrong parent", {"parents": ("c" * 40,)}, "single parent"),
            ("merge commit", {"parents": (TESTED_COMMIT, "c" * 40)}, "single parent"),
            ("non-commit object", {"commit_type": b"tree\n"}, "not a commit object"),
        )
        for name, kwargs, message in boundary_cases:
            with self.subTest(name=name), mock.patch.object(
                BUILDER,
                "_git_stdout",
                side_effect=commit_blob_reader(expected, **kwargs),
            ):
                with self.assertRaisesRegex(BUILDER.EvidenceBuildError, message):
                    BUILDER._validate_commit_evidence_blobs(
                        commit=COMMIT_A,
                        tested_commit=TESTED_COMMIT,
                        expected_blobs=expected,
                    )

    def test_close_preserves_partial_when_commit_a_blob_binding_fails(self) -> None:
        case = self.new_case()
        case.prepare()
        case.package()
        case.seal()
        descriptor_path = case.results / BUILDER.DESCRIPTOR_NAME
        partial_raw = descriptor_path.read_bytes()
        provenance = case.commit_provenance()
        with mock.patch.object(
            BUILDER, "CANONICAL_RESULTS_ROOT", case.results
        ), mock.patch.object(
            BUILDER,
            "_validate_commit_evidence_blobs",
            side_effect=BUILDER.EvidenceBuildError("commit blob differs"),
        ), mock.patch.object(
            BUILDER, "_validate_live_repository_state"
        ), mock.patch.object(
            BUILDER, "_validate_implementation_base_commit",
            return_value=BUILDER.IMPLEMENTATION_BASE_COMMIT,
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "commit blob"):
                BUILDER.close_evidence(
                    run_root_value=case.run,
                    results_root_value=case.results,
                    schema_value=SCHEMA_PATH,
                    commit_provenance_value=provenance,
                    package_report_value=case.package_report_path,
                    implementation_base_commit=BUILDER.IMPLEMENTATION_BASE_COMMIT,
                    exact_storage_locator="private-external:level1/test",
                    sanitized_storage_locator="sanitized-staging:level1/test",
                    commit_a=COMMIT_A,
                    exact_archive_value=case.exact_archive,
                    sanitized_archive_value=case.sanitized_archive,
                    sanitized_staging_dir_value=case.sanitized_staging,
                    verification_timestamp_utc=TIMESTAMP,
                )
        self.assertEqual(descriptor_path.read_bytes(), partial_raw)
        self.assertEqual(
            json.loads(partial_raw)["evidence_status"],
            "partial",
        )

    def test_prepare_and_close_fail_before_publish_on_live_git_rejection(self) -> None:
        prepare_case = self.new_case()
        with mock.patch.object(
            BUILDER, "CANONICAL_RESULTS_ROOT", prepare_case.results
        ), mock.patch.object(
            BUILDER,
            "_validate_live_repository_state",
            side_effect=BUILDER.EvidenceBuildError("live state rejected"),
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "live state"):
                BUILDER.prepare_evidence(
                    run_root_value=prepare_case.run,
                    results_root_value=prepare_case.results,
                )
        self.assertFalse(
            (prepare_case.results / BUILDER.SUMMARY_JSON_NAME).exists()
        )
        self.assertFalse(
            (prepare_case.run / BUILDER.PACKAGE_INPUT_RELATIVE).exists()
        )

        close_case = self.new_case()
        close_case.prepare()
        close_case.package()
        close_case.seal()
        descriptor_path = close_case.results / BUILDER.DESCRIPTOR_NAME
        partial_raw = descriptor_path.read_bytes()
        provenance = close_case.commit_provenance()
        with mock.patch.object(
            BUILDER, "CANONICAL_RESULTS_ROOT", close_case.results
        ), mock.patch.object(
            BUILDER, "_validate_commit_evidence_blobs"
        ), mock.patch.object(
            BUILDER,
            "_validate_live_repository_state",
            side_effect=BUILDER.EvidenceBuildError("live state rejected"),
        ), mock.patch.object(
            BUILDER, "_validate_implementation_base_commit",
            return_value=BUILDER.IMPLEMENTATION_BASE_COMMIT,
        ):
            with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "live state"):
                BUILDER.close_evidence(
                    run_root_value=close_case.run,
                    results_root_value=close_case.results,
                    schema_value=SCHEMA_PATH,
                    commit_provenance_value=provenance,
                    package_report_value=close_case.package_report_path,
                    implementation_base_commit=BUILDER.IMPLEMENTATION_BASE_COMMIT,
                    exact_storage_locator="private-external:level1/test",
                    sanitized_storage_locator="sanitized-staging:level1/test",
                    commit_a=COMMIT_A,
                    exact_archive_value=close_case.exact_archive,
                    sanitized_archive_value=close_case.sanitized_archive,
                    sanitized_staging_dir_value=close_case.sanitized_staging,
                    verification_timestamp_utc=TIMESTAMP,
                )
        self.assertEqual(descriptor_path.read_bytes(), partial_raw)

    def test_archive_locations_are_enforced_from_actual_paths(self) -> None:
        staging_mismatch = self.new_case()
        staging_mismatch.prepare()
        staging_mismatch.package()
        wrong_staging = staging_mismatch.root / "wrong-staging"
        wrong_staging.mkdir()
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "direct child"):
            staging_mismatch.seal(staging=wrong_staging)

        repository_case = self.new_case()
        repository_case.prepare()
        repository_case.package()
        fake_repository = repository_case.root / "fake-repository"
        fake_repository.mkdir()
        bad_exact = fake_repository / repository_case.exact_archive.name
        bad_exact.write_bytes(repository_case.exact_archive.read_bytes())
        repository_case.exact_archive = bad_exact
        with mock.patch.object(
            BUILDER, "CANONICAL_REPOSITORY_ROOT", fake_repository
        ), mock.patch.object(
            BUILDER, "_registered_worktrees", return_value=(fake_repository,)
        ):
            with self.assertRaisesRegex(
                BUILDER.EvidenceBuildError,
                "outside the canonical repository and every Git worktree",
            ):
                repository_case.seal()

        worktree_case = self.new_case()
        worktree_case.prepare()
        worktree_case.package()
        fake_worktree = worktree_case.root / "registered-worktree"
        fake_worktree.mkdir()
        bad_exact = fake_worktree / worktree_case.exact_archive.name
        bad_exact.write_bytes(worktree_case.exact_archive.read_bytes())
        worktree_case.exact_archive = bad_exact
        with mock.patch.object(
            BUILDER, "_registered_worktrees", return_value=(fake_worktree,)
        ):
            with self.assertRaisesRegex(
                BUILDER.EvidenceBuildError,
                "outside the canonical repository and every Git worktree",
            ):
                worktree_case.seal()

        nested_exact_case = self.new_case()
        nested_exact_case.prepare()
        nested_exact_case.package()
        private_nested = nested_exact_case.sanitized_staging / "private-nested"
        private_nested.mkdir()
        bad_exact = private_nested / nested_exact_case.exact_archive.name
        bad_exact.write_bytes(nested_exact_case.exact_archive.read_bytes())
        nested_exact_case.exact_archive = bad_exact
        with self.assertRaisesRegex(
            BUILDER.EvidenceBuildError,
            "retained separately from sanitized staging",
        ):
            nested_exact_case.seal()

        sanitized_case = self.new_case()
        sanitized_case.prepare()
        sanitized_case.package()
        fake_worktree = sanitized_case.root / "sanitized-worktree"
        bad_staging = fake_worktree / "staging"
        bad_staging.mkdir(parents=True)
        bad_sanitized = bad_staging / sanitized_case.sanitized_archive.name
        bad_sanitized.write_bytes(sanitized_case.sanitized_archive.read_bytes())
        sanitized_case.sanitized_archive = bad_sanitized
        sanitized_case.sanitized_staging = bad_staging
        with mock.patch.object(
            BUILDER, "_registered_worktrees", return_value=(fake_worktree,)
        ):
            with self.assertRaisesRegex(
                BUILDER.EvidenceBuildError,
                "outside the canonical repository and every Git worktree",
            ):
                sanitized_case.seal()

    def test_atomic_replace_requires_the_exact_expected_bytes(self) -> None:
        path = self.root / "atomic-descriptor.json"
        path.write_bytes(b"before\n")
        path.chmod(0o600)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "changed before"):
            BUILDER._atomic_replace_expected(
                path,
                expected_raw=b"wrong\n",
                replacement_raw=b"after\n",
                label="test descriptor",
            )
        self.assertEqual(path.read_bytes(), b"before\n")
        BUILDER._atomic_replace_expected(
            path,
            expected_raw=b"before\n",
            replacement_raw=b"after\n",
            label="test descriptor",
        )
        self.assertEqual(path.read_bytes(), b"after\n")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_package_set_mismatch_and_duplicate_alias_are_rejected(self) -> None:
        missing = self.new_case()
        missing.prepare()
        package_path = missing.run / BUILDER.PACKAGE_INPUT_RELATIVE
        document = json.loads(package_path.read_text())
        document["artifacts"].pop()
        write_canonical(package_path, document)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "source set differs"):
            BUILDER._validated_package_input(
                missing.run,
                missing.results,
                tested_commit=missing.fixture.tested_commit,
            )

        duplicate = self.new_case()
        duplicate.prepare()
        package_path = duplicate.run / BUILDER.PACKAGE_INPUT_RELATIVE
        document = json.loads(package_path.read_text())
        document["artifacts"][-1]["source_root"] = document["artifacts"][0][
            "source_root"
        ]
        document["artifacts"][-1]["source_path"] = document["artifacts"][0][
            "source_path"
        ]
        write_canonical(package_path, document)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "duplicate package source alias"):
            BUILDER._validated_package_input(
                duplicate.run,
                duplicate.results,
                tested_commit=duplicate.fixture.tested_commit,
            )

    def test_seal_rejects_archive_primary_report_and_package_report_tampering(self) -> None:
        archive_case = self.new_case()
        archive_case.prepare()
        archive_case.package()
        archive_case.exact_archive.write_bytes(
            archive_case.exact_archive.read_bytes() + b"tampered"
        )
        with self.assertRaises(IntegrityError):
            archive_case.seal()

        report_case = self.new_case()
        report_case.prepare()
        report_case.package()
        report = json.loads(report_case.sanitized_primary_path.read_text())
        report["verified_at_utc"] = "2026-08-09T10:00:01Z"
        write_canonical(report_case.sanitized_primary_path, report)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "differs from independent"):
            report_case.seal()

        package_case = self.new_case()
        package_case.prepare()
        package_case.package()
        report = json.loads(package_case.package_report_path.read_text())
        report["archives"][0]["archive_filename"] = "wrong.tar.gz"
        write_canonical(package_case.package_report_path, report)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "filename differs"):
            package_case.seal()

        future_case = self.new_case()
        future_case.prepare()
        future_case.package()
        report = json.loads(future_case.package_report_path.read_text())
        for archive in report["archives"]:
            archive["created_at_utc"] = "2026-08-09T10:00:01Z"
        write_canonical(future_case.package_report_path, report)
        with self.assertRaisesRegex(
            BUILDER.EvidenceBuildError, "creation timestamp is after"
        ):
            future_case.seal()

        output_case = self.new_case()
        output_case.prepare()
        output_case.package()
        report_path = output_case.results / BUILDER.EXACT_VERIFICATION_NAME
        report_path.write_bytes(b"existing report\n")
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "overwrite"):
            output_case.seal()
        self.assertEqual(report_path.read_bytes(), b"existing report\n")
        self.assertFalse(
            (output_case.results / BUILDER.SANITIZED_VERIFICATION_NAME).exists()
        )
        self.assertFalse((output_case.results / BUILDER.DESCRIPTOR_NAME).exists())

    def test_close_rejects_dirty_commit_report_and_descriptor_hash_drift(self) -> None:
        dirty = self.new_case()
        dirty.prepare()
        dirty.package()
        dirty.seal()
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "not a recorded clean"):
            dirty.close(dirty.commit_provenance(dirty=True))

        report_drift = self.new_case()
        report_drift.prepare()
        report_drift.package()
        report_drift.seal()
        path = report_drift.results / BUILDER.EXACT_VERIFICATION_NAME
        report = json.loads(path.read_text())
        report["verifier_version"] = "9.9.9"
        write_canonical(path, report)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "report hash drift"):
            report_drift.close()

        descriptor_drift = self.new_case()
        descriptor_drift.prepare()
        descriptor_drift.package()
        descriptor_drift.seal()
        path = descriptor_drift.results / BUILDER.DESCRIPTOR_NAME
        descriptor = json.loads(path.read_text())
        descriptor["archives"]["exact_private"][
            "verification_report_sha256"
        ] = "f" * 64
        write_canonical(path, descriptor)
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "report hash drift"):
            descriptor_drift.close()

        private_report = self.new_case()
        private_report.prepare()
        private_report.package()
        private_report.seal()
        path = private_report.results / BUILDER.EXACT_VERIFICATION_NAME
        report = json.loads(path.read_text())
        report["private_probe"] = "/srv/private/evidence"
        write_canonical(path, report)
        with self.assertRaisesRegex(
            BUILDER.EvidenceBuildError, "private path or credential"
        ):
            private_report.close()

    def test_close_atomically_replaces_the_canonical_partial_descriptor(self) -> None:
        case = self.new_case()
        case.prepare()
        case.package()
        case.seal()
        provenance = case.commit_provenance()
        case.close(provenance)
        descriptor_path = case.results / BUILDER.DESCRIPTOR_NAME
        descriptor = json.loads(descriptor_path.read_text())
        self.assertEqual(descriptor["evidence_status"], "complete")
        with self.assertRaisesRegex(BUILDER.EvidenceBuildError, "seal closure boundary"):
            case.close(provenance)
        self.assertEqual(json.loads(descriptor_path.read_text()), descriptor)

    def test_sanitization_projection_is_minimal_and_verified(self) -> None:
        case = self.new_case()
        case.prepare()
        case.package()
        report = json.loads(case.sanitized_primary_path.read_text())
        projection = report["checks"]["sanitization"]["descriptor_values"]
        self.assertEqual(
            set(projection),
            {
                "files_scanned",
                "replacement_count",
                "report_sha256",
                "rules_applied",
            },
        )
        self.assertEqual(projection["files_scanned"], 133)
        self.assertRegex(projection["report_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
