"""Adversarial tests for validation-provenance collector hardening."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.collect_validation_provenance import (  # noqa: E402
    PROVENANCE_FORMAT_VERSION,
    collect_provenance,
    collect_worktree,
    main as provenance_main,
    redact_remote_url,
)


TIMESTAMP = "2026-08-05T00:00:00Z"
TOKEN = "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
FINE_GRAINED_TOKEN = "github_pat_11AAAAAAAAAAAAAAAAAAAA_BBBBBBBBBBBBBBBBBBBB"


def _git(repo: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _initialize_git_repository(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Synthetic Reviewer")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    (repo / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "initial fixture")
    return repo


class CollectorReviewContractTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = _initialize_git_repository(Path(temporary.name), "repository")
        self.head = _git(self.repo, "rev-parse", "HEAD").decode("ascii").strip()

    def _kwargs(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "repo": self.repo,
            "reviewers": ["explicit-reviewer"],
            "review_method": "manual raw-event review",
            "review_commit": self.head,
            "raw_events_reviewed": True,
            "timestamp_utc": TIMESTAMP,
            "env_value": "",
        }
        values.update(overrides)
        return values

    def _cli_args(self, **overrides: object) -> list[str]:
        values: dict[str, object] = {
            "repo": os.fspath(self.repo),
            "reviewer": "explicit-reviewer",
            "review_method": "manual raw-event review",
            "review_commit": self.head,
            "raw_events_reviewed": "true",
            "timestamp_utc": TIMESTAMP,
        }
        values.update(overrides)
        arguments = ["--repo", str(values["repo"])]
        for key, option in (
            ("reviewer", "--reviewer"),
            ("review_method", "--review-method"),
            ("review_commit", "--review-commit"),
            ("raw_events_reviewed", "--raw-events-reviewed"),
            ("timestamp_utc", "--timestamp-utc"),
        ):
            value = values[key]
            if value is not None:
                arguments.extend((option, str(value)))
        return arguments

    def _run_main(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.dict(
            os.environ,
            {"MULTISCREEN_EVIDENCE_REVIEWERS": ""},
            clear=False,
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = provenance_main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_recorded_review_has_only_schema_reviewer_fields(self) -> None:
        provenance = collect_provenance(**self._kwargs())
        reviewer = provenance["acceptance_review"]["reviewers"][0]
        self.assertEqual(
            set(reviewer),
            {
                "identifier",
                "raw_events_reviewed",
                "review_commit",
                "review_method",
                "reviewed_at_utc",
                "role",
            },
        )
        self.assertEqual(reviewer["identifier"], "explicit-reviewer")
        self.assertEqual(reviewer["role"], "evidence_reviewer")
        self.assertTrue(reviewer["raw_events_reviewed"])
        self.assertEqual(provenance["format_version"], PROVENANCE_FORMAT_VERSION)
        self.assertNotIn("schema_version", provenance)
        self.assertEqual(
            provenance["repository"]["worktree"]["collected_at_utc"],
            TIMESTAMP,
        )

    def test_full_sha1_and_sha256_review_commits_are_accepted(self) -> None:
        for value in ("A" * 40, "B" * 64):
            with self.subTest(length=len(value)):
                provenance = collect_provenance(
                    **self._kwargs(review_commit=value)
                )
                reviewer = provenance["acceptance_review"]["reviewers"][0]
                self.assertEqual(reviewer["review_commit"], value.lower())

    def test_each_recorded_review_input_is_required(self) -> None:
        cases = (
            ("reviewers", "reviewer"),
            ("review_method", "review_method"),
            ("review_commit", "review_commit"),
            ("raw_events_reviewed", "raw_events_reviewed"),
        )
        for keyword, cli_keyword in cases:
            with self.subTest(field=keyword):
                with self.assertRaises(ValueError):
                    collect_provenance(**self._kwargs(**{keyword: None}))
                code, stdout, stderr = self._run_main(
                    self._cli_args(**{cli_keyword: None})
                )
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("error:", stderr)

    def test_empty_method_and_short_commit_are_usage_errors(self) -> None:
        invalid_values = (
            ("review_method", "review_method", ""),
            ("review_method", "review_method", "   "),
            ("review_commit", "review_commit", self.head[:7]),
        )
        for keyword, cli_keyword, value in invalid_values:
            with self.subTest(field=keyword, value=value):
                with self.assertRaises(ValueError):
                    collect_provenance(**self._kwargs(**{keyword: value}))
                code, _stdout, stderr = self._run_main(
                    self._cli_args(**{cli_keyword: value})
                )
                self.assertEqual(code, 2)
                self.assertIn("error:", stderr)

    def test_cli_exit_codes_are_stable(self) -> None:
        success, _stdout, success_stderr = self._run_main(self._cli_args())
        self.assertEqual(success, 0)
        self.assertEqual(success_stderr, "")

        invalid, _stdout, invalid_stderr = self._run_main(
            self._cli_args(review_commit="not-a-commit")
        )
        self.assertEqual(invalid, 2)
        self.assertIn("error:", invalid_stderr)

        missing_repo = self.repo.parent / "does-not-exist"
        runtime, _stdout, runtime_stderr = self._run_main(
            self._cli_args(repo=missing_repo)
        )
        self.assertEqual(runtime, 4)
        self.assertIn("error:", runtime_stderr)


class CollectorRemotePrivacyTests(unittest.TestCase):
    def test_remote_redaction_is_fail_closed(self) -> None:
        cases = (
            ("/home/private/repo.git", "[REDACTED_LOCAL_REMOTE]"),
            ("file:///home/private/repo.git", "[REDACTED_LOCAL_REMOTE]"),
            (
                f"https://example.invalid/team/repo.git?token={TOKEN}#private",
                "https://example.invalid/team/repo.git",
            ),
            (
                f"https://example.invalid/team/{TOKEN}/repo.git",
                "https://example.invalid/team/REDACTED_SECRET/repo.git",
            ),
            (
                f"https://example.invalid/team/{FINE_GRAINED_TOKEN}/repo.git",
                "https://example.invalid/team/REDACTED_SECRET/repo.git",
            ),
            (
                "token@example.invalid:team/repo.git?secret=value#private",
                "example.invalid:team/repo.git",
            ),
            (
                f"token@example.invalid:team/{TOKEN}/repo.git",
                "example.invalid:team/REDACTED_SECRET/repo.git",
            ),
        )
        for remote, expected in cases:
            with self.subTest(remote=remote):
                self.assertEqual(redact_remote_url(remote), expected)

    def test_collected_remotes_never_emit_local_paths_or_known_tokens(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        repo = _initialize_git_repository(Path(temporary.name), "repository")
        remotes = {
            "local": "/home/private/repo.git",
            "file": "file:///home/private/repo.git",
            "query": f"https://example.invalid/team/repo.git?token={TOKEN}#private",
            "token-path": f"https://example.invalid/team/{TOKEN}/repo.git",
            "fine-grained-token-path": (
                f"https://example.invalid/team/{FINE_GRAINED_TOKEN}/repo.git"
            ),
        }
        for name, url in remotes.items():
            _git(repo, "remote", "add", name, url)

        head = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
        provenance = collect_provenance(
            repo,
            reviewers=["explicit-reviewer"],
            review_method="manual raw-event review",
            review_commit=head,
            raw_events_reviewed=True,
            timestamp_utc=TIMESTAMP,
            env_value="",
        )
        encoded = json.dumps(provenance, sort_keys=True)
        self.assertNotIn("/home/private", encoded)
        self.assertNotIn("file://", encoded)
        self.assertNotIn(TOKEN, encoded)
        self.assertNotIn(FINE_GRAINED_TOKEN, encoded)
        self.assertNotIn("?token=", encoded)
        self.assertNotIn("#private", encoded)
        self.assertIn("[REDACTED_LOCAL_REMOTE]", encoded)
        self.assertIn("REDACTED_SECRET", encoded)


class CollectorSubmoduleTests(unittest.TestCase):
    def test_recursive_submodule_status_is_hashed_without_paths(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        child = _initialize_git_repository(root, "private-child-source")
        parent = _initialize_git_repository(root, "parent")

        _git(
            parent,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "--quiet",
            os.fspath(child),
            "modules/child",
        )
        _git(parent, "commit", "--quiet", "-m", "add submodule")
        _git(parent, "config", "status.ignoreSubmodules", "all")
        (parent / "modules" / "child" / "tracked.txt").write_text(
            "dirty submodule\n",
            encoding="utf-8",
        )

        observed = collect_worktree(parent, timestamp_utc=TIMESTAMP)
        repeated = collect_worktree(parent, timestamp_utc=TIMESTAMP)
        exact_status = _git(
            parent,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        exact_submodules = _git(parent, "submodule", "status", "--recursive")
        submodules = observed["submodules"]

        self.assertEqual(
            observed["porcelain"]["command"],
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
        )
        self.assertEqual(
            observed["porcelain"]["sha256"],
            hashlib.sha256(exact_status).hexdigest(),
        )
        self.assertTrue(observed["unstaged_changes_present"])
        self.assertEqual(observed["collected_at_utc"], TIMESTAMP)

        self.assertEqual(
            submodules["command"],
            ["git", "submodule", "status", "--recursive"],
        )
        self.assertEqual(submodules["sha256"], hashlib.sha256(exact_submodules).hexdigest())
        self.assertEqual(submodules["byte_count"], len(exact_submodules))
        self.assertEqual(submodules["count"], len(exact_submodules.splitlines()))
        self.assertEqual(submodules["state"], "at_recorded_commit")
        self.assertEqual(submodules["state_counts"]["at_recorded_commit"], 1)
        self.assertEqual(submodules["collected_at_utc"], TIMESTAMP)
        self.assertEqual(submodules, repeated["submodules"])

        encoded = json.dumps(observed, sort_keys=True)
        self.assertNotIn("modules/child", encoded)
        self.assertNotIn(os.fspath(child), encoded)


if __name__ == "__main__":
    unittest.main()
