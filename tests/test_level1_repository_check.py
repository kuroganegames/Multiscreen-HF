"""Focused fixtures for the Level 1 read-only repository checker."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_level1_repository as checker


def git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8", newline="\n")


class RepositoryFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        git(root, "init", "--quiet")
        write(root / "README.md", "# Fixture\n\n[Docs](docs/guide.md#usage)\n")
        write(root / "docs/guide.md", "# Guide\n\n## Usage\n\n```md\n[ignored](missing.md)\n```\n")
        write(root / "config.json", '{"finite":1.25,"nested":{"ok":true}}\n')
        write(
            root / ".github/workflows/test.yml",
            "name: test\n"
            "on:\n"
            "  pull_request:\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: Run\n"
            "        run: |\n"
            "          python -m unittest\n",
        )
        self.commit()

    def commit(self) -> None:
        git(self.root, "add", "-A")
        git(
            self.root,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        )


class Level1RepositoryCheckTests(unittest.TestCase):
    def make_fixture(self, temporary: str) -> RepositoryFixture:
        return RepositoryFixture(Path(temporary))

    def assert_rejected(self, fixture: RepositoryFixture, check: str, message: str) -> None:
        with self.assertRaisesRegex(checker.RepositoryCheckError, message):
            checker.run_check(fixture.root, check)

    def test_all_checks_pass_and_success_reports_are_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            for name in checker.CHECKS:
                with self.subTest(check=name):
                    report = checker.run_check(fixture.root, name)
                    self.assertEqual(report["status"], "passed")
                    serialized = checker._canonical_bytes(report)
                    self.assertNotIn(str(fixture.root).encode(), serialized)
                    self.assertEqual(json.loads(serialized)["check"], name)
            hygiene = checker.run_check(fixture.root, "hygiene")["result"]
            self.assertTrue(hygiene["worktree"]["clean"])
            self.assertEqual(hygiene["worktree"]["porcelain_byte_count"], 0)
            self.assertEqual(hygiene["submodules"]["record_count"], 0)
            self.assertEqual(hygiene["privacy"]["status"], "passed")
            self.assertGreater(hygiene["privacy"]["artifact_count"], 0)

    def test_cli_emits_one_canonical_json_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            stdout = io.BytesIO()

            class BytesStdout:
                buffer = stdout

            with contextlib.redirect_stdout(BytesStdout()):
                result = checker.main(
                    ["--repo-root", str(fixture.root), "--check", "json"]
                )
            self.assertEqual(result, 0)
            value = json.loads(stdout.getvalue())
            self.assertEqual(stdout.getvalue(), checker._canonical_bytes(value))

    def test_duplicate_and_nonfinite_json_fail_closed(self) -> None:
        for invalid in ('{"same":1,"same":2}\n', '{"overflow":1e9999}\n', '{"x":NaN}\n'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_fixture(temporary)
                write(fixture.root / "config.json", invalid)
                fixture.commit()
                self.assert_rejected(fixture, "json", "JSON")

    def test_markdown_missing_escape_and_fenced_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            write(fixture.root / "README.md", "[missing](docs/missing.md)\n")
            fixture.commit()
            self.assert_rejected(fixture, "markdown-links", "missing")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            write(fixture.root / "README.md", "[escape](%2e%2e/outside.md)\n")
            fixture.commit()
            self.assert_rejected(fixture, "markdown-links", "escapes")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            report = checker.run_check(fixture.root, "markdown-links")
            self.assertEqual(report["result"]["local_link_count"], 1)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            write(
                fixture.root / "README.md",
                "[guide][local]\n\n[local]: docs/missing-reference.md\n",
            )
            fixture.commit()
            self.assert_rejected(fixture, "markdown-links", "missing")

    def test_workflow_tabs_missing_jobs_and_duplicate_jobs_fail(self) -> None:
        invalid_documents = (
            "name: bad\njobs:\n\tbad:\n    runs-on: ubuntu-latest\n",
            "name: bad\non: push\n",
            "jobs:\n  test:\n    runs-on: ubuntu\n  test:\n    runs-on: ubuntu\n",
            "name: bad\njobs:\nenv:\n  VALUE: present\n",
            "name: bad\njobs:\n  test:\n    steps:\n      - run: true\n",
            "name: bad\non: *undefined\njobs:\n  test:\n    runs-on: ubuntu\n    steps:\n      - run: true\n",
            "name: bad\njobs:\n  test:\n    runs-on: ubuntu\n",
            "name: bad\njobs:\n  test:\n    runs-on: ubuntu\n    steps: []\n",
            "name: bad\njobs:\n  test:\n    uses: owner/repo/.github/workflows/reuse.yml@main\n    runs-on: ubuntu\n    steps:\n      - run: true\n",
            "name: scalar\n  child: invalid\njobs:\n  test:\n    uses: owner/repo/.github/workflows/reuse.yml@main\n",
            "name: value: invalid\njobs:\n  test:\n    uses: owner/repo/.github/workflows/reuse.yml@main\n",
            "name: @invalid\njobs:\n  test:\n    uses: owner/repo/.github/workflows/reuse.yml@main\n",
            "name: %invalid\njobs:\n  test:\n    uses: owner/repo/.github/workflows/reuse.yml@main\n",
            "name: [a,,b]\njobs:\n  test:\n    uses: owner/repo/.github/workflows/reuse.yml@main\n",
            "jobs:\n  test:\n    runs-on: ubuntu\n    steps:\n      - name: no-command\n",
            "jobs:\n  test:\n    runs-on: ubuntu\n    steps:\n      - run: true\n        uses: owner/action@v1\n",
            "jobs:\n  test:\n    runs-on: ubuntu\n    env:\n      VALUE: one\n      VALUE: two\n    steps:\n      - run: true\n",
        )
        for value in invalid_documents:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_fixture(temporary)
                write(fixture.root / ".github/workflows/test.yml", value)
                fixture.commit()
                self.assert_rejected(fixture, "workflow-yaml", "workflow")

    def test_reusable_workflow_job_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            write(
                fixture.root / ".github/workflows/test.yml",
                "name: reuse\n"
                "on: push\n"
                "jobs:\n"
                "  delegated:\n"
                "    uses: owner/repo/.github/workflows/reuse.yml@main\n",
            )
            fixture.commit()
            report = checker.run_check(fixture.root, "workflow-yaml")
            self.assertEqual(report["result"]["job_count"], 1)

    def test_strict_json_flow_sequence_is_accepted(self) -> None:
        document = (
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu\n"
            "    strategy:\n"
            "      matrix:\n"
            '        python: ["3.10", "3.11"]\n'
            "    steps:\n"
            "      - run: true\n"
        )
        self.assertEqual(checker._validate_workflow_subset(document), 1)

    def test_unsupported_indicator_leading_scalars_are_rejected(self) -> None:
        for scalar in (
            "|invalid",
            ">invalid",
            "&",
            "*",
            "? invalid",
            "- invalid",
            ",invalid",
        ):
            with self.subTest(scalar=scalar):
                document = (
                    f"name: {scalar}\n"
                    "jobs:\n"
                    "  test:\n"
                    "    uses: owner/repo/.github/workflows/reuse.yml@main\n"
                )
                with self.assertRaises(checker.RepositoryCheckError):
                    checker._validate_workflow_subset(document)

    def test_oversized_tracked_file_fails_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            target = fixture.root / "config.json"
            write(target, b"x" * (checker.MAX_TRACKED_FILE_BYTES + 1))
            fixture.commit()
            original_read_bytes = Path.read_bytes

            def guarded_read_bytes(path: Path) -> bytes:
                if path == target:
                    self.fail("oversized tracked artifact was read into memory")
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", guarded_read_bytes):
                self.assert_rejected(fixture, "json", "maximum allowed size")

    def test_hygiene_rejects_dirty_and_forbidden_tracked_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            write(fixture.root / "untracked.txt", "dirty\n")
            self.assert_rejected(fixture, "hygiene", "clean")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            write(fixture.root / "weights.safetensors", b"not a model")
            fixture.commit()
            self.assert_rejected(fixture, "hygiene", "forbidden")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            write(fixture.root / "outputs/result.txt", "forbidden\n")
            fixture.commit()
            self.assert_rejected(fixture, "hygiene", "forbidden")

        for path in (
            "raw-evidence.7z",
            "raw-events.jsonl",
            "raw-output.log",
            "terminal.out",
            "checkpoint-10/report.json",
            "optimizer_state/state.json",
            "weights.npy",
        ):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_fixture(temporary)
                write(fixture.root / path, "forbidden\n")
                fixture.commit()
                self.assert_rejected(fixture, "hygiene", "forbidden")

    def test_hygiene_privacy_scan_rejects_private_roots_tokens_and_secrets(self) -> None:
        cases = (
            f"local={Path.home() / 'private' / 'result.json'}\n",
            "token=" + "hf_" + ("A" * 30) + "\n",
            "api_key=actual-secret-value\n",
            "remote=" + "https://" + "user:password@example.invalid/repo.git\n",
        )
        for index, value in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                fixture = self.make_fixture(temporary)
                write(fixture.root / "docs/private.md", value)
                fixture.commit()
                self.assert_rejected(fixture, "hygiene", "tracked")

    def test_known_sanitizer_fixture_literals_are_bounded_exemptions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            write(
                fixture.root / "tests/test_validation_evidence.py",
                'TOKEN = "' + "hf_" + ("A" * 30) + '"\n'
                'PATH = "/' + "home/synthetic-user/private/result.json" + '"\n',
            )
            fixture.commit()
            hygiene = checker.run_check(fixture.root, "hygiene")["result"]
            self.assertEqual(
                hygiene["privacy"]["fixture_exemption_artifact_count"], 1
            )

    def test_checker_test_source_is_privacy_scannable_without_exemption(self) -> None:
        path = "tests/test_level1_repository_check.py"
        self.assertNotIn(path, checker.PRIVACY_FIXTURE_PATHS)
        report = checker._privacy_summary(
            Path("/") / "synthetic" / "repository",
            [(path, Path(__file__).read_bytes())],
        )
        self.assertEqual(report["fixture_exemption_artifact_count"], 0)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_tracked_symlink_is_rejected_for_every_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            os.symlink("config.json", fixture.root / "alias.json")
            fixture.commit()
            for name in checker.CHECKS:
                with self.subTest(check=name):
                    self.assert_rejected(fixture, name, "symlinks")

    def test_repo_root_must_be_absolute_top_level_and_not_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(temporary)
            with self.assertRaisesRegex(checker.RepositoryCheckError, "absolute"):
                checker.run_check(".", "json")
            child = fixture.root / "docs"
            with self.assertRaisesRegex(checker.RepositoryCheckError, "top-level"):
                checker.run_check(child, "json")

            link = fixture.root.parent / f"{fixture.root.name}-link"
            try:
                os.symlink(fixture.root, link)
                with self.assertRaisesRegex(checker.RepositoryCheckError, "symlink"):
                    checker.run_check(link, "json")
            finally:
                link.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
