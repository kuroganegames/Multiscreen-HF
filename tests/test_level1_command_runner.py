"""Focused tests for the lossless Level 1 command recorder."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "run_level1_requalification_command.py"


class Level1CommandRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="level1-command-runner-")
        self.addCleanup(temporary.cleanup)
        self.temp_root = Path(temporary.name)
        self.run_root = self.temp_root / "evidence-run"

    def _base(self, *, name: str, run_root: Path | None = None) -> list[str]:
        return [
            sys.executable,
            os.fspath(SCRIPT),
            "--repo-root",
            os.fspath(REPOSITORY_ROOT),
            "--run-root",
            os.fspath(self.run_root if run_root is None else run_root),
            "--name",
            name,
        ]

    def _run(
        self,
        *,
        name: str,
        child: list[str] | None = None,
        extra: list[str] | None = None,
        run_root: Path | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        command = self._base(name=name, run_root=run_root)
        command.extend(extra or ())
        if child is not None:
            command.append("--")
            command.extend(child)
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _record(self, name: str) -> tuple[dict[str, object], bytes]:
        raw = (self.run_root / "records" / f"{name}.json").read_bytes()
        return json.loads(raw), raw

    def test_binary_merged_stream_is_lossless_and_record_is_canonical(self) -> None:
        stdout = b"stdout:\x00\xff\n"
        stderr = b"stderr:\x80\xfe\n"
        child_code = (
            "import os; "
            f"os.write(1, {stdout!r}); "
            f"os.write(2, {stderr!r})"
        )
        completed = self._run(
            name="binary-stream",
            child=[sys.executable, "-c", child_code],
        )
        expected = stdout + stderr
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode(errors="replace"),
        )
        self.assertEqual(completed.stdout, expected)
        self.assertEqual(completed.stderr, b"")

        log = (self.run_root / "logs" / "binary-stream.log").read_bytes()
        self.assertEqual(log, expected)
        record, raw_record = self._record("binary-stream")
        self.assertEqual(record["argv"], [sys.executable, "-c", child_code])
        self.assertEqual(
            record["cwd"],
            {"base": "repository_root", "path": "."},
        )
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["returncode"], 0)
        self.assertEqual(record["preconditions"], {"absent_paths": []})
        self.assertIsNone(record["termination_signal"])
        self.assertEqual(record["log"]["size_bytes"], len(expected))
        self.assertEqual(
            record["log"]["sha256"],
            hashlib.sha256(expected).hexdigest(),
        )
        self.assertGreaterEqual(record["duration_ns"], 0)
        self.assertTrue(raw_record.endswith(b"\n"))
        self.assertEqual(raw_record.count(b"\n"), 1)
        self.assertEqual(
            (self.run_root / "commands.jsonl").read_bytes(),
            raw_record,
        )

    def test_arguments_are_not_interpreted_by_a_shell(self) -> None:
        literal = "$(printf shell-was-used) ; * | >"
        completed = self._run(
            name="no-shell",
            child=[
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                literal,
            ],
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, (literal + "\n").encode())
        record, _raw = self._record("no-shell")
        self.assertEqual(record["argv"][-1], literal)

    def test_child_exit_code_is_preserved(self) -> None:
        completed = self._run(
            name="exit-23",
            child=[sys.executable, "-c", "raise SystemExit(23)"],
        )
        self.assertEqual(completed.returncode, 23)
        record, _raw = self._record("exit-23")
        self.assertEqual(record["exit_code"], 23)
        self.assertEqual(record["returncode"], 23)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "requires POSIX signals")
    def test_child_signal_is_recorded_and_reraised(self) -> None:
        child_code = "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
        completed = self._run(
            name="signal-term",
            child=[sys.executable, "-c", child_code],
        )
        self.assertEqual(completed.returncode, -signal.SIGTERM)
        record, _raw = self._record("signal-term")
        self.assertIsNone(record["exit_code"])
        self.assertEqual(record["returncode"], -signal.SIGTERM)
        self.assertEqual(
            record["termination_signal"],
            {"name": "SIGTERM", "number": signal.SIGTERM},
        )

    def test_private_root_is_initialized_once_and_names_are_unique(self) -> None:
        first = self._run(
            name="unique",
            child=[sys.executable, "-c", "print('first')"],
        )
        self.assertEqual(first.returncode, 0)
        self.assertEqual(stat.S_IMODE(self.run_root.stat().st_mode), 0o700)
        for directory in ("logs", "records", "reservations"):
            self.assertEqual(
                stat.S_IMODE((self.run_root / directory).stat().st_mode),
                0o700,
            )

        log_before = (self.run_root / "logs" / "unique.log").read_bytes()
        record_before = (self.run_root / "records" / "unique.json").read_bytes()
        duplicate = self._run(
            name="unique",
            child=[sys.executable, "-c", "print('second')"],
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn(b"refusing to overwrite", duplicate.stderr)
        self.assertEqual(
            (self.run_root / "logs" / "unique.log").read_bytes(),
            log_before,
        )
        self.assertEqual(
            (self.run_root / "records" / "unique.json").read_bytes(),
            record_before,
        )

        second = self._run(
            name="another",
            child=[sys.executable, "-c", "pass"],
        )
        self.assertEqual(second.returncode, 0)
        self.assertEqual(
            len((self.run_root / "commands.jsonl").read_bytes().splitlines()),
            2,
        )

    def test_fresh_outputs_are_checked_and_recorded_canonically(self) -> None:
        first_relative = "outputs/a-result.bin"
        second_relative = "outputs/z-result.bin"
        first = self.run_root / first_relative
        second = self.run_root / second_relative
        child_code = (
            "from pathlib import Path; import sys; "
            "[(Path(value).parent.mkdir(parents=True, exist_ok=True), "
            "Path(value).write_bytes(value.encode())) for value in sys.argv[1:]]"
        )
        completed = self._run(
            name="fresh-outputs",
            extra=[
                "--require-absent",
                second_relative,
                "--require-absent",
                first_relative,
            ],
            child=[
                sys.executable,
                "-c",
                child_code,
                os.fspath(first),
                os.fspath(second),
            ],
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode(errors="replace"),
        )
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        record, _raw = self._record("fresh-outputs")
        self.assertEqual(
            record["preconditions"]["absent_paths"],
            [first_relative, second_relative],
        )

    def test_existing_file_and_directory_fail_before_child_spawn(self) -> None:
        initialized = self._run(
            name="initialize",
            extra=["--environment-record"],
        )
        self.assertEqual(initialized.returncode, 0)
        output_parent = self.run_root / "outputs"
        output_parent.mkdir()
        existing_file = output_parent / "existing.txt"
        existing_file.write_text("existing\n", encoding="utf-8")
        existing_directory = output_parent / "existing-directory"
        existing_directory.mkdir()

        for index, relative in enumerate(
            ("outputs/existing.txt", "outputs/existing-directory")
        ):
            with self.subTest(relative=relative):
                sentinel = output_parent / f"spawned-{index}"
                completed = self._run(
                    name=f"existing-{index}",
                    extra=["--require-absent", relative],
                    child=[
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import sys; "
                        "Path(sys.argv[1]).write_text('spawned')",
                        os.fspath(sentinel),
                    ],
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(b"already exists", completed.stderr)
                self.assertFalse(sentinel.exists())
                self.assertFalse(
                    (self.run_root / "records" / f"existing-{index}.json").exists()
                )

    def test_symlinked_fresh_output_fails_before_child_spawn(self) -> None:
        initialized = self._run(
            name="initialize",
            extra=["--environment-record"],
        )
        self.assertEqual(initialized.returncode, 0)
        output_parent = self.run_root / "outputs"
        output_parent.mkdir()
        link = output_parent / "linked-output"
        link.symlink_to(self.temp_root / "outside-output")
        sentinel = output_parent / "spawned"

        completed = self._run(
            name="symlink-output",
            extra=["--require-absent", "outputs/linked-output"],
            child=[
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; "
                "Path(sys.argv[1]).write_text('spawned')",
                os.fspath(sentinel),
            ],
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"symbolic link", completed.stderr)
        self.assertFalse(sentinel.exists())

    def test_duplicate_and_traversal_fresh_paths_are_rejected(self) -> None:
        duplicate = self._run(
            name="duplicate-output",
            extra=[
                "--require-absent",
                "outputs/result",
                "--require-absent",
                "outputs/result",
            ],
            child=[sys.executable, "-c", "pass"],
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn(b"duplicate", duplicate.stderr)
        self.assertFalse(self.run_root.exists())

        overlap_root = self.temp_root / "overlap"
        overlap = self._run(
            name="overlap-output",
            run_root=overlap_root,
            extra=[
                "--require-absent",
                "a",
                "--require-absent",
                "a-b",
                "--require-absent",
                "a/result",
            ],
            child=[sys.executable, "-c", "pass"],
        )
        self.assertEqual(overlap.returncode, 2)
        self.assertIn(b"overlapping", overlap.stderr)
        self.assertFalse(overlap_root.exists())

        for index, relative in enumerate(
            ("../escape", "outputs/../escape", "/absolute", "C:/absolute")
        ):
            with self.subTest(relative=relative):
                root = self.temp_root / f"traversal-{index}"
                completed = self._run(
                    name=f"traversal-{index}",
                    run_root=root,
                    extra=["--require-absent", relative],
                    child=[sys.executable, "-c", "pass"],
                )
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(root.exists())

    def test_fresh_paths_cannot_collide_with_run_controls(self) -> None:
        collisions = (
            ".level1-requalification-run.json",
            ".level1-requalification-run.json/output",
            "commands.jsonl",
            "commands.jsonl/output",
            "environment.jsonl",
            "logs",
            "logs/output.log",
            "records/output.json",
            "reservations/output",
        )
        for index, relative in enumerate(collisions):
            with self.subTest(relative=relative):
                root = self.temp_root / f"control-{index}"
                completed = self._run(
                    name=f"control-{index}",
                    run_root=root,
                    extra=["--require-absent", relative],
                    child=[sys.executable, "-c", "pass"],
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(b"control paths", completed.stderr)
                self.assertFalse(root.exists())

    def test_environment_record_is_dry_and_privacy_scoped(self) -> None:
        completed = self._run(
            name="runtime",
            extra=["--environment-record"],
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode(errors="replace"),
        )
        self.assertEqual(completed.stdout, b"")
        record, raw = self._record("runtime")
        self.assertEqual(record["record_type"], "environment")
        self.assertNotIn("argv", record)
        self.assertNotIn("log", record)
        self.assertNotIn("environment", record)
        self.assertIn("python", record["runtime"])
        self.assertIn("operating_system", record["runtime"])
        self.assertEqual(record["runtime"]["python"]["optimization_level"], 0)
        self.assertIs(record["runtime"]["python"]["assertions_enabled"], True)
        self.assertNotIn(os.fspath(REPOSITORY_ROOT).encode(), raw)
        self.assertEqual(
            (self.run_root / "environment.jsonl").read_bytes(),
            raw,
        )
        self.assertFalse((self.run_root / "commands.jsonl").exists())

    def test_environment_mode_rejects_a_command(self) -> None:
        completed = self._run(
            name="bad-environment",
            extra=["--environment-record"],
            child=[sys.executable, "--version"],
        )
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(self.run_root.exists())

    def test_environment_mode_rejects_fresh_output_preconditions(self) -> None:
        completed = self._run(
            name="bad-environment-output",
            extra=[
                "--environment-record",
                "--require-absent",
                "outputs/result",
            ],
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"must not be combined", completed.stderr)
        self.assertFalse(self.run_root.exists())

    def test_unsafe_names_are_rejected_before_root_creation(self) -> None:
        for index, name in enumerate(
            ("../escape", ".hidden", "Upper", "ends-", "a/b")
        ):
            with self.subTest(name=name):
                root = self.temp_root / f"unsafe-{index}"
                completed = self._run(
                    name=name,
                    run_root=root,
                    child=[sys.executable, "-c", "pass"],
                )
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(root.exists())

    def test_run_root_inside_worktree_is_rejected_without_creation(self) -> None:
        candidate = REPOSITORY_ROOT / "forbidden-stage5-evidence-root"
        self.assertFalse(candidate.exists())
        completed = self._run(
            name="inside",
            run_root=candidate,
            child=[sys.executable, "-c", "pass"],
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            b"outside every registered Git worktree",
            completed.stderr,
        )
        self.assertFalse(candidate.exists())

    def test_symlinked_run_root_component_is_rejected(self) -> None:
        real_parent = self.temp_root / "real-parent"
        real_parent.mkdir()
        link_parent = self.temp_root / "link-parent"
        link_parent.symlink_to(real_parent, target_is_directory=True)
        completed = self._run(
            name="symlink-root",
            run_root=link_parent / "evidence",
            child=[sys.executable, "-c", "pass"],
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"symbolic-link", completed.stderr)
        self.assertFalse((real_parent / "evidence").exists())

    def test_existing_uninitialized_or_permissive_root_is_rejected(self) -> None:
        arbitrary = self.temp_root / "arbitrary"
        arbitrary.mkdir(mode=0o700)
        no_marker = self._run(
            name="no-marker",
            run_root=arbitrary,
            child=[sys.executable, "-c", "pass"],
        )
        self.assertEqual(no_marker.returncode, 2)

        permissive = self.temp_root / "permissive"
        permissive.mkdir(mode=0o755)
        os.chmod(permissive, 0o755)
        bad_mode = self._run(
            name="bad-mode",
            run_root=permissive,
            child=[sys.executable, "-c", "pass"],
        )
        self.assertEqual(bad_mode.returncode, 2)
        self.assertIn(b"mode 0700", bad_mode.stderr)

    def test_cwd_is_privacy_safe_and_must_not_escape(self) -> None:
        completed = self._run(
            name="oracle-cwd",
            extra=["--cwd", "oracle"],
            child=[
                sys.executable,
                "-c",
                "import os; print(os.path.basename(os.getcwd()))",
            ],
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"oracle\n")
        record, raw = self._record("oracle-cwd")
        self.assertEqual(
            record["cwd"],
            {"base": "repository_root", "path": "oracle"},
        )
        self.assertNotIn(os.fspath(REPOSITORY_ROOT).encode(), raw)

        escaped_root = self.temp_root / "escaped-run"
        escaped = self._run(
            name="escape-cwd",
            run_root=escaped_root,
            extra=["--cwd", "../"],
            child=[sys.executable, "-c", "pass"],
        )
        self.assertEqual(escaped.returncode, 2)
        self.assertFalse(escaped_root.exists())


if __name__ == "__main__":
    unittest.main()
