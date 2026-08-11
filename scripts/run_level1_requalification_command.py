#!/usr/bin/env python3
"""Run one Level 1 requalification command with lossless local evidence.

The recorder is intentionally standard-library-only. It creates (or reuses)
an explicitly named private run root outside every worktree of the selected
repository, streams the child's merged stdout/stderr bytes to both the console
and an immutable log, and writes one canonical JSON record. It never invokes
a shell.

Example:

    python scripts/run_level1_requalification_command.py \
      --repo-root "$PWD" \
      --run-root /private/evidence/level1-2026-08-09 \
      --name formula-units \
      -- python oracle/test_formula_units.py

Use --environment-record without -- to record only the recorder's privacy-safe
Python/platform metadata. It does not inspect or copy the process environment
and it does not launch another command.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence


TOOL_VERSION = "1.0.0"
RECORD_FORMAT_VERSION = "level1-requalification-command-record-v1"
RUN_FORMAT_VERSION = "level1-requalification-run-v1"
RUN_MARKER = ".level1-requalification-run.json"
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,78}[a-z0-9])?$")
COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:/")
RUN_CONTROL_PATHS = frozenset(
    {
        RUN_MARKER,
        "commands.jsonl",
        "environment.jsonl",
    }
)
RUN_CONTROL_DIRECTORIES = frozenset({"logs", "records", "reservations"})


class RecorderInputError(ValueError):
    """A caller-supplied path, name, or mode is unsafe or invalid."""


class RecorderRuntimeError(RuntimeError):
    """A required process or filesystem operation failed."""


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one stable UTF-8 JSON object followed by exactly one newline."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8", errors="strict")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise RecorderRuntimeError("a filesystem write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RecorderRuntimeError(f"cannot open evidence directory for fsync: {exc}") from exc
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, mode)
    except FileExistsError as exc:
        raise RecorderInputError(f"refusing to overwrite existing evidence: {path.name}") from exc
    except OSError as exc:
        raise RecorderRuntimeError(f"cannot create evidence file {path.name!r}: {exc}") from exc
    try:
        _write_all(fd, data)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        raise
    else:
        os.close(fd)
    _fsync_directory(path.parent)


def _validate_private_regular_file(fd: int, *, label: str) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise RecorderRuntimeError(f"{label} is not a regular file")
    if info.st_nlink != 1:
        raise RecorderRuntimeError(f"{label} must not have hard links")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise RecorderRuntimeError(f"{label} is not owned by the current user")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise RecorderRuntimeError(f"{label} must have mode 0600")


def _append_jsonl(path: Path, data: bytes) -> None:
    """Append one already-canonical line under an advisory exclusive lock."""

    if not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise RecorderRuntimeError("JSONL records must contain exactly one line")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RecorderRuntimeError(f"cannot open JSONL ledger {path.name!r}: {exc}") from exc
    try:
        _validate_private_regular_file(fd, label=path.name)
        fcntl.flock(fd, fcntl.LOCK_EX)
        _write_all(fd, data)
        os.fsync(fd)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _safe_name(value: str) -> str:
    if not isinstance(value, str) or NAME_RE.fullmatch(value) is None:
        raise RecorderInputError(
            "command name must be 1-80 lowercase ASCII letters, digits, dots, "
            "underscores, or hyphens; it must start and end with a letter or digit"
        )
    return value


def _safe_relative_cwd(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecorderInputError("--cwd must be a non-empty repository-relative path")
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise RecorderInputError("--cwd must contain only ASCII characters") from exc
    if value == ".":
        return value
    if "\\" in value or "\x00" in value or any(ord(char) < 32 for char in value):
        raise RecorderInputError("--cwd contains an unsafe character")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RecorderInputError("--cwd must be a canonical repository-relative path")
    if path.as_posix() != value:
        raise RecorderInputError("--cwd must use canonical POSIX separators")
    return value


def _safe_run_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecorderInputError(
            "--require-absent must be a non-empty run-root-relative path"
        )
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise RecorderInputError(
            "--require-absent paths must contain only ASCII characters"
        ) from exc
    if (
        "\\" in value
        or "\x00" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise RecorderInputError("--require-absent contains an unsafe character")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or DRIVE_PATH_RE.match(value) is not None
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RecorderInputError(
            "--require-absent must be a canonical POSIX relative path"
        )
    if path.as_posix() != value:
        raise RecorderInputError(
            "--require-absent must use canonical POSIX separators"
        )
    if path.parts[0] in RUN_CONTROL_PATHS | RUN_CONTROL_DIRECTORIES:
        raise RecorderInputError(
            "--require-absent must not collide with run-root control paths"
        )
    return value


def _canonical_absent_paths(values: Sequence[str]) -> list[str]:
    paths = [_safe_run_relative_path(value) for value in values]
    if len(set(paths)) != len(paths):
        raise RecorderInputError("duplicate --require-absent paths are forbidden")
    paths.sort()
    path_set = set(paths)
    for current in paths:
        parts = PurePosixPath(current).parts
        if any(
            "/".join(parts[:index]) in path_set
            for index in range(1, len(parts))
        ):
            raise RecorderInputError(
                "overlapping --require-absent paths are forbidden"
            )
    return paths


def _assert_absent_paths(run_root: Path, paths: Sequence[str]) -> None:
    """Observe each fresh-output leaf with lstat immediately before spawn."""

    for relative in paths:
        current = run_root
        parts = PurePosixPath(relative).parts
        for index, part in enumerate(parts):
            current = current / part
            is_leaf = index == len(parts) - 1
            try:
                info = os.lstat(current)
            except FileNotFoundError:
                # A missing parent or leaf necessarily makes the leaf absent
                # at this observation point.
                break
            except OSError as exc:
                raise RecorderInputError(
                    f"cannot verify fresh output path {relative!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise RecorderInputError(
                    f"fresh output path {relative!r} contains a symbolic link"
                )
            if is_leaf:
                kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
                raise RecorderInputError(
                    f"fresh output path {relative!r} already exists as a {kind}"
                )
            if not stat.S_ISDIR(info.st_mode):
                raise RecorderInputError(
                    f"fresh output path {relative!r} has a non-directory parent"
                )


def _absolute_canonical_path(value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RecorderInputError(f"{field} must be a non-empty absolute path")
    try:
        value.encode(sys.getfilesystemencoding(), errors="strict")
    except UnicodeEncodeError as exc:
        raise RecorderInputError(f"{field} cannot be represented safely") from exc
    path = Path(value)
    if not path.is_absolute() or os.fspath(path) != value:
        raise RecorderInputError(f"{field} must be a canonical absolute path")
    if path == Path(path.anchor):
        raise RecorderInputError(f"{field} must not be a filesystem root")
    return path


def _reject_symlink_components(path: Path, *, allow_missing_leaf: bool) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and is_leaf:
                return
            raise RecorderInputError(f"path component does not exist: {current}") from None
        except OSError as exc:
            raise RecorderInputError(f"cannot inspect path component {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RecorderInputError(f"symbolic-link path components are forbidden: {current}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _run_git(repo: Path, arguments: Sequence[str], *, check: bool = True) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repo), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RecorderRuntimeError(f"cannot execute git: {exc}") from exc
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").splitlines()
        summary = detail[0] if detail else f"exit status {completed.returncode}"
        raise RecorderInputError(f"git repository validation failed: {summary}")
    return completed.stdout if completed.returncode == 0 else b""


def _validate_repository_root(value: str) -> Path:
    path = _absolute_canonical_path(value, field="--repo-root")
    _reject_symlink_components(path, allow_missing_leaf=False)
    if not path.is_dir():
        raise RecorderInputError("--repo-root must name a directory")
    raw_top = _run_git(path, ["rev-parse", "--show-toplevel"])
    try:
        top = Path(os.fsdecode(raw_top.rstrip(b"\n"))).resolve(strict=True)
    except (OSError, UnicodeError) as exc:
        raise RecorderInputError(f"cannot resolve Git worktree root: {exc}") from exc
    if top != path.resolve(strict=True):
        raise RecorderInputError("--repo-root must be the Git worktree top level")
    return top


def _repository_worktrees(repo_root: Path) -> tuple[Path, ...]:
    raw = _run_git(repo_root, ["worktree", "list", "--porcelain", "-z"])
    worktrees: list[Path] = []
    for field in raw.split(b"\x00"):
        if not field.startswith(b"worktree "):
            continue
        candidate = Path(os.fsdecode(field[len(b"worktree ") :]))
        try:
            worktrees.append(candidate.resolve(strict=True))
        except OSError as exc:
            raise RecorderRuntimeError(f"cannot resolve a registered Git worktree: {exc}") from exc
    if repo_root not in worktrees:
        raise RecorderRuntimeError("Git did not report the selected repository worktree")
    return tuple(worktrees)


def _reject_git_containment(run_root: Path, repo_root: Path) -> None:
    resolved_candidate = run_root.resolve(strict=False)
    for worktree in _repository_worktrees(repo_root):
        if _is_within(resolved_candidate, worktree) or _is_within(worktree, resolved_candidate):
            raise RecorderInputError("--run-root must be outside every registered Git worktree")

    # Also reject a path nested in an unrelated enclosing Git worktree.
    parent = run_root.parent.resolve(strict=True)
    raw_top = _run_git(parent, ["rev-parse", "--show-toplevel"], check=False)
    if raw_top:
        enclosing = Path(os.fsdecode(raw_top.rstrip(b"\n"))).resolve(strict=True)
        if _is_within(resolved_candidate, enclosing):
            raise RecorderInputError("--run-root must not be inside a Git worktree")


def _private_directory(path: Path, *, label: str) -> None:
    _reject_symlink_components(path, allow_missing_leaf=False)
    try:
        info = path.stat()
    except OSError as exc:
        raise RecorderRuntimeError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise RecorderInputError(f"{label} must be a directory")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise RecorderInputError(f"{label} must be owned by the current user")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise RecorderInputError(f"{label} must have mode 0700")


def _repository_identity(repo_root: Path) -> dict[str, str]:
    commit = _run_git(repo_root, ["rev-parse", "HEAD"]).decode("ascii").strip().lower()
    if COMMIT_RE.fullmatch(commit) is None:
        raise RecorderRuntimeError("Git returned an invalid full commit identifier")
    root_digest = hashlib.sha256(os.fsencode(repo_root)).hexdigest()
    return {"head_commit": commit, "worktree_path_sha256": root_digest}


def _read_marker(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RecorderInputError(f"existing run root has no safe marker: {exc}") from exc
    try:
        _validate_private_regular_file(fd, label=RUN_MARKER)
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise RecorderInputError("run marker is unexpectedly large")
    finally:
        os.close(fd)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecorderInputError(f"run marker is invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RecorderInputError("run marker must be a JSON object")
    return value


def prepare_run_root(value: str, repo_root: Path) -> Path:
    """Create a private run root once, or validate a prior recorder-owned root."""

    run_root = _absolute_canonical_path(value, field="--run-root")
    _reject_symlink_components(run_root, allow_missing_leaf=not run_root.exists())
    _reject_git_containment(run_root, repo_root)
    identity = _repository_identity(repo_root)

    if not run_root.exists():
        try:
            os.mkdir(run_root, 0o700)
            os.chmod(run_root, 0o700)
        except OSError as exc:
            raise RecorderRuntimeError(f"cannot create private run root: {exc}") from exc
        for child in ("logs", "records", "reservations"):
            try:
                os.mkdir(run_root / child, 0o700)
                os.chmod(run_root / child, 0o700)
            except OSError as exc:
                raise RecorderRuntimeError(f"cannot create run-root directory {child!r}: {exc}") from exc
        marker = {
            "created_at_utc": utc_now(),
            "format_version": RUN_FORMAT_VERSION,
            "repository": identity,
            "tool_version": TOOL_VERSION,
        }
        _exclusive_write(run_root / RUN_MARKER, canonical_json_bytes(marker))
        _fsync_directory(run_root)
    else:
        _private_directory(run_root, label="--run-root")
        marker = _read_marker(run_root / RUN_MARKER)
        expected = {
            "format_version": RUN_FORMAT_VERSION,
            "repository": identity,
        }
        for field, expected_value in expected.items():
            if marker.get(field) != expected_value:
                raise RecorderInputError(
                    "existing run root was not initialized for this repository and commit"
                )
        for child in ("logs", "records", "reservations"):
            _private_directory(run_root / child, label=f"run-root {child}/")
    return run_root


def _resolve_cwd(repo_root: Path, relative: str) -> Path:
    safe = _safe_relative_cwd(relative)
    candidate = repo_root if safe == "." else repo_root.joinpath(*PurePosixPath(safe).parts)
    _reject_symlink_components(candidate, allow_missing_leaf=False)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RecorderInputError(f"cannot resolve --cwd: {exc}") from exc
    if not resolved.is_dir() or not _is_within(resolved, repo_root):
        raise RecorderInputError("--cwd must resolve to a directory inside --repo-root")
    return resolved


def _reserve_name(run_root: Path, name: str) -> None:
    reservation = {
        "format_version": RUN_FORMAT_VERSION,
        "name": name,
        "reserved_at_utc": utc_now(),
    }
    _exclusive_write(
        run_root / "reservations" / name,
        canonical_json_bytes(reservation),
    )


def runtime_metadata() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "operating_system": {
            "libc_name": libc_name or None,
            "libc_version": libc_version or None,
            "machine": platform.machine() or None,
            "release": platform.release() or None,
            "system": platform.system() or None,
        },
        "python": {
            "assertions_enabled": __debug__,
            "cache_tag": getattr(sys.implementation, "cache_tag", None),
            "compiler": platform.python_compiler() or None,
            "implementation": platform.python_implementation() or None,
            "optimization_level": int(sys.flags.optimize),
            "version": platform.python_version(),
        },
        "recorder": {
            "name": "run_level1_requalification_command.py",
            "version": TOOL_VERSION,
        },
    }


def _duration(start_ns: int, end_ns: int) -> tuple[int, float]:
    duration_ns = max(0, end_ns - start_ns)
    return duration_ns, round(duration_ns / 1_000_000_000, 9)


def _command_arguments(arguments: Sequence[str]) -> list[str]:
    if not arguments:
        raise RecorderInputError("a command is required after --")
    result: list[str] = []
    for argument in arguments:
        if "\x00" in argument:
            raise RecorderInputError("command arguments must not contain NUL bytes")
        try:
            argument.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise RecorderInputError("command arguments must be valid Unicode") from exc
        result.append(argument)
    return result


def _open_log(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RecorderInputError(f"refusing to overwrite existing log {path.name!r}") from exc
    except OSError as exc:
        raise RecorderRuntimeError(f"cannot create command log: {exc}") from exc


def _stream_child(
    process: subprocess.Popen[bytes],
    log_fd: int,
    console_fd: int,
) -> tuple[int, str, str | None]:
    if process.stdout is None:
        raise RecorderRuntimeError("child stdout pipe is unavailable")
    size = 0
    digest = hashlib.sha256()
    console_error: str | None = None
    while True:
        try:
            chunk = os.read(process.stdout.fileno(), 1024 * 1024)
        except InterruptedError:
            continue
        if not chunk:
            break
        _write_all(log_fd, chunk)
        digest.update(chunk)
        size += len(chunk)
        if console_error is None:
            try:
                _write_all(console_fd, chunk)
            except OSError as exc:
                console_error = errno.errorcode.get(exc.errno or 0, type(exc).__name__)
            except RecorderRuntimeError as exc:
                console_error = type(exc).__name__
    process.stdout.close()
    return size, digest.hexdigest(), console_error


def run_command(
    *,
    run_root: Path,
    repo_root: Path,
    name: str,
    cwd_relative: str,
    arguments: Sequence[str],
    absent_paths: Sequence[str] = (),
    console_fd: int = 1,
) -> tuple[int | None, dict[str, Any]]:
    safe_name = _safe_name(name)
    argv = _command_arguments(arguments)
    canonical_absent_paths = _canonical_absent_paths(absent_paths)
    cwd = _resolve_cwd(repo_root, cwd_relative)
    _reserve_name(run_root, safe_name)

    log_relative = f"logs/{safe_name}.log"
    log_path = run_root / log_relative
    log_fd = _open_log(log_path)
    started_at = utc_now()
    start_ns = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    launch_error: OSError | None = None
    console_error: str | None = None
    log_size = 0
    log_sha256 = hashlib.sha256(b"").hexdigest()
    try:
        try:
            _assert_absent_paths(run_root, canonical_absent_paths)
            process = subprocess.Popen(
                argv,
                cwd=os.fspath(cwd),
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                bufsize=0,
                close_fds=True,
            )
        except OSError as exc:
            launch_error = exc
        if process is not None:
            log_size, log_sha256, console_error = _stream_child(
                process,
                log_fd,
                console_fd,
            )
            process.wait()
        os.fsync(log_fd)
    finally:
        os.close(log_fd)
    end_ns = time.monotonic_ns()
    ended_at = utc_now()

    duration_ns, duration_seconds = _duration(start_ns, end_ns)
    returncode = process.returncode if process is not None else None
    termination_signal: dict[str, Any] | None = None
    exit_code: int | None = returncode
    if returncode is not None and returncode < 0:
        number = -returncode
        try:
            signal_name = signal.Signals(number).name
        except ValueError:
            signal_name = f"SIG{number}"
        termination_signal = {"name": signal_name, "number": number}
        exit_code = None

    record: dict[str, Any] = {
        "argv": argv,
        "cwd": {"base": "repository_root", "path": cwd_relative},
        "duration_ns": duration_ns,
        "duration_seconds": duration_seconds,
        "ended_at_utc": ended_at,
        "exit_code": exit_code,
        "format_version": RECORD_FORMAT_VERSION,
        "log": {
            "path": log_relative,
            "sha256": log_sha256,
            "size_bytes": log_size,
        },
        "name": safe_name,
        "preconditions": {
            "absent_paths": canonical_absent_paths,
        },
        "record_type": "command",
        "returncode": returncode,
        "runtime": runtime_metadata(),
        "started_at_utc": started_at,
        "termination_signal": termination_signal,
    }
    recorder_exit: int | None = returncode
    if launch_error is not None:
        recorder_exit = 127 if launch_error.errno == errno.ENOENT else 126
        record["launch_error"] = {
            "errno": launch_error.errno,
            "type": type(launch_error).__name__,
        }
        record["recorder_exit_code"] = recorder_exit
    if console_error is not None:
        record["console_stream_error"] = console_error

    encoded = canonical_json_bytes(record)
    _exclusive_write(run_root / "records" / f"{safe_name}.json", encoded)
    _append_jsonl(run_root / "commands.jsonl", encoded)
    return recorder_exit, record


def record_environment(
    *,
    run_root: Path,
    repo_root: Path,
    name: str,
    cwd_relative: str,
) -> dict[str, Any]:
    """Record only local standard-library runtime/version metadata."""

    safe_name = _safe_name(name)
    _resolve_cwd(repo_root, cwd_relative)
    _reserve_name(run_root, safe_name)
    started_at = utc_now()
    start_ns = time.monotonic_ns()
    metadata = runtime_metadata()
    commit = _repository_identity(repo_root)["head_commit"]
    end_ns = time.monotonic_ns()
    ended_at = utc_now()
    duration_ns, duration_seconds = _duration(start_ns, end_ns)
    record = {
        "cwd": {"base": "repository_root", "path": cwd_relative},
        "duration_ns": duration_ns,
        "duration_seconds": duration_seconds,
        "ended_at_utc": ended_at,
        "format_version": RECORD_FORMAT_VERSION,
        "name": safe_name,
        "record_type": "environment",
        "repository": {"head_commit": commit},
        "runtime": metadata,
        "started_at_utc": started_at,
    }
    encoded = canonical_json_bytes(record)
    _exclusive_write(run_root / "records" / f"{safe_name}.json", encoded)
    _append_jsonl(run_root / "environment.jsonl", encoded)
    return record


def _reraise_signal(number: int) -> NoReturn:
    signal.signal(number, signal.SIG_DFL)
    os.kill(os.getpid(), number)
    os._exit(128 + number)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--cwd",
        default=".",
        help="canonical POSIX path relative to --repo-root (default: .)",
    )
    parser.add_argument(
        "--environment-record",
        action="store_true",
        help="record only privacy-safe Python/platform versions; do not run a command",
    )
    parser.add_argument(
        "--require-absent",
        action="append",
        default=[],
        metavar="RUN_ROOT_RELATIVE_PATH",
        help=(
            "require a canonical run-root-relative output leaf to be absent "
            "immediately before child spawn; repeat for multiple outputs"
        ),
    )
    return parser


def _split_arguments(arguments: Sequence[str]) -> tuple[list[str], list[str] | None]:
    values = list(arguments)
    try:
        separator = values.index("--")
    except ValueError:
        return values, None
    return values[:separator], values[separator + 1 :]


def main(arguments: Sequence[str] | None = None) -> int:
    cli_arguments = list(sys.argv[1:] if arguments is None else arguments)
    option_arguments, command = _split_arguments(cli_arguments)
    parser = _parser()
    try:
        args = parser.parse_args(option_arguments)
        repo_root = _validate_repository_root(args.repo_root)
        safe_name = _safe_name(args.name)
        cwd_relative = _safe_relative_cwd(args.cwd)
        absent_paths = _canonical_absent_paths(args.require_absent)
        if args.environment_record:
            if command is not None:
                raise RecorderInputError("--environment-record must not be combined with --")
            if absent_paths:
                raise RecorderInputError(
                    "--environment-record must not be combined with --require-absent"
                )
        elif command is None:
            raise RecorderInputError("a command must follow an explicit -- separator")
        run_root = prepare_run_root(args.run_root, repo_root)
        if args.environment_record:
            record_environment(
                run_root=run_root,
                repo_root=repo_root,
                name=safe_name,
                cwd_relative=cwd_relative,
            )
            return 0
        assert command is not None
        returncode, record = run_command(
            run_root=run_root,
            repo_root=repo_root,
            name=safe_name,
            cwd_relative=cwd_relative,
            arguments=command,
            absent_paths=absent_paths,
        )
    except RecorderInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RecorderRuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    if returncode is None:
        print("error: command could not be launched", file=sys.stderr)
        return int(record.get("recorder_exit_code", 126))
    if returncode < 0:
        _reraise_signal(-returncode)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
