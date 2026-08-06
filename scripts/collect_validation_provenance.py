#!/usr/bin/env python
"""Collect truthful Git and acceptance-review provenance for validation evidence.

The collector deliberately uses only the Python standard library.  Reviewer
identity must be supplied explicitly; repository ownership, Git configuration,
environment usernames, and authenticated service accounts are never used as
reviewer evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit, urlunsplit


PROVENANCE_FORMAT_VERSION = "validation-provenance-v1"
STATUS_COMMAND = (
    "git",
    "status",
    "--porcelain=v1",
    "--untracked-files=all",
    "--ignore-submodules=none",
)
SUBMODULE_STATUS_COMMAND = ("git", "submodule", "status", "--recursive")
REVIEWER_ENV = "MULTISCREEN_EVIDENCE_REVIEWERS"
_REVIEWER_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z"
)
_COMMIT_RE = re.compile(r"(?:[0-9A-Fa-f]{40}|[0-9A-Fa-f]{64})\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SCP_REMOTE_RE = re.compile(
    r"^(?:(?P<userinfo>[^@\s]+)@)?(?P<host>\[[^\]]+\]|[^:/\s]+):(?P<path>.+)$"
)
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}={0,2}"),
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<name>api[_-]?key|access[_-]?token|auth[_-]?token|token|password|secret)"
    r"(?P<separator>=|%3[dD])(?P<value>[^&/?#\s]+)"
)
_NETWORK_SCHEMES = frozenset({"git", "http", "https", "ssh"})
_REDACTED_LOCAL_REMOTE = "[REDACTED_LOCAL_REMOTE]"
_REDACTED_REMOTE = "[REDACTED_REMOTE]"
_REDACTED_SECRET = "REDACTED_SECRET"


class ProvenanceRuntimeError(RuntimeError):
    """A Git or filesystem operation required for collection failed."""


def parse_reviewers(
    explicit_values: Sequence[str] | str | None = None,
    env_value: str | None = None,
) -> list[str]:
    """Parse, validate, de-duplicate, and sort explicit reviewer handles.

    Repeatable CLI values and the environment value are combined.  Each value
    may contain comma- or whitespace-separated GitHub-style handles.  A leading
    ``@`` is accepted for convenience and removed.  No implicit identity source
    is consulted.
    """

    if isinstance(explicit_values, str):
        values = [explicit_values]
    else:
        values = list(explicit_values or ())
    if env_value is not None:
        values.append(env_value)

    parsed: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("reviewer identifiers must be strings")
        for token in re.split(r"[\s,]+", value.strip()):
            if not token:
                continue
            identifier = token[1:] if token.startswith("@") else token
            if not _REVIEWER_RE.fullmatch(identifier) or "--" in identifier:
                raise ValueError(
                    f"invalid reviewer identifier {token!r}; expected a "
                    "GitHub-style handle"
                )
            parsed.add(identifier)

    if not parsed:
        raise ValueError(
            f"at least one explicit reviewer is required via --reviewer or {REVIEWER_ENV}"
        )
    return sorted(parsed, key=lambda value: (value.casefold(), value))


def _contains_recognized_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in _TOKEN_PATTERNS) or bool(
        _SECRET_ASSIGNMENT_RE.search(value)
    )


def _redact_recognized_secrets(value: str) -> str:
    redacted = value
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub(_REDACTED_SECRET, redacted)
    return _SECRET_ASSIGNMENT_RE.sub(
        lambda match: (
            match.group("name")
            + match.group("separator")
            + _REDACTED_SECRET
        ),
        redacted,
    )


def _is_local_remote(value: str) -> bool:
    return (
        value.startswith(("/", "~/", "./", "../", "\\\\"))
        or _WINDOWS_ABSOLUTE_RE.match(value) is not None
    )


def _safe_remote_path(path: str) -> str:
    decoded = unquote(path)
    if decoded != path and _contains_recognized_secret(decoded):
        return "/" + _REDACTED_SECRET
    return _redact_recognized_secrets(path)


def redact_remote_url(url: str) -> str:
    """Return a privacy-safe network remote or a fail-closed marker.

    Query strings and fragments are never retained. Local filesystem remotes,
    file URIs, malformed URLs, and unsupported remote helpers are replaced as a
    whole. Standard URLs and scp-style SSH remotes lose userinfo and recognized
    secret material.
    """

    if not isinstance(url, str):
        raise TypeError("remote URL must be a string")
    value = url.strip()
    if not value:
        return _REDACTED_REMOTE
    if _is_local_remote(value):
        return _REDACTED_LOCAL_REMOTE

    try:
        parsed = urlsplit(value)
    except ValueError:
        return _REDACTED_REMOTE
    scheme = parsed.scheme.casefold()
    if scheme:
        if scheme == "file":
            return _REDACTED_LOCAL_REMOTE
        if scheme not in _NETWORK_SCHEMES or not parsed.netloc:
            return _REDACTED_REMOTE
        redacted_netloc = parsed.netloc.rsplit("@", 1)[-1]
        if _contains_recognized_secret(unquote(redacted_netloc)):
            return _REDACTED_REMOTE
        return urlunsplit(
            (
                scheme,
                redacted_netloc,
                _safe_remote_path(parsed.path),
                "",
                "",
            )
        )

    scp_value = value.split("#", 1)[0].split("?", 1)[0]
    match = _SCP_REMOTE_RE.fullmatch(scp_value)
    if match is None:
        return _REDACTED_LOCAL_REMOTE
    host = match.group("host")
    if _contains_recognized_secret(unquote(host)):
        return _REDACTED_REMOTE
    return f"{host}:{_safe_remote_path(match.group('path'))}"


def _run_git(
    repo: Path,
    arguments: Sequence[str],
    *,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> bytes:
    command = ["git", "-C", os.fspath(repo), *arguments]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ProvenanceRuntimeError(f"could not execute git: {exc}") from exc
    if completed.returncode not in allowed_returncodes:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        if detail:
            detail = detail.splitlines()[0]
        else:
            detail = f"exit status {completed.returncode}"
        raise ProvenanceRuntimeError(
            f"git {arguments[0] if arguments else '<command>'} failed: {detail}"
        )
    return completed.stdout


def _decode_lines(value: bytes) -> list[str]:
    return [
        line
        for line in value.decode("utf-8", errors="replace").splitlines()
        if line
    ]


def _status_classification(status_bytes: bytes) -> dict[str, Any]:
    staged_count = 0
    unstaged_count = 0
    untracked_count = 0
    conflicted = False
    conflict_codes = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}

    # Porcelain v1 quotes and escapes unusual path bytes by default, so every
    # record produced by the required non-``-z`` command remains one line.
    for record in status_bytes.splitlines():
        if len(record) < 2:
            raise ProvenanceRuntimeError("git returned a malformed porcelain-v1 record")
        code = record[:2].decode("ascii", errors="strict")
        if code == "??":
            untracked_count += 1
            continue
        if code == "!!":
            continue
        if code in conflict_codes:
            conflicted = True
        if code[0] not in {" ", "?", "!"}:
            staged_count += 1
        if code[1] not in {" ", "?", "!"}:
            unstaged_count += 1

    clean = staged_count == 0 and unstaged_count == 0 and untracked_count == 0
    return {
        "clean": clean,
        "conflicted_changes_present": conflicted,
        "staged_change_count": staged_count,
        "staged_changes_present": staged_count > 0,
        "unstaged_change_count": unstaged_count,
        "unstaged_changes_present": unstaged_count > 0,
        "untracked_path_count": untracked_count,
    }


def _submodule_status(
    status_bytes: bytes,
    *,
    timestamp_utc: str,
) -> dict[str, Any]:
    state_names = {
        ord(" "): "at_recorded_commit",
        ord("+"): "commit_mismatch",
        ord("-"): "uninitialized",
        ord("U"): "conflicted",
    }
    state_counts = {state: 0 for state in state_names.values()}
    lines = status_bytes.splitlines()
    for line in lines:
        if not line or line[0] not in state_names:
            raise ProvenanceRuntimeError(
                "git returned a malformed recursive submodule-status record"
            )
        state_counts[state_names[line[0]]] += 1

    if not lines:
        state = "none"
    elif state_counts["conflicted"]:
        state = "conflicted"
    elif state_counts["commit_mismatch"]:
        state = "commit_mismatch"
    elif state_counts["uninitialized"]:
        state = "uninitialized"
    else:
        state = "at_recorded_commit"

    return {
        "byte_count": len(status_bytes),
        "collected_at_utc": timestamp_utc,
        "command": list(SUBMODULE_STATUS_COMMAND),
        "count": len(lines),
        "sha256": hashlib.sha256(status_bytes).hexdigest(),
        "state": state,
        "state_counts": state_counts,
        "status": "recorded",
    }


def collect_worktree(
    repo: str | os.PathLike[str],
    *,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Collect privacy-safe worktree and recursive-submodule observations.

    Raw porcelain and submodule-status content are intentionally not emitted
    because filenames can contain private paths. Exact stdout bytes are hashed
    before decoding, with byte and record counts retained for independent
    verification.
    """

    timestamp = _utc_timestamp(timestamp_utc)
    repo_path = Path(repo).expanduser()
    status_bytes = _run_git(repo_path, STATUS_COMMAND[1:])
    submodule_bytes = _run_git(repo_path, SUBMODULE_STATUS_COMMAND[1:])
    classification = _status_classification(status_bytes)
    return {
        "status": "recorded",
        **classification,
        "collected_at_utc": timestamp,
        "porcelain": {
            "byte_count": len(status_bytes),
            "command": list(STATUS_COMMAND),
            "sha256": hashlib.sha256(status_bytes).hexdigest(),
        },
        "submodules": _submodule_status(
            submodule_bytes,
            timestamp_utc=timestamp,
        ),
    }


def _utc_timestamp(override: str | None) -> str:
    if override is None:
        current = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        return current.isoformat().replace("+00:00", "Z")
    if not isinstance(override, str) or not override.strip():
        raise ValueError("timestamp UTC override must be a non-empty string")
    value = override.strip()
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid UTC timestamp: {override!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("timestamp must include an explicit UTC offset or trailing Z")
    return parsed.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _optional_text(value: str | None, *, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(normalized) > maximum or any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field} contains invalid characters or is too long")
    return normalized


def _optional_bool(value: bool | str | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError("raw-events-reviewed must be true or false")


def _collect_remotes(repo: Path) -> list[dict[str, Any]]:
    remote_names = sorted(_decode_lines(_run_git(repo, ("remote",))))
    remotes: list[dict[str, Any]] = []
    for name in remote_names:
        fetch_urls = sorted(
            {
                redact_remote_url(url)
                for url in _decode_lines(
                    _run_git(repo, ("remote", "get-url", "--all", name))
                )
            }
        )
        push_urls = sorted(
            {
                redact_remote_url(url)
                for url in _decode_lines(
                    _run_git(repo, ("remote", "get-url", "--all", "--push", name))
                )
            }
        )
        remotes.append(
            {
                "fetch_urls": fetch_urls,
                "name": _redact_recognized_secrets(name),
                "push_urls": push_urls,
            }
        )
    return remotes


def collect_provenance(
    repo: str | os.PathLike[str] = ".",
    reviewers: Sequence[str] | str | None = None,
    review_method: str | None = None,
    review_commit: str | None = None,
    raw_events_reviewed: bool | str | None = None,
    timestamp_utc: str | None = None,
    env_value: str | None = None,
) -> dict[str, Any]:
    """Collect deterministic evidence-handoff provenance."""

    reviewer_ids = parse_reviewers(reviewers, env_value)
    timestamp = _utc_timestamp(timestamp_utc)

    # Resolve the requested repository before validating review-only fields so a
    # missing or unreadable repository remains an operational error (exit 4).
    requested_repo = Path(repo).expanduser()
    top_level_bytes = _run_git(requested_repo, ("rev-parse", "--show-toplevel"))
    top_level_text = top_level_bytes.decode("utf-8", errors="strict").strip()
    if not top_level_text:
        raise ProvenanceRuntimeError("git returned an empty worktree root")
    repo_path = Path(top_level_text)

    method = _optional_text(review_method, field="review method", maximum=256)
    if method is None:
        raise ValueError("a non-empty review method is required")
    raw_reviewed = _optional_bool(raw_events_reviewed)
    if raw_reviewed is None:
        raise ValueError("raw-events-reviewed must be supplied explicitly")
    commit = _optional_text(review_commit, field="review commit", maximum=64)
    if commit is None or not _COMMIT_RE.fullmatch(commit):
        raise ValueError(
            "review commit must be a full 40- or 64-character hexadecimal object ID"
        )
    commit = commit.lower()
    head_commit = _run_git(repo_path, ("rev-parse", "--verify", "HEAD"))
    head = head_commit.decode("ascii", errors="strict").strip().lower()
    if not _COMMIT_RE.fullmatch(head):
        raise ProvenanceRuntimeError("git returned an invalid HEAD object ID")

    branch_bytes = _run_git(
        repo_path,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        allowed_returncodes=(0, 1),
    )
    branch_value = branch_bytes.decode("utf-8", errors="replace").strip() or None
    branch = {
        "status": "recorded" if branch_value is not None else "not_applicable",
        "value": branch_value,
    }
    worktree = collect_worktree(repo_path, timestamp_utc=timestamp)

    reviewer_entries = [
        {
            "identifier": identifier,
            "raw_events_reviewed": raw_reviewed,
            "review_commit": commit,
            "review_method": method,
            "reviewed_at_utc": timestamp,
            "role": "evidence_reviewer",
        }
        for identifier in reviewer_ids
    ]
    return {
        "acceptance_review": {
            "reviewers": reviewer_entries,
            "status": "recorded",
        },
        "collected_at_utc": timestamp,
        "context": "evidence_handoff",
        "repository": {
            "branch": branch,
            "detached_head": branch_value is None,
            "head_commit": head,
            "name": repo_path.name,
            "remotes": _collect_remotes(repo_path),
            "root_kind": "git_worktree",
            "worktree": worktree,
        },
        "format_version": PROVENANCE_FORMAT_VERSION,
    }


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _human_summary(provenance: dict[str, Any]) -> str:
    repository = provenance["repository"]
    worktree = repository["worktree"]
    reviewers = provenance["acceptance_review"]["reviewers"]
    branch = repository["branch"]["value"] or "(detached HEAD)"
    reviewer_text = ", ".join(entry["identifier"] for entry in reviewers)
    return "\n".join(
        (
            "Validation provenance collected",
            f"  timestamp UTC: {provenance['collected_at_utc']}",
            f"  repository: {repository['name']}",
            f"  branch: {branch}",
            f"  HEAD: {repository['head_commit']}",
            f"  worktree clean: {str(worktree['clean']).lower()}",
            f"  staged changes: {worktree['staged_change_count']}",
            f"  unstaged changes: {worktree['unstaged_change_count']}",
            f"  untracked paths: {worktree['untracked_path_count']}",
            f"  reviewers: {reviewer_text}",
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Git and explicit reviewer provenance for validation evidence."
    )
    parser.add_argument("--repo", default=".", help="path within the Git worktree")
    parser.add_argument(
        "--reviewer",
        action="append",
        default=None,
        help="explicit reviewer handle; repeat for multiple reviewers",
    )
    parser.add_argument("--review-method", default=None)
    parser.add_argument("--review-commit", default=None)
    parser.add_argument(
        "--raw-events-reviewed",
        choices=("true", "false"),
        default=None,
    )
    parser.add_argument(
        "--timestamp-utc",
        default=None,
        help="deterministic UTC ISO-8601 timestamp override",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="write canonical machine-readable JSON to standard output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        provenance = collect_provenance(
            repo=args.repo,
            reviewers=args.reviewer,
            review_method=args.review_method,
            review_commit=args.review_commit,
            raw_events_reviewed=args.raw_events_reviewed,
            timestamp_utc=args.timestamp_utc,
            env_value=os.environ.get(REVIEWER_ENV),
        )
        serialized = _canonical_json(provenance)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError, ProvenanceRuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    try:
        if args.output is not None:
            with args.output.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
    except OSError as exc:
        print(f"error: could not write {args.output}: {exc}", file=sys.stderr)
        return 4

    if args.json:
        sys.stdout.write(serialized)
    else:
        print(_human_summary(provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
