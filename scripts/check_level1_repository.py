#!/usr/bin/env python3
"""Read-only, standard-library repository checks for Level 1 evidence.

Only paths recorded in the Git index are treated as repository artifacts.  A
successful report contains aggregate counts and hashes, never filesystem paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


REPORT_VERSION = "multiscreen-level1-repository-check-v1"
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024
HEX40_OR_64 = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})(?:[^`]*|[^~]*)$")
MARKDOWN_LINK_START = re.compile(r"!?\[[^\]\n]*\]\(")
MARKDOWN_REFERENCE_DEFINITION = re.compile(
    r"(?m)^[ ]{0,3}\[[^\]\n]+\]:[ \t]*(?:<([^>\n]+)>|(\S+))"
)
WORKFLOW_KEY = re.compile(r"^([A-Za-z0-9_.-]+):(?:[ \t]*(.*))?$")
YAML_GRAPH_TOKEN = re.compile(r"(?:^|(?<=[\s\[{,:-]))[&*!][A-Za-z0-9_.-]+")
PRIVACY_TEXT_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)
PRIVACY_FIXTURE_PATHS = frozenset(
    {
        "tests/test_validation_evidence.py",
        "tests/test_validation_evidence_collector_hardening.py",
        "tests/test_validation_evidence_common_hardening.py",
        "tests/test_validation_evidence_verifier_hardening.py",
    }
)
PRIVATE_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|Users|media)/[A-Za-z0-9._-]+(?:/[^\s\"'<>]+)?"
)
PRIVATE_WINDOWS_PATH = re.compile(
    r"\b[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9._-]+(?:[\\/][^\s\"'<>]+)?"
)
CREDENTIAL_URL = re.compile(r"(?i)(?:https?|ssh|git)://[^/@\s]+@")
PRIVATE_FILE_URI = re.compile(
    r"(?i)(?<![A-Za-z0-9])file://(?:localhost)?/(?!<REDACTED)[^\s\"'<>]+"
)
TOKEN_PATTERNS = (
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}={0,2}"),
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])"
    r"(?:password|api[-_]?key|access[-_]?token|auth[-_]?token|token|secret|"
    r"client[-_]?secret|private[-_]?key|aws_access_key_id|aws_secret_access_key|"
    r"aws_session_token)(?![A-Za-z0-9_-])[ \t]*[:=][ \t]*"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;#}\]]+)"
)


class RepositoryCheckError(ValueError):
    """A repository input or checked contract is invalid."""


class RepositoryRuntimeError(RuntimeError):
    """A required read-only operation could not be completed."""


def _fail(message: str) -> None:
    raise RepositoryCheckError(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RepositoryRuntimeError(f"cannot inspect {label}") from exc
        if stat.S_ISLNK(mode):
            _fail(f"{label} must not traverse a symlink")


def _git(repo: Path, arguments: Sequence[str], *, allow: Iterable[int] = (0,)) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repo), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RepositoryRuntimeError("git could not be executed") from exc
    if result.returncode not in set(allow):
        raise RepositoryRuntimeError(
            f"git {' '.join(arguments[:2])} failed with exit {result.returncode}"
        )
    return result.stdout


def validate_repository_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        _fail("--repo-root must be absolute")
    _reject_symlink_components(candidate, label="repository root")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise RepositoryRuntimeError("repository root cannot be resolved") from exc
    if not root.is_dir():
        _fail("--repo-root must name a directory")
    raw = _git(root, ["rev-parse", "--show-toplevel"])
    try:
        top = Path(raw.decode("utf-8", errors="strict").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError) as exc:
        raise RepositoryRuntimeError("Git top-level path is invalid") from exc
    if top != root:
        _fail("--repo-root must be the Git top-level directory")
    return root


class TrackedEntry:
    __slots__ = ("mode", "object_id", "path")

    def __init__(self, mode: str, object_id: str, path: str) -> None:
        self.mode = mode
        self.object_id = object_id
        self.path = path


def _safe_tracked_path(raw: bytes) -> str:
    try:
        path = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RepositoryCheckError("tracked path is not valid UTF-8") from exc
    if not path or path.startswith(("/", "~")) or "\\" in path:
        _fail("tracked path is non-canonical")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("tracked path is unsafe")
    if pure.as_posix() != path:
        _fail("tracked path is non-canonical")
    return path


def tracked_entries(repo: Path) -> list[TrackedEntry]:
    raw = _git(repo, ["ls-files", "--stage", "-z"])
    entries: list[TrackedEntry] = []
    seen: set[str] = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            metadata, raw_path = item.split(b"\t", 1)
            mode_raw, object_raw, stage_raw = metadata.split(b" ", 2)
            mode = mode_raw.decode("ascii")
            object_id = object_raw.decode("ascii")
            stage = stage_raw.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RepositoryCheckError("malformed Git index entry") from exc
        path = _safe_tracked_path(raw_path)
        if stage != "0" or path in seen:
            _fail("Git index contains an unmerged or duplicate path")
        if mode == "120000":
            _fail("tracked symlinks are forbidden")
        if mode not in {"100644", "100755", "160000"}:
            _fail("tracked entry has an unsupported file mode")
        if not HEX40_OR_64.fullmatch(object_id):
            _fail("tracked entry has an invalid object id")
        seen.add(path)
        entries.append(TrackedEntry(mode, object_id, path))
    entries.sort(key=lambda entry: entry.path.encode("utf-8"))
    return entries


def _regular_file(repo: Path, entry: TrackedEntry) -> Path:
    if entry.mode == "160000":
        _fail("a Git submodule cannot be read as a regular tracked file")
    candidate = repo.joinpath(*PurePosixPath(entry.path).parts)
    _reject_symlink_components(candidate, label="tracked file")
    try:
        status = candidate.stat()
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo)
    except (OSError, ValueError) as exc:
        raise RepositoryCheckError("tracked file is missing or escapes the repository") from exc
    if not stat.S_ISREG(status.st_mode):
        _fail("tracked artifact is not a regular file")
    return resolved


def _blob_object_id(raw: bytes, expected: str) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    if len(expected) == 40:
        digest = hashlib.sha1()
    elif len(expected) == 64:
        digest = hashlib.sha256()
    else:
        _fail("tracked entry has an unsupported object id length")
    digest.update(header)
    digest.update(raw)
    return digest.hexdigest()


def _read_tracked(repo: Path, entry: TrackedEntry) -> bytes:
    path = _regular_file(repo, entry)
    try:
        before = path.stat()
        if before.st_size > MAX_TRACKED_FILE_BYTES:
            _fail("tracked artifact exceeds the maximum allowed size")
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise RepositoryRuntimeError("tracked file could not be read") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(raw) != after.st_size:
        _fail("tracked file changed while it was inspected")
    if _blob_object_id(raw, entry.object_id) != entry.object_id:
        _fail("tracked working-tree bytes do not match the Git index object")
    return raw


def _aggregate(records: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for path, raw in sorted(records, key=lambda item: item[0].encode("utf-8")):
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def _duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    _fail(f"JSON contains forbidden non-finite value {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail("JSON float overflows to a non-finite value")
    return parsed


def check_json(repo: Path, entries: Sequence[TrackedEntry]) -> dict[str, Any]:
    records: list[tuple[str, bytes]] = []
    for entry in entries:
        if PurePosixPath(entry.path).suffix.lower() != ".json":
            continue
        raw = _read_tracked(repo, entry)
        if raw.startswith(b"\xef\xbb\xbf"):
            _fail("JSON must not contain a UTF-8 BOM")
        try:
            text = raw.decode("utf-8", errors="strict")
            json.loads(
                text,
                object_pairs_hook=_duplicate_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except RepositoryCheckError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise RepositoryCheckError("tracked JSON is malformed") from exc
        records.append((entry.path, raw))
    if not records:
        _fail("repository has no tracked JSON files")
    return {
        "artifact_count": len(records),
        "artifact_manifest_sha256": _aggregate(records),
        "bytes_checked": sum(len(raw) for _, raw in records),
    }


def _outside_fenced_code(text: str) -> str:
    output: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in text.splitlines(keepends=True):
        match = FENCE.match(line.rstrip("\r\n"))
        if fence_character is None:
            if match:
                marker = match.group(2)
                fence_character = marker[0]
                fence_length = len(marker)
                output.append("\n" if line.endswith("\n") else "")
            else:
                output.append(line)
            continue
        stripped = line.lstrip(" ")
        marker = fence_character * fence_length
        if len(line) - len(stripped) <= 3 and stripped.startswith(marker):
            tail = stripped[len(marker) :].strip()
            if not tail or set(tail) <= {fence_character}:
                fence_character = None
                fence_length = 0
        output.append("\n" if line.endswith("\n") else "")
    if fence_character is not None:
        _fail("Markdown contains an unterminated fenced code block")
    return "".join(output)


def _link_destinations(text: str) -> list[str]:
    destinations: list[str] = []
    position = 0
    while True:
        match = MARKDOWN_LINK_START.search(text, position)
        if match is None:
            return destinations
        cursor = match.end()
        depth = 1
        escaped = False
        angle = False
        buffer: list[str] = []
        while cursor < len(text):
            character = text[cursor]
            cursor += 1
            if escaped:
                buffer.append(character)
                escaped = False
                continue
            if character == "\\":
                escaped = True
                buffer.append(character)
                continue
            if character == "<" and depth == 1 and not buffer:
                angle = True
                buffer.append(character)
                continue
            if character == ">" and angle:
                angle = False
                buffer.append(character)
                continue
            if not angle and character == "(":
                depth += 1
            elif not angle and character == ")":
                depth -= 1
                if depth == 0:
                    break
            buffer.append(character)
        if depth != 0:
            _fail("Markdown contains an unterminated inline link")
        body = "".join(buffer).strip()
        if body.startswith("<"):
            end = body.find(">")
            if end < 0:
                _fail("Markdown angle-bracket destination is malformed")
            destination = body[1:end]
        else:
            destination = body.split(None, 1)[0] if body else ""
        destinations.append(destination)
        position = cursor


def _decode_link(value: str) -> str:
    try:
        raw = urllib.parse.unquote_to_bytes(value)
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RepositoryCheckError("Markdown link is not valid UTF-8 after URL decoding") from exc


def check_markdown_links(repo: Path, entries: Sequence[TrackedEntry]) -> dict[str, Any]:
    tracked_paths = {entry.path for entry in entries if entry.mode != "160000"}
    tracked_directories: set[str] = {"."}
    for path in tracked_paths:
        parent = PurePosixPath(path).parent
        while parent.as_posix() not in {".", ""}:
            tracked_directories.add(parent.as_posix())
            parent = parent.parent
    records: list[tuple[str, bytes]] = []
    link_count = 0
    local_count = 0
    for entry in entries:
        if PurePosixPath(entry.path).suffix.lower() != ".md":
            continue
        raw = _read_tracked(repo, entry)
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RepositoryCheckError("tracked Markdown is not valid UTF-8") from exc
        if text.startswith("\ufeff"):
            _fail("Markdown must not contain a UTF-8 BOM")
        searchable = _outside_fenced_code(text)
        destinations = _link_destinations(searchable)
        destinations.extend(
            first or second
            for first, second in MARKDOWN_REFERENCE_DEFINITION.findall(searchable)
        )
        for raw_destination in destinations:
            link_count += 1
            destination_without_fragment = raw_destination.split("#", 1)[0]
            if not destination_without_fragment:
                continue
            parsed = urllib.parse.urlsplit(destination_without_fragment)
            if parsed.scheme or parsed.netloc or destination_without_fragment.startswith("//"):
                continue
            decoded = _decode_link(parsed.path)
            if (
                not decoded
                or decoded.startswith(("/", "~"))
                or WINDOWS_ABSOLUTE.match(decoded)
                or "\\" in decoded
            ):
                _fail("Markdown local link has an unsafe destination")
            base = PurePosixPath(entry.path).parent
            parts: list[str] = []
            for component in (base / PurePosixPath(decoded)).parts:
                if component in {"", "."}:
                    continue
                if component == "..":
                    if not parts:
                        _fail("Markdown local link escapes the repository")
                    parts.pop()
                else:
                    parts.append(component)
            normalized = PurePosixPath(*parts).as_posix() if parts else "."
            if normalized not in tracked_paths and normalized not in tracked_directories:
                _fail("Markdown local link target is missing from tracked artifacts")
            target = repo if normalized == "." else repo.joinpath(*PurePosixPath(normalized).parts)
            _reject_symlink_components(target, label="Markdown local link target")
            local_count += 1
        records.append((entry.path, raw))
    if not records:
        _fail("repository has no tracked Markdown files")
    return {
        "artifact_count": len(records),
        "artifact_manifest_sha256": _aggregate(records),
        "bytes_checked": sum(len(raw) for _, raw in records),
        "link_count": link_count,
        "local_link_count": local_count,
    }


def _strip_yaml_comment(value: str) -> str:
    single = False
    double = False
    escaped = False
    output: list[str] = []
    for index, character in enumerate(value):
        if escaped:
            output.append(character)
            escaped = False
            continue
        if character == "\\" and double:
            output.append(character)
            escaped = True
            continue
        if character == "'" and not double:
            single = not single
        elif character == '"' and not single:
            double = not double
        if character == "#" and not single and not double and (
            index == 0 or value[index - 1].isspace()
        ):
            break
        output.append(character)
    if single or double:
        _fail("workflow YAML contains an unterminated quoted scalar")
    return "".join(output).rstrip()


def _validate_flow_balance(value: str) -> None:
    stack: list[str] = []
    pairs = {"]": "[", "}": "{"}
    single = False
    double = False
    escaped = False
    for character in value:
        if escaped:
            escaped = False
            continue
        if character == "\\" and double:
            escaped = True
            continue
        if character == "'" and not double:
            single = not single
            continue
        if character == '"' and not single:
            double = not double
            continue
        if single or double:
            continue
        if character in "[{":
            stack.append(character)
        elif character in "]}":
            if not stack or stack.pop() != pairs[character]:
                _fail("workflow YAML has unbalanced flow collections")
    if stack or single or double:
        _fail("workflow YAML has an unterminated flow collection")



def _reject_yaml_graph_syntax(value: str) -> None:
    if YAML_GRAPH_TOKEN.search(value):
        _fail("workflow YAML anchors, aliases, and tags are unsupported")


def _validate_restricted_scalar(value: str) -> None:
    if not value or value in {"|", ">", "|-", ">-", "|+", ">+"}:
        return
    if value[0] in {"[", "{"}:
        try:
            json.loads(
                value,
                object_pairs_hook=_duplicate_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except RepositoryCheckError:
            raise
        except (json.JSONDecodeError, RecursionError) as exc:
            raise RepositoryCheckError(
                "workflow YAML flow collections must use strict JSON syntax"
            ) from exc
        return
    if value[0] in {"'", '"'}:
        return
    if value[0] in set("-?:,[]{}#&*!|>'\"%@`"):
        _fail("workflow YAML plain scalar starts with a reserved indicator")
    if ": " in value:
        _fail("workflow YAML plain scalar contains an unquoted colon-space")


def _validate_restricted_yaml_structure(text: str) -> None:
    """Reject YAML outside the small, indentation-based workflow subset."""

    allowed_child_indents: set[int] = set()
    mapping_keys: dict[int, set[str]] = {}
    block_scalar_indent: int | None = None
    for raw_line in text.splitlines():
        if "\t" in raw_line:
            _fail("workflow YAML must not contain tab characters")
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if block_scalar_indent is not None:
            if indent >= block_scalar_indent:
                continue
            block_scalar_indent = None
        if indent % 2:
            _fail("workflow YAML indentation must use two-space levels")
        if indent != 0 and indent not in allowed_child_indents:
            _fail("workflow YAML indentation follows a scalar or unsupported parent")
        allowed_child_indents = {
            value for value in allowed_child_indents if value <= indent
        }
        mapping_keys = {
            level: keys for level, keys in mapping_keys.items() if level <= indent
        }
        content = _strip_yaml_comment(raw_line[indent:])
        if not content:
            continue
        _validate_flow_balance(content)
        _reject_yaml_graph_syntax(content)
        if content.startswith(("---", "...", "%")):
            _fail("workflow YAML directives and extra documents are unsupported")
        opens_children = False
        if content.startswith("- "):
            candidate = content[2:].strip()
            if not candidate:
                _fail("workflow YAML sequence entry must be non-empty")
            inline = WORKFLOW_KEY.match(candidate)
            if inline is not None:
                key, raw_scalar = inline.groups()
                mapping_keys.pop(indent + 2, None)
                mapping_keys[indent + 2] = {key}
                _validate_restricted_scalar((raw_scalar or "").strip())
                opens_children = True
            else:
                _validate_restricted_scalar(candidate)
        else:
            match = WORKFLOW_KEY.match(content)
            if match is None:
                _fail("workflow YAML contains unsupported mapping syntax")
            key, raw_scalar = match.groups()
            keys = mapping_keys.setdefault(indent, set())
            if key in keys:
                _fail("workflow YAML contains a duplicate mapping key")
            keys.add(key)
            scalar = (raw_scalar or "").strip()
            _validate_restricted_scalar(scalar)
            opens_children = not scalar
            if scalar in {"|", ">", "|-", ">-", "|+", ">+"}:
                block_scalar_indent = indent + 2
        if opens_children:
            allowed_child_indents.add(indent + 2)


def _validate_workflow_subset(text: str) -> int:
    """Validate the restricted structural subset used by this repository."""

    _validate_restricted_yaml_structure(text)
    top_keys: set[str] = set()
    job_keys: set[str] = set()
    jobs_indent: int | None = None
    block_scalar_indent: int | None = None
    current_job: str | None = None
    current_steps_job: str | None = None
    current_step_commands: set[str] | None = None
    job_has_body: dict[str, bool] = {}
    job_has_runs_on: dict[str, bool] = {}
    job_has_steps: dict[str, bool] = {}
    job_has_uses: dict[str, bool] = {}
    job_level_keys: dict[str, set[str]] = {}

    def finish_step() -> None:
        nonlocal current_step_commands
        if current_step_commands is not None and len(current_step_commands) != 1:
            _fail("workflow step must declare exactly one of run or uses")
        current_step_commands = None

    def record_step_command(key: str, scalar: str) -> None:
        if current_step_commands is None:
            _fail("workflow steps must be a block sequence")
        if not scalar:
            _fail("workflow step run or uses must be a non-empty scalar")
        if key in current_step_commands:
            _fail("workflow step contains a duplicate command key")
        current_step_commands.add(key)

    for raw_line in text.splitlines():
        if "\t" in raw_line:
            _fail("workflow YAML must not contain tab characters")
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if block_scalar_indent is not None:
            if indent >= block_scalar_indent:
                continue
            block_scalar_indent = None
        if current_steps_job is not None and indent <= 4:
            finish_step()
            current_steps_job = None
        if indent % 2:
            _fail("workflow YAML indentation must use two-space levels")
        content = _strip_yaml_comment(raw_line[indent:])
        if not content:
            continue
        _validate_flow_balance(content)
        _reject_yaml_graph_syntax(content)
        if content.startswith(("---", "...", "%")):
            _fail("workflow YAML directives and extra documents are unsupported")
        if content.startswith("- "):
            if indent < 2:
                _fail("workflow YAML has a sequence at an invalid structural level")
            candidate = content[2:].strip()
            if not candidate:
                _fail("workflow YAML sequence entry must be non-empty")
            if current_steps_job is not None and indent == 6:
                finish_step()
                current_step_commands = set()
                job_has_steps[current_steps_job] = True
                inline = WORKFLOW_KEY.match(candidate)
                if inline is None:
                    _fail("workflow step must be a mapping")
                step_key, raw_step_scalar = inline.groups()
                if step_key in {"run", "uses"}:
                    record_step_command(
                        step_key, (raw_step_scalar or "").strip()
                    )
            continue
        match = WORKFLOW_KEY.match(content)
        if match is None:
            _fail("workflow YAML contains unsupported mapping syntax")
        key, raw_scalar = match.groups()
        scalar = (raw_scalar or "").strip()
        if current_steps_job is not None:
            if indent == 6:
                _fail("workflow steps must be a block sequence")
            if indent == 8 and key in {"run", "uses"}:
                record_step_command(key, scalar)
        if scalar in {"|", ">", "|-", ">-", "|+", ">+"}:
            block_scalar_indent = indent + 2
        if indent == 0:
            if key in top_keys:
                _fail("workflow YAML contains a duplicate top-level key")
            top_keys.add(key)
            current_job = None
            current_steps_job = None
            if key == "jobs":
                if scalar:
                    _fail("workflow jobs must be a block mapping")
                jobs_indent = 0
            else:
                jobs_indent = None
        elif jobs_indent == 0 and indent == 2:
            if key in job_keys:
                _fail("workflow YAML contains a duplicate job id")
            job_keys.add(key)
            current_job = key
            current_steps_job = None
            job_has_body[key] = False
            job_has_runs_on[key] = False
            job_has_steps[key] = False
            job_has_uses[key] = False
            job_level_keys[key] = set()
            if scalar:
                _fail("workflow job must be a block mapping")
        elif current_job is not None and indent >= 4:
            job_has_body[current_job] = True
            if indent == 4:
                if key in job_level_keys[current_job]:
                    _fail("workflow job contains a duplicate top-level key")
                job_level_keys[current_job].add(key)
                if key == "runs-on":
                    if not scalar:
                        _fail("workflow runs-on must be a non-empty scalar")
                    job_has_runs_on[current_job] = True
                elif key == "uses":
                    if not scalar:
                        _fail("workflow uses must be a non-empty scalar")
                    job_has_uses[current_job] = True
                elif key == "steps":
                    if scalar:
                        _fail("workflow steps must be a block sequence")
                    current_steps_job = current_job
                    current_step_commands = None
    finish_step()
    if "jobs" not in top_keys or not job_keys:
        _fail("workflow YAML must contain a non-empty jobs mapping")
    if any(not job_has_body[name] for name in job_keys):
        _fail("workflow YAML job mapping must not be empty")
    for name in job_keys:
        if job_has_uses[name]:
            if job_has_runs_on[name] or job_has_steps[name]:
                _fail("reusable workflow jobs must use uses without runs-on or steps")
        elif not job_has_runs_on[name] or not job_has_steps[name]:
            _fail("workflow job must declare runs-on and a non-empty steps sequence")
    return len(job_keys)

def check_workflow_yaml(repo: Path, entries: Sequence[TrackedEntry]) -> dict[str, Any]:
    records: list[tuple[str, bytes]] = []
    job_count = 0
    for entry in entries:
        pure = PurePosixPath(entry.path)
        if (
            len(pure.parts) < 3
            or pure.parts[:2] != (".github", "workflows")
            or pure.suffix.lower() not in {".yml", ".yaml"}
        ):
            continue
        raw = _read_tracked(repo, entry)
        if not raw or raw.startswith(b"\xef\xbb\xbf"):
            _fail("workflow YAML must be non-empty UTF-8 without a BOM")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RepositoryCheckError("workflow YAML is not valid UTF-8") from exc
        job_count += _validate_workflow_subset(text)
        records.append((entry.path, raw))
    if not records:
        _fail("repository has no tracked GitHub Actions workflow YAML")
    return {
        "artifact_count": len(records),
        "artifact_manifest_sha256": _aggregate(records),
        "bytes_checked": sum(len(raw) for _, raw in records),
        "job_count": job_count,
        "parser": "stdlib_github_actions_restricted_v2",
    }


FORBIDDEN_COMPONENTS = {
    "__pycache__",
    ".pytest_cache",
    ".validation_evidence_staging",
    "checkpoint",
    "checkpoints",
    "model-weights",
    "model_weights",
    "optimizer-state",
    "optimizer_state",
    "outputs",
    "runs",
    "wandb",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bin",
    ".bz2",
    ".cab",
    ".ckpt",
    ".gguf",
    ".gz",
    ".h5",
    ".hdf5",
    ".jsonl",
    ".lz",
    ".lz4",
    ".log",
    ".npy",
    ".npz",
    ".out",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".rar",
    ".safetensors",
    ".tar",
    ".tgz",
    ".tflite",
    ".xz",
    ".zip",
    ".zst",
}


def _forbidden_artifact(path: str) -> bool:
    pure = PurePosixPath(path)
    lowered_parts = tuple(part.casefold() for part in pure.parts)
    if any(
        part in FORBIDDEN_COMPONENTS or part.startswith("checkpoint-")
        for part in lowered_parts
    ):
        return True
    if lowered_parts and lowered_parts[0] == "third_party" and ".git" in lowered_parts:
        return True
    lowered_name = pure.name.casefold()
    return any(lowered_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)

def _submodule_summary(raw: bytes) -> dict[str, Any]:
    counts = {"clean": 0, "missing": 0, "modified": 0, "conflict": 0}
    for line in raw.splitlines():
        if not line:
            continue
        state = chr(line[0])
        category = {" ": "clean", "-": "missing", "+": "modified", "U": "conflict"}.get(state)
        if category is None:
            _fail("git submodule status contains an unknown state")
        counts[category] += 1
    return {
        "byte_count": len(raw),
        "record_count": sum(counts.values()),
        "sha256": _sha256(raw),
        "states": counts,
    }



def _safe_secret_assignment(match: re.Match[str]) -> bool:
    value = match.group("value").strip("\"'")
    normalized = value.casefold()
    return (
        not value
        or value.startswith("$")
        or (value.startswith("<") and value.endswith(">"))
        or normalized
        in {"false", "none", "not_recorded", "null", "true", "unset"}
    )


def _privacy_summary(
    repo: Path,
    records: Sequence[tuple[str, bytes]],
) -> dict[str, Any]:
    try:
        home = Path.home().resolve()
    except OSError as exc:
        raise RepositoryRuntimeError("cannot resolve the current account home") from exc
    sensitive_literals = {os.fspath(repo)}
    if home != Path("/"):
        sensitive_literals.add(os.fspath(home))
        if home.name:
            sensitive_literals.add(f"/media/{home.name}")
    scanned: list[tuple[str, bytes]] = []
    fixture_exemptions = 0
    for path, raw in records:
        suffix = PurePosixPath(path).suffix.casefold()
        if suffix not in PRIVACY_TEXT_SUFFIXES:
            continue
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RepositoryCheckError("tracked text artifact is not valid UTF-8") from exc
        scanned.append((path, raw))
        if any(value and value in text for value in sensitive_literals):
            _fail("tracked text contains the tested repository or account private root")
        fixture = path in PRIVACY_FIXTURE_PATHS
        if fixture:
            fixture_exemptions += 1
            continue
        if PRIVATE_POSIX_PATH.search(text) or PRIVATE_WINDOWS_PATH.search(text):
            _fail("tracked text contains a private user or media path")
        if CREDENTIAL_URL.search(text):
            _fail("tracked text contains a credential-bearing URL")
        if PRIVATE_FILE_URI.search(text):
            _fail("tracked text contains a private file URI")
        if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
            _fail("tracked text contains a high-confidence credential token")
        if suffix != ".py" and any(
            not _safe_secret_assignment(match)
            for match in SECRET_ASSIGNMENT.finditer(text)
        ):
            _fail("tracked shareable text contains a secret assignment")
    if not scanned:
        _fail("repository has no tracked UTF-8 text artifacts for privacy review")
    return {
        "artifact_count": len(scanned),
        "artifact_manifest_sha256": _aggregate(scanned),
        "bytes_checked": sum(len(raw) for _, raw in scanned),
        "fixture_exemption_artifact_count": fixture_exemptions,
        "rules": [
            "account-and-repository-root",
            "credential-token",
            "credential-url",
            "file-uri",
            "private-user-path",
            "shareable-secret-assignment",
        ],
        "status": "passed",
    }

def check_hygiene(repo: Path, entries: Sequence[TrackedEntry]) -> dict[str, Any]:
    porcelain = _git(
        repo,
        ["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
    )
    if porcelain:
        _fail("worktree must be clean")
    _git(repo, ["diff", "--check"])
    _git(repo, ["diff", "--cached", "--check"])
    submodule = _git(repo, ["submodule", "status", "--recursive"])
    records: list[tuple[str, bytes]] = []
    gitlink_count = 0
    for entry in entries:
        if _forbidden_artifact(entry.path):
            _fail("tracked forbidden output, checkpoint, archive, weight, or cache artifact")
        if entry.mode == "160000":
            gitlink_count += 1
            continue
        raw = _read_tracked(repo, entry)
        status = _regular_file(repo, entry).stat()
        if status.st_nlink != 1:
            _fail("tracked file must not be hard-linked")
        if len(raw) > MAX_TRACKED_FILE_BYTES:
            _fail("tracked file exceeds the Level 1 size limit")
        records.append((entry.path, raw))
    head = _git(repo, ["rev-parse", "HEAD"]).strip().decode("ascii", errors="strict")
    if not HEX40_OR_64.fullmatch(head):
        _fail("repository HEAD is not a full commit id")
    return {
        "artifact_count": len(records),
        "artifact_manifest_sha256": _aggregate(records),
        "bytes_checked": sum(len(raw) for _, raw in records),
        "gitlink_count": gitlink_count,
        "head_commit": head,
        "index_diff_check": "passed",
        "maximum_tracked_file_bytes": MAX_TRACKED_FILE_BYTES,
        "privacy": _privacy_summary(repo, records),
        "submodules": _submodule_summary(submodule),
        "worktree": {
            "clean": True,
            "porcelain_byte_count": len(porcelain),
            "porcelain_sha256": _sha256(porcelain),
        },
        "worktree_diff_check": "passed",
    }


CHECKS: dict[str, Callable[[Path, Sequence[TrackedEntry]], dict[str, Any]]] = {
    "json": check_json,
    "markdown-links": check_markdown_links,
    "workflow-yaml": check_workflow_yaml,
    "hygiene": check_hygiene,
}


def _clean_head(repo: Path) -> str:
    porcelain = _git(
        repo,
        ["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
    )
    if porcelain:
        _fail("repository check requires a clean tested-source worktree")
    head = _git(repo, ["rev-parse", "HEAD"]).strip().decode("ascii", errors="strict")
    if not HEX40_OR_64.fullmatch(head):
        _fail("repository HEAD is not a full commit id")
    return head


def run_check(repo_root: str | os.PathLike[str], check: str) -> dict[str, Any]:
    repo = validate_repository_root(repo_root)
    head_before = _clean_head(repo)
    entries = tracked_entries(repo)
    if not entries:
        _fail("Git index contains no tracked artifacts")
    result = CHECKS[check](repo, entries)
    head_after = _clean_head(repo)
    if head_after != head_before:
        _fail("repository HEAD changed while it was checked")
    return {
        "check": check,
        "format_version": REPORT_VERSION,
        "head_commit": head_before,
        "result": result,
        "status": "passed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--check", required=True, choices=tuple(CHECKS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_check(args.repo_root, args.check)
    except RepositoryCheckError as exc:
        print(f"repository check failed: {exc}", file=sys.stderr)
        return 2
    except (OSError, RepositoryRuntimeError) as exc:
        print(f"repository check runtime error: {exc}", file=sys.stderr)
        return 4
    sys.stdout.buffer.write(_canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
