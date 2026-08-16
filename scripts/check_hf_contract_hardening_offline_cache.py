#!/usr/bin/env python3
"""Fail-closed offline-cache proof for Stage E P0-3 and P0-4 inputs.

The Stage E final requalification creates fresh P0-3 and P0-4 evidence, but it
does not create fresh P0.5-C3 data or CUDA artifacts.  This checker therefore
proves only the public inputs that Stage E actually consumes.  Its successful
report is canonical and path-free; loader diagnostics and cache paths are
never emitted.

The input identity validators are shared with the accepted Level 1 checker.
The Level 1 entry point and its C3-inclusive report remain unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

def _load_level1_checker() -> Any:
    """Load the fixed sibling module even under Python's safe-path mode."""

    source = Path(__file__).resolve(strict=True).with_name(
        "check_level1_offline_cache.py"
    )
    try:
        mode = source.stat().st_mode
    except OSError as exc:  # pragma: no cover - repository corruption
        raise RuntimeError("Level 1 offline-cache checker is unavailable") from exc
    if not stat.S_ISREG(mode):  # pragma: no cover - repository corruption
        raise RuntimeError("Level 1 offline-cache checker is not a regular file")
    specification = importlib.util.spec_from_file_location(
        "_multiscreen_level1_offline_cache",
        source,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Level 1 offline-cache checker cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_level1 = _load_level1_checker()


SCHEMA_VERSION = "multiscreen-hf-contract-hardening-offline-cache-v1"
GPT2_REPOSITORY = _level1.GPT2_REPOSITORY
GPT2_CONTEXT_LENGTH = _level1.GPT2_CONTEXT_LENGTH
TINY_STORIES_REPOSITORY = _level1.TINY_STORIES_REPOSITORY
TINY_STORIES_SPLIT = _level1.TINY_STORIES_SPLIT
TINY_STORIES_TEXT_COLUMN = _level1.TINY_STORIES_TEXT_COLUMN
TINY_STORIES_ROWS = _level1.TINY_STORIES_ROWS
TINY_STORIES_PINNED_REVISION = _level1.TINY_STORIES_PINNED_REVISION
REQUIRED_OFFLINE_ENVIRONMENT = dict(_level1.REQUIRED_OFFLINE_ENVIRONMENT)
OfflineCacheError = _level1.OfflineCacheError


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise OfflineCacheError("invalid_arguments")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return _level1.canonical_json_bytes(value)


def validate_repository_root(value: str) -> Path:
    return _level1.validate_repository_root(value)


def validate_cache_directory(value: str) -> Path:
    return _level1.validate_cache_directory(value)


def validate_offline_environment(environment: Mapping[str, str]) -> None:
    _level1.validate_offline_environment(environment)


def check_offline_cache(
    repository: Path,
    cache: Path,
    *,
    tokenizer_loader: Callable[..., Any] | None = None,
    tokenizer_projector: Callable[[Any], Mapping[str, Any]] | None = None,
    dataset_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Verify the exact Stage E public inputs against one explicit cache."""

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        tokenizer_loader = tokenizer_loader or _level1._default_tokenizer_loader()
        tokenizer_projector = (
            tokenizer_projector
            or _level1._default_tokenizer_projector(repository)
        )
        dataset_loader = dataset_loader or _level1._default_dataset_loader()
        p0_4_tokenizer = _level1._check_default_gpt2(
            cache,
            tokenizer_loader,
            tokenizer_projector,
        )
        p0_4_dataset = _level1._load_tiny_stories(
            cache,
            dataset_loader,
            revision=None,
            failure_prefix="p0_4_dataset",
        )
        p0_3_dataset = _level1._load_tiny_stories(
            cache,
            dataset_loader,
            revision=TINY_STORIES_PINNED_REVISION,
            failure_prefix="p0_3_dataset",
        )

    return {
        "cache": {
            "explicit": True,
            "path_recorded": False,
            "single_cache": True,
        },
        "checks": {
            "p0_3_tinystories": p0_3_dataset,
            "p0_4_gpt2_tokenizer": p0_4_tokenizer,
            "p0_4_tinystories": p0_4_dataset,
        },
        "offline_environment": dict(sorted(REQUIRED_OFFLINE_ENVIRONMENT.items())),
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "fresh_p0_3": True,
            "fresh_p0_4": True,
            "fresh_p0_5_c3": False,
        },
        "status": "passed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    return parser


def _failure_report(code: str) -> dict[str, str]:
    return {
        "failure": code,
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    stdout: Any | None = None,
) -> int:
    output = sys.stdout.buffer if stdout is None else stdout
    try:
        arguments = build_parser().parse_args(argv)
        active_environment = os.environ if environment is None else environment
        validate_offline_environment(active_environment)
        repository = validate_repository_root(arguments.repo_root)
        cache = validate_cache_directory(arguments.cache_dir)
        report = check_offline_cache(repository, cache)
        exit_code = 0
    except OfflineCacheError as exc:
        report = _failure_report(exc.code)
        exit_code = 1
    except Exception:
        report = _failure_report("internal_failure")
        exit_code = 1
    output.write(canonical_json_bytes(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
