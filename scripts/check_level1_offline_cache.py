#!/usr/bin/env python3
"""Fail-closed Stage 5 proof that one explicit cache serves every HF input.

The successful report is deliberately compact and shareable.  It records only
public Hub identities and checked aggregate metadata; cache paths, repository
paths, source text, and loader diagnostics never enter the report.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "multiscreen-level1-offline-cache-v1"
GPT2_REPOSITORY = "gpt2"
GPT2_CONTEXT_LENGTH = 4_096
TINY_STORIES_REPOSITORY = "roneneldan/TinyStories"
TINY_STORIES_SPLIT = "train[:20000]"
TINY_STORIES_TEXT_COLUMN = "text"
TINY_STORIES_ROWS = 20_000
TINY_STORIES_PINNED_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
REQUIRED_OFFLINE_ENVIRONMENT = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
C3_MANIFEST_RELATIVE = Path("configs/p0_5_c3_paper_training_contract.json")
P0_4_EVIDENCE_CONTRACT_RELATIVE = Path("scripts/p0_4_evidence_contract.py")


class OfflineCacheError(RuntimeError):
    """A safe, stable failure code for the path-free CLI report."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise OfflineCacheError("invalid_arguments")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize one canonical, finite JSON object with one trailing newline."""

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


def _fail(code: str) -> None:
    raise OfflineCacheError(code)


def _reject_symlink_components(path: Path, *, failure_code: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except OSError:
            _fail(failure_code)
        if stat.S_ISLNK(mode):
            _fail(failure_code)


def validate_canonical_directory(value: str, *, failure_code: str) -> Path:
    """Require an absolute, existing directory with no lexical or symlink alias."""

    if not value or "\x00" in value:
        _fail(failure_code)
    candidate = Path(value)
    if not candidate.is_absolute() or os.fspath(candidate) != value:
        _fail(failure_code)
    _reject_symlink_components(candidate, failure_code=failure_code)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _fail(failure_code)
    if os.fspath(resolved) != value or not resolved.is_dir():
        _fail(failure_code)
    return resolved


def _git_top_level(repository: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repository), "rev-parse", "--show-toplevel"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        _fail("invalid_repo_root")
    if result.returncode != 0:
        _fail("invalid_repo_root")
    try:
        raw = result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail("invalid_repo_root")
    if not raw.endswith("\n") or raw.count("\n") != 1:
        _fail("invalid_repo_root")
    candidate = raw[:-1]
    if candidate != os.fspath(repository):
        _fail("invalid_repo_root")
    return repository


def validate_repository_root(value: str) -> Path:
    repository = validate_canonical_directory(value, failure_code="invalid_repo_root")
    return _git_top_level(repository)


def validate_cache_directory(value: str) -> Path:
    return validate_canonical_directory(value, failure_code="invalid_cache_dir")


def validate_offline_environment(environment: Mapping[str, str]) -> None:
    if any(environment.get(name) != expected for name, expected in REQUIRED_OFFLINE_ENVIRONMENT.items()):
        _fail("offline_environment_required")


def _validate_repository_file(
    repository: Path,
    relative: Path,
    *,
    failure_code: str = "invalid_c3_manifest",
) -> Path:
    candidate = repository.joinpath(*relative.parts)
    _reject_symlink_components(candidate, failure_code=failure_code)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository)
        mode = resolved.stat().st_mode
    except (OSError, ValueError):
        _fail(failure_code)
    if not stat.S_ISREG(mode):
        _fail(failure_code)
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        del value
        raise ValueError("non-finite JSON value")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        _fail("invalid_c3_manifest")
    if not isinstance(value, dict):
        _fail("invalid_c3_manifest")
    return value


def _default_tokenizer_loader() -> Callable[..., Any]:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained


def _default_tokenizer_projector(
    repository: Path,
) -> Callable[[Any], Mapping[str, Any]]:
    """Import the exact in-repository P0-4 projection without PYTHONPATH."""

    expected_module = _validate_repository_file(
        repository,
        P0_4_EVIDENCE_CONTRACT_RELATIVE,
        failure_code="p0_4_tokenizer_projection_failed",
    )
    repository_text = os.fspath(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    try:
        module = importlib.import_module("scripts.p0_4_evidence_contract")
        loaded_from = Path(module.__file__).resolve(strict=True)
        projector = getattr(module, "build_tokenizer_projection")
    except Exception:
        _fail("p0_4_tokenizer_projection_failed")
    if loaded_from != expected_module or not callable(projector):
        _fail("p0_4_tokenizer_projection_failed")
    return projector


def _default_dataset_loader() -> Callable[..., Any]:
    from datasets import load_dataset

    return load_dataset


def _default_c3_module(repository: Path) -> Any:
    expected_module = _validate_repository_file(
        repository,
        Path("scripts/p0_5_c3_paper_training_contract.py"),
    )
    repository_text = os.fspath(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    module = importlib.import_module("scripts.p0_5_c3_paper_training_contract")
    try:
        loaded_from = Path(module.__file__).resolve(strict=True)
    except (AttributeError, OSError, TypeError):
        _fail("c3_contract_check_failed")
    if loaded_from != expected_module:
        _fail("c3_contract_check_failed")
    return module


def _dataset_summary(dataset: Any, *, failure_prefix: str) -> dict[str, Any]:
    try:
        row_count = len(dataset)
        fingerprint = getattr(dataset, "_fingerprint", None)
        columns = getattr(dataset, "column_names", None)
    except Exception:
        _fail(f"{failure_prefix}_identity_mismatch")
    if row_count != TINY_STORIES_ROWS:
        _fail(f"{failure_prefix}_identity_mismatch")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        _fail(f"{failure_prefix}_identity_mismatch")
    if not isinstance(columns, (list, tuple)) or TINY_STORIES_TEXT_COLUMN not in columns:
        _fail(f"{failure_prefix}_identity_mismatch")
    try:
        texts = dataset[TINY_STORIES_TEXT_COLUMN]
        text_rows = len(texts)
        all_strings = all(isinstance(text, str) for text in texts)
    except Exception:
        _fail(f"{failure_prefix}_identity_mismatch")
    if text_rows != TINY_STORIES_ROWS or not all_strings:
        _fail(f"{failure_prefix}_identity_mismatch")
    return {
        "fingerprint_nonempty": True,
        "fingerprint_sha256": hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(),
        "repository": TINY_STORIES_REPOSITORY,
        "row_count": row_count,
        "split": TINY_STORIES_SPLIT,
        "text_column": TINY_STORIES_TEXT_COLUMN,
        "text_rows": text_rows,
    }


def _path_free_symbol(value: Any) -> bool:
    first_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
    remaining_characters = first_characters + "0123456789.-"
    return (
        isinstance(value, str)
        and 0 < len(value) <= 128
        and value[0] in first_characters
        and all(character in remaining_characters for character in value)
    )


def _validated_tokenizer_projection(value: Any) -> dict[str, Any]:
    """Copy only the exact, path-free public P0-4 projection schema."""

    expected_top_level = {
        "class",
        "counts",
        "hashes",
        "is_fast",
        "operationalization",
    }
    if not isinstance(value, Mapping) or set(value) != expected_top_level:
        _fail("p0_4_tokenizer_projection_failed")
    if not _path_free_symbol(value["class"]) or type(value["is_fast"]) is not bool:
        _fail("p0_4_tokenizer_projection_failed")

    expected_count_fields = {
        "added_vocabulary",
        "all_special_tokens",
        "probes",
        "special_token_boundary_probes",
        "tokenizer_length",
        "vocab_size",
        "vocabulary",
    }
    counts = value["counts"]
    if not isinstance(counts, Mapping) or set(counts) != expected_count_fields:
        _fail("p0_4_tokenizer_projection_failed")
    normalized_counts: dict[str, int] = {}
    for field in sorted(expected_count_fields):
        count = counts[field]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            _fail("p0_4_tokenizer_projection_failed")
        normalized_counts[field] = count
    for field in (
        "all_special_tokens",
        "probes",
        "tokenizer_length",
        "vocab_size",
        "vocabulary",
    ):
        if normalized_counts[field] <= 0:
            _fail("p0_4_tokenizer_projection_failed")

    expected_hash_fields = {
        "probe_manifest_sha256",
        "special_tokens_manifest_sha256",
        "vocabulary_manifest_sha256",
    }
    hashes = value["hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != expected_hash_fields:
        _fail("p0_4_tokenizer_projection_failed")
    normalized_hashes: dict[str, str] = {}
    for field in sorted(expected_hash_fields):
        digest = hashes[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            _fail("p0_4_tokenizer_projection_failed")
        normalized_hashes[field] = digest

    operationalization = value["operationalization"]
    expected_operational_fields = {
        "model_input_names",
        "model_max_length",
        "padding_side",
        "truncation_side",
    }
    if (
        not isinstance(operationalization, Mapping)
        or set(operationalization) != expected_operational_fields
    ):
        _fail("p0_4_tokenizer_projection_failed")
    model_input_names = operationalization["model_input_names"]
    if (
        not isinstance(model_input_names, list)
        or not model_input_names
        or any(not _path_free_symbol(name) for name in model_input_names)
    ):
        _fail("p0_4_tokenizer_projection_failed")
    model_max_length = operationalization["model_max_length"]
    if (
        not isinstance(model_max_length, int)
        or isinstance(model_max_length, bool)
        or model_max_length <= 0
        or operationalization["padding_side"] not in {"left", "right"}
        or operationalization["truncation_side"] not in {"left", "right"}
    ):
        _fail("p0_4_tokenizer_projection_failed")
    return {
        "class": value["class"],
        "counts": normalized_counts,
        "hashes": normalized_hashes,
        "is_fast": value["is_fast"],
        "operationalization": {
            "model_input_names": list(model_input_names),
            "model_max_length": model_max_length,
            "padding_side": operationalization["padding_side"],
            "truncation_side": operationalization["truncation_side"],
        },
    }


def _check_default_gpt2(
    cache: Path,
    loader: Callable[..., Any],
    projector: Callable[[Any], Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        tokenizer = loader(
            pretrained_model_name_or_path=GPT2_REPOSITORY,
            cache_dir=os.fspath(cache),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception:
        _fail("p0_4_tokenizer_load_failed")
    try:
        vocab_size = len(tokenizer)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        is_fast = getattr(tokenizer, "is_fast", None)
    except Exception:
        _fail("p0_4_tokenizer_identity_mismatch")
    if vocab_size != 50_257 or eos_token_id != 50_256 or is_fast is not True:
        _fail("p0_4_tokenizer_identity_mismatch")
    try:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = GPT2_CONTEXT_LENGTH
        projection = _validated_tokenizer_projection(projector(tokenizer))
        normalized_identity = {
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "is_fast": getattr(tokenizer, "is_fast", None),
            "length": len(tokenizer),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "model_input_names": getattr(tokenizer, "model_input_names", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "padding_side": getattr(tokenizer, "padding_side", None),
            "truncation_side": getattr(tokenizer, "truncation_side", None),
        }
    except Exception:
        _fail("p0_4_tokenizer_projection_failed")
    if normalized_identity != {
        "eos_token_id": 50_256,
        "is_fast": True,
        "length": 50_257,
        "model_max_length": GPT2_CONTEXT_LENGTH,
        "model_input_names": ["input_ids", "attention_mask"],
        "pad_token_id": 50_256,
        "padding_side": "right",
        "truncation_side": "right",
    }:
        _fail("p0_4_tokenizer_identity_mismatch")
    if (
        projection["is_fast"] is not True
        or projection["counts"]["tokenizer_length"] != vocab_size
        or projection["counts"]["vocab_size"] != vocab_size
        or projection["counts"]["vocabulary"] != vocab_size
        or projection["operationalization"]
        != {
            "model_input_names": ["input_ids", "attention_mask"],
            "model_max_length": GPT2_CONTEXT_LENGTH,
            "padding_side": "right",
            "truncation_side": "right",
        }
    ):
        _fail("p0_4_tokenizer_projection_failed")
    return {
        "eos_token_id": eos_token_id,
        "identity_projection": projection,
        "repository": GPT2_REPOSITORY,
        "revision": "default",
        "use_fast": True,
        "vocab_size": vocab_size,
    }


def _load_tiny_stories(
    cache: Path,
    loader: Callable[..., Any],
    *,
    revision: str | None,
    failure_prefix: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cache_dir": os.fspath(cache),
        "split": TINY_STORIES_SPLIT,
        "streaming": False,
    }
    if revision is not None:
        kwargs["revision"] = revision
    try:
        dataset = loader(TINY_STORIES_REPOSITORY, **kwargs)
    except Exception:
        _fail(f"{failure_prefix}_load_failed")
    summary = _dataset_summary(dataset, failure_prefix=failure_prefix)
    summary["revision"] = revision if revision is not None else "default"
    return summary


def _check_c3(
    repository: Path,
    cache: Path,
    c3: Any,
) -> dict[str, Any]:
    manifest_path = _validate_repository_file(repository, C3_MANIFEST_RELATIVE)
    manifest = _load_json_object(manifest_path)
    required = ("validate_manifest", "load_pinned_tokenizer", "load_pinned_rows")
    if any(not callable(getattr(c3, name, None)) for name in required):
        _fail("c3_contract_check_failed")
    try:
        contract = c3.validate_manifest(manifest)
        tokenizer, tokenizer_provenance = c3.load_pinned_tokenizer(
            manifest,
            cache_dir=os.fspath(cache),
        )
        selected_rows = c3.load_pinned_rows(
            manifest,
            cache_dir=os.fspath(cache),
        )
    except Exception:
        _fail("c3_contract_check_failed")
    try:
        tokenizer_config = manifest["tokenizer"]
        dataset_config = manifest["dataset"]
        selection_config = dataset_config["selection"]
        data_contract_config = dataset_config["expected_data_contract"]
        provenance = selected_rows.provenance
        selection = provenance["selection"]
        asset_hash = tokenizer_provenance["asset_manifest_sha256"]
        row_count = len(selected_rows.texts)
        values = {
            "contract_status": contract["status"],
            "tokenizer_revision": tokenizer_provenance["revision"],
            "tokenizer_vocabulary": len(tokenizer),
            "tokenizer_eos": tokenizer.eos_token_id,
            "asset_manifest_sha256": asset_hash,
            "dataset_revision": provenance["revision"],
            "dataset_data_files": provenance["data_files"],
            "dataset_file_sha256": provenance["file"]["sha256"],
            "dataset_file_size": provenance["file"]["size_bytes"],
            "full_fingerprint": provenance["full_fingerprint"],
            "selected_fingerprint": selection["fingerprint"],
            "selected_rows": row_count,
            "row_manifest_sha256": provenance["row_manifest_sha256"],
        }
    except Exception:
        _fail("c3_contract_check_failed")
    expected = {
        "contract_status": "passed",
        "tokenizer_revision": tokenizer_config["revision"],
        "tokenizer_vocabulary": tokenizer_config["expected_vocab_size"],
        "tokenizer_eos": tokenizer_config["expected_eos_token_id"],
        "asset_manifest_sha256": tokenizer_config["asset_manifest_sha256"],
        "dataset_revision": dataset_config["revision"],
        "dataset_data_files": dataset_config["data_files"],
        "dataset_file_sha256": dataset_config["expected_file_sha256"],
        "dataset_file_size": dataset_config["expected_file_size_bytes"],
        "full_fingerprint": dataset_config["expected_full_fingerprint"],
        "selected_fingerprint": selection_config["expected_fingerprint"],
        "selected_rows": selection_config["stop"] - selection_config["start"],
        "row_manifest_sha256": data_contract_config["row_manifest_sha256"],
    }
    if canonical_json_bytes(values) != canonical_json_bytes(expected):
        _fail("c3_contract_check_failed")
    return {
        "dataset": {
            "data_files": dict(dataset_config["data_files"]),
            "file_sha256": values["dataset_file_sha256"],
            "file_size_bytes": values["dataset_file_size"],
            "full_fingerprint": values["full_fingerprint"],
            "repository": dataset_config["repository"],
            "revision": values["dataset_revision"],
            "selected_fingerprint": values["selected_fingerprint"],
            "selected_rows": values["selected_rows"],
            "row_manifest_sha256": values["row_manifest_sha256"],
            "split": dataset_config["split"],
            "text_column": dataset_config["text_column"],
        },
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "tokenizer": {
            "asset_manifest_sha256": values["asset_manifest_sha256"],
            "eos_token_id": values["tokenizer_eos"],
            "repository": tokenizer_config["repository"],
            "revision": values["tokenizer_revision"],
            "vocab_size": values["tokenizer_vocabulary"],
        },
    }


def check_offline_cache(
    repository: Path,
    cache: Path,
    *,
    tokenizer_loader: Callable[..., Any] | None = None,
    tokenizer_projector: Callable[[Any], Mapping[str, Any]] | None = None,
    dataset_loader: Callable[..., Any] | None = None,
    c3_module: Any | None = None,
) -> dict[str, Any]:
    """Run all identity checks against the exact same explicit cache directory."""

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tokenizer_loader = tokenizer_loader or _default_tokenizer_loader()
        tokenizer_projector = tokenizer_projector or _default_tokenizer_projector(
            repository
        )
        dataset_loader = dataset_loader or _default_dataset_loader()
        c3_module = c3_module or _default_c3_module(repository)
        p0_4_tokenizer = _check_default_gpt2(
            cache,
            tokenizer_loader,
            tokenizer_projector,
        )
        p0_4_dataset = _load_tiny_stories(
            cache,
            dataset_loader,
            revision=None,
            failure_prefix="p0_4_dataset",
        )
        p0_3_dataset = _load_tiny_stories(
            cache,
            dataset_loader,
            revision=TINY_STORIES_PINNED_REVISION,
            failure_prefix="p0_3_dataset",
        )
        c3 = _check_c3(repository, cache, c3_module)
    return {
        "cache": {"explicit": True, "path_recorded": False, "single_cache": True},
        "checks": {
            "p0_3_tinystories": p0_3_dataset,
            "p0_4_gpt2_tokenizer": p0_4_tokenizer,
            "p0_4_tinystories": p0_4_dataset,
            "p0_5_c3": c3,
        },
        "offline_environment": dict(sorted(REQUIRED_OFFLINE_ENVIRONMENT.items())),
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    return parser


def _failure_report(code: str) -> dict[str, str]:
    return {"failure": code, "schema_version": SCHEMA_VERSION, "status": "failed"}


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
