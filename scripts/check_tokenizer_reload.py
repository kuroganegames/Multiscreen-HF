#!/usr/bin/env python3
"""Verify that a checkpoint tokenizer is an exact reload of its source.

The report is intentionally compact and shareable: complete vocabulary,
special-token, and probe manifests are compared and hashed, but are never
written to the report.  Filesystem paths likewise remain process-local.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast


SCHEMA_VERSION = "multiscreen-tokenizer-reload-check-v1"
VERIFIER_VERSION = "1.0.0"
LOGICAL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
TOKENIZERS_BACKEND_ERROR = "Tokenizer class TokenizersBackend does not exist"
TOKENIZER_ASSET_NAMES = frozenset(
    {
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
    }
)
PROBES = (
    ("story", "Once upon a time"),
    ("counting", "A tiny dragon counted 1, 2, 3."),
    ("whitespace", " leading and trailing whitespace "),
    ("newline", "Line one.\nLine two."),
    ("unicode", "café 東京 🚀"),
)
SPECIAL_TOKEN_BOUNDARIES = (
    ("exact", "", ""),
    ("word_prefix", "word", ""),
    ("word_suffix", "", "word"),
    ("space_prefix", " ", ""),
    ("space_suffix", "", " "),
    ("newline_prefix", "\n", ""),
    ("newline_suffix", "", "\n"),
)
CHECKED_FIELDS = (
    "full_vocabulary_mapping",
    "vocab_size",
    "tokenizer_length",
    "added_vocabulary_mapping",
    "added_tokens_decoder",
    "special_tokens_map",
    "special_tokens_map_extended",
    "special_token_attributes",
    "special_token_ids",
    "all_special_tokens",
    "all_special_ids",
    "model_input_names",
    "padding_side",
    "model_max_length",
    "truncation_side",
    "probe_encodings_without_special_tokens",
    "probe_encodings_with_special_tokens",
    "probe_decodings",
    "special_token_boundary_encodings",
    "special_token_boundary_decodings",
    "tokenizer_class",
    "is_fast",
    "checkpoint_reload_origin",
    "source_normalization",
)


class TokenizerReloadError(RuntimeError):
    """An expected input, reload-origin, or equality check failed."""


@dataclass(frozen=True)
class LoadedTokenizer:
    tokenizer: Any
    loader: str
    direct_local_directory: Path | None


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable, UTF-8, pretty JSON with one trailing newline."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _sha256_manifest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _reject_symlink_components(path: Path, *, field: str) -> None:
    """Reject a symlink at the leaf or in any existing parent component."""

    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TokenizerReloadError(f"cannot inspect {field}") from exc
        if stat.S_ISLNK(mode):
            raise TokenizerReloadError(f"{field} must not be a symlink or traverse one")


def validate_checkpoint_directory(value: str | Path) -> Path:
    candidate = _absolute_without_resolving(Path(value))
    _reject_symlink_components(candidate, field="checkpoint")
    if not candidate.exists() or not candidate.is_dir():
        raise TokenizerReloadError("checkpoint must be an existing directory")
    try:
        resolved = candidate.resolve(strict=True)
        children = tuple(resolved.iterdir())
    except OSError as exc:
        raise TokenizerReloadError("checkpoint directory cannot be inspected") from exc
    if any(child.is_symlink() for child in children):
        raise TokenizerReloadError("checkpoint must not contain symlinked top-level files")
    config = resolved / "tokenizer_config.json"
    if not config.is_file():
        raise TokenizerReloadError("checkpoint is missing tokenizer_config.json")
    if not any((resolved / name).is_file() for name in TOKENIZER_ASSET_NAMES):
        raise TokenizerReloadError("checkpoint has no recognized tokenizer vocabulary asset")
    return resolved


def _validate_shareable_identifier(value: str, *, field: str) -> str:
    if not value or len(value) > 200:
        raise TokenizerReloadError(f"{field} must be a non-empty identifier of at most 200 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TokenizerReloadError(f"{field} contains control characters")
    if (
        value.startswith(("/", "~", "file:"))
        or WINDOWS_ABSOLUTE_RE.match(value)
        or "\\" in value
        or any(component in {"", ".", ".."} for component in value.split("/"))
    ):
        raise TokenizerReloadError(f"{field} must not contain a private filesystem path")
    return value


def validate_logical_name(value: str) -> str:
    if LOGICAL_NAME_RE.fullmatch(value) is None:
        raise TokenizerReloadError(
            "logical name must match [a-z0-9][a-z0-9._-]{0,63}"
        )
    return value


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _default_identifier(reference: str | Path) -> str:
    text = str(reference)
    path = Path(text).expanduser()
    if path.is_absolute() or path.exists() or text.startswith((".", "~")):
        return path.name
    return text


def _read_tokenizer_config(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenizerReloadError("invalid tokenizer_config.json for compatibility reload") from exc
    if not isinstance(value, dict):
        raise TokenizerReloadError("tokenizer_config.json must contain an object")
    return value


def _existing_local_directory(reference: str | Path) -> Path | None:
    candidate = _absolute_without_resolving(Path(reference))
    if not candidate.is_dir():
        return None
    try:
        return candidate.resolve(strict=True)
    except OSError:
        return None


def load_tokenizer(
    reference: str | Path,
    *,
    cache_dir: str | None,
    local_files_only: bool,
) -> LoadedTokenizer:
    """Load through AutoTokenizer, with the repository's narrow 4.x fallback."""

    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": False,
        "use_fast": True,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(reference), **kwargs)
        return LoadedTokenizer(tokenizer, "auto_tokenizer_from_pretrained", None)
    except ValueError as exc:
        if TOKENIZERS_BACKEND_ERROR not in str(exc):
            raise
        directory = _existing_local_directory(reference)
        if directory is None:
            raise
        config = _read_tokenizer_config(directory)
        tokenizer_file = directory / "tokenizer.json"
        if config.get("tokenizer_class") != "TokenizersBackend" or not tokenizer_file.is_file():
            raise
        if tokenizer_file.is_symlink():
            raise TokenizerReloadError("compatibility tokenizer asset must not be a symlink")
        forwarded = {
            name: config[name]
            for name in (
                "bos_token",
                "eos_token",
                "unk_token",
                "pad_token",
                "sep_token",
                "cls_token",
                "mask_token",
                "additional_special_tokens",
                "model_max_length",
                "padding_side",
                "truncation_side",
            )
            if name in config
        }
        forwarded["model_input_names"] = list(
            config.get("model_input_names", ["input_ids", "attention_mask"])
        )
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **forwarded)
        return LoadedTokenizer(tokenizer, "pretrained_tokenizer_fast_compat", directory)


def _verify_checkpoint_reload_origin(loaded: LoadedTokenizer, checkpoint: Path) -> None:
    if loaded.direct_local_directory is not None:
        if loaded.direct_local_directory != checkpoint:
            raise TokenizerReloadError("checkpoint compatibility reload used the wrong directory")
        return
    name_or_path = getattr(loaded.tokenizer, "name_or_path", None)
    if not isinstance(name_or_path, str) or not name_or_path:
        raise TokenizerReloadError("checkpoint tokenizer did not record its reload origin")
    origin = _existing_local_directory(name_or_path)
    if origin is None or origin != checkpoint:
        raise TokenizerReloadError("checkpoint tokenizer was not reloaded from the checkpoint")


def normalize_source_tokenizer(
    tokenizer: Any,
    *,
    pad_token_from_eos: bool,
    padding_side: str | None,
    model_max_length: int | None,
) -> dict[str, Any]:
    """Apply only explicitly requested source-side runtime operationalization."""

    if pad_token_from_eos:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is None or eos_token_id is None:
            raise TokenizerReloadError("source tokenizer has no EOS token for pad-token normalization")
        tokenizer.pad_token = eos_token
        if getattr(tokenizer, "pad_token_id", None) != eos_token_id:
            raise TokenizerReloadError("source pad-token normalization did not preserve the EOS id")
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length
    return {
        "model_max_length": model_max_length,
        "pad_token_from_eos": pad_token_from_eos,
        "padding_side": padding_side,
    }


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (value == value and abs(value) != float("inf")):
            raise TokenizerReloadError("tokenizer metadata contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TokenizerReloadError("tokenizer metadata has a non-string object key")
            normalized[key] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if all(hasattr(value, name) for name in ("content", "single_word", "lstrip", "rstrip")):
        return {
            "content": str(value.content),
            "lstrip": bool(value.lstrip),
            "normalized": bool(getattr(value, "normalized", True)),
            "rstrip": bool(value.rstrip),
            "single_word": bool(value.single_word),
            "special": bool(getattr(value, "special", False)),
        }
    if hasattr(value, "tolist"):
        return _normalize_json(value.tolist())
    raise TokenizerReloadError(f"unsupported tokenizer metadata type: {type(value).__name__}")


def _normalize_special_token_content(value: Any) -> Any:
    """Ignore string-vs-AddedToken wrappers while preserving token content."""

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_special_token_content(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_special_token_content(item) for item in value]
    if hasattr(value, "content"):
        return str(value.content)
    raise TokenizerReloadError("unsupported extended special-token representation")


def _vocabulary_mapping(tokenizer: Any, method: str) -> dict[str, int]:
    try:
        value = getattr(tokenizer, method)()
    except Exception as exc:
        raise TokenizerReloadError(f"{method}() failed") from exc
    if not isinstance(value, Mapping):
        raise TokenizerReloadError(f"{method}() did not return a mapping")
    result: dict[str, int] = {}
    for token, token_id in value.items():
        if not isinstance(token, str) or not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TokenizerReloadError(f"{method}() returned an invalid entry")
        result[token] = token_id
    return result


def vocabulary_manifest(tokenizer: Any) -> dict[str, Any]:
    try:
        vocab_size = int(tokenizer.vocab_size)
        tokenizer_length = int(len(tokenizer))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TokenizerReloadError("tokenizer does not expose integer vocabulary sizes") from exc
    return {
        "added_vocabulary_mapping": _vocabulary_mapping(tokenizer, "get_added_vocab"),
        "full_vocabulary_mapping": _vocabulary_mapping(tokenizer, "get_vocab"),
        "tokenizer_length": tokenizer_length,
        "vocab_size": vocab_size,
    }


def _special_id_attribute(attribute: str) -> str:
    if attribute == "additional_special_tokens":
        return "additional_special_tokens_ids"
    return f"{attribute}_id"


def _added_tokens_decoder_manifest(tokenizer: Any) -> list[dict[str, Any]]:
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if not isinstance(decoder, Mapping):
        raise TokenizerReloadError("tokenizer does not expose added_tokens_decoder")
    records: list[dict[str, Any]] = []
    for token_id, token in decoder.items():
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
        ):
            raise TokenizerReloadError("added_tokens_decoder has an invalid token id")
        normalized = _normalize_json(token)
        expected_fields = {
            "content",
            "lstrip",
            "normalized",
            "rstrip",
            "single_word",
            "special",
        }
        if not isinstance(normalized, Mapping) or set(normalized) != expected_fields:
            raise TokenizerReloadError(
                "added_tokens_decoder must preserve every AddedToken behavior flag"
            )
        records.append({"token": normalized, "token_id": token_id})
    records.sort(key=lambda value: value["token_id"])
    return records


def special_tokens_manifest(tokenizer: Any) -> dict[str, Any]:
    attributes = sorted(
        str(value)
        for value in getattr(
            tokenizer,
            "SPECIAL_TOKENS_ATTRIBUTES",
            (
                "bos_token",
                "eos_token",
                "unk_token",
                "sep_token",
                "pad_token",
                "cls_token",
                "mask_token",
                "additional_special_tokens",
            ),
        )
    )
    special_tokens_map = dict(tokenizer.special_tokens_map)
    special_tokens_map_extended = getattr(tokenizer, "special_tokens_map_extended", None)
    if special_tokens_map_extended is None:
        special_tokens_map_extended = special_tokens_map
    return {
        "added_tokens_decoder": _added_tokens_decoder_manifest(tokenizer),
        "all_special_ids": _normalize_json(list(tokenizer.all_special_ids)),
        "all_special_tokens": _normalize_json(list(tokenizer.all_special_tokens)),
        "special_token_attributes": {
            attribute: _normalize_json(getattr(tokenizer, attribute, None))
            for attribute in attributes
        },
        "special_token_ids": {
            attribute: _normalize_json(
                getattr(tokenizer, _special_id_attribute(attribute), None)
            )
            for attribute in attributes
        },
        "special_tokens_map": _normalize_json(special_tokens_map),
        "special_tokens_map_extended": _normalize_special_token_content(
            dict(special_tokens_map_extended)
        ),
    }


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TokenizerReloadError("probe encoding did not return input_ids")
    normalized = _normalize_json(dict(encoded))
    if not isinstance(normalized.get("input_ids"), list):
        raise TokenizerReloadError("probe input_ids must be a list")
    return normalized


def _decode(tokenizer: Any, token_ids: list[int], *, skip_special_tokens: bool) -> str:
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise TokenizerReloadError("probe decode did not return text")
    return decoded


def probe_manifest(tokenizer: Any) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for probe_id, text in PROBES:
        without_special = _encode(tokenizer, text, add_special_tokens=False)
        with_special = _encode(tokenizer, text, add_special_tokens=True)
        without_ids = without_special["input_ids"]
        with_ids = with_special["input_ids"]
        probes.append(
            {
                "decodings": {
                    "with_special_tokens_kept": _decode(
                        tokenizer, with_ids, skip_special_tokens=False
                    ),
                    "with_special_tokens_skipped": _decode(
                        tokenizer, with_ids, skip_special_tokens=True
                    ),
                    "without_special_tokens": _decode(
                        tokenizer, without_ids, skip_special_tokens=False
                    ),
                },
                "encoding_with_special_tokens": with_special,
                "encoding_without_special_tokens": without_special,
                "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "probe_id": probe_id,
            }
        )
    boundary_probes: list[dict[str, Any]] = []
    seen_special_tokens: set[str] = set()
    for special_index, token in enumerate(tokenizer.all_special_tokens):
        if not isinstance(token, str) or not token:
            raise TokenizerReloadError("all_special_tokens contains an invalid token")
        if token in seen_special_tokens:
            continue
        seen_special_tokens.add(token)
        for boundary_id, prefix, suffix in SPECIAL_TOKEN_BOUNDARIES:
            text = f"{prefix}{token}{suffix}"
            encoded = _encode(tokenizer, text, add_special_tokens=False)
            token_ids = encoded["input_ids"]
            boundary_probes.append(
                {
                    "decodings": {
                        "special_tokens_kept": _decode(
                            tokenizer, token_ids, skip_special_tokens=False
                        ),
                        "special_tokens_skipped": _decode(
                            tokenizer, token_ids, skip_special_tokens=True
                        ),
                    },
                    "encoding": encoded,
                    "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "probe_id": f"special_{special_index}_{boundary_id}",
                }
            )
    return {
        "probes": probes,
        "special_token_boundary_probes": boundary_probes,
    }


def operationalization_manifest(tokenizer: Any) -> dict[str, Any]:
    try:
        model_max_length = int(tokenizer.model_max_length)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TokenizerReloadError("tokenizer does not expose an integer model_max_length") from exc
    if model_max_length <= 0:
        raise TokenizerReloadError("tokenizer model_max_length must be positive")
    padding_side = getattr(tokenizer, "padding_side", None)
    truncation_side = getattr(tokenizer, "truncation_side", None)
    if padding_side not in {"left", "right"}:
        raise TokenizerReloadError("tokenizer padding_side must be left or right")
    if truncation_side not in {"left", "right"}:
        raise TokenizerReloadError("tokenizer truncation_side must be left or right")
    model_input_names = _normalize_json(list(tokenizer.model_input_names))
    if not all(isinstance(value, str) and value for value in model_input_names):
        raise TokenizerReloadError("tokenizer model_input_names must contain non-empty strings")
    return {
        "model_input_names": model_input_names,
        "model_max_length": model_max_length,
        "padding_side": padding_side,
        "truncation_side": truncation_side,
    }


def _assert_equal(source: Any, checkpoint: Any, *, field: str) -> None:
    if source != checkpoint:
        raise TokenizerReloadError(f"checkpoint mismatch: {field}")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_report(
    *,
    source_reference: str,
    checkpoint: Path,
    logical_name: str,
    source_identifier: str,
    checkpoint_identifier: str,
    cache_dir: str | None,
    allow_nonlocal: bool,
    source_pad_token_from_eos: bool,
    source_padding_side: str | None,
    source_model_max_length: int | None,
) -> dict[str, Any]:
    source = load_tokenizer(
        source_reference,
        cache_dir=cache_dir,
        local_files_only=not allow_nonlocal,
    )
    reloaded = load_tokenizer(
        checkpoint,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if source.tokenizer is reloaded.tokenizer:
        raise TokenizerReloadError("source and checkpoint loads returned the same tokenizer object")
    _verify_checkpoint_reload_origin(reloaded, checkpoint)
    source_normalization = normalize_source_tokenizer(
        source.tokenizer,
        pad_token_from_eos=source_pad_token_from_eos,
        padding_side=source_padding_side,
        model_max_length=source_model_max_length,
    )

    source_class = type(source.tokenizer).__name__
    checkpoint_class = type(reloaded.tokenizer).__name__
    source_is_fast = bool(getattr(source.tokenizer, "is_fast", False))
    checkpoint_is_fast = bool(getattr(reloaded.tokenizer, "is_fast", False))
    _assert_equal(source_class, checkpoint_class, field="tokenizer_class")
    _assert_equal(source_is_fast, checkpoint_is_fast, field="is_fast")

    source_vocab = vocabulary_manifest(source.tokenizer)
    checkpoint_vocab = vocabulary_manifest(reloaded.tokenizer)
    for field in (
        "full_vocabulary_mapping",
        "vocab_size",
        "tokenizer_length",
        "added_vocabulary_mapping",
    ):
        _assert_equal(source_vocab[field], checkpoint_vocab[field], field=field)

    source_special = special_tokens_manifest(source.tokenizer)
    checkpoint_special = special_tokens_manifest(reloaded.tokenizer)
    for field in (
        "added_tokens_decoder",
        "special_tokens_map",
        "special_tokens_map_extended",
        "special_token_attributes",
        "special_token_ids",
        "all_special_tokens",
        "all_special_ids",
    ):
        _assert_equal(source_special[field], checkpoint_special[field], field=field)

    source_operationalization = operationalization_manifest(source.tokenizer)
    checkpoint_operationalization = operationalization_manifest(reloaded.tokenizer)
    for field in (
        "model_input_names",
        "padding_side",
        "model_max_length",
        "truncation_side",
    ):
        _assert_equal(
            source_operationalization[field],
            checkpoint_operationalization[field],
            field=field,
        )

    source_probes = probe_manifest(source.tokenizer)
    checkpoint_probes = probe_manifest(reloaded.tokenizer)
    _assert_equal(source_probes, checkpoint_probes, field="probe encodings/decodings")

    vocab_hash = _sha256_manifest(source_vocab)
    special_hash = _sha256_manifest(source_special)
    probe_hash = _sha256_manifest(source_probes)
    return {
        "checked_fields": list(CHECKED_FIELDS),
        "checkpoint": {
            "class": checkpoint_class,
            "identifier": checkpoint_identifier,
            "is_fast": checkpoint_is_fast,
            "reload_method": reloaded.loader,
            "reloaded_from_checkpoint": True,
        },
        "counts": {
            "added_vocabulary": len(source_vocab["added_vocabulary_mapping"]),
            "all_special_tokens": len(source_special["all_special_tokens"]),
            "probes": len(PROBES),
            "special_token_boundary_probes": len(
                source_probes["special_token_boundary_probes"]
            ),
            "tokenizer_length": source_vocab["tokenizer_length"],
            "vocab_size": source_vocab["vocab_size"],
            "vocabulary": len(source_vocab["full_vocabulary_mapping"]),
        },
        "hashes": {
            "probe_manifest_sha256": probe_hash,
            "special_tokens_manifest_sha256": special_hash,
            "vocabulary_manifest_sha256": vocab_hash,
        },
        "logical_name": logical_name,
        "operationalization": source_operationalization,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "class": source_class,
            "identifier": source_identifier,
            "is_fast": source_is_fast,
        },
        "status": "passed",
        "source_normalization": source_normalization,
        "versions": {
            "python": platform.python_version(),
            "tokenizers": _package_version("tokenizers"),
            "transformers": transformers.__version__,
            "verifier": VERIFIER_VERSION,
        },
    }


def safe_write_new(path: Path, data: bytes) -> None:
    if not path.expanduser().is_absolute():
        raise TokenizerReloadError("output path must be absolute")
    target = _absolute_without_resolving(path)
    parent = target.parent
    _reject_symlink_components(parent, field="output parent")
    if not parent.exists() or not parent.is_dir():
        raise TokenizerReloadError("output parent must be an existing directory")
    if target.exists() or target.is_symlink():
        raise TokenizerReloadError("output already exists; overwrite is forbidden")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise TokenizerReloadError("output already exists; overwrite is forbidden") from exc
    except OSError as exc:
        raise TokenizerReloadError("cannot create output safely") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--logical-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--source-pad-token-from-eos", action="store_true")
    parser.add_argument("--source-padding-side", choices=("left", "right"), default=None)
    parser.add_argument("--source-model-max-length", type=positive_int, default=None)
    parser.add_argument(
        "--allow-nonlocal",
        action="store_true",
        help="allow the source tokenizer loader to access nonlocal files; checkpoint reload stays local",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        logical_name = validate_logical_name(args.logical_name)
        checkpoint = validate_checkpoint_directory(args.checkpoint)
        source_identifier = _validate_shareable_identifier(
            args.source_id or _default_identifier(args.source_tokenizer),
            field="source identifier",
        )
        checkpoint_identifier = _validate_shareable_identifier(
            args.checkpoint_id or checkpoint.name,
            field="checkpoint identifier",
        )
        report = build_report(
            source_reference=args.source_tokenizer,
            checkpoint=checkpoint,
            logical_name=logical_name,
            source_identifier=source_identifier,
            checkpoint_identifier=checkpoint_identifier,
            cache_dir=args.cache_dir,
            allow_nonlocal=args.allow_nonlocal,
            source_pad_token_from_eos=args.source_pad_token_from_eos,
            source_padding_side=args.source_padding_side,
            source_model_max_length=args.source_model_max_length,
        )
        safe_write_new(Path(args.output), canonical_json_bytes(report))
    except (OSError, TokenizerReloadError, ValueError) as exc:
        print(f"tokenizer reload check failed: {exc}", file=sys.stderr)
        return 1
    print(f"tokenizer reload check passed: {logical_name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
