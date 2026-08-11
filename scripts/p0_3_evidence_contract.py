#!/usr/bin/env python3
"""Build the path-free P0-3 selected-text and packed-token evidence contract."""

from __future__ import annotations

import hashlib
import json
import operator
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "multiscreen-p0-3-data-contract-v1"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
DATASET_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{16,64}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
TOKENIZER_CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
LOCAL_TEXT_FILE_MARKER = "provided_path_not_recorded"
TEXT_MANIFEST_ALGORITHM = "sha256-length-framed-utf8-texts-v1"
TOKEN_STREAM_ALGORITHM = "sha256-uint32-le-packed-token-stream-v1"


class P0ThreeEvidenceContractError(ValueError):
    """P0-3 data identity cannot be represented without ambiguity."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _required_text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise P0ThreeEvidenceContractError(
            f"{field} must be non-empty printable text"
        )
    return value


def _public_identifier(value: Any, *, field: str) -> str:
    text = _required_text(value, field=field)
    if (
        len(text) > 200
        or text.startswith(("/", "~"))
        or text.lower().startswith("file:")
        or WINDOWS_ABSOLUTE_RE.match(text)
        or "\\" in text
        or any(part in {"", ".", ".."} for part in text.split("/"))
    ):
        raise P0ThreeEvidenceContractError(
            f"{field} must be a path-free public identifier"
        )
    return text


def _exact_nonnegative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise P0ThreeEvidenceContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def build_tokenizer_projection(
    tokenizer: Any,
    *,
    verifier: Any | None = None,
) -> dict[str, Any]:
    """Project through the exact tokenizer-reload verifier manifests."""

    if verifier is None:
        from scripts import check_tokenizer_reload as verifier

    required = (
        "vocabulary_manifest",
        "special_tokens_manifest",
        "probe_manifest",
        "operationalization_manifest",
        "_sha256_manifest",
    )
    if any(not callable(getattr(verifier, name, None)) for name in required):
        raise P0ThreeEvidenceContractError(
            "tokenizer verifier projection is incomplete"
        )
    try:
        vocabulary = verifier.vocabulary_manifest(tokenizer)
        special = verifier.special_tokens_manifest(tokenizer)
        probes = verifier.probe_manifest(tokenizer)
        operationalization = verifier.operationalization_manifest(tokenizer)
        projection = {
            "class": type(tokenizer).__name__,
            "counts": {
                "added_vocabulary": len(vocabulary["added_vocabulary_mapping"]),
                "all_special_tokens": len(special["all_special_tokens"]),
                "probes": len(probes["probes"]),
                "special_token_boundary_probes": len(
                    probes["special_token_boundary_probes"]
                ),
                "tokenizer_length": vocabulary["tokenizer_length"],
                "vocab_size": vocabulary["vocab_size"],
                "vocabulary": len(vocabulary["full_vocabulary_mapping"]),
            },
            "hashes": {
                "probe_manifest_sha256": verifier._sha256_manifest(probes),
                "special_tokens_manifest_sha256": verifier._sha256_manifest(
                    special
                ),
                "vocabulary_manifest_sha256": verifier._sha256_manifest(
                    vocabulary
                ),
            },
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "operationalization": dict(operationalization),
        }
    except Exception as exc:
        raise P0ThreeEvidenceContractError(
            "tokenizer verifier projection could not be built"
        ) from exc
    return _validate_tokenizer_projection(projection)


def _validate_tokenizer_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "class",
        "counts",
        "hashes",
        "is_fast",
        "operationalization",
    }:
        raise P0ThreeEvidenceContractError(
            "tokenizer projection fields are incomplete"
        )
    tokenizer_class = value["class"]
    if (
        not isinstance(tokenizer_class, str)
        or TOKENIZER_CLASS_RE.fullmatch(tokenizer_class) is None
    ):
        raise P0ThreeEvidenceContractError(
            "tokenizer class must be a path-free class identifier"
        )
    if type(value["is_fast"]) is not bool:
        raise P0ThreeEvidenceContractError(
            "tokenizer is_fast must be an exact boolean"
        )

    counts = value["counts"]
    expected_count_fields = {
        "added_vocabulary",
        "all_special_tokens",
        "probes",
        "special_token_boundary_probes",
        "tokenizer_length",
        "vocab_size",
        "vocabulary",
    }
    if not isinstance(counts, Mapping) or set(counts) != expected_count_fields:
        raise P0ThreeEvidenceContractError(
            "tokenizer count fields are incomplete"
        )
    normalized_counts = {
        field: _exact_nonnegative_int(
            counts[field], field=f"tokenizer.counts.{field}"
        )
        for field in sorted(expected_count_fields)
    }
    for positive in (
        "all_special_tokens",
        "probes",
        "tokenizer_length",
        "vocab_size",
        "vocabulary",
    ):
        if normalized_counts[positive] <= 0:
            raise P0ThreeEvidenceContractError(
                f"tokenizer.counts.{positive} must be positive"
            )

    hashes = value["hashes"]
    expected_hash_fields = {
        "probe_manifest_sha256",
        "special_tokens_manifest_sha256",
        "vocabulary_manifest_sha256",
    }
    if not isinstance(hashes, Mapping) or set(hashes) != expected_hash_fields:
        raise P0ThreeEvidenceContractError(
            "tokenizer hash fields are incomplete"
        )
    normalized_hashes: dict[str, str] = {}
    for field in sorted(expected_hash_fields):
        digest = hashes[field]
        if not isinstance(digest, str) or HEX64_RE.fullmatch(digest) is None:
            raise P0ThreeEvidenceContractError(
                f"tokenizer.hashes.{field} must be lowercase SHA-256"
            )
        normalized_hashes[field] = digest

    operational = value["operationalization"]
    if not isinstance(operational, Mapping) or set(operational) != {
        "model_input_names",
        "model_max_length",
        "padding_side",
        "truncation_side",
    }:
        raise P0ThreeEvidenceContractError(
            "tokenizer operationalization fields are incomplete"
        )
    model_input_names = operational["model_input_names"]
    if (
        not isinstance(model_input_names, list)
        or not model_input_names
        or any(
            not isinstance(name, str)
            or TOKENIZER_CLASS_RE.fullmatch(name) is None
            for name in model_input_names
        )
    ):
        raise P0ThreeEvidenceContractError(
            "tokenizer model_input_names are invalid"
        )
    model_max_length = _exact_nonnegative_int(
        operational["model_max_length"],
        field="tokenizer.operationalization.model_max_length",
    )
    if model_max_length <= 0:
        raise P0ThreeEvidenceContractError(
            "tokenizer model_max_length must be positive"
        )
    if operational["padding_side"] not in {"left", "right"}:
        raise P0ThreeEvidenceContractError(
            "tokenizer padding_side is invalid"
        )
    if operational["truncation_side"] not in {"left", "right"}:
        raise P0ThreeEvidenceContractError(
            "tokenizer truncation_side is invalid"
        )
    return {
        "class": tokenizer_class,
        "counts": normalized_counts,
        "hashes": normalized_hashes,
        "is_fast": value["is_fast"],
        "operationalization": {
            "model_input_names": list(model_input_names),
            "model_max_length": model_max_length,
            "padding_side": operational["padding_side"],
            "truncation_side": operational["truncation_side"],
        },
    }


def _text_manifest(texts: Sequence[str]) -> dict[str, Any]:
    if isinstance(texts, (str, bytes)) or not texts:
        raise P0ThreeEvidenceContractError("selected texts must be a non-empty sequence")
    digest = hashlib.sha256(b"multiscreen-p0-3-text-manifest-v1\0")
    total_bytes = 0
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise P0ThreeEvidenceContractError("selected texts must be non-empty strings")
        encoded = text.encode("utf-8")
        total_bytes += len(encoded)
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(hashlib.sha256(encoded).digest())
    return {
        "algorithm": TEXT_MANIFEST_ALGORITHM,
        "selected_text_count": len(texts),
        "selected_text_manifest_sha256": digest.hexdigest(),
        "selected_text_utf8_bytes": total_bytes,
    }


def _packed_manifest(packed_tokens: Any, *, seq_len: int) -> dict[str, Any]:
    shape = getattr(packed_tokens, "shape", None)
    flat = getattr(packed_tokens, "flat", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or flat is None:
        raise P0ThreeEvidenceContractError("packed tokens must expose a two-dimensional shape and flat iterator")
    chunk_count = operator.index(shape[0])
    chunk_size = operator.index(shape[1])
    if chunk_count <= 0 or chunk_size != seq_len + 1:
        raise P0ThreeEvidenceContractError("packed token shape does not match shifted-label chunking")
    digest = hashlib.sha256(b"multiscreen-p0-3-packed-token-stream-v1\0")
    count = 0
    for value in flat:
        token_id = operator.index(value)
        if not 0 <= token_id < 2**32:
            raise P0ThreeEvidenceContractError("packed token id is outside uint32")
        digest.update(token_id.to_bytes(4, "little"))
        count += 1
    expected_count = chunk_count * chunk_size
    if count != expected_count:
        raise P0ThreeEvidenceContractError("packed token count differs from its shape")
    return {
        "algorithm": TOKEN_STREAM_ALGORITHM,
        "chunk_count": chunk_count,
        "chunk_size": chunk_size,
        "packed_token_stream_sha256": digest.hexdigest(),
        "usable_token_count": count,
    }


def build_data_contract(
    *,
    source_kind: str,
    dataset_name: str | None,
    dataset_config: str | None,
    train_split: str | None,
    revision: str | None,
    text_column: str | None,
    dataset_fingerprint: str | None,
    data_files: Any,
    data_dir: str | None,
    text_file: str | None,
    max_texts: int,
    max_train_tokens: int,
    texts: Sequence[str],
    packed_tokens: Any,
    seq_len: int,
    eos_token_id: int,
    tokenizer: Mapping[str, Any],
) -> dict[str, Any]:
    if source_kind not in {"huggingface_dataset", "local_text_file"}:
        raise P0ThreeEvidenceContractError("source_kind is unsupported")
    for field, value in (
        ("max_texts", max_texts),
        ("max_train_tokens", max_train_tokens),
        ("seq_len", seq_len),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise P0ThreeEvidenceContractError(
                f"{field} must be a positive integer"
            )
    if (
        not isinstance(eos_token_id, int)
        or isinstance(eos_token_id, bool)
        or eos_token_id < 0
    ):
        raise P0ThreeEvidenceContractError(
            "eos_token_id must be a non-negative integer"
        )

    if source_kind == "huggingface_dataset":
        dataset_name = _public_identifier(
            dataset_name, field="dataset_name"
        )
        if dataset_config is not None:
            dataset_config = _public_identifier(
                dataset_config, field="dataset_config"
            )
        train_split = _public_identifier(train_split, field="train_split")
        text_column = _public_identifier(
            text_column, field="text_column"
        )
        if (
            not isinstance(dataset_fingerprint, str)
            or DATASET_FINGERPRINT_RE.fullmatch(dataset_fingerprint) is None
        ):
            raise P0ThreeEvidenceContractError(
                "dataset_fingerprint must be 16-64 lowercase hexadecimal characters"
            )
        if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
            raise P0ThreeEvidenceContractError(
                "Hugging Face dataset revision must be a full lowercase commit"
            )
        if data_files is not None or data_dir is not None or text_file is not None:
            raise P0ThreeEvidenceContractError(
                "qualifying Hub data forbids data_files, data_dir, and text_file overrides"
            )
        recorded_text_file: str | None = None
    else:
        if data_files is not None or data_dir is not None:
            raise P0ThreeEvidenceContractError(
                "local text data forbids data_files and data_dir overrides"
            )
        if not isinstance(text_file, str) or not text_file:
            raise P0ThreeEvidenceContractError(
                "local text data requires an unrecorded source path"
            )
        dataset_name = None
        dataset_config = None
        train_split = None
        revision = None
        text_column = None
        dataset_fingerprint = None
        recorded_text_file = LOCAL_TEXT_FILE_MARKER

    text_manifest = _text_manifest(texts)
    packed_manifest = _packed_manifest(packed_tokens, seq_len=seq_len)
    tokenizer_projection = _validate_tokenizer_projection(tokenizer)
    return {
        "packing": {
            **packed_manifest,
            "eos_token_id": eos_token_id,
            "legacy_shifted_labels": True,
            "max_train_tokens": max_train_tokens,
            "return_labels_are_shifted": True,
            "seq_len": seq_len,
        },
        "schema_version": SCHEMA_VERSION,
        "source": {
            "data_dir": None,
            "data_files": None,
            "dataset_config": dataset_config,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_name": dataset_name,
            "max_texts": max_texts,
            "revision": revision,
            "source_kind": source_kind,
            "text_column": text_column,
            "text_file": recorded_text_file,
            **text_manifest,
            "train_split": train_split,
        },
        "status": "recorded",
        "tokenizer": tokenizer_projection,
    }


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except OSError as exc:
            raise P0ThreeEvidenceContractError(
                "data-contract output parent cannot be inspected"
            ) from exc
        if stat.S_ISLNK(mode):
            raise P0ThreeEvidenceContractError(
                "data-contract output parent must not traverse a symlink"
            )


def _canonical_output_parent(path: Path) -> Path:
    if not path.is_absolute() or not path.name:
        raise P0ThreeEvidenceContractError(
            "data-contract output requires an absolute file path"
        )
    parent = path.parent
    _reject_symlink_components(parent)
    try:
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise P0ThreeEvidenceContractError(
            "data-contract output parent cannot be resolved"
        ) from exc
    if resolved != parent or not resolved.is_dir():
        raise P0ThreeEvidenceContractError(
            "data-contract output parent must be canonical and existing"
        )
    return resolved


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise P0ThreeEvidenceContractError(
                "data-contract output write made no progress"
            )
        view = view[written:]


def _new_temporary_file(parent_descriptor: int) -> tuple[str, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _ in range(32):
        name = f".p0-3-data-contract-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                stat.S_IRUSR | stat.S_IWUSR,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        except OSError as exc:
            raise P0ThreeEvidenceContractError(
                "data-contract temporary output could not be created"
            ) from exc
        return name, descriptor
    raise P0ThreeEvidenceContractError(
        "data-contract temporary output name collisions exceeded the limit"
    )


def write_new_report(path: Path, report: Mapping[str, Any]) -> str:
    target = Path(path)
    parent = _canonical_output_parent(target)
    data = canonical_json_bytes(report)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(parent, directory_flags)
    except OSError as exc:
        raise P0ThreeEvidenceContractError(
            "data-contract output parent could not be opened safely"
        ) from exc

    temporary_name: str | None = None
    published = False
    try:
        expected_parent = parent.stat()
        actual_parent = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(actual_parent.st_mode)
            or (expected_parent.st_dev, expected_parent.st_ino)
            != (actual_parent.st_dev, actual_parent.st_ino)
        ):
            raise P0ThreeEvidenceContractError(
                "data-contract output parent identity changed"
            )
        try:
            os.stat(
                target.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise P0ThreeEvidenceContractError(
                "data-contract final output cannot be inspected"
            ) from exc
        else:
            raise P0ThreeEvidenceContractError(
                "data-contract final output already exists"
            )

        temporary_name, temporary_descriptor = _new_temporary_file(
            parent_descriptor
        )
        try:
            os.fchmod(
                temporary_descriptor,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            _write_all(temporary_descriptor, data)
            info = os.fstat(temporary_descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise P0ThreeEvidenceContractError(
                    "data-contract temporary output identity is invalid"
                )
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                raise P0ThreeEvidenceContractError(
                    "data-contract temporary output owner is invalid"
                )
            os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)

        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise P0ThreeEvidenceContractError(
                "data-contract final output already exists"
            ) from exc
        except OSError as exc:
            raise P0ThreeEvidenceContractError(
                "data-contract output could not be published atomically"
            ) from exc
        published = True
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = None

        final_info = os.stat(
            target.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(final_info.st_mode)
            or final_info.st_nlink != 1
            or stat.S_IMODE(final_info.st_mode) != 0o600
        ):
            raise P0ThreeEvidenceContractError(
                "data-contract final output identity is invalid"
            )
        os.fsync(parent_descriptor)
    except P0ThreeEvidenceContractError:
        raise
    except BaseException as exc:
        raise P0ThreeEvidenceContractError(
            "data-contract output could not be written safely"
        ) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)

    if not published:
        raise P0ThreeEvidenceContractError(
            "data-contract output was not published"
        )
    return hashlib.sha256(data).hexdigest()
