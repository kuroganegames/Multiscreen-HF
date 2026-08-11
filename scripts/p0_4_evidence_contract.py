#!/usr/bin/env python3
"""Build the path-free P0-4 selected-data and packed-token contract."""

from __future__ import annotations

import hashlib
import operator
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import p0_3_evidence_contract as _common


SCHEMA_VERSION = "multiscreen-p0-4-data-contract-v1"
DATASET_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{16,64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
LOCAL_TEXT_FILE_MARKER = "provided_path_not_recorded"
CONFIGURED_OVERRIDE_MARKER = "configured_value_not_recorded"
TEXT_MANIFEST_ALGORITHM = "sha256-length-framed-utf8-texts-v1"
TOKEN_STREAM_ALGORITHM = "sha256-uint32-le-packed-token-stream-v1"


class P0FourEvidenceContractError(ValueError):
    """P0-4 data identity cannot be represented without ambiguity."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return _common.canonical_json_bytes(value)


def build_tokenizer_projection(
    tokenizer: Any,
    *,
    verifier: Any | None = None,
) -> dict[str, Any]:
    """Project through the exact tokenizer-reload verifier manifests."""

    try:
        return _common.build_tokenizer_projection(tokenizer, verifier=verifier)
    except _common.P0ThreeEvidenceContractError as exc:
        raise P0FourEvidenceContractError(str(exc)) from exc


def _required_text(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise P0FourEvidenceContractError(
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
        raise P0FourEvidenceContractError(
            f"{field} must be a path-free public identifier"
        )
    return text


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise P0FourEvidenceContractError(f"{field} must be a positive integer")
    return value


def _text_manifest(texts: Sequence[str]) -> dict[str, Any]:
    if isinstance(texts, (str, bytes)) or not texts:
        raise P0FourEvidenceContractError(
            "selected texts must be a non-empty sequence"
        )
    digest = hashlib.sha256(b"multiscreen-p0-4-text-manifest-v1\0")
    total_bytes = 0
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise P0FourEvidenceContractError(
                "selected texts must be non-empty strings"
            )
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
        raise P0FourEvidenceContractError(
            "packed tokens must expose a two-dimensional shape and flat iterator"
        )
    chunk_count = operator.index(shape[0])
    chunk_size = operator.index(shape[1])
    if chunk_count <= 0 or chunk_size != seq_len + 1:
        raise P0FourEvidenceContractError(
            "packed token shape does not match shifted-label chunking"
        )
    digest = hashlib.sha256(b"multiscreen-p0-4-packed-token-stream-v1\0")
    token_count = 0
    for value in flat:
        token_id = operator.index(value)
        if not 0 <= token_id < 2**32:
            raise P0FourEvidenceContractError(
                "packed token id is outside uint32"
            )
        digest.update(token_id.to_bytes(4, "little"))
        token_count += 1
    if token_count != chunk_count * chunk_size:
        raise P0FourEvidenceContractError(
            "packed token count differs from its shape"
        )
    return {
        "algorithm": TOKEN_STREAM_ALGORITHM,
        "chunk_count": chunk_count,
        "chunk_size": chunk_size,
        "packed_token_stream_sha256": digest.hexdigest(),
        "usable_token_count": token_count,
    }


def _override_marker(value: Any) -> str | None:
    return None if value is None else CONFIGURED_OVERRIDE_MARKER


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
    streaming: bool,
    max_texts: int,
    max_train_tokens: int,
    texts: Sequence[str],
    packed_tokens: Any,
    seq_len: int,
    eos_token_id: int,
    tokenizer: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical path-free identity of the P0-4 training data."""

    if source_kind not in {"huggingface_dataset", "local_text_file"}:
        raise P0FourEvidenceContractError("source_kind is unsupported")
    max_texts = _positive_int(max_texts, field="max_texts")
    max_train_tokens = _positive_int(
        max_train_tokens, field="max_train_tokens"
    )
    seq_len = _positive_int(seq_len, field="seq_len")
    if (
        not isinstance(eos_token_id, int)
        or isinstance(eos_token_id, bool)
        or eos_token_id < 0
    ):
        raise P0FourEvidenceContractError(
            "eos_token_id must be a non-negative integer"
        )
    if type(streaming) is not bool:
        raise P0FourEvidenceContractError("streaming must be an exact boolean")

    if source_kind == "huggingface_dataset":
        dataset_name = _public_identifier(dataset_name, field="dataset_name")
        if dataset_config is not None:
            dataset_config = _public_identifier(
                dataset_config, field="dataset_config"
            )
        train_split = _public_identifier(train_split, field="train_split")
        text_column = _public_identifier(text_column, field="text_column")
        if (
            not isinstance(dataset_fingerprint, str)
            or DATASET_FINGERPRINT_RE.fullmatch(dataset_fingerprint) is None
        ):
            raise P0FourEvidenceContractError(
                "dataset_fingerprint must be 16-64 lowercase hexadecimal characters"
            )
        if revision is not None:
            if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
                raise P0FourEvidenceContractError(
                    "revision must be null for the default ref or a full lowercase commit"
                )
            revision_resolution = "explicit_commit"
        else:
            revision_resolution = "default_ref"
        recorded_text_file = None
    else:
        dataset_name = None
        dataset_config = None
        train_split = None
        revision = None
        revision_resolution = None
        text_column = None
        dataset_fingerprint = None
        data_files = None
        data_dir = None
        if not isinstance(text_file, str) or not text_file:
            raise P0FourEvidenceContractError(
                "local text data requires an unrecorded source path"
            )
        recorded_text_file = LOCAL_TEXT_FILE_MARKER

    try:
        tokenizer_projection = _common._validate_tokenizer_projection(tokenizer)
    except _common.P0ThreeEvidenceContractError as exc:
        raise P0FourEvidenceContractError(str(exc)) from exc
    return {
        "packing": {
            **_packed_manifest(packed_tokens, seq_len=seq_len),
            "eos_token_id": eos_token_id,
            "legacy_shifted_labels": True,
            "max_train_tokens": max_train_tokens,
            "return_labels_are_shifted": True,
            "seq_len": seq_len,
        },
        "schema_version": SCHEMA_VERSION,
        "source": {
            "data_dir": _override_marker(data_dir),
            "data_files": _override_marker(data_files),
            "dataset_config": dataset_config,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_name": dataset_name,
            "max_texts": max_texts,
            "revision": revision,
            "revision_resolution": revision_resolution,
            "source_kind": source_kind,
            "streaming": streaming,
            "text_column": text_column,
            "text_file": recorded_text_file,
            **_text_manifest(texts),
            "train_split": train_split,
        },
        "status": "recorded",
        "tokenizer": tokenizer_projection,
    }


def write_new_report(path: Path, report: Mapping[str, Any]) -> str:
    """Publish one canonical owner-only report with exclusive atomic creation."""

    try:
        digest = _common.write_new_report(path, report)
    except _common.P0ThreeEvidenceContractError as exc:
        raise P0FourEvidenceContractError(str(exc)) from exc
    if HEX64_RE.fullmatch(digest) is None:  # defensive contract on dependency
        raise P0FourEvidenceContractError("data-contract digest is invalid")
    return digest
