#!/usr/bin/env python
"""P0.5-C3 paper-training-contract checks and bounded CUDA diagnostics.

The checked manifest is authoritative. Contract and data checks fail closed.
Operational modes are workstation diagnostics only; they do not reproduce the
paper's global batch, duration, corpus, or quality results.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import platform
import random
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


STAGE = "P0.5-C3"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "p0_5_c3_paper_training_contract.json"
)

TOKENIZER_ASSETS = {
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}

EXPECTED_VALUES: dict[str, Any] = {
    "schema_version": "1.0.0",
    "stage": STAGE,
    "description": "Checked paper-training contract plus bounded workstation diagnostics",
    "tokenizer.repository": "gpt2",
    "tokenizer.revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "tokenizer.expected_vocab_size": 50_257,
    "tokenizer.expected_eos_token_id": 50_256,
    "tokenizer.asset_manifest_sha256": "07c45937a89b33f30016aef5b3982f13f25bf2c6ba940c535d1b5daa90459a71",
    "tokenizer.assets": TOKENIZER_ASSETS,
    "tokenizer.use_fast": True,
    "dataset.family": "SlimPajama",
    "dataset.repository": "gmongaras/SlimPajama-627B_Reupload",
    "dataset.revision": "c34c22dbb10ae6b264a2f357a909d1a537141b36",
    "dataset.data_files": {"test": "data/test-00000-of-00030.parquet"},
    "dataset.split": "test",
    "dataset.text_column": "text",
    "dataset.verification_mode": "no_checks",
    "dataset.expected_file_size_bytes": 43_263_929,
    "dataset.expected_file_sha256": "d9a83d59b72f4c303f0c0e46d0e73a8446eabb56b9aa5fd992347c358ab65743",
    "dataset.expected_full_fingerprint": "507a47fcec5cbfdc",
    "dataset.selection": {
        "kind": "contiguous_rows",
        "start": 0,
        "stop": 64,
        "expected_fingerprint": "f1e6c1c09434a7e4",
    },
    "dataset.recorded_datasets_version": "5.0.1",
    "dataset.source_scope": (
        "third-party reupload test shard; not claimed byte-identical to the paper "
        "corpus or representative of its train split"
    ),
    "dataset.expected_data_contract": {
        "selected_rows": 64,
        "nonempty_documents": 64,
        "text_tokens": 58_645,
        "eos_tokens": 64,
        "concatenated_tokens": 58_709,
        "packed_chunks": 14,
        "usable_tokens": 57_358,
        "discarded_tail_tokens": 1_351,
        "row_manifest_sha256": "942f9b3397ff7073342973082efa4cddf3ace16bc7e3d180c827df3203243831",
        "token_stream_sha256": "3232bc3996272d563b6cc4e63a8d7a7d3769c7ec33e74d3d008d97cd290d7496",
        "token_encoding": "uint32_little_endian",
    },
    "model.psi_values": [8, 16],
    "model.vocab_size": 50_257,
    "model.sequence_length": 4_096,
    "model.key_dim": 16,
    "model.value_dim": 64,
    "model.mipe_threshold": 256.0,
    "model.initializer_range": 0.1,
    "model.mipe_position_mode": "paper_absolute",
    "model.mipe_compute_dtype": "fp32",
    "model.softmask_compute_dtype": "fp32",
    "model.tie_word_embeddings": True,
    "model.gradient_checkpointing": {"enabled": True, "use_reentrant": False},
    "packing.document_handling": "eos_concatenated_continuous_stream",
    "packing.sequence_length": 4_096,
    "packing.stored_chunk_tokens": 4_097,
    "packing.legacy_shifted_labels": True,
    "packing.return_labels_are_shifted": True,
    "packing.discard_incomplete_tail_only": True,
    "paper_recipe.paper_global_batch_tokens": 4_194_304,
    "paper_recipe.optimizer": {
        "name": "AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "eps": 1e-8,
        "eps_source": "repository_operationalization_paper_unspecified",
    },
    "paper_recipe.scheduler": {
        "name": "linear_warmup_then_constant",
        "step_indexing": "zero_based_optimizer_update_set_before_update",
        "warmup_steps": 4_096,
        "peak_learning_rate": 0.0625,
        "post_warmup": "constant",
    },
    "paper_recipe.gradient_clipping": {"enabled": False, "max_norm": None},
    "diagnostics.operational": {
        "qualification": "diagnostic_only_reduced_warmup_and_learning_rate",
        "optimizer_steps": 3,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 2,
        "sequence_length": 4_096,
        "warmup_steps": 2,
        "peak_learning_rate": 0.0006,
        "amp_dtype": "bf16",
        "gradient_clipping": False,
    },
    "diagnostics.peak_exposure": {
        "qualification": "diagnostic_only_bounded_exact_peak_exposure",
        "optimizer_steps": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_length": 4_096,
        "warmup_steps": 1,
        "peak_learning_rate": 0.0625,
        "amp_dtype": "bf16",
        "gradient_clipping": False,
        "require_loss_decrease": False,
    },
    "runtime.seed": 42,
    "runtime.device": "cuda:0",
    "runtime.num_workers": 0,
    "runtime.fused_adamw": False,
    "runtime.raw_output_policy": (
        "explicit absolute path outside repository; refuse overwrite; never include source text"
    ),
}

EXPECTED_KEYS: dict[str, set[str]] = {
    "": {
        "schema_version",
        "stage",
        "description",
        "tokenizer",
        "dataset",
        "model",
        "packing",
        "paper_recipe",
        "diagnostics",
        "runtime",
    },
    "tokenizer": {
        "repository",
        "revision",
        "expected_vocab_size",
        "expected_eos_token_id",
        "asset_manifest_sha256",
        "assets",
        "use_fast",
    },
    "dataset": {
        "family",
        "repository",
        "revision",
        "data_files",
        "split",
        "text_column",
        "verification_mode",
        "expected_file_size_bytes",
        "expected_file_sha256",
        "expected_full_fingerprint",
        "selection",
        "recorded_datasets_version",
        "source_scope",
        "expected_data_contract",
    },
    "dataset.selection": {"kind", "start", "stop", "expected_fingerprint"},
    "dataset.expected_data_contract": {
        "selected_rows",
        "nonempty_documents",
        "text_tokens",
        "eos_tokens",
        "concatenated_tokens",
        "packed_chunks",
        "usable_tokens",
        "discarded_tail_tokens",
        "row_manifest_sha256",
        "token_stream_sha256",
        "token_encoding",
    },
    "model": {
        "psi_values",
        "vocab_size",
        "sequence_length",
        "key_dim",
        "value_dim",
        "mipe_threshold",
        "initializer_range",
        "mipe_position_mode",
        "mipe_compute_dtype",
        "softmask_compute_dtype",
        "tie_word_embeddings",
        "gradient_checkpointing",
    },
    "model.gradient_checkpointing": {"enabled", "use_reentrant"},
    "packing": {
        "document_handling",
        "sequence_length",
        "stored_chunk_tokens",
        "legacy_shifted_labels",
        "return_labels_are_shifted",
        "discard_incomplete_tail_only",
    },
    "paper_recipe": {
        "paper_global_batch_tokens",
        "optimizer",
        "scheduler",
        "gradient_clipping",
    },
    "paper_recipe.optimizer": {
        "name",
        "betas",
        "weight_decay",
        "eps",
        "eps_source",
    },
    "paper_recipe.scheduler": {
        "name",
        "step_indexing",
        "warmup_steps",
        "peak_learning_rate",
        "post_warmup",
    },
    "paper_recipe.gradient_clipping": {"enabled", "max_norm"},
    "diagnostics": {"operational", "peak_exposure"},
    "diagnostics.operational": {
        "qualification",
        "optimizer_steps",
        "microbatch_size",
        "gradient_accumulation_steps",
        "sequence_length",
        "warmup_steps",
        "peak_learning_rate",
        "amp_dtype",
        "gradient_clipping",
    },
    "diagnostics.peak_exposure": {
        "qualification",
        "optimizer_steps",
        "microbatch_size",
        "gradient_accumulation_steps",
        "sequence_length",
        "warmup_steps",
        "peak_learning_rate",
        "amp_dtype",
        "gradient_clipping",
        "require_loss_decrease",
    },
    "runtime": {
        "seed",
        "device",
        "num_workers",
        "fused_adamw",
        "raw_output_policy",
    },
}


@dataclasses.dataclass(frozen=True)
class SelectedRows:
    texts: tuple[str, ...]
    provenance: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PackedData:
    dataset: Any
    summary: dict[str, Any]


class AccountingTokenizer:
    """Record the exact EOS-concatenated stream while forwarding encode calls."""

    def __init__(self, tokenizer: Any, *, eos_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = int(eos_token_id)
        self.document_count = 0
        self.text_token_count = 0
        self.stream_ids: list[int] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("paper packing requires add_special_tokens=False")
        ids = [
            int(token)
            for token in self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
        ]
        self.document_count += 1
        self.text_token_count += len(ids)
        self.stream_ids.extend(ids)
        self.stream_ids.append(self.eos_token_id)
        return ids


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _value_at(document: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = document
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"manifest is missing {dotted_path}")
        current = current[part]
    return current


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every checked field and reject unknown contract keys."""

    for path, expected_keys in EXPECTED_KEYS.items():
        value = manifest if path == "" else _value_at(manifest, path)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path or 'manifest'} must be an object")
        actual_keys = set(value)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"{path or 'manifest'} keys differ; missing={missing}, extra={extra}"
            )

    for path, expected in EXPECTED_VALUES.items():
        actual = _value_at(manifest, path)
        actual_canonical = canonical_json_bytes(actual)
        expected_canonical = canonical_json_bytes(expected)
        if actual_canonical != expected_canonical:
            raise ValueError(
                f"checked manifest value {path} changed: expected {expected!r}, got {actual!r}"
            )

    asset_payload = {
        "repo_id": _value_at(manifest, "tokenizer.repository"),
        "revision": _value_at(manifest, "tokenizer.revision"),
        "files": dict(_value_at(manifest, "tokenizer.assets")),
    }
    aggregate = hashlib.sha256(canonical_json_bytes(asset_payload)).hexdigest()
    expected_aggregate = _value_at(manifest, "tokenizer.asset_manifest_sha256")
    if aggregate != expected_aggregate:
        raise ValueError(
            "tokenizer asset aggregate does not match its canonical checked payload"
        )

    checkpoints = {
        str(step): paper_learning_rate(
            step,
            peak_learning_rate=float(
                _value_at(manifest, "paper_recipe.scheduler.peak_learning_rate")
            ),
            warmup_steps=int(
                _value_at(manifest, "paper_recipe.scheduler.warmup_steps")
            ),
        )
        for step in (0, 1, 4095, 4096, 4097)
    }
    return {
        "stage": STAGE,
        "status": "passed",
        "mode": "contract",
        "scheduler_checkpoints": checkpoints,
        "paper_global_batch_tokens": _value_at(
            manifest, "paper_recipe.paper_global_batch_tokens"
        ),
        "gradient_clipping_enabled": False,
        "dataset_scope": _value_at(manifest, "dataset.source_scope"),
    }


def paper_learning_rate(
    step: int,
    *,
    peak_learning_rate: float,
    warmup_steps: int,
) -> float:
    """Reference-aligned zero-based linear warmup followed by a constant LR."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps <= 0
    ):
        raise ValueError("warmup_steps must be a positive integer")
    peak = float(peak_learning_rate)
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("peak_learning_rate must be finite and positive")
    if step < warmup_steps:
        return peak * float(step + 1) / float(warmup_steps)
    return peak


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hub_file_identity(
    *,
    repository: str,
    revision: str,
    filename: str,
    repo_type: str,
    expected_size: int | None,
    expected_sha256: str,
    cache_dir: str | None,
    hub_download_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    if hub_download_fn is None:
        from huggingface_hub import hf_hub_download

        hub_download_fn = hf_hub_download
    kwargs: dict[str, Any] = {
        "repo_id": repository,
        "revision": revision,
        "filename": filename,
        "repo_type": repo_type,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    local_path = Path(hub_download_fn(**kwargs))
    size = local_path.stat().st_size
    digest = file_sha256(local_path)
    if expected_size is not None and size != expected_size:
        raise AssertionError(
            f"{repository}:{filename} size mismatch: expected {expected_size}, got {size}"
        )
    if digest != expected_sha256:
        raise AssertionError(
            f"{repository}:{filename} SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return {
        "filename": filename,
        "size_bytes": size,
        "sha256": digest,
        "revision": revision,
    }


def validate_tokenizer_assets(
    tokenizer_config: Mapping[str, Any],
    *,
    cache_dir: str | None,
    hub_download_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for filename, expected_hash in sorted(tokenizer_config["assets"].items()):
        assets[filename] = _verify_hub_file_identity(
            repository=str(tokenizer_config["repository"]),
            revision=str(tokenizer_config["revision"]),
            filename=filename,
            repo_type="model",
            expected_size=None,
            expected_sha256=str(expected_hash),
            cache_dir=cache_dir,
            hub_download_fn=hub_download_fn,
        )
    actual_hashes = {
        filename: record["sha256"] for filename, record in sorted(assets.items())
    }
    aggregate_payload = {
        "repo_id": str(tokenizer_config["repository"]),
        "revision": str(tokenizer_config["revision"]),
        "files": actual_hashes,
    }
    aggregate = hashlib.sha256(canonical_json_bytes(aggregate_payload)).hexdigest()
    if aggregate != tokenizer_config["asset_manifest_sha256"]:
        raise AssertionError("downloaded tokenizer asset aggregate SHA-256 mismatch")
    return {
        "repository": tokenizer_config["repository"],
        "revision": tokenizer_config["revision"],
        "assets": assets,
        "asset_manifest_sha256": aggregate,
    }


def load_pinned_tokenizer(
    manifest: Mapping[str, Any],
    *,
    cache_dir: str | None,
    hub_download_fn: Callable[..., str] | None = None,
    tokenizer_loader: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    config = _value_at(manifest, "tokenizer")
    assets = validate_tokenizer_assets(
        config,
        cache_dir=cache_dir,
        hub_download_fn=hub_download_fn,
    )
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer_loader = AutoTokenizer.from_pretrained
    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": config["repository"],
        "revision": config["revision"],
        "use_fast": config["use_fast"],
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    tokenizer = tokenizer_loader(**kwargs)
    if len(tokenizer) != config["expected_vocab_size"]:
        raise AssertionError(
            f"GPT-2 vocabulary mismatch: expected {config['expected_vocab_size']}, got {len(tokenizer)}"
        )
    if tokenizer.eos_token_id != config["expected_eos_token_id"]:
        raise AssertionError(
            f"GPT-2 EOS mismatch: expected {config['expected_eos_token_id']}, "
            f"got {tokenizer.eos_token_id}"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # GPT-2's published model context is 1024, but C3 tokenizes whole documents
    # without truncation before packing them into 4096-token training chunks.
    tokenizer.model_max_length = 1_000_000_000
    return tokenizer, {
        **assets,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "eos_token_id": int(tokenizer.eos_token_id),
        "document_tokenization_truncation": False,
    }


def load_pinned_rows(
    manifest: Mapping[str, Any],
    *,
    cache_dir: str | None,
    load_dataset_fn: Callable[..., Any] | None = None,
    hub_download_fn: Callable[..., str] | None = None,
    datasets_version: str | None = None,
) -> SelectedRows:
    config = _value_at(manifest, "dataset")
    file_record = _verify_hub_file_identity(
        repository=str(config["repository"]),
        revision=str(config["revision"]),
        filename=str(config["data_files"][config["split"]]),
        repo_type="dataset",
        expected_size=int(config["expected_file_size_bytes"]),
        expected_sha256=str(config["expected_file_sha256"]),
        cache_dir=cache_dir,
        hub_download_fn=hub_download_fn,
    )

    if load_dataset_fn is None:
        import datasets

        load_dataset_fn = datasets.load_dataset
        datasets_version = datasets.__version__
    if datasets_version != config["recorded_datasets_version"]:
        raise RuntimeError(
            "dataset fingerprints are qualified only under datasets "
            f"{config['recorded_datasets_version']}; got {datasets_version}"
        )

    kwargs: dict[str, Any] = {
        "data_files": dict(config["data_files"]),
        "split": config["split"],
        "revision": config["revision"],
        "streaming": False,
        "verification_mode": config["verification_mode"],
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    full_dataset = load_dataset_fn(config["repository"], **kwargs)
    full_fingerprint = getattr(full_dataset, "_fingerprint", None)
    if full_fingerprint != config["expected_full_fingerprint"]:
        raise AssertionError(
            "canonical Hub map-style dataset fingerprint mismatch: "
            f"expected {config['expected_full_fingerprint']}, got {full_fingerprint}"
        )

    selection = config["selection"]
    if selection["kind"] != "contiguous_rows":
        raise AssertionError("only contiguous_rows selection is supported")
    row_indices = range(int(selection["start"]), int(selection["stop"]))
    selected = full_dataset.select(row_indices)
    selected_fingerprint = getattr(selected, "_fingerprint", None)
    if selected_fingerprint != selection["expected_fingerprint"]:
        raise AssertionError(
            "selected dataset fingerprint mismatch: "
            f"expected {selection['expected_fingerprint']}, got {selected_fingerprint}"
        )

    text_column = str(config["text_column"])
    texts: list[str] = []
    row_records: list[dict[str, Any]] = []
    for offset, row in enumerate(selected):
        value = row[text_column]
        text = "" if value is None else str(value)
        raw = text.encode("utf-8")
        row_index = int(selection["start"]) + offset
        texts.append(text)
        row_records.append(
            {
                "row_index": row_index,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "utf8_bytes": len(raw),
            }
        )
    expected_rows = int(selection["stop"]) - int(selection["start"])
    if len(texts) != expected_rows:
        raise AssertionError(
            f"selected row count mismatch: expected {expected_rows}, got {len(texts)}"
        )
    row_manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(row_records)
    ).hexdigest()

    return SelectedRows(
        texts=tuple(texts),
        provenance={
            "family": config["family"],
            "repository": config["repository"],
            "revision": config["revision"],
            "data_files": dict(config["data_files"]),
            "split": config["split"],
            "file": file_record,
            "datasets_version": datasets_version,
            "full_fingerprint": full_fingerprint,
            "selection": {
                "kind": selection["kind"],
                "start": selection["start"],
                "stop": selection["stop"],
                "fingerprint": selected_fingerprint,
                "row_count": len(texts),
            },
            "row_records": row_records,
            "row_manifest_sha256": row_manifest_sha256,
            "source_scope": config["source_scope"],
            "streaming": False,
            "canonical_hub_loader": True,
            "local_parquet_fingerprint_not_accepted": True,
        },
    )


def uint32_little_endian_sha256(token_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token in token_ids:
        value = int(token)
        if value < 0 or value >= 2**32:
            raise ValueError(f"token ID cannot be encoded as uint32: {value}")
        digest.update(value.to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def pack_selected_rows(
    manifest: Mapping[str, Any],
    *,
    selected_rows: SelectedRows,
    tokenizer: Any,
    tokenizer_provenance: Mapping[str, Any],
    enforce_checked_accounting: bool = True,
) -> PackedData:
    from multiscreen_transformers import PackedTextDataset

    packing = _value_at(manifest, "packing")
    eos_token_id = int(_value_at(manifest, "tokenizer.expected_eos_token_id"))
    accounting_tokenizer = AccountingTokenizer(
        tokenizer,
        eos_token_id=eos_token_id,
    )
    dataset = PackedTextDataset(
        texts=selected_rows.texts,
        tokenizer=accounting_tokenizer,
        seq_len=int(packing["sequence_length"]),
        eos_token_id=eos_token_id,
        max_tokens=None,
        legacy_shifted_labels=bool(packing["legacy_shifted_labels"]),
        return_labels_are_shifted=bool(packing["return_labels_are_shifted"]),
    )

    concatenated_tokens = len(accounting_tokenizer.stream_ids)
    usable_tokens = int(dataset.tokens.size)
    accounting = {
        "selected_rows": len(selected_rows.texts),
        "nonempty_documents": accounting_tokenizer.document_count,
        "text_tokens": accounting_tokenizer.text_token_count,
        "eos_tokens": accounting_tokenizer.document_count,
        "concatenated_tokens": concatenated_tokens,
        "packed_chunks": len(dataset),
        "usable_tokens": usable_tokens,
        "discarded_tail_tokens": concatenated_tokens - usable_tokens,
        "sequence_length": int(packing["sequence_length"]),
        "stored_chunk_tokens": int(packing["stored_chunk_tokens"]),
        "prediction_tokens_per_chunk": int(packing["sequence_length"]),
    }
    hashes = {
        "row_manifest_sha256": selected_rows.provenance["row_manifest_sha256"],
        "packed_chunk_sha256": [
            uint32_little_endian_sha256(chunk.tolist())
            for chunk in dataset.tokens
        ],
        "token_stream_sha256": uint32_little_endian_sha256(
            accounting_tokenizer.stream_ids
        ),
        "token_encoding": "uint32_little_endian",
    }
    checked = {**accounting, **hashes}
    expected = _value_at(manifest, "dataset.expected_data_contract")
    comparison = {key: checked[key] for key in expected}
    if enforce_checked_accounting and comparison != expected:
        differences = {
            key: {"expected": expected[key], "actual": comparison[key]}
            for key in expected
            if comparison[key] != expected[key]
        }
        raise AssertionError(f"checked data/token accounting mismatch: {differences}")

    summary = {
        "stage": STAGE,
        "status": "passed",
        "mode": "data",
        "qualification": "diagnostic_pinned_slimpajama_family_shard",
        "tokenizer": dict(tokenizer_provenance),
        "source": selected_rows.provenance,
        "packing": {
            "document_handling": packing["document_handling"],
            "legacy_shifted_labels": packing["legacy_shifted_labels"],
            "return_labels_are_shifted": packing["return_labels_are_shifted"],
            "discard_incomplete_tail_only": packing["discard_incomplete_tail_only"],
        },
        "accounting": accounting,
        "hashes": hashes,
        "checked_accounting_match": comparison == expected,
        "limitations": [
            "The source is a pinned third-party SlimPajama reupload test shard.",
            "It is not claimed byte-identical to the paper corpus or representative of its train split.",
            "No raw source text is retained in this output.",
        ],
    }
    return PackedData(dataset=dataset, summary=summary)


def build_data_bundle(
    manifest: Mapping[str, Any],
    *,
    cache_dir: str | None,
) -> PackedData:
    tokenizer, tokenizer_provenance = load_pinned_tokenizer(
        manifest,
        cache_dir=cache_dir,
    )
    selected_rows = load_pinned_rows(
        manifest,
        cache_dir=cache_dir,
    )
    return pack_selected_rows(
        manifest,
        selected_rows=selected_rows,
        tokenizer=tokenizer,
        tokenizer_provenance=tokenizer_provenance,
        enforce_checked_accounting=True,
    )


def make_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    manifest: Mapping[str, Any],
    learning_rate: float,
) -> torch.optim.AdamW:
    recipe = _value_at(manifest, "paper_recipe.optimizer")
    if _value_at(manifest, "paper_recipe.gradient_clipping.enabled") is not False:
        raise AssertionError("C3 requires gradient clipping to be disabled")
    return torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=(float(recipe["betas"][0]), float(recipe["betas"][1])),
        weight_decay=float(recipe["weight_decay"]),
        eps=float(recipe["eps"]),
        fused=bool(_value_at(manifest, "runtime.fused_adamw")),
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    value = float(learning_rate)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("optimizer learning rate must be finite and positive")
    for group in optimizer.param_groups:
        group["lr"] = value


def gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = 0.0
    found = False
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach().float()
        if not bool(torch.isfinite(detached).all()):
            raise RuntimeError("non-finite gradient")
        squared += float(detached.square().sum().cpu())
        found = True
    if not found:
        raise RuntimeError("optimizer update has no gradients")
    value = math.sqrt(squared)
    if not math.isfinite(value):
        raise RuntimeError("non-finite gradient norm")
    return value


def _tracked_gradient_coordinate(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[str, torch.nn.Parameter, int, float]:
    best: tuple[str, torch.nn.Parameter, int, float] | None = None
    for name, parameter in named_parameters:
        if parameter.grad is None or parameter.numel() == 0:
            continue
        flat = parameter.grad.detach().float().reshape(-1)
        absolute = flat.abs()
        maximum, index = absolute.max(dim=0)
        value = float(maximum.cpu())
        if math.isfinite(value) and value > 0 and (best is None or value > best[3]):
            best = (name, parameter, int(index.cpu()), value)
    if best is None:
        raise RuntimeError("no nonzero finite gradient coordinate found")
    return best


def optimizer_step_without_clipping(
    optimizer: torch.optim.Optimizer,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    """Take one optimizer step after read-only norm/update checks.

    This function intentionally never accepts a max norm and never calls a
    clipping helper.
    """

    norm = gradient_l2_norm(parameter for _, parameter in named_parameters)
    name, parameter, index, gradient_abs = _tracked_gradient_coordinate(
        named_parameters
    )
    flat = parameter.detach().reshape(-1)
    before = float(flat[index].float().cpu())
    optimizer.step()
    after = float(parameter.detach().reshape(-1)[index].float().cpu())
    delta = after - before
    if not math.isfinite(delta) or delta == 0.0:
        raise RuntimeError("optimizer step did not produce a finite nonzero update")

    for parameter_name, candidate in named_parameters:
        if not bool(torch.isfinite(candidate.detach()).all()):
            raise RuntimeError(f"non-finite parameter after update: {parameter_name}")
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(
                torch.isfinite(value.detach()).all()
            ):
                raise RuntimeError("non-finite optimizer state after update")
    return {
        "gradient_l2_norm": norm,
        "tracked_parameter": name,
        "tracked_gradient_abs": gradient_abs,
        "tracked_parameter_delta": delta,
        "gradient_clipping_applied": False,
    }


def cycle_loader(loader: DataLoader) -> Iterator[Mapping[str, Any]]:
    while True:
        yield from loader


def move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def cuda_bf16_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def build_model(
    manifest: Mapping[str, Any],
    *,
    psi: int,
    tokenizer_summary: Mapping[str, Any],
):
    from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM

    model_config = _value_at(manifest, "model")
    if psi not in model_config["psi_values"]:
        raise ValueError(f"Psi={psi} is outside the checked C3 values")
    sequence_length = int(model_config["sequence_length"])
    eos = int(tokenizer_summary["eos_token_id"])
    config = MultiscreenConfig.from_psi(
        psi=psi,
        vocab_size=int(model_config["vocab_size"]),
        max_seq_len=sequence_length,
        key_dim=int(model_config["key_dim"]),
        value_dim=int(model_config["value_dim"]),
        mipe_threshold=float(model_config["mipe_threshold"]),
        initializer_range=float(model_config["initializer_range"]),
        mipe_position_mode=str(model_config["mipe_position_mode"]),
        mipe_reference_wrap_boundary=sequence_length,
        mipe_compute_dtype=str(model_config["mipe_compute_dtype"]),
        softmask_compute_dtype=str(model_config["softmask_compute_dtype"]),
        tie_word_embeddings=bool(model_config["tie_word_embeddings"]),
        gradient_checkpointing=False,
        use_cache=False,
        labels_are_shifted=False,
        strict_position_ids=True,
        strict_cache_positions=True,
        bos_token_id=eos,
        eos_token_id=eos,
        pad_token_id=eos,
    )
    model = MultiscreenForCausalLM(config)
    checkpointing = model_config["gradient_checkpointing"]
    if checkpointing["enabled"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": checkpointing["use_reentrant"]
            }
        )
    return model


def memory_summary(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def runtime_environment(device: torch.device) -> dict[str, Any]:
    import datasets
    import huggingface_hub
    import transformers

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                dict(value),
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def prepare_output_directory(value: str, *, repository_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("output directory must be an explicit absolute path")
    resolved = candidate.resolve()
    repository = repository_root.resolve()
    if resolved == repository or repository in resolved.parents:
        raise ValueError("output directory must be outside the current repository")
    for ancestor in (resolved, *resolved.parents):
        if (ancestor / ".git").exists():
            raise ValueError(
                "output directory must be outside every Git repository and worktree"
            )
    if resolved.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_training_diagnostic(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    psi: int,
    device_name: str,
    data: PackedData,
    output_dir: Path,
) -> dict[str, Any]:
    mode_config_keys = {
        "operational": "operational",
        "peak-exposure": "peak_exposure",
    }
    if mode not in mode_config_keys:
        raise ValueError(f"unsupported training diagnostic mode: {mode}")
    settings = _value_at(manifest, f"diagnostics.{mode_config_keys[mode]}")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C3 training diagnostics require CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("C3 training diagnostics require CUDA bf16 support")
    if settings["amp_dtype"] != "bf16":
        raise AssertionError("C3 training diagnostics require bf16 autocast")
    if settings["gradient_clipping"] is not False:
        raise AssertionError("C3 training diagnostics prohibit gradient clipping")
    if int(settings["sequence_length"]) != int(
        _value_at(manifest, "model.sequence_length")
    ):
        raise AssertionError("diagnostic sequence length must remain 4096")

    seed = int(_value_at(manifest, "runtime.seed")) + psi * 1009
    if mode == "peak-exposure":
        seed += 1_000_003
    _seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    model = build_model(
        manifest,
        psi=psi,
        tokenizer_summary=data.summary["tokenizer"],
    ).to(device)
    model.train()
    named_parameters = list(model.named_parameters())
    parameter_count = sum(parameter.numel() for _, parameter in named_parameters)

    loader = DataLoader(
        data.dataset,
        batch_size=int(settings["microbatch_size"]),
        shuffle=False,
        drop_last=True,
        num_workers=int(_value_at(manifest, "runtime.num_workers")),
        pin_memory=True,
    )
    if len(loader) == 0:
        raise RuntimeError("packed data produced an empty DataLoader")
    iterator = cycle_loader(loader)

    optimizer = make_optimizer(
        (parameter for _, parameter in named_parameters),
        manifest=manifest,
        learning_rate=float(settings["peak_learning_rate"]),
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8", newline="\n")
    torch.cuda.reset_peak_memory_stats(device)

    events: list[dict[str, Any]] = []
    for step in range(int(settings["optimizer_steps"])):
        learning_rate = paper_learning_rate(
            step,
            peak_learning_rate=float(settings["peak_learning_rate"]),
            warmup_steps=int(settings["warmup_steps"]),
        )
        set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(int(settings["gradient_accumulation_steps"])):
            batch = move_batch(next(iterator), device)
            if int(batch["input_ids"].shape[-1]) != int(settings["sequence_length"]):
                raise AssertionError("training batch sequence length changed")
            with cuda_bf16_context(device):
                output = model(
                    **batch,
                    use_cache=False,
                    return_dict=True,
                )
                loss = output.loss
            if loss is None:
                raise RuntimeError("model returned loss=None")
            loss_value = float(loss.detach().float().cpu())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite loss at optimizer step {step}")
            micro_losses.append(loss_value)
            (
                loss / int(settings["gradient_accumulation_steps"])
            ).backward()

        update = optimizer_step_without_clipping(
            optimizer,
            named_parameters,
        )
        event = {
            "event": "optimizer_step",
            "stage": STAGE,
            "mode": mode,
            "psi": psi,
            "optimizer_step_zero_based": step,
            "learning_rate": learning_rate,
            "mean_micro_loss": sum(micro_losses) / len(micro_losses),
            "micro_losses": micro_losses,
            **update,
            **memory_summary(device),
        }
        events.append(event)
        append_jsonl(metrics_path, event)

    model.eval()
    post_batch = move_batch(next(iterator), device)
    with torch.no_grad(), cuda_bf16_context(device):
        post_loss_tensor = model(
            **post_batch,
            use_cache=False,
            return_dict=True,
        ).loss
    if post_loss_tensor is None:
        raise RuntimeError("post-update model returned loss=None")
    post_loss = float(post_loss_tensor.detach().float().cpu())
    if not math.isfinite(post_loss):
        raise RuntimeError("post-update loss is non-finite")

    effective_tokens = (
        int(settings["sequence_length"])
        * int(settings["microbatch_size"])
        * int(settings["gradient_accumulation_steps"])
    )
    paper_tokens = int(
        _value_at(manifest, "paper_recipe.paper_global_batch_tokens")
    )
    summary = {
        "stage": STAGE,
        "status": "diagnostic_passed",
        "mode": mode,
        "qualification": settings["qualification"],
        "timestamp_utc": utc_now(),
        "psi": psi,
        "parameter_count": parameter_count,
        "model": {
            "vocab_size": int(model.config.vocab_size),
            "sequence_length": int(model.config.max_position_embeddings),
            "mipe_position_mode": model.config.mipe_position_mode,
            "mipe_compute_dtype": model.config.mipe_compute_dtype,
            "softmask_compute_dtype": model.config.softmask_compute_dtype,
            "gradient_checkpointing": bool(model.multiscreen.gradient_checkpointing),
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "betas": list(optimizer.defaults["betas"]),
            "weight_decay": optimizer.defaults["weight_decay"],
            "eps": optimizer.defaults["eps"],
            "eps_source": _value_at(
                manifest, "paper_recipe.optimizer.eps_source"
            ),
            "fused": bool(optimizer.defaults.get("fused", False)),
            "gradient_clipping": False,
        },
        "scheduler": {
            "name": "linear_warmup_then_constant",
            "paper_warmup_steps": int(
                _value_at(manifest, "paper_recipe.scheduler.warmup_steps")
            ),
            "paper_peak_learning_rate": float(
                _value_at(manifest, "paper_recipe.scheduler.peak_learning_rate")
            ),
            "executed_warmup_steps": int(settings["warmup_steps"]),
            "executed_peak_learning_rate": float(settings["peak_learning_rate"]),
            "observed_learning_rates": [
                event["learning_rate"] for event in events
            ],
            "diagnostic_reduced_from_paper": (
                int(settings["warmup_steps"]) != 4096
                or float(settings["peak_learning_rate"]) != 0.0625
            ),
        },
        "training": {
            "optimizer_steps": int(settings["optimizer_steps"]),
            "world_size": 1,
            "microbatch_size": int(settings["microbatch_size"]),
            "sequences_per_optimizer_step": (
                int(settings["microbatch_size"])
                * int(settings["gradient_accumulation_steps"])
            ),
            "gradient_accumulation_steps": int(
                settings["gradient_accumulation_steps"]
            ),
            "effective_tokens_per_optimizer_step": effective_tokens,
            "paper_global_batch_tokens": paper_tokens,
            "local_to_paper_batch_ratio": effective_tokens / paper_tokens,
            "losses_finite": True,
            "gradients_finite": True,
            "parameters_finite": True,
            "optimizer_updates_nonzero": True,
            "post_update_loss": post_loss,
            "loss_decrease_required": bool(
                settings.get("require_loss_decrease", False)
            ),
        },
        "data": data.summary,
        "environment": runtime_environment(device),
        "memory": memory_summary(device),
        "limitations": [
            "This is a bounded dense-reference workstation diagnostic.",
            "It does not reproduce the paper global batch, duration, corpus, quality, or efficiency.",
            "The peak-exposure mode requires finite updates, not loss decrease or model quality.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    marker_name = (
        "P0_5_C3_OPERATIONAL_COMPLETE.json"
        if mode == "operational"
        else "P0_5_C3_PEAK_EXPOSURE_COMPLETE.json"
    )
    write_json(
        output_dir / marker_name,
        {
            "stage": STAGE,
            "status": "diagnostic_passed",
            "mode": mode,
            "psi": psi,
            "timestamp_utc": summary["timestamp_utc"],
        },
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--mode",
        choices=("contract", "data", "operational", "peak-exposure"),
        default="contract",
    )
    parser.add_argument("--psi", type=int, choices=(8, 16), default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    contract_summary = validate_manifest(manifest)
    if args.mode == "contract":
        sys.stdout.write(
            json.dumps(
                contract_summary,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return 0

    if args.output_dir is None:
        raise ValueError("--output-dir is required outside contract mode")
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = prepare_output_directory(
        args.output_dir,
        repository_root=repository_root,
    )
    try:
        data = build_data_bundle(
            manifest,
            cache_dir=args.cache_dir,
        )
        if args.mode == "data":
            write_json(output_dir / "data_contract.json", data.summary)
            write_json(
                output_dir / "P0_5_C3_DATA_CONTRACT_COMPLETE.json",
                {
                    "stage": STAGE,
                    "status": "passed",
                    "mode": "data",
                    "timestamp_utc": utc_now(),
                },
            )
            return 0
        if args.psi is None:
            raise ValueError("--psi is required for training diagnostics")
        device = args.device or str(_value_at(manifest, "runtime.device"))
        run_training_diagnostic(
            manifest,
            mode=args.mode,
            psi=args.psi,
            device_name=device,
            data=data,
            output_dir=output_dir,
        )
        return 0
    except BaseException as exc:
        write_json(
            output_dir / "failure.json",
            {
                "stage": STAGE,
                "status": "failed",
                "mode": args.mode,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "timestamp_utc": utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
