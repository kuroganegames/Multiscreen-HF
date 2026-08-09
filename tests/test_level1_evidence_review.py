from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import stat
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "review_level1_requalification",
    REPO_ROOT / "scripts" / "review_level1_requalification.py",
)
assert SPEC is not None and SPEC.loader is not None
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)

CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_level1_repository",
    REPO_ROOT / "scripts" / "check_level1_repository.py",
)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
REPOSITORY_CHECK = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(REPOSITORY_CHECK)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=True, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=True, allow_nan=True, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
        newline="\n",
    )


class EvidenceFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        self.p0_3 = root / "artifacts" / "p0-3"
        self.p0_3_stdout = root / "p0_3.stdout.log"
        self.p0_4_psi8 = root / "artifacts" / "p0-4" / "psi8"
        self.p0_4_psi16 = root / "artifacts" / "p0-4" / "psi16"
        self.c3_roots = {
            "c3_psi8_operational": root / "artifacts/c3/cuda/psi8/operational",
            "c3_psi8_peak_exposure": root / "artifacts/c3/cuda/psi8/peak-exposure",
            "c3_psi16_operational": root / "artifacts/c3/cuda/psi16/operational",
            "c3_psi16_peak_exposure": root / "artifacts/c3/cuda/psi16/peak-exposure",
        }
        self.p0_4_contract_sha256: dict[int, str] = {}
        self.c3_data = root / "artifacts/c3/data"
        self.tokenizer_reports: dict[str, Path] = {}
        self.ledger = root / "commands.jsonl"
        self.tested_commit = "d" * 40
        self.repository = Path(REVIEW.__file__).resolve().parents[1]
        self.cache = root / "fixture-cache"
        self.cache.mkdir(mode=0o700)
        self.tf4576_python = "/opt/multiscreen/tf4576/bin/python"
        self.tf5141_python = "/opt/multiscreen/tf5141/bin/python"
        (root / "logs").mkdir(mode=0o700)
        (root / "records").mkdir(mode=0o700)
        self.p0_3_stdout = root / "logs" / "p0-3-checkpointed.log"
        self._make_p0_3()
        self._make_p0_4(self.p0_4_psi8, psi=8)
        self._make_p0_4(self.p0_4_psi16, psi=16)
        self._make_c3_data()
        self._make_c3()
        self._make_tokenizers()
        self._make_ledger()
        self._make_focused_review()

    def _p0_3_metric(
        self, psi: int, data_contract_sha256: str
    ) -> dict[str, object]:
        steps = REVIEW.P0_3_STEPS[psi]
        first = round(10.0 - 0.01, 10)
        last = round(10.0 - 0.01 * steps, 10)
        checkpoint = self.p0_3 / f"psi{psi}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        return {
            "psi": psi,
            "data_contract_sha256": data_contract_sha256,
            "steps": steps,
            "params": 123,
            "device": "cuda:0",
            "amp_dtype": "bf16",
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "seq_len": 128,
            "batch_size": 4,
            "tokens_per_step": 512,
            "approx_tokens_seen": steps * 512,
            "initial_probe_loss": 10.0,
            "final_probe_loss": 8.0,
            "abs_loss_drop": 2.0,
            "rel_loss_drop": 0.2,
            "train_loss_first": first,
            "train_loss_last": last,
            "train_loss_min": last,
            "grad_norm_max": 1.0,
            "save_load_logits_max_abs": 0.0,
            "cache_split_logits_max_abs": 0.01,
            "generation": {
                "prompt_len": 4,
                "generated_len": 16,
                "sample_text": "fixture",
            },
            "checkpoint_dir": str(checkpoint),
        }

    def _make_p0_3(self) -> None:
        self.p0_3.mkdir(parents=True)
        data_contract = {
            "packing": {
                "algorithm": "sha256-uint32-le-packed-token-stream-v1",
                "chunk_count": 2_032,
                "chunk_size": 129,
                "eos_token_id": 2,
                "legacy_shifted_labels": True,
                "max_train_tokens": 262_144,
                "packed_token_stream_sha256": "2" * 64,
                "return_labels_are_shifted": True,
                "seq_len": 128,
                "usable_token_count": 262_128,
            },
            "schema_version": "multiscreen-p0-3-data-contract-v1",
            "source": {
                "algorithm": "sha256-length-framed-utf8-texts-v1",
                "data_dir": None,
                "data_files": None,
                "dataset_config": None,
                "dataset_fingerprint": "0123456789abcdef",
                "dataset_name": "roneneldan/TinyStories",
                "max_texts": 20_000,
                "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                "selected_text_count": 20_000,
                "selected_text_manifest_sha256": "1" * 64,
                "selected_text_utf8_bytes": 1_000_000,
                "source_kind": "huggingface_dataset",
                "text_column": "text",
                "text_file": None,
                "train_split": "train[:20000]",
            },
            "status": "recorded",
            "tokenizer": {
                "class": "FixtureTokenizer",
                "counts": {
                    "added_vocabulary": 4,
                    "all_special_tokens": 4,
                    "probes": 5,
                    "special_token_boundary_probes": 28,
                    "tokenizer_length": 768,
                    "vocab_size": 768,
                    "vocabulary": 768,
                },
                "hashes": {
                    "probe_manifest_sha256": "a" * 64,
                    "special_tokens_manifest_sha256": "a" * 64,
                    "vocabulary_manifest_sha256": "a" * 64,
                },
                "is_fast": True,
                "operationalization": {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "model_max_length": 512,
                    "padding_side": "right",
                    "truncation_side": "right",
                },
            },
        }
        contract_raw = REVIEW._canonical_bytes(data_contract)
        (self.p0_3 / "data_contract.json").write_bytes(contract_raw)
        contract_sha256 = hashlib.sha256(contract_raw).hexdigest()
        metrics = [
            self._p0_3_metric(8, contract_sha256),
            self._p0_3_metric(16, contract_sha256),
        ]
        write_json(self.p0_3 / "p0_3_results.json", metrics)
        for metric in metrics:
            write_json(
                self.p0_3 / f"psi{metric['psi']}" / "p0_3_metrics.json", metric
            )
        (self.p0_3 / "P0-3_COMPLETE.md").write_text(
            "# P0-3\n\nPassed.\n", encoding="utf-8", newline="\n"
        )
        lines: list[str] = [
            f"[P0-3] data_contract sha256={contract_sha256}"
        ]
        for psi in (8, 16):
            steps = REVIEW.P0_3_STEPS[psi]
            lines.append(
                f"[P0-3] Psi={psi} params=123 steps={steps} device=cuda:0 amp=bf16"
            )
            for step in range(1, steps + 1):
                lines.append(
                    f"[P0-3][Psi={psi}] step={step:04d}/{steps} "
                    f"loss={10.0 - 0.01 * step:.4f} grad_norm=1.0000"
                )
            lines.append(
                f"[P0-3][Psi={psi}] probe_loss initial=10.0000 "
                "final=8.0000 drop=2.0000 rel=20.0000%"
            )
        lines.extend(
            [
                "P0-3 TinyStories stability checks passed.",
                f"[P0-3] wrote metrics to {self.p0_3 / 'p0_3_results.json'}",
                f"[P0-3] wrote note to {self.p0_3 / 'P0-3_COMPLETE.md'}",
            ]
        )
        self.p0_3_stdout.write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )

    def _p0_4_tokenizer_projection(self) -> dict[str, object]:
        digest = "a" * 64
        return {
            "class": "GPT2TokenizerFast",
            "counts": {
                "added_vocabulary": 4,
                "all_special_tokens": 4,
                "probes": 5,
                "special_token_boundary_probes": 28,
                "tokenizer_length": 50_257,
                "vocab_size": 50_257,
                "vocabulary": 50_257,
            },
            "hashes": {
                "probe_manifest_sha256": digest,
                "special_tokens_manifest_sha256": digest,
                "vocabulary_manifest_sha256": digest,
            },
            "is_fast": True,
            "operationalization": {
                "model_input_names": ["input_ids", "attention_mask"],
                "model_max_length": 4_096,
                "padding_side": "right",
                "truncation_side": "right",
            },
        }

    def _make_p0_4_data_contract(
        self, root: Path, *, psi: int
    ) -> dict[str, str]:
        contract = {
            "packing": {
                "algorithm": "sha256-uint32-le-packed-token-stream-v1",
                "chunk_count": 128,
                "chunk_size": 4_097,
                "eos_token_id": 50_256,
                "legacy_shifted_labels": True,
                "max_train_tokens": 524_416,
                "packed_token_stream_sha256": "c" * 64,
                "return_labels_are_shifted": True,
                "seq_len": 4_096,
                "usable_token_count": 524_416,
            },
            "schema_version": "multiscreen-p0-4-data-contract-v1",
            "source": {
                "algorithm": "sha256-length-framed-utf8-texts-v1",
                "data_dir": None,
                "data_files": None,
                "dataset_config": None,
                "dataset_fingerprint": "fedcba9876543210",
                "dataset_name": "roneneldan/TinyStories",
                "max_texts": 20_000,
                "revision": None,
                "revision_resolution": "default_ref",
                "selected_text_count": 20_000,
                "selected_text_manifest_sha256": "b" * 64,
                "selected_text_utf8_bytes": 123_456,
                "source_kind": "huggingface_dataset",
                "streaming": False,
                "text_column": "text",
                "text_file": None,
                "train_split": "train[:20000]",
            },
            "status": "recorded",
            "tokenizer": self._p0_4_tokenizer_projection(),
        }
        raw = REVIEW._canonical_bytes(contract)
        (root / "data_contract.json").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        self.p0_4_contract_sha256[psi] = digest
        return {
            "file": "data_contract.json",
            "schema_version": "multiscreen-p0-4-data-contract-v1",
            "sha256": digest,
        }

    def _p0_4_summary(self, root: Path, psi: int) -> dict[str, object]:
        data_contract_ref = {
            "file": "data_contract.json",
            "schema_version": "multiscreen-p0-4-data-contract-v1",
            "sha256": self.p0_4_contract_sha256[psi],
        }
        settings = {
            "expected_vocab_size": 50257,
            "seq_len": 4096,
            "steps": 50,
            "batch_size": 1,
            "grad_accum": 8,
            "amp_dtype": "bf16",
            "gradient_checkpointing": True,
            "allow_cpu": False,
            "device": "cuda:0",
            "output_dir": str(root),
            "cache_dir": str(self.cache),
            "tokenizer_name": "gpt2",
            "tokenizer_use_fast": True,
            "max_texts": 20000,
            "max_train_tokens": 524416,
            "lr": 0.0006,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "max_grad_norm": 1.0,
            "reload_atol": 1e-5,
            "reload_rtol": 1e-5,
            "cache_atol": 0.03,
            "cache_rtol": 0.03,
            "fused_adamw": True,
            "probe_replay_every": 4,
            "seed": 42,
            "log_every": 1,
            "repo_root": str(self.repository),
            "config_dir": str(self.repository / f"configs/p0_4_multiscreen_psi{psi}_gpt2_ctx4096"),
            "dataset_name": "roneneldan/TinyStories",
            "dataset_config": None,
            "train_split": "train[:20000]",
            "text_column": "text",
            "text_file": None,
            "data_files": None,
            "data_dir": None,
            "revision": None,
            "streaming": False,
            "min_loss_drop": 0.01,
            "min_rel_loss_drop": 0.001,
            "reload_tokens": 16,
            "cache_tokens": 24,
            "prompt": "Once upon a time",
            "max_new_tokens": 8,
            "num_workers": 0,
        }
        environment = {
            "python": "3.12.11 (fixture)",
            "platform": "Linux-fixture",
            "torch": "2.7.1+cu128",
            "transformers": "4.57.6",
            "datasets": "5.0.1",
            "device": "cuda:0",
            "cuda_available": True,
            "cuda_version": "12.8",
            "gpu_name": "Fixture GPU",
            "gpu_total_memory_bytes": 1024,
            "bf16_supported": True,
        }
        model = {
            "psi": psi,
            "parameter_count": {8: 4_134_146, 16: 27_546_626}[psi],
            "vocab_size": 50257,
            "hidden_size": psi * psi,
            "num_hidden_layers": psi,
            "num_attention_heads": psi,
            "key_dim": 16,
            "value_dim": 64,
            "max_position_embeddings": 4096,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dense_similarity_one_layer_lower_bound_bytes": {
                8: 268_435_456,
                16: 536_870_912,
            }[psi],
        }
        data = {
            "source": "roneneldan/TinyStories",
            "train_split": "train[:20000]",
            "texts_loaded": 20000,
            "packed_chunks": 128,
            "seq_len": 4096,
            "max_train_tokens": 524416,
            "tokenizer_class": "GPT2TokenizerFast",
            "tokenizer_vocab_size": 50257,
            "data_contract": data_contract_ref,
        }
        training = {
            "optimizer_steps": 50,
            "gradient_accumulation_steps": 8,
            "microbatch_size": 1,
            "effective_batch_tokens": 32768,
            "initial_probe_loss": 10.0,
            "final_probe_loss": 8.0,
            "abs_loss_drop": 2.0,
            "rel_loss_drop": 0.2,
            "train_loss_first": 9.99,
            "train_loss_last": 9.5,
            "train_loss_min": 9.5,
            "grad_norm_max": 1.0,
            "elapsed_sec": 50.0,
            "allocated_bytes": 1,
            "reserved_bytes": 2,
            "peak_allocated_bytes": 3,
            "peak_reserved_bytes": 4,
        }
        checks = {
            "save_reload": {
                "checkpoint_dir": str(root / "checkpoint"),
                "loaded_logits_max_abs": 0.0,
                "reload_check_tokens": 16,
            },
            "cache": {
                "cache_split_logits_max_abs": 0.01,
                "cache_check_tokens": 24,
                "cache_split": 12,
            },
            "generation": {
                "prompt": "Once upon a time",
                "prompt_len": 4,
                "generated_len": 12,
                "sample_text": "Once upon a time fixture story",
            },
        }
        qualification = {
            "qualified": True,
            "conditions": {
                "gpt2_vocab_50257": True,
                "context_4096": True,
                "cuda_device": True,
                "bf16_amp": True,
                "optimizer_steps_at_least_50": True,
            },
        }
        return {
            "stage": "P0-4",
            "status": "passed",
            "timestamp_utc": f"2026-08-09T00:0{psi // 8}:00+00:00",
            "qualification": qualification,
            "settings": settings,
            "environment": environment,
            "model": model,
            "data": data,
            "training": training,
            "checks": checks,
        }

    def _make_p0_4(self, root: Path, *, psi: int) -> None:
        root.mkdir(parents=True)
        (root / "checkpoint").mkdir()
        data_contract_ref = self._make_p0_4_data_contract(root, psi=psi)
        (root / "P0-4_COMPLETE.md").write_text(
            "# P0-4\n\nPassed.\n", encoding="utf-8", newline="\n"
        )
        summary = self._p0_4_summary(root, psi)
        write_json(root / "summary.json", summary)
        events: list[dict[str, object]] = [
            {
                "event": "run_start",
                "timestamp_utc": summary["timestamp_utc"],
                "stage": "P0-4",
                "settings": summary["settings"],
                "environment": summary["environment"],
                "data_contract": data_contract_ref,
            },
            {
                "event": "preflight_complete",
                "timestamp_utc": summary["timestamp_utc"],
                "stage": "P0-4",
                "model": summary["model"],
                "data": summary["data"],
            },
        ]
        for step in range(1, 51):
            micro = [10.0 - step / 100.0] * 8
            events.append(
                {
                    "event": "train_step",
                    "timestamp_utc": summary["timestamp_utc"],
                    "stage": "P0-4",
                    "optimizer_step": step,
                    "optimizer_steps": 50,
                    "mean_loss": sum(micro) / 8,
                    "micro_losses": micro,
                    "grad_norm": 1.0,
                    "elapsed_sec": float(step),
                    "allocated_bytes": 1,
                    "reserved_bytes": 2,
                    "peak_allocated_bytes": 3,
                    "peak_reserved_bytes": 4,
                }
            )
        events.extend(
            [
                {
                    "event": "training_complete",
                    "timestamp_utc": summary["timestamp_utc"],
                    "stage": "P0-4",
                    **summary["training"],
                },
                {
                    "event": "save_reload_check",
                    "timestamp_utc": summary["timestamp_utc"],
                    "stage": "P0-4",
                    **summary["checks"]["save_reload"],
                },
                {
                    "event": "cache_split_check",
                    "timestamp_utc": summary["timestamp_utc"],
                    "stage": "P0-4",
                    **summary["checks"]["cache"],
                },
                {
                    "event": "generation_check",
                    "timestamp_utc": summary["timestamp_utc"],
                    "stage": "P0-4",
                    **summary["checks"]["generation"],
                },
                {
                    "event": "run_complete",
                    "timestamp_utc": summary["timestamp_utc"],
                    "stage": "P0-4",
                    "status": "passed",
                    "qualification": summary["qualification"],
                    "data_contract": data_contract_ref,
                },
            ]
        )
        write_jsonl(root / "metrics.jsonl", events)

    def _make_c3_lane(self, logical: str, psi: int, mode: str) -> None:
        root = self.c3_roots[logical]
        root.mkdir(parents=True)
        operational = mode == "operational"
        lrs = [0.0003, 0.0006, 0.0006] if operational else [0.0625]
        accum = 2 if operational else 1
        qualification = (
            "diagnostic_only_reduced_warmup_and_learning_rate"
            if operational
            else "diagnostic_only_bounded_exact_peak_exposure"
        )
        order = {
            "c3_psi8_operational": 1,
            "c3_psi8_peak_exposure": 2,
            "c3_psi16_operational": 3,
            "c3_psi16_peak_exposure": 4,
        }[logical]
        timestamp = f"2026-08-09T01:00:0{order}+00:00"
        summary = {
            "stage": "P0.5-C3",
            "status": "diagnostic_passed",
            "mode": mode,
            "qualification": qualification,
            "timestamp_utc": timestamp,
            "psi": psi,
            "parameter_count": {8: 4_134_146, 16: 27_546_626}[psi],
            "model": {
                "vocab_size": 50257,
                "sequence_length": 4096,
                "mipe_position_mode": "paper_absolute",
                "mipe_compute_dtype": "fp32",
                "softmask_compute_dtype": "fp32",
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "tie_word_embeddings": True,
            },
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.95],
                "weight_decay": 0.0,
                "eps": 1e-8,
                "eps_source": "repository_operationalization_paper_unspecified",
                "fused": False,
                "gradient_clipping": False,
            },
            "scheduler": {
                "name": "linear_warmup_then_constant",
                "paper_warmup_steps": 4096,
                "paper_peak_learning_rate": 0.0625,
                "executed_warmup_steps": 2 if operational else 1,
                "executed_peak_learning_rate": 0.0006 if operational else 0.0625,
                "observed_learning_rates": lrs,
                "diagnostic_reduced_from_paper": operational,
            },
            "training": {
                "optimizer_steps": len(lrs),
                "world_size": 1,
                "microbatch_size": 1,
                "sequences_per_optimizer_step": accum,
                "gradient_accumulation_steps": accum,
                "effective_tokens_per_optimizer_step": 4096 * accum,
                "paper_global_batch_tokens": 4_194_304,
                "local_to_paper_batch_ratio": (4096 * accum) / 4_194_304,
                "losses_finite": True,
                "gradients_finite": True,
                "parameters_finite": True,
                "optimizer_updates_nonzero": True,
                "post_update_loss": 9.0,
                "loss_decrease_required": False,
            },
            "data": copy.deepcopy(self.c3_contract),
            "environment": {
                "python": "3.12.11 (fixture)",
                "platform": "Linux-fixture",
                "torch": "2.7.1+cu128",
                "transformers": "4.57.6",
                "datasets": "5.0.1",
                "huggingface_hub": "0.34.3",
                "device": "cuda:0",
                "cuda_runtime": "12.8",
                "gpu_name": "Fixture GPU",
                "gpu_total_memory_bytes": 1024,
                "bf16_supported": True,
            },
            "memory": {
                "allocated_bytes": 1,
                "reserved_bytes": 2,
                "peak_allocated_bytes": 3,
                "peak_reserved_bytes": 4,
            },
            "limitations": [
                "This is a bounded dense-reference workstation diagnostic.",
                "It does not reproduce the paper global batch, duration, corpus, quality, or efficiency.",
                "The peak-exposure mode requires finite updates, not loss decrease or model quality.",
            ],
        }
        events: list[dict[str, object]] = []
        for step, lr in enumerate(lrs):
            micro = [10.0 - step / 10.0] * accum
            events.append(
                {
                    "event": "optimizer_step",
                    "stage": "P0.5-C3",
                    "mode": mode,
                    "psi": psi,
                    "optimizer_step_zero_based": step,
                    "learning_rate": lr,
                    "mean_micro_loss": sum(micro) / len(micro),
                    "micro_losses": micro,
                    "gradient_l2_norm": 1.0,
                    "tracked_gradient_abs": 0.5,
                    "tracked_parameter": "multiscreen.layers.0.q_proj.weight",
                    "tracked_parameter_delta": -lr,
                    "gradient_clipping_applied": False,
                    "allocated_bytes": 1,
                    "reserved_bytes": 2,
                    "peak_allocated_bytes": 3,
                    "peak_reserved_bytes": 4,
                }
            )
        marker_name = (
            "P0_5_C3_OPERATIONAL_COMPLETE.json"
            if operational
            else "P0_5_C3_PEAK_EXPOSURE_COMPLETE.json"
        )
        marker = {
            "stage": "P0.5-C3",
            "status": "diagnostic_passed",
            "mode": mode,
            "psi": psi,
            "timestamp_utc": timestamp,
        }
        write_json(root / "summary.json", summary)
        write_jsonl(root / "metrics.jsonl", events)
        write_json(root / marker_name, marker)

    def _make_c3_data(self) -> None:
        self.c3_data.mkdir(parents=True)
        digest = "d" * 64
        rows = [
            {"row_index": index, "sha256": digest, "utf8_bytes": index}
            for index in range(64)
        ]
        self.c3_row_manifest_sha256 = hashlib.sha256(
            REVIEW._canonical_bytes(rows)
        ).hexdigest()
        self.c3_contract = {
            "stage": "P0.5-C3",
            "status": "passed",
            "mode": "data",
            "qualification": "diagnostic_pinned_slimpajama_family_shard",
            "tokenizer": {
                "repository": "gpt2",
                "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                "asset_manifest_sha256": "07c45937a89b33f30016aef5b3982f13f25bf2c6ba940c535d1b5daa90459a71",
                "assets": {
                    "merges.txt": {
                        "filename": "merges.txt",
                        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                        "sha256": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
                        "size_bytes": 456318,
                    },
                    "tokenizer.json": {
                        "filename": "tokenizer.json",
                        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                        "sha256": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
                        "size_bytes": 1355256,
                    },
                    "tokenizer_config.json": {
                        "filename": "tokenizer_config.json",
                        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                        "sha256": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
                        "size_bytes": 26,
                    },
                    "vocab.json": {
                        "filename": "vocab.json",
                        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                        "sha256": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
                        "size_bytes": 1042301,
                    },
                },
                "vocab_size": 50257,
                "eos_token_id": 50256,
                "tokenizer_class": "GPT2TokenizerFast",
                "document_tokenization_truncation": False,
            },
            "source": {
                "family": "SlimPajama",
                "repository": "gmongaras/SlimPajama-627B_Reupload",
                "revision": "c34c22dbb10ae6b264a2f357a909d1a537141b36",
                "data_files": {"test": "data/test-00000-of-00030.parquet"},
                "split": "test",
                "file": {
                    "filename": "data/test-00000-of-00030.parquet",
                    "size_bytes": 43263929,
                    "sha256": "d9a83d59b72f4c303f0c0e46d0e73a8446eabb56b9aa5fd992347c358ab65743",
                    "revision": "c34c22dbb10ae6b264a2f357a909d1a537141b36",
                },
                "datasets_version": "5.0.1",
                "full_fingerprint": "507a47fcec5cbfdc",
                "selection": {
                    "kind": "contiguous_rows",
                    "start": 0,
                    "stop": 64,
                    "row_count": 64,
                    "fingerprint": "f1e6c1c09434a7e4",
                },
                "row_manifest_sha256": self.c3_row_manifest_sha256,
                "row_records": [
                    {"row_index": index, "sha256": digest, "utf8_bytes": index}
                    for index in range(64)
                ],
                "streaming": False,
                "source_scope": "third-party reupload test shard; not claimed byte-identical to the paper corpus or representative of its train split",
                "canonical_hub_loader": True,
                "local_parquet_fingerprint_not_accepted": True,
            },
            "packing": {
                "document_handling": "eos_concatenated_continuous_stream",
                "legacy_shifted_labels": True,
                "return_labels_are_shifted": True,
                "discard_incomplete_tail_only": True,
            },
            "accounting": {
                "selected_rows": 64,
                "nonempty_documents": 64,
                "text_tokens": 58645,
                "eos_tokens": 64,
                "concatenated_tokens": 58709,
                "packed_chunks": 14,
                "usable_tokens": 57358,
                "discarded_tail_tokens": 1351,
                "sequence_length": 4096,
                "stored_chunk_tokens": 4097,
                "prediction_tokens_per_chunk": 4096,
            },
            "hashes": {
                "row_manifest_sha256": self.c3_row_manifest_sha256,
                "token_stream_sha256": "3232bc3996272d563b6cc4e63a8d7a7d3769c7ec33e74d3d008d97cd290d7496",
                "token_encoding": "uint32_little_endian",
                "packed_chunk_sha256": [
                    "371d6dd52faa5b4278469eea935708ad99113621bb4ce3eed31cb8e519b00076",
                    "4b4e51618393b9deaa9cc0b6dc0b7b093fd9fef6567ee5f4e15fb2745a9ccccb",
                    "b1bbe605c93eb17d8b36f9f3cbbc73e2e4c5096cb447325fcea1e318420dbb7e",
                    "e13251a530b7eb5636ad426ffe6a5b42923121f223a2089f41936e5a4ac2c39a",
                    "ef05aa2f9860867e77617c88bbf693d222f7b61561bd5795e06c33a237ea4735",
                    "75f04cd07a7ae0e927ae86eec630110a069805e36acb6fc4c7cf49db28661656",
                    "e34beac42f4ca180661ab289a0cd79237b12c7e6de399396b821742347e5b175",
                    "ba9a311adeba69458424a36591fdaef5b2582db8074b61c79f942006b4404e18",
                    "4229f001a2d3967bc4a2f803cad2e79b0ae9ed48b1892896e7e77f49fd432873",
                    "c1cdaefbf45852700c0b58db5a212acd20ce616a68ba1f50ddb1d7e9c2f9b010",
                    "98bbdc89c73fbec7fb96a24bc73a8cd73af51bc8184f79846677aab007c49c3d",
                    "aec3c2e7ea78f3ed7c7c6f77299b21164c5dcb9d4cb9f5b1e19c05ed9f2534c2",
                    "0c749304c428d82d2d2333bde432cdd2617afb24f8aabc6716977cc3b0f22a72",
                    "ac023f84f579520ede3aa139fc459eb2d17cff6f3a9a0558b330df94495d5eda",
                ],
            },
            "checked_accounting_match": True,
            "limitations": [
                "The source is a pinned third-party SlimPajama reupload test shard.",
                "It is not claimed byte-identical to the paper corpus or representative of its train split.",
                "No raw source text is retained in this output.",
            ],
        }
        write_json(self.c3_data / "data_contract.json", self.c3_contract)
        write_json(
            self.c3_data / "P0_5_C3_DATA_CONTRACT_COMPLETE.json",
            {
                "stage": "P0.5-C3",
                "status": "passed",
                "mode": "data",
                "timestamp_utc": "2026-08-09T00:30:00+00:00",
            },
        )

    def _make_c3(self) -> None:
        for logical, psi, mode in REVIEW.C3_LANES:
            self._make_c3_lane(logical, psi, mode)

    def _make_tokenizers(self) -> None:
        digest = "a" * 64
        checkpoint_identifiers = {
            "p0_3_psi8": "p0-3-psi8-checkpoint",
            "p0_3_psi16": "p0-3-psi16-checkpoint",
            "p0_4_psi8": "p0-4-psi8-checkpoint",
            "p0_4_psi16": "p0-4-psi16-checkpoint",
        }
        for logical in REVIEW.TOKENIZER_NAMES:
            vocab = 768 if logical.startswith("p0_3") else 50257
            report = {
                "schema_version": REVIEW.TOKENIZER_SCHEMA_VERSION,
                "status": "passed",
                "logical_name": logical,
                "source_normalization": (
                    {
                        "pad_token_from_eos": False,
                        "padding_side": None,
                        "model_max_length": None,
                    }
                    if logical.startswith("p0_3")
                    else {
                        "pad_token_from_eos": True,
                        "padding_side": "right",
                        "model_max_length": 4096,
                    }
                ),
                "operationalization": {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "padding_side": "right",
                    "truncation_side": "right",
                    "model_max_length": 512 if logical.startswith("p0_3") else 4096,
                },
                "source": {
                    "identifier": (
                        "tinystories-spm768"
                        if logical.startswith("p0_3")
                        else "gpt2"
                    ),
                    "class": (
                        "FixtureTokenizer"
                        if logical.startswith("p0_3")
                        else "GPT2TokenizerFast"
                    ),
                    "is_fast": True,
                },
                "checkpoint": {
                    "identifier": checkpoint_identifiers[logical],
                    "class": (
                        "FixtureTokenizer"
                        if logical.startswith("p0_3")
                        else "GPT2TokenizerFast"
                    ),
                    "is_fast": True,
                    "reload_method": "AutoTokenizer.from_pretrained",
                    "reloaded_from_checkpoint": True,
                },
                "versions": {
                    "verifier": "1.0.0",
                    "python": "3.12.11",
                    "transformers": "4.57.6",
                    "tokenizers": "0.22.0",
                },
                "hashes": {
                    "vocabulary_manifest_sha256": digest,
                    "special_tokens_manifest_sha256": digest,
                    "probe_manifest_sha256": digest,
                },
                "counts": {
                    "vocabulary": vocab,
                    "vocab_size": vocab,
                    "tokenizer_length": vocab,
                    "added_vocabulary": 4,
                    "all_special_tokens": 4,
                    "probes": 5,
                    "special_token_boundary_probes": 28,
                },
                "checked_fields": list(REVIEW.TOKENIZER_CHECKED_FIELDS),
            }
            if logical == "p0_3_psi8":
                path = self.p0_3 / "tokenizer-reload-psi8.json"
            elif logical == "p0_3_psi16":
                path = self.p0_3 / "tokenizer-reload-psi16.json"
            elif logical == "p0_4_psi8":
                path = self.p0_4_psi8 / "tokenizer-reload.json"
            else:
                path = self.p0_4_psi16 / "tokenizer-reload.json"
            write_json(path, report)
            self.tokenizer_reports[logical] = path

    def _runtime(self, version: str = "3.12.11") -> dict[str, object]:
        return {
            "operating_system": {
                "libc_name": "glibc",
                "libc_version": "2.39",
                "machine": "x86_64",
                "release": "fixture",
                "system": "Linux",
            },
            "python": {
                "assertions_enabled": True,
                "cache_tag": "cpython-312",
                "compiler": "GCC fixture",
                "implementation": "CPython",
                "optimization_level": 0,
                "version": version,
            },
            "recorder": {
                "name": "run_level1_requalification_command.py",
                "version": "1.0.0",
            },
        }

    def _time(self, seconds: int) -> str:
        value = dt.datetime(2026, 8, 9, tzinfo=dt.timezone.utc) + dt.timedelta(
            seconds=seconds
        )
        return value.isoformat().replace("+00:00", "Z")

    def _environment_log(self, name: str) -> dict[str, object]:
        packages_4576 = {
            "PyYAML": "6.0.1",
            "accelerate": "1.6.0",
            "datasets": "5.0.1",
            "huggingface-hub": "0.34.3",
            "numpy": "1.26.4",
            "pyarrow": "25.0.0",
            "safetensors": "0.5.3",
            "sentencepiece": "0.2.0",
            "tokenizers": "0.22.0",
            "torch": "2.7.1+cu128",
            "transformers": "4.57.6",
            "trl": "1.9.2",
        }
        common = {
            "schema_version": REVIEW.ENVIRONMENT_SCHEMA_VERSION,
            "status": "passed",
            "tool_version": "1.0.0",
        }
        if name == "environment-cuda0":
            return {
                **common,
                "lane": "cuda0",
                "selection": {
                    "cuda_visible_devices": "0",
                    "logical_device": "cuda:0",
                },
                "packages": packages_4576,
                "python": {
                    "assertions_enabled": True,
                    "implementation": "CPython",
                    "optimization_level": 0,
                    "version": "3.12.11",
                },
                "runtime": {
                    "torch": "2.7.1+cu128",
                    "transformers": "4.57.6",
                },
                "nvidia_smi": {
                    "compute_capability": "12.0",
                    "device_name": "Fixture GPU",
                    "driver_version": "999.0",
                    "memory_free_mib": 900,
                    "memory_total_mib": 1024,
                    "other_compute_process_count": 0,
                    "other_compute_used_memory_mib": 0,
                    "physical_index": 0,
                    "reporter_compute_process_present": True,
                    "reporter_used_memory_mib": 1,
                },
                "cuda": {
                    "allocated_memory_bytes": 1,
                    "bf16_supported": True,
                    "capability": [12, 0],
                    "cudnn_version": 90701,
                    "device_count": 1,
                    "device_name": "Fixture GPU",
                    "free_memory_bytes": 900,
                    "reserved_memory_bytes": 1,
                    "runtime_version": "12.8",
                    "total_memory_bytes": 1024,
                },
            }
        if name == "environment-tf4576":
            return {
                **common,
                "lane": "tf4576",
                "packages": packages_4576,
                "python": {
                    "assertions_enabled": True,
                    "implementation": "CPython",
                    "optimization_level": 0,
                    "version": "3.12.11",
                },
                "runtime": {
                    "torch": "2.7.1+cu128",
                    "transformers": "4.57.6",
                },
            }
        return {
            **common,
            "lane": "tf5141",
            "packages": {
                "PyYAML": "6.0.2",
                "accelerate": None,
                "datasets": None,
                "huggingface-hub": "1.27.0",
                "numpy": "2.3.2",
                "pyarrow": None,
                "safetensors": "0.8.0",
                "sentencepiece": "0.2.0",
                "tokenizers": "0.22.2",
                "torch": "2.8.0",
                "transformers": "5.14.1",
                "trl": None,
            },
            "python": {
                "assertions_enabled": True,
                "implementation": "CPython",
                "optimization_level": 0,
                "version": "3.12.10",
            },
            "runtime": {
                "torch": "2.8.0+cu128",
                "transformers": "5.14.1",
            },
        }

    def _offline_cache_log(self) -> dict[str, object]:
        return {
            "cache": {
                "explicit": True,
                "path_recorded": False,
                "single_cache": True,
            },
            "checks": {
                "p0_3_tinystories": {
                    "fingerprint_nonempty": True,
                    "fingerprint_sha256": hashlib.sha256(
                        b"0123456789abcdef"
                    ).hexdigest(),
                    "repository": "roneneldan/TinyStories",
                    "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                    "row_count": 20000,
                    "split": "train[:20000]",
                    "text_column": "text",
                    "text_rows": 20000,
                },
                "p0_4_gpt2_tokenizer": {
                    "eos_token_id": 50256,
                    "identity_projection": self._p0_4_tokenizer_projection(),
                    "repository": "gpt2",
                    "revision": "default",
                    "use_fast": True,
                    "vocab_size": 50257,
                },
                "p0_4_tinystories": {
                    "fingerprint_nonempty": True,
                    "fingerprint_sha256": hashlib.sha256(
                        b"fedcba9876543210"
                    ).hexdigest(),
                    "repository": "roneneldan/TinyStories",
                    "revision": "default",
                    "row_count": 20000,
                    "split": "train[:20000]",
                    "text_column": "text",
                    "text_rows": 20000,
                },
                "p0_5_c3": {
                    "dataset": {
                        "data_files": {
                            "test": "data/test-00000-of-00030.parquet"
                        },
                        "file_sha256": "d9a83d59b72f4c303f0c0e46d0e73a8446eabb56b9aa5fd992347c358ab65743",
                        "file_size_bytes": 43263929,
                        "full_fingerprint": "507a47fcec5cbfdc",
                        "repository": "gmongaras/SlimPajama-627B_Reupload",
                        "revision": "c34c22dbb10ae6b264a2f357a909d1a537141b36",
                        "row_manifest_sha256": self.c3_row_manifest_sha256,
                        "selected_fingerprint": "f1e6c1c09434a7e4",
                        "selected_rows": 64,
                        "split": "test",
                        "text_column": "text",
                    },
                    "manifest_sha256": "480c127a8db02acb839d49e55d3a468cf452e816a04780d1b2a9fa8fe2c16060",
                    "tokenizer": {
                        "asset_manifest_sha256": "07c45937a89b33f30016aef5b3982f13f25bf2c6ba940c535d1b5daa90459a71",
                        "eos_token_id": 50256,
                        "repository": "gpt2",
                        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                        "vocab_size": 50257,
                    },
                },
            },
            "offline_environment": {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            "schema_version": "multiscreen-level1-offline-cache-v1",
            "status": "passed",
        }

    def _repository_log(self, name: str) -> dict[str, object]:
        check = {
            "json-validation": "json",
            "workflow-yaml": "workflow-yaml",
            "markdown-links": "markdown-links",
            "repository-hygiene": "hygiene",
            "repository-hygiene-final": "hygiene",
        }[name]
        result: dict[str, object] = {
            "artifact_count": 2,
            "artifact_manifest_sha256": "a" * 64,
            "bytes_checked": 100,
        }
        if check == "markdown-links":
            result.update({"link_count": 2, "local_link_count": 1})
        elif check == "workflow-yaml":
            result.update(
                {
                    "job_count": 1,
                    "parser": "stdlib_github_actions_restricted_v2",
                }
            )
        elif check == "hygiene":
            empty_sha = hashlib.sha256(b"").hexdigest()
            result.update(
                {
                    "gitlink_count": 0,
                    "head_commit": self.tested_commit,
                    "index_diff_check": "passed",
                    "maximum_tracked_file_bytes": 5 * 1024 * 1024,
                    "privacy": {
                        "artifact_count": 1,
                        "artifact_manifest_sha256": "b" * 64,
                        "bytes_checked": 80,
                        "fixture_exemption_artifact_count": 0,
                        "rules": [
                            "account-and-repository-root",
                            "credential-token",
                            "credential-url",
                            "file-uri",
                            "private-user-path",
                            "shareable-secret-assignment",
                        ],
                        "status": "passed",
                    },
                    "submodules": {
                        "byte_count": 0,
                        "record_count": 0,
                        "sha256": empty_sha,
                        "states": {
                            "clean": 0,
                            "conflict": 0,
                            "missing": 0,
                            "modified": 0,
                        },
                    },
                    "worktree": {
                        "clean": True,
                        "porcelain_byte_count": 0,
                        "porcelain_sha256": empty_sha,
                    },
                    "worktree_diff_check": "passed",
                }
            )
        return {
            "check": check,
            "format_version": REVIEW.REPOSITORY_CHECK_SCHEMA_VERSION,
            "head_commit": self.tested_commit,
            "result": result,
            "status": "passed",
        }

    def _command_contract(
        self, name: str
    ) -> tuple[dict[str, str], list[str]]:
        options: dict[str, str] = {}
        absent: list[Path] = []
        c3_command_paths = {
            "c3-data": (self.c3_data, "data", None),
            "c3-psi8-operational": (
                self.c3_roots["c3_psi8_operational"],
                "operational",
                8,
            ),
            "c3-psi8-peak-exposure": (
                self.c3_roots["c3_psi8_peak_exposure"],
                "peak-exposure",
                8,
            ),
            "c3-psi16-operational": (
                self.c3_roots["c3_psi16_operational"],
                "operational",
                16,
            ),
            "c3-psi16-peak-exposure": (
                self.c3_roots["c3_psi16_peak_exposure"],
                "peak-exposure",
                16,
            ),
        }
        if name == "offline-cache-preflight":
            options = {
                "--repo-root": str(self.repository),
                "--cache-dir": str(self.cache),
            }
        elif name == "syntax-level1":
            absent = [self.root / "pycache/syntax-level1"]
        elif name in c3_command_paths:
            path, mode, psi = c3_command_paths[name]
            options = {"--mode": mode, "--output-dir": str(path)}
            if psi is not None:
                options["--psi"] = str(psi)
            absent = [path]
        elif name == "p0-3-checkpointed":
            options = {
                "--output-dir": str(self.p0_3),
                "--log-every": "1",
                "--revision": REVIEW.P0_3_DATASET_REVISION,
            }
            absent = [self.p0_3]
        elif name in {"p0-3-tokenizer-psi8", "p0-3-tokenizer-psi16"}:
            psi = 8 if name.endswith("psi8") else 16
            report = self.tokenizer_reports[f"p0_3_psi{psi}"]
            options = {
                "--logical-name": f"p0_3_psi{psi}",
                "--checkpoint": str(self.p0_3 / f"psi{psi}"),
                "--output": str(report),
                "--source-id": "tinystories-spm768",
                "--checkpoint-id": f"p0-3-psi{psi}-checkpoint",
            }
            absent = [report]
        elif name in {"p0-4-psi8", "p0-4-psi16"}:
            root = self.p0_4_psi8 if name.endswith("psi8") else self.p0_4_psi16
            options = {"--output-dir": str(root)}
            absent = [root]
        elif name in {"p0-4-tokenizer-psi8", "p0-4-tokenizer-psi16"}:
            psi = 8 if name.endswith("psi8") else 16
            root = self.p0_4_psi8 if psi == 8 else self.p0_4_psi16
            report = self.tokenizer_reports[f"p0_4_psi{psi}"]
            options = {
                "--logical-name": f"p0_4_psi{psi}",
                "--checkpoint": str(root / "checkpoint"),
                "--output": str(report),
                "--source-id": "gpt2",
                "--checkpoint-id": f"p0-4-psi{psi}-checkpoint",
            }
            absent = [report]
        elif name == "p0-4-review-psi8":
            options = {
                "--mode": "p0-4-lane",
                "--tested-commit": self.tested_commit,
                "--p0-4-root": str(self.p0_4_psi8),
                "--tokenizer-reload-report": (
                    f"p0_4_psi8={self.tokenizer_reports['p0_4_psi8']}"
                ),
                "--command-ledger": str(self.ledger),
                "--output": str(self.p0_4_psi8 / "raw-review.json"),
            }
            absent = [self.p0_4_psi8 / "raw-review.json"]
        return options, sorted(
            path.relative_to(self.root).as_posix() for path in absent
        )

    def _make_ledger(self) -> None:
        marker = {
            "created_at_utc": self._time(0),
            "format_version": "level1-requalification-run-v1",
            "repository": {
                "head_commit": self.tested_commit,
                "worktree_path_sha256": hashlib.sha256(
                    os.fsencode(self.repository)
                ).hexdigest(),
            },
            "tool_version": "1.0.0",
        }
        (self.root / ".level1-requalification-run.json").write_bytes(
            REVIEW._runner_canonical_bytes(marker)
        )

        environment_records: list[dict[str, object]] = []
        for index, name in enumerate(REVIEW.REQUIRED_ENVIRONMENT_NAMES):
            started = 1 + index * 2
            record = {
                "cwd": {"base": "repository_root", "path": "."},
                "duration_ns": 1_000_000_000,
                "duration_seconds": 1.0,
                "ended_at_utc": self._time(started + 1),
                "format_version": "level1-requalification-command-record-v1",
                "name": name,
                "record_type": "environment",
                "repository": {"head_commit": self.tested_commit},
                "runtime": self._runtime("3.12.10" if name == "runtime-tf5141" else "3.12.11"),
                "started_at_utc": self._time(started),
            }
            raw = REVIEW._runner_canonical_bytes(record)
            (self.root / "records" / f"{name}.json").write_bytes(raw)
            environment_records.append(record)
        (self.root / "environment.jsonl").write_bytes(
            b"".join(REVIEW._runner_canonical_bytes(item) for item in environment_records)
        )

        special = {
            "c3-data",
            "c3-psi8-operational",
            "c3-psi8-peak-exposure",
            "c3-psi16-operational",
            "c3-psi16-peak-exposure",
            "p0-3-checkpointed",
            "p0-3-tokenizer-psi8",
            "p0-3-tokenizer-psi16",
            "p0-4-psi8-preflight",
            "p0-4-psi8",
            "p0-4-tokenizer-psi8",
            "p0-4-review-psi8",
            "p0-4-psi16-preflight",
            "p0-4-psi16",
            "p0-4-tokenizer-psi16",
        }
        prefix = [
            "environment-tf4576",
            "environment-tf5141",
            "environment-cuda0",
            "offline-cache-preflight",
            "repository-hygiene",
        ]
        static = [
            name
            for name in REVIEW.REQUIRED_COMMAND_NAMES
            if name not in set(prefix) | special | {"repository-hygiene-final"}
        ]
        order = [
            *prefix,
            *static,
            "c3-data",
            "c3-psi8-operational",
            "c3-psi8-peak-exposure",
            "c3-psi16-operational",
            "c3-psi16-peak-exposure",
            "p0-3-checkpointed",
            "p0-3-tokenizer-psi8",
            "p0-3-tokenizer-psi16",
            "p0-4-psi8-preflight",
            "p0-4-psi8",
            "p0-4-tokenizer-psi8",
            "p0-4-review-psi8",
            "p0-4-psi16-preflight",
            "p0-4-psi16",
            "p0-4-tokenizer-psi16",
            "repository-hygiene-final",
        ]
        assert len(order) == len(set(order)) == len(REVIEW.REQUIRED_COMMAND_NAMES)
        assert set(order) == set(REVIEW.REQUIRED_COMMAND_NAMES)

        command_records: list[dict[str, object]] = []
        semantic_names = {
            "environment-tf4576",
            "environment-tf5141",
            "environment-cuda0",
        }
        repository_names = {
            "json-validation",
            "workflow-yaml",
            "markdown-links",
            "repository-hygiene",
            "repository-hygiene-final",
        }
        tf5141_names = {
            "environment-tf5141",
            "tokenizer-reload-tests-tf5141",
            "gradient-checkpointing-tf5141",
            "c3-contracts-tf5141",
        }
        expected_tails = REVIEW._expected_command_tails(
            required_names=REVIEW.REQUIRED_COMMAND_NAMES,
            repository=REPO_ROOT,
            run_root=self.root,
            cache=self.cache,
            tf4576_python=self.tf4576_python,
            tf5141_python=self.tf5141_python,
        )
        expected_tails["p0-4-review-psi8"] = tuple(
            self.tested_commit if value == "__TESTED_COMMIT__" else value
            for value in expected_tails["p0-4-review-psi8"]
        )
        for index, name in enumerate(order):
            started = 10 + index * 3
            _, absent = self._command_contract(name)
            argv = [
                "/usr/bin/env",
                "-i",
                *REVIEW._expected_hermetic_environment(name=name, run_root=self.root),
                *expected_tails[name],
            ]
            log_path = self.root / "logs" / f"{name}.log"
            if name == "p0-3-checkpointed":
                log_raw = log_path.read_bytes()
            elif name == "offline-cache-preflight":
                log_raw = REVIEW._runner_canonical_bytes(self._offline_cache_log())
                log_path.write_bytes(log_raw)
            elif name in {"p0-4-psi8", "p0-4-psi16"}:
                psi = 8 if name.endswith("psi8") else 16
                log_raw = (
                    f"[P0-4] data_contract sha256={self.p0_4_contract_sha256[psi]}\n"
                    "fixture passed\n"
                ).encode("utf-8")
                log_path.write_bytes(log_raw)
            elif name in semantic_names:
                log_raw = REVIEW._runner_canonical_bytes(
                    self._environment_log(name)
                )
                log_path.write_bytes(log_raw)
            elif name in repository_names:
                log_raw = REVIEW._runner_canonical_bytes(
                    self._repository_log(name)
                )
                log_path.write_bytes(log_raw)
            else:
                log_raw = b"fixture passed\n"
                log_path.write_bytes(log_raw)
            record = {
                "argv": argv,
                "cwd": {"base": "repository_root", "path": "."},
                "duration_ns": 1_000_000_000,
                "duration_seconds": 1.0,
                "ended_at_utc": self._time(started + 1),
                "exit_code": 0,
                "format_version": "level1-requalification-command-record-v1",
                "log": {
                    "path": f"logs/{name}.log",
                    "sha256": hashlib.sha256(log_raw).hexdigest(),
                    "size_bytes": len(log_raw),
                },
                "name": name,
                "preconditions": {"absent_paths": absent},
                "record_type": "command",
                "returncode": 0,
                "runtime": self._runtime("3.12.10" if name in tf5141_names else "3.12.11"),
                "started_at_utc": self._time(started),
                "termination_signal": None,
            }
            raw = REVIEW._runner_canonical_bytes(record)
            (self.root / "records" / f"{name}.json").write_bytes(raw)
            command_records.append(record)
        self.ledger.write_bytes(
            b"".join(REVIEW._runner_canonical_bytes(item) for item in command_records)
        )

    def command_records(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in self.ledger.read_text(encoding="utf-8").splitlines()
        ]

    def rewrite_environment(self, name: str, mutate: object) -> None:
        ledger = self.root / "environment.jsonl"
        records = [
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
        ]
        for record in records:
            if record["name"] == name:
                mutate(record)
                raw = REVIEW._runner_canonical_bytes(record)
                (self.root / "records" / f"{name}.json").write_bytes(raw)
                break
        else:
            raise AssertionError(f"missing fixture environment record {name}")
        ledger.write_bytes(
            b"".join(REVIEW._runner_canonical_bytes(item) for item in records)
        )

    def rewrite_command(
        self,
        name: str,
        mutate: object,
    ) -> None:
        records = self.command_records()
        for record in records:
            if record["name"] == name:
                mutate(record)
                raw = REVIEW._runner_canonical_bytes(record)
                (self.root / "records" / f"{name}.json").write_bytes(raw)
                break
        else:
            raise AssertionError(f"missing fixture command {name}")
        self.ledger.write_bytes(
            b"".join(REVIEW._runner_canonical_bytes(item) for item in records)
        )

    def replace_log(self, name: str, raw: bytes) -> None:
        path = self.root / "logs" / f"{name}.log"
        path.write_bytes(raw)

        def mutate(record: dict[str, object]) -> None:
            record["log"] = {
                "path": f"logs/{name}.log",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }

        self.rewrite_command(name, mutate)

    def rewrite_p0_3_data_contract(self, mutate: object) -> None:
        contract_path = self.p0_3 / "data_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        mutate(contract)
        contract_raw = REVIEW._canonical_bytes(contract)
        contract_path.write_bytes(contract_raw)
        contract_sha256 = hashlib.sha256(contract_raw).hexdigest()

        results_path = self.p0_3 / "p0_3_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        for result in results:
            result["data_contract_sha256"] = contract_sha256
            write_json(
                self.p0_3 / f"psi{result['psi']}" / "p0_3_metrics.json",
                result,
            )
        write_json(results_path, results)

        lines = self.p0_3_stdout.read_text(encoding="utf-8").splitlines()
        digest_lines = [
            index
            for index, line in enumerate(lines)
            if line.startswith("[P0-3] data_contract")
        ]
        if len(digest_lines) != 1:
            raise AssertionError("fixture must have one P0-3 data-contract digest")
        lines[digest_lines[0]] = f"[P0-3] data_contract sha256={contract_sha256}"
        self.replace_log(
            "p0-3-checkpointed", ("\n".join(lines) + "\n").encode("utf-8")
        )

    def _make_focused_review(self) -> None:
        self.p0_4_psi8_review = self.p0_4_psi8 / "raw-review.json"
        with mock.patch.object(
            REVIEW, "C3_ROW_MANIFEST_SHA256", self.c3_row_manifest_sha256
        ):
            report = REVIEW.review_p0_4_lane_inputs(
                p0_4_root=self.p0_4_psi8,
                tokenizer_reports={
                    "p0_4_psi8": self.tokenizer_reports["p0_4_psi8"]
                },
                command_ledger=self.ledger,
                tested_commit=self.tested_commit,
            )
        self.p0_4_psi8_review.write_bytes(REVIEW._pretty_canonical_bytes(report))

    def kwargs(self) -> dict[str, object]:
        return {
            "p0_3_root": self.p0_3,
            "p0_3_stdout": self.p0_3_stdout,
            "p0_4_psi8_root": self.p0_4_psi8,
            "p0_4_psi16_root": self.p0_4_psi16,
            "p0_4_psi8_review": self.p0_4_psi8_review,
            "c3_data_root": self.c3_data,
            "c3_psi8_operational_root": self.c3_roots["c3_psi8_operational"],
            "c3_psi8_peak_exposure_root": self.c3_roots["c3_psi8_peak_exposure"],
            "c3_psi16_operational_root": self.c3_roots["c3_psi16_operational"],
            "c3_psi16_peak_exposure_root": self.c3_roots["c3_psi16_peak_exposure"],
            "tokenizer_reports": self.tokenizer_reports,
            "command_ledger": self.ledger,
            "tested_commit": self.tested_commit,
        }


class Level1EvidenceReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = EvidenceFixture(Path(self.temporary.name))

    def review(self) -> dict[str, object]:
        with mock.patch.object(
            REVIEW, "C3_ROW_MANIFEST_SHA256", self.fixture.c3_row_manifest_sha256
        ):
            return REVIEW.review_inputs(**self.fixture.kwargs())

    def assert_rejected(self, pattern: str = "") -> None:
        with self.assertRaisesRegex(REVIEW.ReviewError, pattern or ".+"):
            self.review()

    def test_complete_fixture_passes_deterministically(self) -> None:
        first = self.review()
        second = self.review()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "passed")
        self.assertEqual(first["aggregate"]["raw_event_counts"]["total"], 187)
        self.assertEqual(first["aggregate"]["artifact_count"], 130)
        self.assertRegex(first["aggregate"]["review_material_sha256"], r"^[0-9a-f]{64}$")

    def test_p0_3_rejects_missing_duplicate_extra_or_out_of_order_steps(self) -> None:
        lines = self.fixture.p0_3_stdout.read_text(encoding="utf-8").splitlines()
        step_line = next(index for index, line in enumerate(lines) if "step=0002/40" in line)
        lines.insert(step_line, lines[step_line])
        self.fixture.replace_log(
            "p0-3-checkpointed", ("\n".join(lines) + "\n").encode("utf-8")
        )
        self.assert_rejected("missing, duplicate, extra, or out-of-order")

    def test_p0_3_stdout_data_contract_digest_is_exact_and_bound(self) -> None:
        cases = ("duplicate", "malformed", "stale")
        for label in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"p0-3-stdout-digest-{label}"
                )
                lines = self.fixture.p0_3_stdout.read_text(
                    encoding="utf-8"
                ).splitlines()
                digest_index = next(
                    index
                    for index, line in enumerate(lines)
                    if line.startswith("[P0-3] data_contract")
                )
                if label == "duplicate":
                    lines.insert(digest_index + 1, lines[digest_index])
                    expected = "exactly one data-contract digest"
                elif label == "malformed":
                    lines[digest_index] = lines[digest_index].replace(
                        "sha256=", "sha256:"
                    )
                    expected = "malformed P0-3 data-contract digest"
                else:
                    lines[digest_index] = (
                        "[P0-3] data_contract sha256=" + "0" * 64
                    )
                    expected = "does not match the contract file"
                self.fixture.replace_log(
                    "p0-3-checkpointed",
                    ("\n".join(lines) + "\n").encode("utf-8"),
                )
                self.assert_rejected(expected)

    def test_p0_3_contract_source_overrides_and_fingerprint_fail_closed(self) -> None:
        cases = (
            ("data-files", "data_files", {"train": "private.parquet"}),
            ("data-dir", "data_dir", "private"),
            ("text-file", "text_file", "provided_path_not_recorded"),
            (
                "fingerprint",
                "dataset_fingerprint",
                "/" + "home/private/fingerprint",
            ),
        )
        for label, field, value in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"p0-3-source-{label}"
                )

                def mutate(contract: dict[str, object]) -> None:
                    contract["source"][field] = value

                self.fixture.rewrite_p0_3_data_contract(mutate)
                self.assert_rejected("data contract source|dataset fingerprint")

    def test_p0_3_contract_rejects_wrong_committed_eos_token_id(self) -> None:
        def mutate(contract: dict[str, object]) -> None:
            contract["packing"]["eos_token_id"] = 3

        self.fixture.rewrite_p0_3_data_contract(mutate)
        self.assert_rejected("eos_token_id")

    def test_p0_3_tokenizer_projection_cross_bind_rejects_mutations(self) -> None:
        def mutate_class(contract: dict[str, object]) -> None:
            contract["tokenizer"]["class"] = "OtherTokenizer"

        def mutate_hash(contract: dict[str, object]) -> None:
            contract["tokenizer"]["hashes"]["probe_manifest_sha256"] = "b" * 64

        def mutate_operationalization(contract: dict[str, object]) -> None:
            contract["tokenizer"]["operationalization"][
                "model_max_length"
            ] = 1024

        for label, mutation in (
            ("class", mutate_class),
            ("hash", mutate_hash),
            ("operationalization", mutate_operationalization),
        ):
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"p0-3-tokenizer-{label}"
                )
                self.fixture.rewrite_p0_3_data_contract(mutation)
                self.assert_rejected("source tokenizer projection differs")

    def test_p0_3_raw_tokenizer_report_binds_back_to_contract(self) -> None:
        path = self.fixture.tokenizer_reports["p0_3_psi8"]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["hashes"]["probe_manifest_sha256"] = "b" * 64
        write_json(path, report)
        self.assert_rejected("source tokenizer projection differs")

    def test_p0_3_dataset_fingerprint_binds_offline_cache_preflight(self) -> None:
        path = self.fixture.root / "logs/offline-cache-preflight.log"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["checks"]["p0_3_tinystories"]["fingerprint_sha256"] = "0" * 64
        self.fixture.replace_log(
            "offline-cache-preflight", REVIEW._runner_canonical_bytes(report)
        )
        self.assert_rejected("offline-cache P0-3 fingerprint does not bind")

    def test_p0_3_rejects_nonfinite_and_ambiguous_values(self) -> None:
        results_path = self.fixture.p0_3 / "p0_3_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        results[0]["grad_norm_max"] = float("nan")
        write_json(results_path, results)
        self.assert_rejected("non-finite")

    def test_p0_3_rejects_checkpoint_path_outside_root(self) -> None:
        results_path = self.fixture.p0_3 / "p0_3_results.json"
        per_path = self.fixture.p0_3 / "psi8" / "p0_3_metrics.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        results[0]["checkpoint_dir"] = str(self.fixture.root)
        write_json(results_path, results)
        write_json(per_path, results[0])
        self.assert_rejected("expected artifact directory")

    def test_p0_3_data_contract_requires_canonical_json(self) -> None:
        path = self.fixture.p0_3 / "data_contract.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        write_json(path, contract)
        self.assert_rejected("must use canonical JSON bytes")

    def test_p0_3_data_contract_rejects_wrong_pinned_revision(self) -> None:
        path = self.fixture.p0_3 / "data_contract.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        contract["source"]["revision"] = "0" * 40
        path.write_bytes(REVIEW._canonical_bytes(contract))
        self.assert_rejected("source.revision")

    def test_p0_3_data_contract_rejects_invalid_selected_text_hash(self) -> None:
        path = self.fixture.p0_3 / "data_contract.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        contract["source"]["selected_text_manifest_sha256"] = "x" * 64
        path.write_bytes(REVIEW._canonical_bytes(contract))
        self.assert_rejected("selected-text manifest")

    def test_p0_3_data_contract_rejects_invalid_packed_token_hash(self) -> None:
        path = self.fixture.p0_3 / "data_contract.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        contract["packing"]["packed_token_stream_sha256"] = "A" * 64
        path.write_bytes(REVIEW._canonical_bytes(contract))
        self.assert_rejected("packed-token stream")

    def test_p0_3_metrics_must_bind_data_contract_file_digest(self) -> None:
        results_path = self.fixture.p0_3 / "p0_3_results.json"
        per_path = self.fixture.p0_3 / "psi8/p0_3_metrics.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        per_metric = json.loads(per_path.read_text(encoding="utf-8"))
        results[0]["data_contract_sha256"] = "f" * 64
        per_metric["data_contract_sha256"] = "f" * 64
        write_json(results_path, results)
        write_json(per_path, per_metric)
        self.assert_rejected("bind the canonical P0-3 data contract")


    def test_p0_3_metric_contract_references_must_match_each_other(self) -> None:
        results_path = self.fixture.p0_3 / "p0_3_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        results[0]["data_contract_sha256"] = "f" * 64
        write_json(results_path, results)
        self.assert_rejected("aggregate and per-Psi data-contract references differ")
    def test_p0_4_rejects_event_sequence_change(self) -> None:
        metrics = self.fixture.p0_4_psi8 / "metrics.jsonl"
        events = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
        events[2], events[3] = events[3], events[2]
        write_jsonl(metrics, events)
        self.assert_rejected("missing, duplicate, extra, or out-of-order|optimizer_step")

    def test_p0_4_rejects_wrong_strict_runtime_or_qualification(self) -> None:
        summary_path = self.fixture.p0_4_psi8 / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["settings"]["grad_accum"] = "8"
        write_json(summary_path, summary)
        self.assert_rejected("unambiguous JSON integer")

    def test_p0_4_rejects_duplicate_json_key(self) -> None:
        summary_path = self.fixture.p0_4_psi8 / "summary.json"
        summary_path.write_text(
            '{"stage":"P0-4","stage":"P0-4"}\n', encoding="utf-8"
        )
        self.assert_rejected("duplicate JSON object key")

    def test_p0_4_exact_schema_fixed_values_and_memory_are_bound(self) -> None:
        cases = (
            (
                "extra-summary-field",
                lambda summary: summary.__setitem__("unreviewed", True),
                "summary fields",
            ),
            (
                "key-dim",
                lambda summary: summary["model"].__setitem__("key_dim", 15),
                "key_dim",
            ),
            (
                "amp-alias",
                lambda summary: summary["settings"].__setitem__(
                    "amp_dtype", "bfloat16"
                ),
                "amp_dtype",
            ),
            (
                "device-alias",
                lambda summary: summary["settings"].__setitem__("device", "cuda"),
                "device",
            ),
            (
                "generation-bound",
                lambda summary: summary["checks"]["generation"].__setitem__(
                    "generated_len", 13
                ),
                "generation",
            ),
            (
                "memory-order",
                lambda summary: summary["training"].__setitem__(
                    "allocated_bytes", 5
                ),
                "allocated bytes exceed reserved",
            ),
            (
                "training-peak-below-step",
                lambda summary: summary["training"].__setitem__(
                    "peak_allocated_bytes", 2
                ),
                "summary peak memory",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"p0-4-summary-{label}"
                )
                summary_path = self.fixture.p0_4_psi8 / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                mutate(summary)
                write_json(summary_path, summary)
                self.assert_rejected(expected)

    def test_p0_4_step_schema_and_peak_memory_are_bound(self) -> None:
        cases = (
            (
                "extra-field",
                lambda events: events[2].__setitem__("unreviewed", True),
                "train_step fields",
            ),
            (
                "peak-regression",
                lambda events: events[3].__setitem__(
                    "peak_allocated_bytes", 2
                ),
                "not monotonic",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"p0-4-event-{label}"
                )
                metrics = self.fixture.p0_4_psi8 / "metrics.jsonl"
                events = [
                    json.loads(line)
                    for line in metrics.read_text(encoding="utf-8").splitlines()
                ]
                mutate(events)
                write_jsonl(metrics, events)
                self.assert_rejected(expected)

    def test_p0_4_contract_stdout_offline_and_tokenizer_are_cross_bound(self) -> None:
        cases = ("contract", "stdout", "offline", "tokenizer")
        for label in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"p0-4-cross-{label}"
                )
                if label == "contract":
                    path = self.fixture.p0_4_psi8 / "data_contract.json"
                    contract = json.loads(path.read_text(encoding="utf-8"))
                    contract["source"]["selected_text_utf8_bytes"] += 1
                    path.write_bytes(REVIEW._canonical_bytes(contract))
                    expected = "data_contract|data-contract"
                elif label == "stdout":
                    raw = (self.fixture.root / "logs/p0-4-psi8.log").read_bytes()
                    self.fixture.replace_log("p0-4-psi8", raw.splitlines(keepends=True)[0] + raw)
                    expected = "exactly one .*data-contract digest"
                elif label == "offline":
                    report = self.fixture._offline_cache_log()
                    report["checks"]["p0_4_tinystories"]["fingerprint_sha256"] = "0" * 64
                    self.fixture.replace_log(
                        "offline-cache-preflight",
                        REVIEW._runner_canonical_bytes(report),
                    )
                    expected = "offline dataset fingerprint"
                else:
                    path = self.fixture.tokenizer_reports["p0_4_psi8"]
                    report = json.loads(path.read_text(encoding="utf-8"))
                    report["hashes"]["probe_manifest_sha256"] = "0" * 64
                    write_json(path, report)
                    expected = "source tokenizer projection"
                self.assert_rejected(expected)

    def test_c3_authentic_data_manifests_are_fixed_and_recomputed(self) -> None:
        self.assertEqual(
            REVIEW.C3_ROW_MANIFEST_SHA256,
            "942f9b3397ff7073342973082efa4cddf3ace16bc7e3d180c827df3203243831",
        )
        cases = ("row", "chunk", "asset")
        for label in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"c3-data-{label}"
                )
                path = self.fixture.c3_data / "data_contract.json"
                contract = json.loads(path.read_text(encoding="utf-8"))
                if label == "row":
                    contract["source"]["row_records"][0]["utf8_bytes"] += 1
                    expected = "row records"
                elif label == "chunk":
                    contract["hashes"]["packed_chunk_sha256"][0] = "0" * 64
                    expected = "packed-chunk digests"
                else:
                    contract["tokenizer"]["assets"]["merges.txt"]["size_bytes"] += 1
                    expected = "size_bytes"
                write_json(path, contract)
                self.assert_rejected(expected)

    def test_c3_lane_exact_schema_schedule_batch_and_memory_are_bound(self) -> None:
        cases = (
            (
                "extra-summary",
                lambda summary: summary.__setitem__("unreviewed", True),
                "summary fields",
            ),
            (
                "warmup",
                lambda summary: summary["scheduler"].__setitem__(
                    "executed_warmup_steps", 3
                ),
                "executed_warmup_steps",
            ),
            (
                "batch-ratio",
                lambda summary: summary["training"].__setitem__(
                    "local_to_paper_batch_ratio", 1.0
                ),
                "local_to_paper_batch_ratio",
            ),
            (
                "environment-extra",
                lambda summary: summary["environment"].__setitem__(
                    "unreviewed", True
                ),
                "environment fields",
            ),
            (
                "memory-order",
                lambda summary: summary["memory"].__setitem__(
                    "allocated_bytes", 5
                ),
                "allocated bytes exceed reserved",
            ),
            (
                "limitations",
                lambda summary: summary["limitations"].append("unreviewed"),
                "limitations",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"c3-lane-{label}"
                )
                root = self.fixture.c3_roots["c3_psi8_operational"]
                path = root / "summary.json"
                summary = json.loads(path.read_text(encoding="utf-8"))
                mutate(summary)
                write_json(path, summary)
                self.assert_rejected(expected)

    def test_c3_event_schema_tracked_parameter_and_peaks_are_bound(self) -> None:
        cases = (
            (
                "extra",
                lambda events: events[0].__setitem__("unreviewed", True),
                "optimizer_step fields",
            ),
            (
                "tracked-parameter",
                lambda events: events[0].__setitem__("tracked_parameter", ""),
                "tracked_parameter",
            ),
            (
                "peak-regression",
                lambda events: events[1].__setitem__(
                    "peak_allocated_bytes", 2
                ),
                "not monotonic",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"c3-event-{label}"
                )
                root = self.fixture.c3_roots["c3_psi8_operational"]
                path = root / "metrics.jsonl"
                events = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                mutate(events)
                write_jsonl(path, events)
                self.assert_rejected(expected)

    def test_c3_offline_row_manifest_is_explicitly_cross_bound(self) -> None:
        report = self.review()
        c3 = report["p0_5_c3"]
        self.assertEqual(len(c3["runs"]), 4)
        self.assertTrue(
            c3["cross_bindings"]["offline_cache_row_manifest_match"]
        )
        reviewed = c3["data"]["row_manifest_sha256"]
        self.assertEqual(c3["cross_bindings"]["row_manifest_sha256"], reviewed)
        bad_ledger = {
            "semantic_logs": {
                "offline_cache": {
                    "checks": {
                        "p0_5_c3": {"row_manifest_sha256": "0" * 64}
                    }
                }
            }
        }
        with self.assertRaisesRegex(REVIEW.ReviewError, "row manifest differs"):
            REVIEW._review_c3_cross_bindings(
                c3_data=c3["data"], ledger=bad_ledger
            )

        offline = self.fixture._offline_cache_log()
        offline["checks"]["p0_5_c3"]["dataset"]["row_manifest_sha256"] = "0" * 64
        self.fixture.replace_log(
            "offline-cache-preflight", REVIEW._runner_canonical_bytes(offline)
        )
        self.assert_rejected("C3 dataset identity")

    def test_c3_rejects_clipping_parameter_stasis_and_wrong_lr(self) -> None:
        root = self.fixture.c3_roots["c3_psi8_operational"]
        metrics = root / "metrics.jsonl"
        events = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
        events[0]["gradient_clipping_applied"] = True
        events[0]["tracked_parameter_delta"] = 0.0
        events[0]["learning_rate"] = 0.1
        write_jsonl(metrics, events)
        self.assert_rejected(r"event\[0\].lr|tracked parameter|gradient_clipping")

    def test_c3_rejects_wrong_or_additional_marker(self) -> None:
        root = self.fixture.c3_roots["c3_psi16_peak_exposure"]
        write_json(root / "P0_5_C3_OPERATIONAL_COMPLETE.json", {})
        self.assert_rejected("exactly the correct completion marker")

    def test_tokenizer_report_requires_exact_hash_count_and_probe_contract(self) -> None:
        path = self.fixture.tokenizer_reports["p0_4_psi8"]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["counts"]["probes"] = 4
        report["hashes"]["probe_manifest_sha256"] = "ABC"
        report["checked_fields"].pop()
        write_json(path, report)
        self.assert_rejected("probes|SHA-256|checked_fields")

    def test_tokenizer_checkpoint_identifiers_bind_all_fixed_lanes(self) -> None:
        expected = {
            "p0_3_psi8": "p0-3-psi8-checkpoint",
            "p0_3_psi16": "p0-3-psi16-checkpoint",
            "p0_4_psi8": "p0-4-psi8-checkpoint",
            "p0_4_psi16": "p0-4-psi16-checkpoint",
        }
        observed = {
            logical: json.loads(path.read_text(encoding="utf-8"))["checkpoint"][
                "identifier"
            ]
            for logical, path in self.fixture.tokenizer_reports.items()
        }
        self.assertEqual(observed, expected)

        for logical in ("p0_3_psi16", "p0_4_psi16"):
            with self.subTest(logical=logical):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"wrong-checkpoint-{logical}"
                )
                path = self.fixture.tokenizer_reports[logical]
                report = json.loads(path.read_text(encoding="utf-8"))
                report["checkpoint"]["identifier"] = expected[logical].replace(
                    "psi16", "psi6"
                )
                write_json(path, report)
                self.assert_rejected(rf"{logical}\.checkpoint\.identifier")

    def test_tokenizer_report_rejects_absolute_private_identifier(self) -> None:
        path = self.fixture.tokenizer_reports["p0_3_psi8"]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["checkpoint"]["identifier"] = "/private/checkpoint"
        write_json(path, report)
        self.assert_rejected("must not expose an absolute path")

    def test_tokenizer_report_rejects_wrong_compatibility_version(self) -> None:
        path = self.fixture.tokenizer_reports["p0_4_psi16"]
        report = json.loads(path.read_text(encoding="utf-8"))
        report["versions"]["transformers"] = "5.14.1"
        write_json(path, report)
        self.assert_rejected("recorded tf4576 lane")

    def test_command_ledger_rejects_failed_or_missing_required_command(self) -> None:
        records = [
            json.loads(line)
            for line in self.fixture.ledger.read_text(encoding="utf-8").splitlines()
        ]
        records[0]["exit_code"] = False
        self.fixture.ledger.write_bytes(
            b"".join(REVIEW._runner_canonical_bytes(item) for item in records)
        )
        self.assert_rejected("unambiguous JSON integer")

    def test_symlink_in_explicit_root_is_rejected(self) -> None:
        target = self.fixture.root / "outside.txt"
        target.write_text("outside\n", encoding="utf-8")
        os.symlink(target, self.fixture.p0_4_psi16 / "linked.txt")
        self.assert_rejected("contains a symlink")

    def test_relative_input_path_is_rejected(self) -> None:
        kwargs = self.fixture.kwargs()
        kwargs["p0_3_root"] = Path("relative")
        with self.assertRaisesRegex(REVIEW.ReviewError, "explicit absolute path"):
            REVIEW.review_inputs(**kwargs)

    def test_noncanonical_absolute_aliases_and_tilde_are_rejected(self) -> None:
        canonical = str(self.fixture.p0_3)
        cases = (
            ("tilde", "~/p0-3"),
            ("parent", canonical + "/../p0_3"),
            ("duplicate-separator", canonical.replace("/p0-3", "//p0-3")),
            ("trailing-separator", canonical + "/"),
        )
        for label, value in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    REVIEW.ReviewError, "explicit absolute|lexical canonical"
                ):
                    REVIEW._absolute_path(value, label=label)

    def test_command_ledger_order_matches_the_fixed_testing_matrix(self) -> None:
        records = self.fixture.command_records()
        first = next(index for index, item in enumerate(records) if item["name"] == "formula-units")
        second = next(index for index, item in enumerate(records) if item["name"] == "oracle-selfcheck")
        records[first], records[second] = records[second], records[first]
        self.fixture.ledger.write_bytes(
            b"".join(REVIEW._runner_canonical_bytes(item) for item in records)
        )
        self.assert_rejected("command ledger order differs")

    def test_runner_rejects_missing_hermetic_env_i(self) -> None:
        def mutate(record: dict[str, object]) -> None:
            del record["argv"][1]

        self.fixture.rewrite_command("formula-units", mutate)
        self.assert_rejected(r"hermetic /usr/bin/env -i")

    def test_runner_rejects_nonzero_optimization_metadata(self) -> None:
        def mutate(record: dict[str, object]) -> None:
            record["runtime"]["python"]["optimization_level"] = 1

        self.fixture.rewrite_command("formula-units", mutate)
        self.assert_rejected("optimization_level")

    def test_runner_runtime_python_versions_are_bound_per_lane(self) -> None:
        cases = (
            ("tf4576-command", "command", "formula-units", "3.12.10"),
            (
                "tf5141-command",
                "command",
                "gradient-checkpointing-tf5141",
                "3.12.11",
            ),
            ("tf5141-environment", "environment", "runtime-tf5141", "3.12.11"),
        )
        for label, kind, name, version in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"runtime-{label}"
                )

                def mutate(record: dict[str, object]) -> None:
                    record["runtime"]["python"]["version"] = version

                if kind == "command":
                    self.fixture.rewrite_command(name, mutate)
                else:
                    self.fixture.rewrite_environment(name, mutate)
                self.assert_rejected(r"runtime\.python\.version")

    def test_runner_rejects_wrong_cuda_selection(self) -> None:
        def mutate(record: dict[str, object]) -> None:
            argv = record["argv"]
            index = argv.index("CUDA_VISIBLE_DEVICES=0")
            argv[index] = "CUDA_VISIBLE_DEVICES=1"

        self.fixture.rewrite_command("p0-1-cuda-bf16", mutate)
        self.assert_rejected("CPU/CUDA selection")

    def test_runner_rejects_wrong_p0_3_revision_option(self) -> None:
        def mutate(record: dict[str, object]) -> None:
            argv = record["argv"]
            index = argv.index("--revision")
            argv[index + 1] = "0" * 40

        self.fixture.rewrite_command("p0-3-checkpointed", mutate)
        self.assert_rejected("fixed TESTING matrix|--revision")

    def test_runner_rejects_extra_reordered_or_duplicate_environment(self) -> None:
        mutations = {
            "extra": lambda argv: argv.insert(2, "UNREVIEWED_ENV=1"),
            "reordered": lambda argv: argv.__setitem__(
                slice(2, 4), [argv[3], argv[2]]
            ),
            "duplicate": lambda argv: argv.insert(
                argv.index("PYTHONOPTIMIZE=0") + 1, "PYTHONOPTIMIZE=0"
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"environment-{label}"
                )

                def mutate(record: dict[str, object]) -> None:
                    mutation(record["argv"])

                self.fixture.rewrite_command("formula-units", mutate)
                self.assert_rejected("hermetic environment assignments")

    def test_offline_cache_preflight_tail_and_order_are_bound(self) -> None:
        def add_site_suppression(record: dict[str, object]) -> None:
            argv = record["argv"]
            script = argv.index("scripts/check_level1_offline_cache.py")
            argv.insert(script, "-S")

        self.fixture.rewrite_command(
            "offline-cache-preflight", add_site_suppression
        )
        self.assert_rejected("exact TF4 Python .*checker tail")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "offline-order")
        records = {item["name"]: item for item in self.fixture.command_records()}
        hygiene = records["repository-hygiene"]

        def overlap_hygiene(record: dict[str, object]) -> None:
            record["started_at_utc"] = hygiene["started_at_utc"]
            record["ended_at_utc"] = hygiene["ended_at_utc"]

        self.fixture.rewrite_command("offline-cache-preflight", overlap_hygiene)
        self.assert_rejected("environment and initial hygiene commands|out of order")

    def test_exact_child_argv_tails_reject_representative_command_classes(self) -> None:
        def replace_value(option: str, value: str):
            def mutate(argv: list[str]) -> None:
                index = argv.index(option)
                argv[index + 1] = value

            return mutate

        def replace_argument(old: str, new: str):
            def mutate(argv: list[str]) -> None:
                argv[argv.index(old)] = new

            return mutate

        def replace_child_with_true(argv: list[str]) -> None:
            executable = argv.index(self.fixture.tf4576_python)
            argv[executable:] = ["/bin/true"]

        cases = (
            (
                "child-executable-replacement",
                "formula-units",
                replace_child_with_true,
            ),
            (
                "environment",
                "environment-cuda0",
                replace_value("--lane", "tf4576"),
            ),
            (
                "repository-check",
                "repository-hygiene",
                replace_value("--check", "json"),
            ),
            (
                "syntax-tracked-set",
                "syntax-level1",
                lambda argv: argv.pop(),
            ),
            (
                "unittest-pattern",
                "c1-architecture",
                replace_value("-p", "test_paper_initialization_contract.py"),
            ),
            (
                "script-replacement",
                "formula-units",
                replace_argument("oracle/test_formula_units.py", "/bin/true"),
            ),
            (
                "forbidden-quick",
                "p0-1-cpu-fp32",
                lambda argv: argv.append("--quick"),
            ),
            (
                "tolerance",
                "p0-2-cuda-bf16",
                replace_value("--rtol", "0.3"),
            ),
            (
                "device-flag",
                "p0-2-cuda-bf16",
                replace_value("--device", "cpu"),
            ),
            (
                "c3-mode",
                "c3-psi8-operational",
                replace_value("--mode", "peak-exposure"),
            ),
            (
                "p0-3-step-contract",
                "p0-3-checkpointed",
                replace_value("--steps-per-psi", "8:4,16:2"),
            ),
            (
                "p0-3-data-files-override",
                "p0-3-checkpointed",
                lambda argv: argv.extend(["--data-files", "private.parquet"]),
            ),
            (
                "p0-3-data-dir-override",
                "p0-3-checkpointed",
                lambda argv: argv.extend(["--data-dir", "private"]),
            ),
            (
                "p0-3-text-file-override",
                "p0-3-checkpointed",
                lambda argv: argv.extend(["--text-file", "private.txt"]),
            ),
            (
                "tokenizer-source",
                "p0-3-tokenizer-psi8",
                replace_value("--source-tokenizer", "gpt2"),
            ),
            (
                "p0-4-steps",
                "p0-4-psi8",
                replace_value("--steps", "49"),
            ),
            (
                "focused-review-safe-path",
                "p0-4-review-psi8",
                lambda argv: argv.remove("-P"),
            ),
            (
                "offline-no-bytecode",
                "offline-cache-preflight",
                lambda argv: argv.remove("-B"),
            ),
            (
                "repository-safe-path",
                "repository-hygiene",
                lambda argv: argv.remove("-P"),
            ),
            (
                "focused-review-mode",
                "p0-4-review-psi8",
                replace_value("--mode", "full"),
            ),
            (
                "wrong-lane-python",
                "gradient-checkpointing-tf5141",
                replace_argument(
                    self.fixture.tf5141_python, self.fixture.tf4576_python
                ),
            ),
        )
        for label, name, mutation in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"exact-tail-{label}"
                )

                def mutate(record: dict[str, object]) -> None:
                    mutation(record["argv"])

                self.fixture.rewrite_command(name, mutate)
                self.assert_rejected(
                    "fixed TESTING matrix|exact .* tail|exact TF|tracked Python"
                )

    def test_offline_cache_semantic_log_rejects_all_contract_tampering(self) -> None:
        def set_nested(*keys_and_value: object):
            *keys, value = keys_and_value

            def mutate(report: dict[str, object]) -> None:
                target = report
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value

            return mutate

        cases = (
            ("status", set_nested("status", "failed"), False, "status"),
            (
                "schema",
                set_nested("schema_version", "multiscreen-level1-offline-cache-v2"),
                False,
                "schema_version",
            ),
            (
                "cache",
                set_nested("cache", "single_cache", False),
                False,
                "single_cache",
            ),
            (
                "flags",
                set_nested("offline_environment", "HF_HUB_OFFLINE", "0"),
                False,
                "offline flags",
            ),
            (
                "revision",
                set_nested(
                    "checks", "p0_3_tinystories", "revision", "0" * 40
                ),
                False,
                "revision",
            ),
            (
                "c3-manifest",
                set_nested("checks", "p0_5_c3", "manifest_sha256", "0" * 64),
                False,
                "manifest_sha256",
            ),
            (
                "private-path",
                set_nested(
                    "checks",
                    "p0_3_tinystories",
                    "fingerprint_sha256",
                    "/" + "home/private/cache",
                ),
                False,
                "path|SHA-256",
            ),
            ("extra-line", lambda report: None, True, "exactly one JSON line"),
        )
        for label, mutation, extra_line, expected in cases:
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"offline-semantic-{label}"
                )
                path = self.fixture.root / "logs/offline-cache-preflight.log"
                report = json.loads(path.read_text(encoding="utf-8"))
                mutation(report)
                raw = REVIEW._runner_canonical_bytes(report)
                if extra_line:
                    raw += b"{}\n"
                self.fixture.replace_log("offline-cache-preflight", raw)
                self.assert_rejected(expected)

    def test_semantic_environment_python_optimization_is_bound(self) -> None:
        path = self.fixture.root / "logs/environment-cuda0.log"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["python"]["optimization_level"] = 1
        self.fixture.replace_log(
            "environment-cuda0", REVIEW._runner_canonical_bytes(report)
        )
        self.assert_rejected("python")

    def test_runner_rejects_tampered_missing_and_traversal_logs(self) -> None:
        log = self.fixture.root / "logs" / "formula-units.log"
        log.write_bytes(log.read_bytes() + b"tamper\n")
        self.assert_rejected("size_bytes|sha256")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "missing")
        (self.fixture.root / "logs" / "oracle-smoke.log").unlink()
        self.assert_rejected("missing required file")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "traversal")

        def traverse(record: dict[str, object]) -> None:
            record["log"]["path"] = "../escape.log"

        self.fixture.rewrite_command("formula-units", traverse)
        self.assert_rejected(r"log\.path")

    def test_runner_rejects_hard_linked_evidence(self) -> None:
        log = self.fixture.root / "logs/formula-units.log"
        os.link(log, self.fixture.root / "formula-units-hardlink.log")
        self.assert_rejected("must not have hard links")

    def test_runner_requires_private_run_root_permissions(self) -> None:
        self.fixture.root.chmod(0o755)
        self.addCleanup(self.fixture.root.chmod, 0o700)
        self.assert_rejected("private mode 0700")

    def test_runner_marker_commit_and_p0_4_order_are_bound(self) -> None:
        marker_path = self.fixture.root / ".level1-requalification-run.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["repository"]["head_commit"] = "a" * 40
        write_json(marker_path, marker)
        self.assert_rejected("head_commit")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "order")
        records = {item["name"]: item for item in self.fixture.command_records()}
        review_record = records["p0-4-review-psi8"]

        def overlap(record: dict[str, object]) -> None:
            record["started_at_utc"] = review_record["started_at_utc"]
            record["ended_at_utc"] = review_record["ended_at_utc"]

        self.fixture.rewrite_command("p0-4-psi16-preflight", overlap)
        self.assert_rejected("strict P0-4|out of order")

    def test_runner_marker_worktree_digest_is_bound_to_reviewer_checkout(self) -> None:
        marker_path = self.fixture.root / ".level1-requalification-run.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = marker["repository"]["worktree_path_sha256"]
        marker["repository"]["worktree_path_sha256"] = (
            "0" * 64 if expected != "0" * 64 else "1" * 64
        )
        marker_path.write_bytes(REVIEW._runner_canonical_bytes(marker))
        self.assert_rejected(r"runner\.marker\.worktree_path_sha256")

    def test_c3_data_contract_and_tokenizer_normalization_are_bound(self) -> None:
        contract_path = self.fixture.c3_data / "data_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["accounting"]["selected_rows"] = 63
        write_json(contract_path, contract)
        self.assert_rejected("selected_rows")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "normalization")
        tokenizer_path = self.fixture.tokenizer_reports["p0_4_psi8"]
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        tokenizer["source_normalization"]["padding_side"] = "left"
        write_json(tokenizer_path, tokenizer)
        self.assert_rejected("source_normalization")

    def test_real_repository_checker_schema_is_consumed_exactly(self) -> None:
        result = {
            "artifact_count": 2,
            "artifact_manifest_sha256": "a" * 64,
            "bytes_checked": 100,
        }
        with (
            mock.patch.object(
                REPOSITORY_CHECK,
                "validate_repository_root",
                return_value=self.fixture.repository,
            ),
            mock.patch.object(
                REPOSITORY_CHECK,
                "_clean_head",
                side_effect=[self.fixture.tested_commit] * 2,
            ),
            mock.patch.object(
                REPOSITORY_CHECK, "tracked_entries", return_value=[object()]
            ),
            mock.patch.dict(
                REPOSITORY_CHECK.CHECKS,
                {"json": lambda repository, entries: dict(result)},
            ),
        ):
            produced = REPOSITORY_CHECK.run_check(
                self.fixture.repository, "json"
            )
        self.assertEqual(
            set(produced),
            {"check", "format_version", "head_commit", "result", "status"},
        )
        reviewed = REVIEW._review_repository_command_log(
            "json-validation",
            REPOSITORY_CHECK._canonical_bytes(produced),
            tested_commit=self.fixture.tested_commit,
        )
        self.assertEqual(reviewed["check"], "json")
        self.assertEqual(
            reviewed["artifact_manifest_sha256"],
            result["artifact_manifest_sha256"],
        )

    def test_repository_checker_top_level_head_is_required_and_bound(self) -> None:
        for variant in ("missing", "wrong"):
            with self.subTest(variant=variant):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"repository-head-{variant}"
                )
                path = self.fixture.root / "logs/json-validation.log"
                report = json.loads(path.read_text(encoding="utf-8"))
                if variant == "missing":
                    report.pop("head_commit")
                else:
                    report["head_commit"] = "a" * 40
                self.fixture.replace_log(
                    "json-validation", REVIEW._runner_canonical_bytes(report)
                )
                self.assert_rejected("fields are incomplete|head_commit")

    def test_semantic_environment_and_hygiene_logs_are_not_hash_only(self) -> None:
        environment_path = self.fixture.root / "logs/environment-tf4576.log"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment["packages"]["numpy"] = "0"
        self.fixture.replace_log(
            "environment-tf4576", REVIEW._runner_canonical_bytes(environment)
        )
        self.assert_rejected("packages")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "hygiene")
        hygiene_path = self.fixture.root / "logs/repository-hygiene.log"
        hygiene = json.loads(hygiene_path.read_text(encoding="utf-8"))
        hygiene["result"]["head_commit"] = "a" * 40
        self.fixture.replace_log(
            "repository-hygiene", REVIEW._runner_canonical_bytes(hygiene)
        )
        self.assert_rejected("head_commit")

    def test_hygiene_privacy_summary_is_strict(self) -> None:
        path = self.fixture.root / "logs/repository-hygiene.log"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["result"]["privacy"]["rules"].reverse()
        self.fixture.replace_log(
            "repository-hygiene", REVIEW._runner_canonical_bytes(report)
        )
        self.assert_rejected("privacy.rules")

    def test_initial_final_hygiene_privacy_identity_is_bound(self) -> None:
        path = self.fixture.root / "logs/repository-hygiene-final.log"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["result"]["privacy"]["artifact_manifest_sha256"] = "d" * 64
        self.fixture.replace_log(
            "repository-hygiene-final", REVIEW._runner_canonical_bytes(report)
        )
        self.assert_rejected("hygiene identities differ")

    def test_successful_harness_tolerance_results_do_not_use_unsound_max_cap(
        self,
    ) -> None:
        summary_path = self.fixture.p0_4_psi8 / "summary.json"
        metrics_path = self.fixture.p0_4_psi8 / "metrics.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["checks"]["save_reload"]["loaded_logits_max_abs"] = 0.5
        summary["checks"]["cache"]["cache_split_logits_max_abs"] = 0.5
        events = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        events[53]["loaded_logits_max_abs"] = 0.5
        events[54]["cache_split_logits_max_abs"] = 0.5
        write_json(summary_path, summary)
        write_jsonl(metrics_path, events)
        lane = REVIEW._review_p0_4_lane(
            self.fixture.p0_4_psi8, psi=8, expected_cache=self.fixture.cache, hashes={}
        )
        self.assertEqual(lane["status"], "passed")

        results_path = self.fixture.p0_3 / "p0_3_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        results[0]["save_load_logits_max_abs"] = 0.5
        results[0]["cache_split_logits_max_abs"] = 0.5
        write_json(results_path, results)
        write_json(self.fixture.p0_3 / "psi8/p0_3_metrics.json", results[0])
        p03 = REVIEW._review_p0_3(
            self.fixture.p0_3, self.fixture.p0_3_stdout, hashes={}
        )
        self.assertEqual(p03["status"], "passed")

    def test_focused_psi8_review_is_revalidated_by_full_review(self) -> None:
        with mock.patch.object(
            REVIEW, "C3_ROW_MANIFEST_SHA256", self.fixture.c3_row_manifest_sha256
        ):
            focused = REVIEW.review_p0_4_lane_inputs(
                p0_4_root=self.fixture.p0_4_psi8,
                tokenizer_reports={
                    "p0_4_psi8": self.fixture.tokenizer_reports["p0_4_psi8"]
                },
                command_ledger=self.fixture.ledger,
                tested_commit=self.fixture.tested_commit,
            )
        self.assertEqual(focused["status"], "passed")
        stored = json.loads(
            self.fixture.p0_4_psi8_review.read_text(encoding="utf-8")
        )
        stored["aggregate"]["review_material_sha256"] = "0" * 64
        write_json(self.fixture.p0_4_psi8_review, stored)
        self.assert_rejected("review_material_sha256")

    def test_live_reviewer_checkout_requires_clean_tested_blob(self) -> None:
        repository = self.fixture.root / "live-reviewer-repository"
        source = repository / "scripts/review_level1_requalification.py"
        source.parent.mkdir(parents=True)
        original = (REPO_ROOT / "scripts/review_level1_requalification.py").read_bytes()
        source.write_bytes(original)

        def git(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        git("init", "-q")
        git("config", "user.name", "Level1 Fixture")
        git("config", "user.email", "level1-fixture@example.invalid")
        git("add", "scripts/review_level1_requalification.py")
        git("commit", "-q", "-m", "fixture")
        tested_commit = git("rev-parse", "HEAD").stdout.strip()
        REVIEW._verify_live_reviewer_checkout(
            tested_commit, repository=repository, reviewer_path=source
        )

        source.write_bytes(original + b"# uncommitted drift\n")
        with self.assertRaisesRegex(REVIEW.ReviewError, "not exactly clean"):
            REVIEW._verify_live_reviewer_checkout(
                tested_commit, repository=repository, reviewer_path=source
            )
        source.write_bytes(original)

        untracked = repository / "untracked.txt"
        untracked.write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(REVIEW.ReviewError, "not exactly clean"):
            REVIEW._verify_live_reviewer_checkout(
                tested_commit, repository=repository, reviewer_path=source
            )
        untracked.unlink()

        with self.assertRaisesRegex(REVIEW.ReviewError, "HEAD differs"):
            REVIEW._verify_live_reviewer_checkout(
                "0" * 40, repository=repository, reviewer_path=source
            )

        git(
            "update-index",
            "--assume-unchanged",
            "scripts/review_level1_requalification.py",
        )
        source.write_bytes(original + b"# hidden drift\n")
        with self.assertRaisesRegex(REVIEW.ReviewError, "source bytes differ"):
            REVIEW._verify_live_reviewer_checkout(
                tested_commit, repository=repository, reviewer_path=source
            )
        source.write_bytes(original)
        git(
            "update-index",
            "--no-assume-unchanged",
            "scripts/review_level1_requalification.py",
        )

    def test_python_safety_flags_are_required_exactly(self) -> None:
        REVIEW._verify_python_safety_flags(
            mock.Mock(safe_path=True, no_site=True, dont_write_bytecode=True)
        )
        for field in ("safe_path", "no_site", "dont_write_bytecode"):
            values = {
                "safe_path": True,
                "no_site": True,
                "dont_write_bytecode": True,
            }
            values[field] = False
            with self.subTest(field=field):
                with self.assertRaisesRegex(REVIEW.ReviewError, field):
                    REVIEW._verify_python_safety_flags(mock.Mock(**values))

    def test_focused_cli_rejects_dirty_executing_checkout(self) -> None:
        repository = self.fixture.root / "focused-dirty-repository"
        source = repository / "scripts/review_level1_requalification.py"
        source.parent.mkdir(parents=True)
        original = (REPO_ROOT / "scripts/review_level1_requalification.py").read_bytes()
        source.write_bytes(original)

        def git(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        git("init", "-q")
        git("config", "user.name", "Level1 Fixture")
        git("config", "user.email", "level1-fixture@example.invalid")
        git("add", "scripts/review_level1_requalification.py")
        git("commit", "-q", "-m", "fixture")
        tested_commit = git("rev-parse", "HEAD").stdout.strip()
        source.write_bytes(original + b"# dirty focused reviewer\n")
        output = repository / "focused.json"
        args = [
            "--mode",
            "p0-4-lane",
            "--tested-commit",
            tested_commit,
            "--command-ledger",
            str(repository / "missing-ledger.jsonl"),
            "--p0-4-root",
            str(repository / "missing-p0-4"),
            "--output",
            str(output),
        ]
        with mock.patch.object(
            REVIEW, "__file__", str(source)
        ), mock.patch.object(REVIEW, "_verify_python_safety_flags"):
            with self.assertRaisesRegex(REVIEW.ReviewError, "not exactly clean"):
                REVIEW.main(args)
        self.assertFalse(output.exists())

    def test_output_creation_is_exclusive_nofollow_and_private(self) -> None:
        output = self.fixture.root / "exclusive.json"
        REVIEW._exclusive_output(output, b"{}\n")
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        with self.assertRaisesRegex(REVIEW.ReviewError, "refusing to overwrite"):
            REVIEW._exclusive_output(output, b"{}\n")

        race_output = self.fixture.root / "race.json"

        def race(path: Path, flags: int, mode: int = 0) -> int:
            self.assertTrue(flags & os.O_EXCL)
            if hasattr(os, "O_NOFOLLOW"):
                self.assertTrue(flags & os.O_NOFOLLOW)
            raise FileExistsError

        with mock.patch.object(REVIEW.os, "open", side_effect=race):
            with self.assertRaisesRegex(REVIEW.ReviewError, "refusing to overwrite"):
                REVIEW._exclusive_output(race_output, b"{}\n")

        symlink_output = self.fixture.root / "symlink-output.json"
        target = self.fixture.root / "target.json"
        target.write_text("preserve\n", encoding="utf-8")
        os.symlink(target, symlink_output)
        with self.assertRaisesRegex(REVIEW.ReviewError, "refusing to overwrite"):
            REVIEW._exclusive_output(symlink_output, b"{}\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "preserve\n")

    def test_focused_lane_cli_writes_canonical_report(self) -> None:
        output = self.fixture.root / "focused-cli.json"
        args = [
            "--mode",
            "p0-4-lane",
            "--tested-commit",
            self.fixture.tested_commit,
            "--command-ledger",
            str(self.fixture.ledger),
            "--p0-4-root",
            str(self.fixture.p0_4_psi8),
            "--tokenizer-reload-report",
            f"p0_4_psi8={self.fixture.tokenizer_reports['p0_4_psi8']}",
            "--output",
            str(output),
        ]
        with mock.patch.object(
            REVIEW, "_verify_live_reviewer_checkout"
        ) as integrity, mock.patch.object(
            REVIEW, "_verify_python_safety_flags"
        ), mock.patch.object(
            REVIEW, "C3_ROW_MANIFEST_SHA256", self.fixture.c3_row_manifest_sha256
        ):
            self.assertEqual(REVIEW.main(args), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "p0-4-lane")
            self.assertEqual(report["status"], "passed")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(integrity.call_count, 2)

    def test_cli_writes_canonical_report_once(self) -> None:
        output = self.fixture.root / "review.json"
        args = [
            "--p0-3-root", str(self.fixture.p0_3),
            "--p0-3-stdout", str(self.fixture.p0_3_stdout),
            "--p0-4-psi8-root", str(self.fixture.p0_4_psi8),
            "--p0-4-psi16-root", str(self.fixture.p0_4_psi16),
            "--p0-4-psi8-review", str(self.fixture.p0_4_psi8_review),
            "--c3-data-root", str(self.fixture.c3_data),
            "--c3-psi8-operational-root", str(self.fixture.c3_roots["c3_psi8_operational"]),
            "--c3-psi8-peak-exposure-root", str(self.fixture.c3_roots["c3_psi8_peak_exposure"]),
            "--c3-psi16-operational-root", str(self.fixture.c3_roots["c3_psi16_operational"]),
            "--c3-psi16-peak-exposure-root", str(self.fixture.c3_roots["c3_psi16_peak_exposure"]),
        ]
        for name, path in self.fixture.tokenizer_reports.items():
            args.extend(["--tokenizer-reload-report", f"{name}={path}"])
        args.extend(
            [
                "--command-ledger", str(self.fixture.ledger),
                "--tested-commit", self.fixture.tested_commit,
                "--output", str(output),
            ]
        )
        with mock.patch.object(
            REVIEW, "_verify_live_reviewer_checkout"
        ) as integrity, mock.patch.object(
            REVIEW, "_verify_python_safety_flags"
        ), mock.patch.object(
            REVIEW, "C3_ROW_MANIFEST_SHA256", self.fixture.c3_row_manifest_sha256
        ):
            self.assertEqual(REVIEW.main(args), 0)
            raw = output.read_bytes()
            parsed = json.loads(raw)
            self.assertEqual(raw, REVIEW._pretty_canonical_bytes(parsed))
            with self.assertRaisesRegex(REVIEW.ReviewError, "refusing to overwrite"):
                REVIEW.main(args)
            self.assertEqual(integrity.call_count, 4)


if __name__ == "__main__":
    unittest.main()
