#!/usr/bin/env python
"""P0-4 GPT-2-vocabulary, context-4096 smoke training for Multiscreen.

The harness is correctness-first and targets the dense P0-qualified baseline;
it is not a throughput or long-context efficiency benchmark. A run writes
``P0-4_COMPLETE.md`` only when it uses GPT-2 vocab 50,257, context 4,096,
CUDA bf16, and at least 50 optimizer steps. Reduced runs that pass all checks
write ``P0-4_DIAGNOSTIC_COMPLETE.md`` instead.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.utils.data import DataLoader

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - environment dependent
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None

GPT2_VOCAB_SIZE = 50_257
STAGE = "P0-4"


def repo_default() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required config not found: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def nested(value: Mapping[str, Any], key: str, default: Any = None) -> Any:
    current: Any = value
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def override(cli_value: Any, configured: Any) -> Any:
    return configured if cli_value is None else cli_value


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value!r}")


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_settings(args: argparse.Namespace) -> argparse.Namespace:
    root = Path(args.repo_root).expanduser().resolve()
    config_dir = resolve_path(args.config_dir, root)
    run = read_json(config_dir / "run.json")
    get = lambda key, default=None: nested(run, key, default)
    pick = lambda cli, key, default=None: override(cli, get(key, default))
    betas = get("training.betas", [0.9, 0.95])
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("training.betas must contain two numbers")
    output = pick(args.output_dir, "runtime.output_dir")
    if not output:
        raise ValueError("runtime.output_dir or --output-dir is required")
    text_file = pick(args.text_file, "dataset.text_file")
    s = argparse.Namespace(
        repo_root=root,
        config_dir=config_dir,
        output_dir=resolve_path(output, root),
        tokenizer_name=str(pick(args.tokenizer_name_or_path, "tokenizer.name_or_path", "gpt2")),
        tokenizer_use_fast=bool(get("tokenizer.use_fast", True)),
        expected_vocab_size=int(get("tokenizer.expected_vocab_size", GPT2_VOCAB_SIZE)),
        dataset_name=str(pick(args.dataset_name, "dataset.name", "roneneldan/TinyStories")),
        dataset_config=pick(args.dataset_config, "dataset.config_name"),
        train_split=str(pick(args.train_split, "dataset.train_split", "train[:20000]")),
        text_column=str(pick(args.text_column, "dataset.text_column", "text")),
        text_file=resolve_path(text_file, root) if text_file else None,
        data_files=get("dataset.data_files"),
        data_dir=get("dataset.data_dir"),
        revision=get("dataset.revision"),
        streaming=bool(pick(args.streaming, "dataset.streaming", False)),
        cache_dir=pick(args.cache_dir, "cache.cache_dir"),
        max_texts=int(pick(args.max_texts, "dataset.max_texts", 20000)),
        max_train_tokens=int(pick(args.max_train_tokens, "dataset.max_train_tokens", 524416)),
        seq_len=int(pick(args.seq_len, "training.seq_len", 4096)),
        steps=int(pick(args.steps, "training.optimizer_steps", 50)),
        batch_size=int(pick(args.microbatch_size, "training.microbatch_size", 1)),
        grad_accum=int(pick(args.gradient_accumulation_steps, "training.gradient_accumulation_steps", 8)),
        lr=float(pick(args.learning_rate, "training.learning_rate", 6e-4)),
        weight_decay=float(pick(args.weight_decay, "training.weight_decay", 0.0)),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(get("training.eps", 1e-8)),
        max_grad_norm=float(pick(args.max_grad_norm, "training.max_grad_norm", 1.0)),
        amp_dtype=str(pick(args.amp_dtype, "training.amp_dtype", "bf16")),
        gradient_checkpointing=bool(pick(args.gradient_checkpointing, "training.gradient_checkpointing", True)),
        fused_adamw=bool(pick(args.fused_adamw, "training.fused_adamw", True)),
        probe_replay_every=int(get("training.probe_replay_every", 4)),
        min_loss_drop=float(get("checks.min_loss_drop", 0.01)),
        min_rel_loss_drop=float(get("checks.min_rel_loss_drop", 0.001)),
        reload_atol=float(get("checks.reload_atol", 1e-5)),
        reload_rtol=float(get("checks.reload_rtol", 1e-5)),
        reload_tokens=int(get("checks.reload_check_tokens", 16)),
        cache_atol=float(get("checks.cache_atol", 3e-2)),
        cache_rtol=float(get("checks.cache_rtol", 3e-2)),
        cache_tokens=int(get("checks.cache_check_tokens", 24)),
        prompt=str(get("checks.prompt", "Once upon a time")),
        max_new_tokens=int(get("checks.max_new_tokens", 8)),
        device=str(pick(args.device, "runtime.device", "cuda:0")),
        allow_cpu=bool(pick(args.allow_cpu, "runtime.allow_cpu", False)),
        num_workers=int(pick(args.num_workers, "runtime.num_workers", 0)),
        seed=int(pick(args.seed, "runtime.seed", 42)),
        log_every=int(pick(args.log_every, "runtime.log_every", 1)),
    )
    for name in (
        "expected_vocab_size", "max_texts", "max_train_tokens", "seq_len", "steps",
        "batch_size", "grad_accum", "reload_tokens", "cache_tokens", "max_new_tokens", "log_every",
    ):
        if getattr(s, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if s.seq_len > 4096:
        raise ValueError("P0-4 supports at most context 4096")
    if s.max_train_tokens < s.seq_len + 1:
        raise ValueError("max_train_tokens is too small for one shifted packed example")
    if s.amp_dtype not in {"none", "bf16", "bfloat16", "fp16", "float16"}:
        raise ValueError(f"Unsupported amp dtype: {s.amp_dtype}")
    return s


def settings_json(s: argparse.Namespace) -> dict[str, Any]:
    result = vars(s).copy()
    for key in ("repo_root", "config_dir", "output_dir", "text_file"):
        if result[key] is not None:
            result[key] = str(result[key])
    result["betas"] = list(result["betas"])
    return result


def validate_config_files(s: argparse.Namespace) -> dict[str, Any]:
    model = read_json(s.config_dir / "config.json")
    run = read_json(s.config_dir / "run.json")
    hidden = int(model.get("hidden_size", 0))
    psi = math.isqrt(hidden)
    checks = {
        "model_type_multiscreen": model.get("model_type") == "multiscreen",
        "vocab_size_50257": int(model.get("vocab_size", -1)) == GPT2_VOCAB_SIZE,
        "max_position_embeddings_4096": int(model.get("max_position_embeddings", -1)) == 4096,
        "hidden_size_is_psi_squared": psi * psi == hidden,
        "psi_is_8_or_16": psi in {8, 16},
        "layers_equal_psi": int(model.get("num_hidden_layers", -1)) == psi,
        "heads_equal_psi": int(model.get("num_attention_heads", -1)) == psi,
        "tie_word_embeddings": model.get("tie_word_embeddings") is True,
        "run_expected_vocab_50257": int(nested(run, "tokenizer.expected_vocab_size", -1)) == GPT2_VOCAB_SIZE,
        "run_seq_len_4096": int(nested(run, "training.seq_len", -1)) == 4096,
        "run_amp_bf16": str(nested(run, "training.amp_dtype", "")) in {"bf16", "bfloat16"},
        "run_microbatch_1": int(nested(run, "training.microbatch_size", -1)) == 1,
        "run_steps_at_least_50": int(nested(run, "training.optimizer_steps", -1)) >= 50,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Static P0-4 config validation failed: {', '.join(failed)}")
    return {"config_dir": str(s.config_dir), "psi": psi, "checks": checks}


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(s: argparse.Namespace) -> torch.device:
    device = torch.device(s.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}) but unavailable")
    if device.type == "cpu" and not s.allow_cpu:
        raise RuntimeError("CPU is diagnostic-only; pass --allow-cpu true explicitly")
    if device.type == "cuda" and s.amp_dtype in {"bf16", "bfloat16"} and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but unsupported by this CUDA device")
    return device


def autocast(device: torch.device, dtype: str):
    if dtype == "none":
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16 if dtype in {"bf16", "bfloat16"} else torch.float16)
    if dtype in {"bf16", "bfloat16"}:
        return torch.autocast("cpu", dtype=torch.bfloat16)
    raise RuntimeError("CPU fp16 autocast is unsupported")


def grad_scaler(device: torch.device, dtype: str):
    enabled = device.type == "cuda" and dtype in {"fp16", "float16"}
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def load_texts(s: argparse.Namespace) -> list[str]:
    if s.text_file:
        raw = s.text_file.read_text(encoding="utf-8")
        texts = [x.strip() for x in raw.split("\n\n") if x.strip()]
        if len(texts) <= 1:
            texts = [x.strip() for x in raw.splitlines() if x.strip()]
        if not texts:
            raise RuntimeError(f"No text found in {s.text_file}")
        return texts[: s.max_texts]
    if load_dataset is None:
        raise RuntimeError(f"datasets unavailable and no --text-file supplied: {_DATASETS_IMPORT_ERROR!r}")
    kwargs = {
        "split": s.train_split, "cache_dir": s.cache_dir, "data_files": s.data_files,
        "data_dir": s.data_dir, "revision": s.revision, "streaming": s.streaming,
    }
    dataset = load_dataset(s.dataset_name, s.dataset_config, **{k: v for k, v in kwargs.items() if v is not None})
    columns = list(getattr(dataset, "column_names", []) or [])
    column = s.text_column
    if column == "auto":
        column = next((x for x in ("text", "story", "content", "document") if x in columns), "")
    if columns and column not in columns:
        raise ValueError(f"text column {column!r} not found; available: {columns}")
    texts: list[str] = []
    for row in dataset:
        value = row.get(column) if isinstance(row, Mapping) else row[column]
        if value is not None and str(value).strip():
            texts.append(str(value))
        if len(texts) >= s.max_texts:
            break
    if not texts:
        raise RuntimeError("Dataset contains no non-empty text")
    return texts


def load_tokenizer(s: argparse.Namespace):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(s.tokenizer_name, use_fast=s.tokenizer_use_fast, cache_dir=s.cache_dir)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("GPT-2 tokenizer has no eos_token_id")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = s.seq_len
    if len(tokenizer) != s.expected_vocab_size or len(tokenizer) != GPT2_VOCAB_SIZE:
        raise AssertionError(f"Expected GPT-2 vocab {GPT2_VOCAB_SIZE}, got {len(tokenizer)}")
    return tokenizer


def build_model(s: argparse.Namespace, tokenizer):
    from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM, register_multiscreen_auto_classes

    register_multiscreen_auto_classes()
    config = MultiscreenConfig.from_pretrained(str(s.config_dir))
    if int(config.vocab_size) != len(tokenizer):
        raise AssertionError("Model/tokenizer vocabulary mismatch")
    configured_context = int(config.max_position_embeddings)
    if s.seq_len > configured_context:
        raise AssertionError(f"seq_len {s.seq_len} exceeds configured context {configured_context}")
    if s.seq_len != configured_context:  # reduced diagnostic
        config.max_position_embeddings = s.seq_len
        config.max_seq_len = s.seq_len
    config.pad_token_id = int(tokenizer.pad_token_id)
    config.eos_token_id = int(tokenizer.eos_token_id)
    config.bos_token_id = int(tokenizer.bos_token_id or tokenizer.eos_token_id)
    config.use_cache = False
    config.gradient_checkpointing = s.gradient_checkpointing
    model = MultiscreenForCausalLM(config)
    if s.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def get_loss(model, batch: Mapping[str, Any], s: argparse.Namespace, device: torch.device) -> torch.Tensor:
    with autocast(device, s.amp_dtype):
        output = model(**batch, use_cache=False, return_dict=True)
    if output.loss is None:
        raise RuntimeError("Model returned loss=None")
    return output.loss


@torch.no_grad()
def probe_loss(model, batch: Mapping[str, Any], s: argparse.Namespace, device: torch.device) -> float:
    model.eval()
    value = float(get_loss(model, batch, s, device).detach().float().cpu())
    model.train()
    return value


def memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def train(model, loader: DataLoader, probe: Mapping[str, Any], s: argparse.Namespace, device: torch.device, metrics: Path):
    kwargs = {"lr": s.lr, "weight_decay": s.weight_decay, "betas": s.betas, "eps": s.eps}
    if s.fused_adamw and device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError) as exc:
            print(f"[warn] fused AdamW unavailable ({exc}); using standard AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    scaler = grad_scaler(device, s.amp_dtype)
    iterator = cycle(loader)
    initial = probe_loss(model, probe, s, device)
    if not math.isfinite(initial):
        raise RuntimeError(f"Initial probe loss is non-finite: {initial}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    grad_norms: list[float] = []
    started = time.perf_counter()
    for step in range(1, s.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(s.grad_accum):
            batch = probe if s.probe_replay_every > 0 and step % s.probe_replay_every == 0 else move(next(iterator), device)
            loss = get_loss(model, batch, s, device)
            value = float(loss.detach().float().cpu())
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite loss at step {step}: {value}")
            micro_losses.append(value)
            scaled = loss / s.grad_accum
            scaler.scale(scaled).backward() if scaler.is_enabled() else scaled.backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), s.max_grad_norm)
        norm_value = float(norm.detach().float().cpu() if isinstance(norm, torch.Tensor) else norm)
        if not math.isfinite(norm_value):
            raise RuntimeError(f"Non-finite grad norm at step {step}: {norm_value}")
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        mean_loss = sum(micro_losses) / len(micro_losses)
        losses.append(mean_loss)
        grad_norms.append(norm_value)
        event = {
            "event": "train_step", "stage": STAGE, "timestamp_utc": utc_now(),
            "optimizer_step": step, "optimizer_steps": s.steps, "mean_loss": mean_loss,
            "micro_losses": micro_losses, "grad_norm": norm_value,
            "elapsed_sec": time.perf_counter() - started, **memory(device),
        }
        append_jsonl(metrics, event)
        if step == 1 or step == s.steps or step % s.log_every == 0:
            peak = event["peak_allocated_bytes"]
            print(f"[P0-4] step={step:04d}/{s.steps} loss={mean_loss:.6f} grad_norm={norm_value:.6f} peak_gib={(peak or 0)/(1024**3):.3f}")
    final = probe_loss(model, probe, s, device)
    if not math.isfinite(final):
        raise RuntimeError(f"Final probe loss is non-finite: {final}")
    absolute = initial - final
    relative = absolute / max(abs(initial), 1e-12)
    if absolute < s.min_loss_drop and relative < s.min_rel_loss_drop:
        raise AssertionError(f"Probe loss did not decrease enough: {initial:.8f} -> {final:.8f}")
    result = {
        "optimizer_steps": s.steps, "gradient_accumulation_steps": s.grad_accum,
        "microbatch_size": s.batch_size, "effective_batch_tokens": s.batch_size * s.grad_accum * s.seq_len,
        "initial_probe_loss": initial, "final_probe_loss": final,
        "abs_loss_drop": absolute, "rel_loss_drop": relative,
        "train_loss_first": losses[0], "train_loss_last": losses[-1], "train_loss_min": min(losses),
        "grad_norm_max": max(grad_norms), "elapsed_sec": time.perf_counter() - started, **memory(device),
    }
    return optimizer, result


@torch.no_grad()
def short_logits(model, probe: Mapping[str, Any], s: argparse.Namespace, device: torch.device) -> torch.Tensor:
    length = min(s.reload_tokens, int(probe["input_ids"].shape[1]))
    ids = probe["input_ids"][:, :length]
    mask = probe.get("attention_mask")
    mask = mask[:, :length] if mask is not None else None
    model.eval()
    with autocast(device, s.amp_dtype):
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
    return logits.detach().float().cpu()


@torch.no_grad()
def save_reload(model, optimizer, tokenizer, probe, s: argparse.Namespace, device: torch.device):
    from multiscreen_transformers import register_multiscreen_auto_classes
    from transformers import AutoModelForCausalLM

    checkpoint = s.output_dir / "checkpoint"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    checkpoint.mkdir(parents=True)
    before = short_logits(model, probe, s, device)
    model.save_pretrained(str(checkpoint), safe_serialization=True)
    tokenizer.save_pretrained(str(checkpoint))
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    register_multiscreen_auto_classes()
    loaded = AutoModelForCausalLM.from_pretrained(str(checkpoint)).to(device).eval()
    after = short_logits(loaded, probe, s, device)
    maximum = float((before - after).abs().max())
    torch.testing.assert_close(before, after, atol=s.reload_atol, rtol=s.reload_rtol)
    return loaded, {"checkpoint_dir": str(checkpoint), "reload_check_tokens": int(before.shape[1]), "loaded_logits_max_abs": maximum}


@torch.no_grad()
def cache_check(model, probe, s: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    count = min(s.cache_tokens, int(probe["input_ids"].shape[1]))
    if count < 4:
        raise ValueError("cache_check_tokens must be at least 4")
    split = count // 2
    ids = probe["input_ids"][:, :count]
    mask = probe.get("attention_mask")
    mask = mask[:, :count] if mask is not None else None
    old = bool(getattr(model.config, "use_cache", False))
    model.config.use_cache = True
    try:
        with autocast(device, s.amp_dtype):
            full = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            prefix = model(input_ids=ids[:, :split], attention_mask=mask[:, :split] if mask is not None else None, use_cache=True, return_dict=True)
            suffix = model(input_ids=ids[:, split:], attention_mask=mask, past_key_values=prefix.past_key_values, use_cache=True, return_dict=True)
        a = suffix.logits.detach().float()
        b = full.logits[:, split:, :].detach().float()
        maximum = float((a - b).abs().max().cpu())
        torch.testing.assert_close(a, b, atol=s.cache_atol, rtol=s.cache_rtol)
        return {"cache_check_tokens": count, "cache_split": split, "cache_split_logits_max_abs": maximum}
    finally:
        model.config.use_cache = old


@torch.no_grad()
def generation_check(model, tokenizer, s: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    encoded = tokenizer(s.prompt, return_tensors="pt", add_special_tokens=False)
    if encoded.input_ids.shape[1] == 0:
        encoded = tokenizer("Once upon a time", return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    old = bool(getattr(model.config, "use_cache", False))
    model.config.use_cache = True
    try:
        generated = model.generate(**encoded, max_new_tokens=s.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    finally:
        model.config.use_cache = old
    prompt_len = int(encoded["input_ids"].shape[1])
    if generated.shape[1] <= prompt_len:
        raise AssertionError("generate(use_cache=True) appended no token")
    return {
        "prompt": s.prompt, "prompt_len": prompt_len, "generated_len": int(generated.shape[1]),
        "sample_text": tokenizer.decode(generated[0].detach().cpu(), skip_special_tokens=True)[:400],
    }


def environment(device: torch.device) -> dict[str, Any]:
    import transformers
    try:
        import datasets
        datasets_version = datasets.__version__
    except Exception:
        datasets_version = None
    result = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "datasets": datasets_version,
        "device": str(device), "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        result.update({
            "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    return result


def qualification(s: argparse.Namespace, vocab: int, device: torch.device) -> dict[str, Any]:
    conditions = {
        "gpt2_vocab_50257": vocab == GPT2_VOCAB_SIZE, "context_4096": s.seq_len == 4096,
        "cuda_device": device.type == "cuda", "bf16_amp": s.amp_dtype in {"bf16", "bfloat16"},
        "optimizer_steps_at_least_50": s.steps >= 50,
    }
    return {"qualified": all(conditions.values()), "conditions": conditions}


def write_note(summary: Mapping[str, Any], output: Path) -> None:
    q = summary["qualification"]
    if q["qualified"]:
        t = summary["training"]
        c = summary["checks"]
        lines = [
            "# P0-4 GPT-2 Vocabulary + Context 4096 Smoke Result", "", "## Result", "", "Passed.", "",
            "## Confirmed behavior", "", "- GPT-2 vocabulary size 50,257", "- context-4096 forward/backward",
            "- finite losses and gradient norms", "- probe loss decrease", "- save/load logits comparison",
            "- generate(use_cache=True)", "- manual cache split comparison", "", "## Metrics", "",
            f"- psi: {summary['model']['psi']}", f"- parameters: {summary['model']['parameter_count']:,}",
            f"- optimizer_steps: {t['optimizer_steps']}", f"- initial_probe_loss: {t['initial_probe_loss']:.8f}",
            f"- final_probe_loss: {t['final_probe_loss']:.8f}", f"- abs_loss_drop: {t['abs_loss_drop']:.8f}",
            f"- rel_loss_drop: {t['rel_loss_drop']:.4%}", f"- peak_allocated_bytes: {t['peak_allocated_bytes']}",
            f"- loaded_logits_max_abs: {c['save_reload']['loaded_logits_max_abs']:.8g}",
            f"- cache_split_logits_max_abs: {c['cache']['cache_split_logits_max_abs']:.8g}", "",
            "This remains a short dense-reference smoke, not an efficiency or paper-scale result.", "",
        ]
        name = "P0-4_COMPLETE.md"
    else:
        unmet = [name for name, passed in q["conditions"].items() if not passed]
        lines = [
            "# P0-4 Diagnostic Smoke Result", "", "## Result", "",
            "Diagnostic checks passed, but this run is not P0-4-qualified.", "",
            "## Unmet qualification conditions", "", *[f"- `{name}`" for name in unmet], "",
            "Use the unmodified config defaults on CUDA bf16 to produce `P0-4_COMPLETE.md`.", "",
        ]
        name = "P0-4_DIAGNOSTIC_COMPLETE.md"
    (output / name).write_text("\n".join(lines), encoding="utf-8")


def write_failure(output: Path, exc: BaseException) -> None:
    output.mkdir(parents=True, exist_ok=True)
    failure = {
        "stage": STAGE, "status": "failed", "timestamp_utc": utc_now(),
        "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
    }
    write_json(output / "failure.json", failure)
    (output / "P0-4_FAILED.md").write_text(
        f"# P0-4 Smoke Result\n\n## Result\n\nFailed.\n\n- exception: `{type(exc).__name__}`\n- message: {exc}\n",
        encoding="utf-8",
    )


def run(s: argparse.Namespace) -> dict[str, Any]:
    s.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("failure.json", "P0-4_FAILED.md", "P0-4_COMPLETE.md", "P0-4_DIAGNOSTIC_COMPLETE.md", "summary.json"):
        (s.output_dir / name).unlink(missing_ok=True)
    metrics = s.output_dir / "metrics.jsonl"
    metrics.write_text("", encoding="utf-8")
    seed_all(s.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(s)
    env = environment(device)
    append_jsonl(metrics, {"event": "run_start", "stage": STAGE, "timestamp_utc": utc_now(), "settings": settings_json(s), "environment": env})

    tokenizer = load_tokenizer(s)
    texts = load_texts(s)
    from multiscreen_transformers import PackedTextDataset
    dataset = PackedTextDataset(
        texts=texts, tokenizer=tokenizer, seq_len=s.seq_len, eos_token_id=tokenizer.eos_token_id,
        max_tokens=s.max_train_tokens, legacy_shifted_labels=True, return_labels_are_shifted=True,
    )
    if len(dataset) < s.batch_size:
        raise RuntimeError(f"Only {len(dataset)} packed chunks for batch_size={s.batch_size}")
    loader = DataLoader(dataset, batch_size=s.batch_size, shuffle=True, drop_last=True, num_workers=s.num_workers, pin_memory=device.type == "cuda")
    probe = move(next(iter(loader)), device)
    if int(probe["input_ids"].shape[1]) != s.seq_len:
        raise AssertionError("Packed batch does not have requested sequence length")

    model = build_model(s, tokenizer).to(device).train()
    psi = int(model.config.num_hidden_layers)
    if int(model.config.num_attention_heads) != psi or int(model.config.hidden_size) != psi * psi:
        raise AssertionError("Config violates Psi scaling")
    parameters = sum(p.numel() for p in model.parameters())
    element_bytes = 2 if s.amp_dtype != "none" else 4
    dense_lower_bound = s.batch_size * psi * s.seq_len * s.seq_len * element_bytes
    model_info = {
        "psi": psi, "parameter_count": parameters, "vocab_size": int(model.config.vocab_size),
        "hidden_size": int(model.config.hidden_size), "num_hidden_layers": psi,
        "num_attention_heads": int(model.config.num_attention_heads), "key_dim": int(model.config.key_dim),
        "value_dim": int(model.config.value_dim), "max_position_embeddings": int(model.config.max_position_embeddings),
        "gradient_checkpointing": s.gradient_checkpointing,
        "dense_similarity_one_layer_lower_bound_bytes": dense_lower_bound,
    }
    data_info = {
        "source": str(s.text_file) if s.text_file else s.dataset_name, "train_split": s.train_split,
        "texts_loaded": len(texts), "packed_chunks": len(dataset), "seq_len": s.seq_len,
        "max_train_tokens": s.max_train_tokens, "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
    }
    append_jsonl(metrics, {"event": "preflight_complete", "stage": STAGE, "timestamp_utc": utc_now(), "model": model_info, "data": data_info})
    print(f"[P0-4] psi={psi} params={parameters:,} vocab={len(tokenizer):,} seq_len={s.seq_len} chunks={len(dataset)} device={device} amp={s.amp_dtype}")
    print(f"[P0-4] one-layer dense similarity lower bound: {dense_lower_bound/(1024**3):.3f} GiB (not total memory)")

    optimizer, training = train(model, loader, probe, s, device, metrics)
    append_jsonl(metrics, {"event": "training_complete", "stage": STAGE, "timestamp_utc": utc_now(), **training})
    loaded, reload_result = save_reload(model, optimizer, tokenizer, probe, s, device)
    del optimizer, model
    append_jsonl(metrics, {"event": "save_reload_check", "stage": STAGE, "timestamp_utc": utc_now(), **reload_result})
    cache_result = cache_check(loaded, probe, s, device)
    append_jsonl(metrics, {"event": "cache_split_check", "stage": STAGE, "timestamp_utc": utc_now(), **cache_result})
    generation_result = generation_check(loaded, tokenizer, s, device)
    append_jsonl(metrics, {"event": "generation_check", "stage": STAGE, "timestamp_utc": utc_now(), **generation_result})

    q = qualification(s, len(tokenizer), device)
    status = "passed" if q["qualified"] else "diagnostic_passed"
    summary = {
        "stage": STAGE, "status": status, "timestamp_utc": utc_now(), "qualification": q,
        "environment": env, "settings": settings_json(s), "model": model_info, "data": data_info,
        "training": training, "checks": {"save_reload": reload_result, "cache": cache_result, "generation": generation_result},
    }
    write_json(s.output_dir / "summary.json", summary)
    append_jsonl(metrics, {"event": "run_complete", "stage": STAGE, "status": status, "qualification": q, "timestamp_utc": summary["timestamp_utc"]})
    write_note(summary, s.output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(repo_default()))
    p.add_argument("--config-dir", default="configs/p0_4_multiscreen_psi8_gpt2_ctx4096")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--tokenizer-name-or-path", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--train-split", default=None)
    p.add_argument("--text-column", default=None)
    p.add_argument("--text-file", default=None)
    p.add_argument("--streaming", type=bool_arg, default=None)
    p.add_argument("--max-texts", type=int, default=None)
    p.add_argument("--max-train-tokens", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--microbatch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--amp-dtype", choices=["none", "bf16", "bfloat16", "fp16", "float16"], default=None)
    p.add_argument("--gradient-checkpointing", type=bool_arg, default=None)
    p.add_argument("--fused-adamw", type=bool_arg, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-cpu", type=bool_arg, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--validate-config-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    settings = None
    try:
        settings = load_settings(args)
        static = validate_config_files(settings)
        if args.validate_config_only:
            print(json.dumps(static, ensure_ascii=False, indent=2, sort_keys=True))
            return
        summary = run(settings)
    except BaseException as exc:
        if settings is not None and not args.validate_config_only:
            write_failure(settings.output_dir, exc)
        raise
    note = "P0-4_COMPLETE.md" if summary["qualification"]["qualified"] else "P0-4_DIAGNOSTIC_COMPLETE.md"
    print("\nP0-4 smoke checks passed.")
    print(f"[P0-4] status: {summary['status']}")
    print(f"[P0-4] metrics: {settings.output_dir / 'metrics.jsonl'}")
    print(f"[P0-4] summary: {settings.output_dir / 'summary.json'}")
    print(f"[P0-4] note: {settings.output_dir / note}")


if __name__ == "__main__":
    main()
