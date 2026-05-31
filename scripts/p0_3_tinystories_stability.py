#!/usr/bin/env python
"""P0-3 TinyStories smoke training for Multiscreen Psi=8/16.

This script is intentionally small and explicit.  It does not use TRL/SFTTrainer;
it verifies the core model training path directly with PyTorch so failures are
easier to localize after P0-1/P0-2.

What it checks per Psi value:
  - TinyStories text can be tokenized and packed.
  - Forward/backward/optimizer steps are finite.
  - Probe-batch loss decreases after a short overfit-style run.
  - save_pretrained / from_pretrained preserves logits.
  - generate() works.
  - cache split after training matches full forward.

Run from the HF implementation repository root, or pass --repo-root.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import torch
from torch.utils.data import DataLoader

try:
    from datasets import Dataset, load_dataset
except Exception as exc:  # pragma: no cover - environment dependent
    Dataset = None
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None

from transformers import AutoModelForCausalLM, AutoTokenizer


def _default_repo_root() -> Path:
    # script is usually copied to <repo>/scripts/p0_3_tinystories_stability.py
    return Path(__file__).resolve().parents[1]


def add_repo_to_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_steps_per_psi(value: str, psi_values: list[int]) -> dict[int, int]:
    """Parse "8:40,16:25" or a single integer for all Psi values."""

    value = str(value).strip()
    if value.isdigit():
        return {psi: int(value) for psi in psi_values}
    out: dict[int, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --steps-per-psi item {item!r}; use e.g. 8:40,16:25")
        k, v = item.split(":", 1)
        out[int(k)] = int(v)
    missing = [psi for psi in psi_values if psi not in out]
    if missing:
        raise ValueError(f"--steps-per-psi missing Psi values: {missing}")
    return out


def choose_text_column(dataset: Any, requested: str) -> str:
    columns = list(getattr(dataset, "column_names", []) or [])
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"text_column={requested!r} not found. Available columns: {columns}")
        return requested
    for name in ("text", "story", "content", "completion", "document"):
        if name in columns:
            return name
    if len(columns) == 1:
        return columns[0]
    raise ValueError(f"Could not infer text column. Available columns: {columns}")


def read_text_file(path: Path, *, max_texts: int) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Prefer blank-line separated stories.  Fall back to non-empty lines.
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
    return parts[:max_texts]


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        texts = read_text_file(Path(args.text_file).expanduser(), max_texts=args.max_texts)
        if not texts:
            raise RuntimeError(f"No texts loaded from --text-file={args.text_file}")
        return texts

    if load_dataset is None:
        raise RuntimeError(
            "datasets is not importable. Install datasets or pass --text-file. "
            f"Original import error: {_DATASETS_IMPORT_ERROR!r}"
        )
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.train_split,
        cache_dir=args.cache_dir,
        data_files=args.data_files,
        data_dir=args.data_dir,
        revision=args.revision,
    )
    if args.max_texts and args.max_texts > 0 and len(ds) > args.max_texts:
        ds = ds.select(range(args.max_texts))
    col = choose_text_column(ds, args.text_column)
    texts = ["" if row[col] is None else str(row[col]) for row in ds]
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise RuntimeError("Loaded dataset contains no non-empty texts")
    return texts


def cycle_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}[amp_dtype]
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype in {"fp16", "float16"}
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # pragma: no cover - older PyTorch fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def get_loss(model, batch: dict[str, torch.Tensor], *, device: torch.device, amp_dtype: str) -> torch.Tensor:
    with autocast_context(device, amp_dtype):
        out = model(**batch, return_dict=True)
        loss = out.loss
    if loss is None:
        raise RuntimeError("model returned loss=None")
    return loss


@torch.no_grad()
def evaluate_probe_loss(model, batch: dict[str, torch.Tensor], *, device: torch.device, amp_dtype: str) -> float:
    model.eval()
    loss = get_loss(model, batch, device=device, amp_dtype=amp_dtype)
    return float(loss.detach().float().cpu().item())


def create_multiscreen_config(
    *,
    psi: int,
    vocab_size: int,
    seq_len: int,
    key_dim: int,
    value_dim: int,
    mipe_threshold: float,
    initializer_range: float,
    compute_dtype: str,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
):
    from multiscreen_transformers import MultiscreenConfig

    return MultiscreenConfig(
        vocab_size=vocab_size,
        hidden_size=psi * psi,
        num_hidden_layers=psi,
        num_attention_heads=psi,
        key_dim=key_dim,
        value_dim=value_dim,
        max_position_embeddings=seq_len,
        mipe_threshold=mipe_threshold,
        initializer_range=initializer_range,
        use_cache=False,
        labels_are_shifted=False,
        mipe_compute_dtype=compute_dtype,
        softmask_compute_dtype=compute_dtype,
        strict_position_ids=True,
        strict_cache_positions=True,
        zero_pad_hidden_states=False,
        tie_word_embeddings=True,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )


def create_model(config):
    from multiscreen_transformers import MultiscreenForCausalLM

    return MultiscreenForCausalLM(config)


def make_optimizer(model: torch.nn.Module, *, lr: float, weight_decay: float, fused: bool) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if fused:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            print("[warn] torch.optim.AdamW(fused=True) unavailable; falling back to non-fused AdamW")
    return torch.optim.AdamW(model.parameters(), **kwargs)


@torch.no_grad()
def assert_save_load_logits(
    *,
    model: torch.nn.Module,
    tokenizer,
    save_dir: Path,
    probe_batch: dict[str, torch.Tensor],
    device: torch.device,
    amp_dtype: str,
    atol: float,
    rtol: float,
) -> float:
    from multiscreen_transformers import register_multiscreen_auto_classes

    register_multiscreen_auto_classes()
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(save_dir))

    model.eval()
    with autocast_context(device, amp_dtype):
        before = model(input_ids=probe_batch["input_ids"], attention_mask=probe_batch.get("attention_mask"), return_dict=True).logits

    loaded = AutoModelForCausalLM.from_pretrained(str(save_dir))
    loaded.to(device)
    loaded.eval()
    with autocast_context(device, amp_dtype):
        after = loaded(input_ids=probe_batch["input_ids"], attention_mask=probe_batch.get("attention_mask"), return_dict=True).logits

    max_abs = float((before.detach().float() - after.detach().float()).abs().max().cpu().item())
    torch.testing.assert_close(before.detach().float(), after.detach().float(), atol=atol, rtol=rtol)
    return max_abs


@torch.no_grad()
def assert_cache_split_after_training(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    device: torch.device,
    amp_dtype: str,
    atol: float,
    rtol: float,
) -> float:
    model.eval()
    # Keep this tiny; use_cache is the thing being tested here.
    t = int(input_ids.shape[1])
    if t < 4:
        raise ValueError("cache split check requires at least 4 tokens")
    t = min(t, 24)
    ids = input_ids[:, :t]
    mask = attention_mask[:, :t] if attention_mask is not None else None
    split = max(1, min(t - 1, t // 2))

    old_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    try:
        with autocast_context(device, amp_dtype):
            full = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            prefix = model(input_ids=ids[:, :split], attention_mask=mask[:, :split] if mask is not None else None, use_cache=True, return_dict=True)
            suffix = model(
                input_ids=ids[:, split:],
                attention_mask=mask,
                past_key_values=prefix.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        a = suffix.logits.detach().float()
        b = full.logits[:, split:, :].detach().float()
        max_abs = float((a - b).abs().max().cpu().item())
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        return max_abs
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache


@torch.no_grad()
def assert_generate(
    *,
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    model.eval()
    old_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    try:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if enc.input_ids.shape[1] == 0:
            enc = tokenizer("Once upon a time", return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if out.shape[1] <= enc["input_ids"].shape[1]:
            raise AssertionError("generate did not append new tokens")
        return {
            "prompt_len": int(enc["input_ids"].shape[1]),
            "generated_len": int(out.shape[1]),
            "sample_text": tokenizer.decode(out[0].detach().cpu().tolist(), skip_special_tokens=True)[:240],
        }
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache


def train_one_psi(
    *,
    psi: int,
    steps: int,
    args: argparse.Namespace,
    tokenizer,
    dataset,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    from multiscreen_transformers import register_multiscreen_auto_classes

    register_multiscreen_auto_classes()
    # Make initialization deterministic per Psi while keeping different Psi runs independent.
    seed = int(args.seed) + psi * 1009
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = create_multiscreen_config(
        psi=psi,
        vocab_size=len(tokenizer),
        seq_len=args.seq_len,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        mipe_threshold=args.mipe_threshold,
        initializer_range=args.initializer_range,
        compute_dtype=args.model_compute_dtype,
        pad_token_id=int(tokenizer.pad_token_id),
        bos_token_id=int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
    )
    model = create_model(cfg).to(device)
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n[P0-3] Psi={psi} params={param_count:,} steps={steps} device={device} amp={args.amp_dtype}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty; increase --max-train-tokens or lower --batch-size")
    probe_batch = move_batch(next(iter(loader)), device)
    train_iter = cycle_loader(loader)

    optimizer = make_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay, fused=args.fused_adamw and device.type == "cuda")
    scaler = make_grad_scaler(device, args.amp_dtype)

    initial_probe_loss = evaluate_probe_loss(model, probe_batch, device=device, amp_dtype=args.amp_dtype)
    if not math.isfinite(initial_probe_loss):
        raise RuntimeError(f"Initial probe loss is not finite for Psi={psi}: {initial_probe_loss}")

    losses: list[float] = []
    grad_norms: list[float] = []
    start_time = time.time()
    model.train()
    for step in range(1, steps + 1):
        if args.train_probe_every > 0 and step % args.train_probe_every == 0:
            # Deliberately revisit the probe batch every few steps.  P0-3 is a
            # stability/overfit smoke test, not a generalization benchmark, so
            # this makes the loss-drop check robust and deterministic.
            batch = probe_batch
        else:
            batch = move_batch(next(train_iter), device)
        optimizer.zero_grad(set_to_none=True)
        loss = get_loss(model, batch, device=device, amp_dtype=args.amp_dtype)
        loss_float = float(loss.detach().float().cpu().item())
        if not math.isfinite(loss_float):
            raise RuntimeError(f"Non-finite train loss for Psi={psi} step={step}: {loss_float}")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        grad_norm_float = float(grad_norm.detach().float().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        if not math.isfinite(grad_norm_float):
            raise RuntimeError(f"Non-finite grad norm for Psi={psi} step={step}: {grad_norm_float}")
        losses.append(loss_float)
        grad_norms.append(grad_norm_float)

        if step == 1 or step % args.log_every == 0 or step == steps:
            print(f"[P0-3][Psi={psi}] step={step:04d}/{steps} loss={loss_float:.4f} grad_norm={grad_norm_float:.4f}")

    elapsed = time.time() - start_time
    final_probe_loss = evaluate_probe_loss(model, probe_batch, device=device, amp_dtype=args.amp_dtype)
    if not math.isfinite(final_probe_loss):
        raise RuntimeError(f"Final probe loss is not finite for Psi={psi}: {final_probe_loss}")

    abs_drop = initial_probe_loss - final_probe_loss
    rel_drop = abs_drop / max(abs(initial_probe_loss), 1e-12)
    print(f"[P0-3][Psi={psi}] probe_loss initial={initial_probe_loss:.4f} final={final_probe_loss:.4f} drop={abs_drop:.4f} rel={rel_drop:.4%}")
    if not args.no_loss_drop_check:
        if abs_drop < args.min_loss_drop and rel_drop < args.min_rel_loss_drop:
            raise AssertionError(
                f"Probe loss did not decrease enough for Psi={psi}: "
                f"initial={initial_probe_loss:.6f}, final={final_probe_loss:.6f}, "
                f"drop={abs_drop:.6f}, rel={rel_drop:.6f}. "
                "Increase --steps-per-psi or lower --min-loss-drop for diagnostic runs."
            )

    psi_dir = output_dir / f"psi{psi}"
    save_load_max_abs = assert_save_load_logits(
        model=model,
        tokenizer=tokenizer,
        save_dir=psi_dir,
        probe_batch=probe_batch,
        device=device,
        amp_dtype=args.amp_dtype,
        atol=args.reload_atol,
        rtol=args.reload_rtol,
    )
    cache_max_abs = assert_cache_split_after_training(
        model=model,
        input_ids=probe_batch["input_ids"],
        attention_mask=probe_batch.get("attention_mask"),
        device=device,
        amp_dtype=args.amp_dtype,
        atol=args.cache_atol,
        rtol=args.cache_rtol,
    )
    gen_info = assert_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    metrics = {
        "psi": psi,
        "steps": steps,
        "params": param_count,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "initial_probe_loss": initial_probe_loss,
        "final_probe_loss": final_probe_loss,
        "abs_loss_drop": abs_drop,
        "rel_loss_drop": rel_drop,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_min": min(losses) if losses else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "elapsed_sec": elapsed,
        "tokens_per_step": args.batch_size * args.seq_len,
        "approx_tokens_seen": args.batch_size * args.seq_len * steps,
        "save_load_logits_max_abs": save_load_max_abs,
        "cache_split_logits_max_abs": cache_max_abs,
        "generation": gen_info,
        "checkpoint_dir": str(psi_dir),
    }
    (psi_dir / "p0_3_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_complete_note(output_dir: Path, all_metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# P0-3 TinyStories Smoke Training Result",
        "",
        "## Result",
        "",
        "Passed.",
        "",
        "## Confirmed behavior",
        "",
        "- TinyStories text was tokenized and packed.",
        "- Training loss stayed finite.",
        "- Gradients stayed finite.",
        "- Probe-batch loss decreased.",
        "- `save_pretrained` / `from_pretrained` preserved logits.",
        "- `generate()` worked with cache enabled.",
        "- Cached suffix logits matched full forward suffix logits after training.",
        "",
        "## Per-Psi metrics",
        "",
    ]
    for m in all_metrics:
        lines.extend([
            f"### Psi={m['psi']}",
            "",
            f"- params: {m['params']:,}",
            f"- steps: {m['steps']}",
            f"- seq_len: {m['seq_len']}",
            f"- batch_size: {m['batch_size']}",
            f"- amp_dtype: {m['amp_dtype']}",
            f"- initial_probe_loss: {m['initial_probe_loss']:.6f}",
            f"- final_probe_loss: {m['final_probe_loss']:.6f}",
            f"- abs_loss_drop: {m['abs_loss_drop']:.6f}",
            f"- rel_loss_drop: {m['rel_loss_drop']:.4%}",
            f"- save_load_logits_max_abs: {m['save_load_logits_max_abs']:.6g}",
            f"- cache_split_logits_max_abs: {m['cache_split_logits_max_abs']:.6g}",
            f"- checkpoint_dir: `{m['checkpoint_dir']}`",
            "",
        ])
    lines.extend([
        "## Scope",
        "",
        "This confirms short-run training stability and basic checkpoint/generation behavior.",
        "It does not confirm paper-scale performance, long-context retrieval, or runtime efficiency.",
        "",
    ])
    (output_dir / "P0-3_COMPLETE.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(_default_repo_root()), help="Root containing multiscreen_transformers/")
    p.add_argument("--tokenizer-path", default="tokenizers/tinystories_spm768")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dataset-name", default="roneneldan/TinyStories")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--train-split", default="train[:20000]")
    p.add_argument("--text-column", default="text")
    p.add_argument("--text-file", default=None, help="Optional local text fallback; still tokenized as TinyStories-style text.")
    p.add_argument("--data-files", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--max-texts", type=int, default=20000)
    p.add_argument("--max-train-tokens", type=int, default=262144, help="Cap packed tokens before chunking; lower for faster smoke tests.")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--psi", type=int, nargs="+", default=[8, 16])
    p.add_argument("--steps-per-psi", default="8:40,16:25", help="Either one int for all Psi or comma map, e.g. 8:40,16:25")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp-dtype", choices=["none", "bf16", "bfloat16", "fp16", "float16"], default="bf16" if torch.cuda.is_available() else "none")
    p.add_argument("--model-compute-dtype", choices=["fp32", "reference"], default="fp32", help="MiPE/Softmask auxiliary compute dtype in model config.")
    p.add_argument("--key-dim", type=int, default=16)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--mipe-threshold", type=float, default=256.0)
    p.add_argument("--initializer-range", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.0, help="P0 smoke defaults to 0 to isolate stability; TRL configs can still test wd=0.1.")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--fused-adamw", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--train-probe-every", type=int, default=4, help="Train on the probe batch every N steps for robust overfit-style loss-drop checks; 0 disables.")
    p.add_argument("--min-loss-drop", type=float, default=0.01)
    p.add_argument("--min-rel-loss-drop", type=float, default=0.001)
    p.add_argument("--no-loss-drop-check", action="store_true")
    p.add_argument("--reload-atol", type=float, default=1e-5)
    p.add_argument("--reload-rtol", type=float, default=1e-5)
    p.add_argument("--cache-atol", type=float, default=3e-2)
    p.add_argument("--cache-rtol", type=float, default=3e-2)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--output-dir", default="outputs/p0_3_tinystories_stability")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    add_repo_to_path(repo_root)

    from multiscreen_transformers import PackedTextDataset, register_multiscreen_auto_classes

    register_multiscreen_auto_classes()
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device}) but torch.cuda.is_available() is False")

    tokenizer_path = Path(args.tokenizer_path).expanduser()
    if not tokenizer_path.is_absolute():
        tokenizer_path = repo_root / tokenizer_path
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer path not found: {tokenizer_path}. Train/create the 768 TinyStories tokenizer first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer must have eos_token_id")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer must have pad_token_id")
    print(f"[P0-3] tokenizer={tokenizer.__class__.__name__} len={len(tokenizer)} path={tokenizer_path}")

    texts = load_texts(args)
    print(f"[P0-3] loaded texts={len(texts)} from {args.text_file or args.dataset_name}:{args.train_split}")
    dataset = PackedTextDataset(
        texts=texts,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        eos_token_id=tokenizer.eos_token_id,
        max_tokens=args.max_train_tokens,
        legacy_shifted_labels=True,
        return_labels_are_shifted=True,
    )
    print(f"[P0-3] packed chunks={len(dataset)} seq_len={args.seq_len} max_train_tokens={args.max_train_tokens}")
    if len(dataset) < args.batch_size:
        raise RuntimeError(f"Not enough chunks ({len(dataset)}) for batch_size={args.batch_size}")

    steps_per_psi = parse_steps_per_psi(args.steps_per_psi, args.psi)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    for psi in args.psi:
        metrics = train_one_psi(
            psi=psi,
            steps=steps_per_psi[psi],
            args=args,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)

    (output_dir / "p0_3_results.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_complete_note(output_dir, all_metrics)
    print("\nP0-3 TinyStories stability checks passed.")
    print(f"[P0-3] wrote metrics to {output_dir / 'p0_3_results.json'}")
    print(f"[P0-3] wrote note to {output_dir / 'P0-3_COMPLETE.md'}")


if __name__ == "__main__":
    main()
