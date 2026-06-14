#!/usr/bin/env python
"""P0-4 GPT-2 vocab + context-4096 smoke pretraining for Multiscreen.

This script extends the P0-3 stability harness to a GPT-2 tokenizer/vocab and
4096-token packed examples. It checks model construction, tokenizer loading,
packed dataset creation, seq_len forward/backward, finite loss/grad norm,
short-run probe-loss decrease, save/load logits, generate(use_cache=True),
manual cache split vs full-forward suffix logits, metrics.jsonl, and generation
of P0-4_COMPLETE.md.

It is a correctness/stability smoke for the current dense research
implementation, not a throughput benchmark or paper-scale validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - environment dependent
    load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


def repo_default() -> Path:
    return Path(__file__).resolve().parents[1]


def add_repo(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_p0_3_helpers(repo_root: Path):
    path = repo_root / "scripts" / "p0_3_tinystories_stability.py"
    spec = importlib.util.spec_from_file_location("_p0_3_tinystories_stability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load P0-3 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_int_csv(text: str) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    row = dict(row)
    row.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_text_file(path: Path, max_texts: int) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    texts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(texts) <= 1:
        texts = [p.strip() for p in raw.splitlines() if p.strip()]
    return texts[:max_texts] if max_texts > 0 else texts


def synthetic_texts(max_texts: int) -> list[str]:
    base = [
        "Once upon a time, a tiny robot learned to read long stories and remember where each chapter began.",
        "The scientist wrote careful notes, checked every number twice, and saved the results before going home.",
        "Maps, poems, recipes, and field reports were packed together into a single training stream.",
        "A small dragon counted stars above the village while its friends prepared breakfast for the festival.",
    ]
    n = max(max_texts if max_texts > 0 else 512, len(base))
    return [f"{base[i % len(base)]} Example {i}." for i in range(n)]


def choose_text_column(ds: Any, requested: str) -> str:
    columns = list(getattr(ds, "column_names", []) or [])
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"text_column={requested!r} not in columns {columns}")
        return requested
    for name in ("text", "story", "content", "completion", "document"):
        if name in columns:
            return name
    if len(columns) == 1:
        return columns[0]
    raise ValueError(f"Could not infer text column from columns {columns}")


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        texts = read_text_file(Path(args.text_file).expanduser(), args.max_texts)
    elif args.synthetic_text:
        texts = synthetic_texts(args.max_texts)
    else:
        if load_dataset is None:
            raise RuntimeError(
                "datasets is not importable. Install datasets, pass --text-file, or pass --synthetic-text. "
                f"Original import error: {DATASETS_IMPORT_ERROR!r}"
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
        if args.max_texts > 0 and len(ds) > args.max_texts:
            ds = ds.select(range(args.max_texts))
        col = choose_text_column(ds, args.text_column)
        texts = ["" if row[col] is None else str(row[col]) for row in ds]
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise RuntimeError("No non-empty texts loaded")
    return texts


def load_gpt2_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("P0-4 expects an EOS token; GPT-2 uses eos_token_id=50256")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(args.seq_len, args.postcheck_tokens, 32)
    return tokenizer


def make_packed_dataset(texts: list[str], tokenizer, args: argparse.Namespace):
    from multiscreen_transformers import PackedTextDataset

    repeated = list(texts)
    best = None
    last_error: Exception | None = None
    for _ in range(8):
        try:
            ds = PackedTextDataset(
                repeated,
                tokenizer,
                seq_len=args.seq_len,
                eos_token_id=tokenizer.eos_token_id,
                max_tokens=args.max_train_tokens,
                legacy_shifted_labels=True,
                return_labels_are_shifted=True,
            )
            best = ds
            if len(ds) >= args.min_packed_examples:
                return ds
            if args.max_train_tokens and args.max_train_tokens > 0:
                return ds
        except ValueError as exc:
            last_error = exc
        if args.no_repeat_texts:
            break
        repeated = repeated + repeated
    if best is not None:
        return best
    raise RuntimeError(f"Could not create one packed seq_len={args.seq_len} example") from last_error


def make_config(psi: int, tokenizer, args: argparse.Namespace):
    from multiscreen_transformers import MultiscreenConfig

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    return MultiscreenConfig.from_psi(
        psi,
        vocab_size=len(tokenizer),
        max_seq_len=args.seq_len,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        mipe_threshold=args.mipe_threshold,
        initializer_range=args.initializer_range,
        gradient_checkpointing=args.gradient_checkpointing,
        use_cache=False,
        labels_are_shifted=False,
        mipe_compute_dtype=args.model_compute_dtype,
        softmask_compute_dtype=args.model_compute_dtype,
        strict_position_ids=True,
        strict_cache_positions=True,
        zero_pad_hidden_states=False,
        tie_word_embeddings=True,
        pad_token_id=int(tokenizer.pad_token_id),
        bos_token_id=int(bos),
        eos_token_id=int(tokenizer.eos_token_id),
    )


def postcheck_batch(batch: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    n = min(int(batch["input_ids"].shape[1]), n)
    out = {"input_ids": batch["input_ids"][:, :n].contiguous()}
    if "attention_mask" in batch:
        out["attention_mask"] = batch["attention_mask"][:, :n].contiguous()
    return out


def cuda_metrics(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def train_one_psi(psi: int, steps: int, args: argparse.Namespace, helper, tokenizer, dataset, device: torch.device, output_dir: Path, metrics_path: Path) -> dict[str, Any]:
    from multiscreen_transformers import MultiscreenForCausalLM, register_multiscreen_auto_classes

    register_multiscreen_auto_classes()
    seed = args.seed + 1009 * psi
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MultiscreenForCausalLM(make_config(psi, tokenizer, args)).to(device)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n[P0-4] Psi={psi} params={param_count:,} steps={steps} seq_len={args.seq_len} microbatch={args.microbatch_size} grad_accum={args.grad_accum_steps} device={device} amp={args.amp_dtype}")

    loader = DataLoader(dataset, batch_size=args.microbatch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty; increase --max-train-tokens or lower --microbatch-size")
    probe_batch = helper.move_batch(next(iter(loader)), device)
    train_iter = cycle(loader)
    optimizer = helper.make_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay, fused=args.fused_adamw and device.type == "cuda")
    scaler = helper.make_grad_scaler(device, args.amp_dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    initial_probe_loss = helper.evaluate_probe_loss(model, probe_batch, device=device, amp_dtype=args.amp_dtype)
    if not math.isfinite(initial_probe_loss):
        raise RuntimeError(f"Initial probe loss is not finite for Psi={psi}: {initial_probe_loss}")

    losses: list[float] = []
    grad_norms: list[float] = []
    start_time = time.time()
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for micro in range(1, args.grad_accum_steps + 1):
            use_probe = args.train_probe_every > 0 and step % args.train_probe_every == 0 and micro == 1
            batch = probe_batch if use_probe else helper.move_batch(next(train_iter), device)
            loss = helper.get_loss(model, batch, device=device, amp_dtype=args.amp_dtype)
            loss_float = float(loss.detach().float().cpu().item())
            if not math.isfinite(loss_float):
                raise RuntimeError(f"Non-finite train loss for Psi={psi} step={step} micro={micro}: {loss_float}")
            micro_losses.append(loss_float)
            scaled_loss = loss / max(1, args.grad_accum_steps)
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        step_loss = sum(micro_losses) / len(micro_losses)
        grad_norm_float = float(grad_norm.detach().float().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        if not math.isfinite(grad_norm_float):
            raise RuntimeError(f"Non-finite grad norm for Psi={psi} step={step}: {grad_norm_float}")
        losses.append(step_loss)
        grad_norms.append(grad_norm_float)
        row = {"event": "train_step", "psi": psi, "step": step, "steps": steps, "train_loss": step_loss, "grad_norm": grad_norm_float, "elapsed_sec": time.time() - start_time}
        row.update(cuda_metrics(device))
        append_jsonl(metrics_path, row)
        if step == 1 or step % args.log_every == 0 or step == steps:
            print(f"[P0-4][Psi={psi}] step={step:04d}/{steps} loss={step_loss:.4f} grad_norm={grad_norm_float:.4f}")

    final_probe_loss = helper.evaluate_probe_loss(model, probe_batch, device=device, amp_dtype=args.amp_dtype)
    if not math.isfinite(final_probe_loss):
        raise RuntimeError(f"Final probe loss is not finite for Psi={psi}: {final_probe_loss}")
    abs_drop = initial_probe_loss - final_probe_loss
    rel_drop = abs_drop / max(abs(initial_probe_loss), 1e-12)
    if not args.no_loss_drop_check and abs_drop < args.min_loss_drop and rel_drop < args.min_rel_loss_drop:
        raise AssertionError(f"Probe loss did not decrease enough for Psi={psi}: initial={initial_probe_loss:.6f}, final={final_probe_loss:.6f}, drop={abs_drop:.6f}, rel={rel_drop:.6f}")

    psi_dir = output_dir / f"psi{psi}"
    check = postcheck_batch(probe_batch, args.postcheck_tokens)
    save_load_max_abs = helper.assert_save_load_logits(model=model, tokenizer=tokenizer, save_dir=psi_dir, probe_batch=check, device=device, amp_dtype=args.amp_dtype, atol=args.reload_atol, rtol=args.reload_rtol)
    cache_max_abs = helper.assert_cache_split_after_training(model=model, input_ids=check["input_ids"], attention_mask=check.get("attention_mask"), device=device, amp_dtype=args.amp_dtype, atol=args.cache_atol, rtol=args.cache_rtol)
    generation = helper.assert_generate(model=model, tokenizer=tokenizer, prompt=args.prompt, device=device, max_new_tokens=args.max_new_tokens)

    metrics = {
        "psi": psi,
        "steps": steps,
        "params": param_count,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "seq_len": args.seq_len,
        "microbatch_size": args.microbatch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_tokens": args.seq_len * args.microbatch_size * args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "initial_probe_loss": initial_probe_loss,
        "final_probe_loss": final_probe_loss,
        "abs_loss_drop": abs_drop,
        "rel_loss_drop": rel_drop,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_loss_min": min(losses),
        "grad_norm_max": max(grad_norms),
        "elapsed_sec": time.time() - start_time,
        "approx_tokens_seen": args.seq_len * args.microbatch_size * args.grad_accum_steps * steps,
        "save_load_logits_max_abs": save_load_max_abs,
        "cache_split_logits_max_abs": cache_max_abs,
        "generation": generation,
        "checkpoint_dir": str(psi_dir),
    }
    metrics.update(cuda_metrics(device))
    (psi_dir / "p0_4_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(metrics_path, {"event": "psi_result", **metrics})
    print(f"[P0-4][Psi={psi}] probe_loss initial={initial_probe_loss:.4f} final={final_probe_loss:.4f} drop={abs_drop:.4f} rel={rel_drop:.4%}")
    return metrics


def write_complete(output_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# P0-4 GPT-2 Context-4096 Smoke Result", "", "## Result", "", "Passed.", "",
        "## Confirmed behavior", "",
        "- GPT-2 tokenizer loaded and used for model vocab size.",
        "- Text was tokenized and packed into fixed context-length examples.",
        "- `seq_len` forward/backward training ran for the requested optimizer steps.",
        "- Training losses stayed finite.",
        "- Gradient norms stayed finite.",
        "- Probe-batch loss decreased.",
        "- `save_pretrained` / `from_pretrained` preserved logits on a post-check slice.",
        "- `generate(use_cache=True)` produced new tokens.",
        "- Manual prefix-cache suffix logits matched full-forward suffix logits on a post-check slice.",
        "", "## Per-Psi metrics", "",
    ]
    for m in results:
        lines.extend([
            f"### Psi={m['psi']}", "",
            f"- params: {m['params']:,}",
            f"- steps: {m['steps']}",
            f"- seq_len: {m['seq_len']}",
            f"- microbatch_size: {m['microbatch_size']}",
            f"- grad_accum_steps: {m['grad_accum_steps']}",
            f"- amp_dtype: {m['amp_dtype']}",
            f"- initial_probe_loss: {m['initial_probe_loss']:.6f}",
            f"- final_probe_loss: {m['final_probe_loss']:.6f}",
            f"- abs_loss_drop: {m['abs_loss_drop']:.6f}",
            f"- rel_loss_drop: {m['rel_loss_drop']:.4%}",
            f"- grad_norm_max: {m['grad_norm_max']:.6f}",
            f"- save_load_logits_max_abs: {m['save_load_logits_max_abs']:.6g}",
            f"- cache_split_logits_max_abs: {m['cache_split_logits_max_abs']:.6g}",
            f"- checkpoint_dir: `{m['checkpoint_dir']}`", "",
        ])
        if "cuda_peak_allocated_bytes" in m:
            lines.extend([
                f"- cuda_peak_allocated_gib: {m['cuda_peak_allocated_bytes'] / 2**30:.3f}",
                f"- cuda_peak_reserved_gib: {m['cuda_peak_reserved_bytes'] / 2**30:.3f}", "",
            ])
    lines.extend(["## Scope", "", "This confirms only a short P0 smoke path for a dense research implementation.", "It does not confirm paper-scale pretraining, long-context efficiency, production generation compatibility, or benchmark reproduction.", ""])
    (output_dir / "P0-4_COMPLETE.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(repo_default()))
    p.add_argument("--tokenizer-name-or-path", default="gpt2")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dataset-name", default="roneneldan/TinyStories")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--train-split", default="train")
    p.add_argument("--text-column", default="auto")
    p.add_argument("--text-file", default=None)
    p.add_argument("--synthetic-text", action="store_true")
    p.add_argument("--data-files", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--max-texts", type=int, default=4096)
    p.add_argument("--max-train-tokens", type=int, default=262144)
    p.add_argument("--min-packed-examples", type=int, default=8)
    p.add_argument("--no-repeat-texts", action="store_true")
    p.add_argument("--psi-values", default="8")
    p.add_argument("--steps-per-psi", default="8:50")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--amp-dtype", default="bf16", choices=["none", "bf16", "bfloat16", "fp16", "float16"])
    p.add_argument("--model-compute-dtype", default="fp32", choices=["fp32", "reference"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--key-dim", type=int, default=16)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--mipe-threshold", type=float, default=256.0)
    p.add_argument("--initializer-range", type=float, default=0.1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--fused-adamw", action="store_true")
    p.add_argument("--train-probe-every", type=int, default=4)
    p.add_argument("--min-loss-drop", type=float, default=1e-4)
    p.add_argument("--min-rel-loss-drop", type=float, default=1e-4)
    p.add_argument("--no-loss-drop-check", action="store_true")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--postcheck-tokens", type=int, default=128)
    p.add_argument("--reload-atol", type=float, default=1e-5)
    p.add_argument("--reload-rtol", type=float, default=1e-5)
    p.add_argument("--cache-atol", type=float, default=2e-2)
    p.add_argument("--cache-rtol", type=float, default=2e-2)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--output-dir", default="outputs/p0_4_gpt2_context4096_smoke")
    p.add_argument("--append-metrics", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    add_repo(repo_root)
    helper = load_p0_3_helpers(repo_root)
    if args.require_cuda and not str(args.device).startswith("cuda"):
        raise RuntimeError("--require-cuda was set but --device is not CUDA")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.append_metrics:
        metrics_path.unlink()

    psi_values = parse_int_csv(args.psi_values)
    steps_by_psi = helper.parse_steps_per_psi(args.steps_per_psi, psi_values)
    tokenizer = load_gpt2_tokenizer(args)
    texts = load_texts(args)
    dataset = make_packed_dataset(texts, tokenizer, args)
    run_info = {
        "event": "run_start",
        "repo_root": str(repo_root),
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "dataset_name": None if args.text_file or args.synthetic_text else args.dataset_name,
        "text_file": args.text_file,
        "synthetic_text": bool(args.synthetic_text),
        "num_texts": len(texts),
        "packed_examples": len(dataset),
        "seq_len": args.seq_len,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "psi_values": psi_values,
        "steps_per_psi": steps_by_psi,
    }
    print(json.dumps(run_info, ensure_ascii=False, indent=2))
    append_jsonl(metrics_path, run_info)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    results = [train_one_psi(psi, steps_by_psi[psi], args, helper, tokenizer, dataset, device, output_dir, metrics_path) for psi in psi_values]
    (output_dir / "p0_4_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_complete(output_dir, results)
    append_jsonl(metrics_path, {"event": "run_complete", "num_psi": len(results), "output_dir": str(output_dir)})
    print(f"[P0-4] complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
