#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
from dataclasses import fields
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, CONFIG_MAPPING
from trl import SFTConfig, SFTTrainer

# Allow running from the repository root with `python scripts/train_pretrain_sft.py`
# while importing local custom architectures such as `multiscreen_transformers`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cache_utils import apply_hf_cache_env, make_cache_paths

FLASH_ATTN_NAMES = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn",
    "kernels-community/flash-attn3",
    "kernels-community/vllm-flash-attn3",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: str | None, *, base_file: str | Path) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    # Prefer repository cwd for paths like configs/... or tokenizers/...
    candidate = Path.cwd() / p
    if candidate.exists() or str(path).startswith(("configs/", "tokenizers/", "outputs/", "custom_arch_examples/")):
        return candidate
    return Path(base_file).resolve().parent / p


def choose_text_column(dataset, requested: str) -> str:
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


def normalize_text_dataset(dataset, *, text_column: str, num_proc: int | None):
    def convert(batch):
        return {"text": ["" if v is None else str(v) for v in batch[text_column]]}

    remove_columns = list(getattr(dataset, "column_names", []) or [])
    kwargs: dict[str, Any] = {"batched": True, "remove_columns": remove_columns}
    if num_proc:
        kwargs["num_proc"] = int(num_proc)
    return dataset.map(convert, **kwargs)


def register_local_custom_architectures() -> None:
    """Register local custom Transformers architectures that ship in this repo.

    At the moment this only registers Multiscreen. Keeping this function tiny
    lets the same training script continue to work for built-in architectures
    such as Llama while enabling `model_type=multiscreen` JSON configs.
    """

    try:
        from multiscreen_transformers import register_multiscreen_auto_classes
    except ImportError:
        return
    register_multiscreen_auto_classes()


def dataset_load_kwargs(ds_cfg: dict[str, Any], *, split_key: str, cache_dir: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "split": ds_cfg.get(split_key),
        "cache_dir": cache_dir,
        "data_files": ds_cfg.get("data_files"),
        "data_dir": ds_cfg.get("data_dir"),
        "revision": ds_cfg.get("revision"),
        "streaming": ds_cfg.get("streaming"),
    }
    return {k: v for k, v in kwargs.items() if v is not None}


def supported_sft_config_keys() -> set[str]:
    """Return SFTConfig constructor keys for the installed TRL version."""

    try:
        return {f.name for f in fields(SFTConfig)}
    except TypeError:
        sig = inspect.signature(SFTConfig.__init__)
        return {k for k in sig.parameters if k != "self"}


def normalize_sft_config_kwargs(sft: dict[str, Any]) -> dict[str, Any]:
    """Drop/translate run-config keys unsupported by the installed TRL version.

    The sample configs keep a few conservative knobs such as ``use_cache`` near
    the SFT settings for readability, while model cache behavior is controlled
    directly through ``model.config.use_cache`` above.  Filtering here makes the
    script robust across small TRL/Transformers argument-name changes.
    """

    normalized = dict(sft)
    supported = supported_sft_config_keys()

    if "eval_strategy" in normalized and "eval_strategy" not in supported and "evaluation_strategy" in supported:
        normalized["evaluation_strategy"] = normalized.pop("eval_strategy")
    if "evaluation_strategy" in normalized and "evaluation_strategy" not in supported and "eval_strategy" in supported:
        normalized["eval_strategy"] = normalized.pop("evaluation_strategy")

    ignored: dict[str, Any] = {}
    for key in list(normalized):
        if key not in supported:
            ignored[key] = normalized.pop(key)
    if ignored:
        print(f"[info] ignored unsupported SFTConfig keys for this TRL version: {sorted(ignored)}")
    return normalized


def load_config(config_path: Path, *, trust_remote_code: bool, cache_dir: str | None):
    if config_path.is_dir():
        return AutoConfig.from_pretrained(str(config_path), trust_remote_code=trust_remote_code, cache_dir=cache_dir)
    cfg = load_json(config_path)
    if "auto_map" in cfg:
        raise ValueError("Custom configs with auto_map must be loaded from a directory, not a single JSON file.")
    model_type = cfg.get("model_type")
    if not model_type or model_type not in CONFIG_MAPPING:
        raise ValueError(
            f"Unknown model_type={model_type!r}. For Multiscreen, run this script from the "
            "repository root so `multiscreen_transformers` can be imported and registered."
        )
    cfg_kwargs = dict(cfg)
    cfg_kwargs.pop("model_type", None)
    return CONFIG_MAPPING[model_type](**cfg_kwargs)


def validate_tokenizer(tokenizer, *, expected_vocab_size: int) -> None:
    actual = len(tokenizer)
    print(f"[check] tokenizer class={tokenizer.__class__.__name__}")
    print(f"[check] tokenizer len={actual}, vocab_size={getattr(tokenizer, 'vocab_size', None)}")
    print(
        "[check] special ids:",
        {"unk": tokenizer.unk_token_id, "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
    )
    if actual != expected_vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab mismatch: expected={expected_vocab_size}, actual={actual}. "
            "Do not train; recreate tokenizer first."
        )
    sample = "Once upon a time, Timmy went to the park."
    ids = tokenizer.encode(sample, add_special_tokens=False)
    print(f"[check] sample token ids: {ids[:60]}")
    if not ids or max(ids) <= 3 or len(set(ids)) < 4:
        raise RuntimeError("Tokenizer seems broken: normal text maps mostly to special/unk IDs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--hf_home", default=None)
    parser.add_argument("--hub_cache_dir", default=None)
    parser.add_argument("--datasets_cache_dir", default=None)
    parser.add_argument("--model_cache_dir", default=None)
    parser.add_argument("--tokenizer_cache_dir", default=None)
    parser.add_argument("--modules_cache_dir", default=None)
    parser.add_argument("--assets_cache_dir", default=None)
    parser.add_argument("--allow_packing_without_flash", action="store_true")
    args = parser.parse_args()

    run = load_json(args.run_config)
    cache_paths = make_cache_paths(
        run.get("cache"),
        cache_dir=args.cache_dir,
        hf_home=args.hf_home,
        hub_cache_dir=args.hub_cache_dir,
        datasets_cache_dir=args.datasets_cache_dir,
        model_cache_dir=args.model_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        modules_cache_dir=args.modules_cache_dir,
        assets_cache_dir=args.assets_cache_dir,
    )
    apply_hf_cache_env(cache_paths)
    register_local_custom_architectures()

    trust_remote_code = bool(run.get("model", {}).get("trust_remote_code", False))
    tokenizer_path = resolve_path(run["tokenizer"]["path"], base_file=args.run_config)
    if tokenizer_path is None:
        raise ValueError("tokenizer.path is required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=bool(run["tokenizer"].get("use_fast", True)),
        cache_dir=str(cache_paths.tokenizer_cache_dir) if cache_paths.tokenizer_cache_dir else None,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = run["tokenizer"].get("padding_side", "right")

    expected_vocab_size = int(run.get("expected_vocab_size", run.get("model", {}).get("expected_vocab_size", 768)))
    validate_tokenizer(tokenizer, expected_vocab_size=expected_vocab_size)

    model_cfg_path = resolve_path(run["model"]["config_path"], base_file=args.run_config)
    if model_cfg_path is None:
        raise ValueError("model.config_path is required")
    config = load_config(
        model_cfg_path,
        trust_remote_code=trust_remote_code,
        cache_dir=str(cache_paths.model_cache_dir) if cache_paths.model_cache_dir else None,
    )

    if int(config.vocab_size) != expected_vocab_size:
        raise RuntimeError(f"Model config vocab_size={config.vocab_size}, expected={expected_vocab_size}.")
    if int(config.vocab_size) != len(tokenizer):
        raise RuntimeError(
            f"Refusing to override config.vocab_size {config.vocab_size} -> {len(tokenizer)}. "
            "Fix tokenizer/config mismatch."
        )
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.use_cache = False
    if "tie_word_embeddings" in run["model"]:
        config.tie_word_embeddings = bool(run["model"]["tie_word_embeddings"])

    model_kwargs: dict[str, Any] = {}
    attn_impl = run["model"].get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    if run["model"].get("from_pretrained"):
        model = AutoModelForCausalLM.from_pretrained(
            run["model"]["pretrained_model_name_or_path"],
            config=config,
            cache_dir=str(cache_paths.model_cache_dir) if cache_paths.model_cache_dir else None,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code, **model_kwargs)
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] model_class={model.__class__.__name__}, config_class={config.__class__.__name__}")
    print(f"[info] params={total:,}, trainable={trainable:,}, vocab={model.config.vocab_size}")

    ds_cfg = run["dataset"]
    train_kwargs = dataset_load_kwargs(
        ds_cfg,
        split_key="train_split",
        cache_dir=str(cache_paths.datasets_cache_dir) if cache_paths.datasets_cache_dir else None,
    )
    if "split" not in train_kwargs:
        train_kwargs["split"] = "train"
    train_ds = load_dataset(ds_cfg["name"], ds_cfg.get("config_name"), **train_kwargs)
    train_col = choose_text_column(train_ds, ds_cfg.get("text_column", "text"))
    train_ds = normalize_text_dataset(train_ds, text_column=train_col, num_proc=ds_cfg.get("num_proc"))
    if ds_cfg.get("shuffle", True):
        train_ds = train_ds.shuffle(seed=int(ds_cfg.get("seed", 42)))

    eval_ds = None
    if ds_cfg.get("eval_split"):
        eval_kwargs = dataset_load_kwargs(
            ds_cfg,
            split_key="eval_split",
            cache_dir=str(cache_paths.datasets_cache_dir) if cache_paths.datasets_cache_dir else None,
        )
        eval_ds = load_dataset(ds_cfg["name"], ds_cfg.get("config_name"), **eval_kwargs)
        eval_col = choose_text_column(eval_ds, ds_cfg.get("text_column", "text"))
        eval_ds = normalize_text_dataset(eval_ds, text_column=eval_col, num_proc=ds_cfg.get("num_proc"))

    sft = dict(run["sft_config"])
    sft["output_dir"] = str(resolve_path(sft["output_dir"], base_file=args.run_config) or Path(sft["output_dir"]))
    sft = normalize_sft_config_kwargs(sft)
    # Avoid deprecated warmup_ratio in newer Transformers if warmup_steps is provided.
    if "warmup_steps" in sft and "warmup_ratio" in sft:
        sft.pop("warmup_ratio")

    packing = bool(sft.get("packing", False))
    if packing and attn_impl not in FLASH_ATTN_NAMES and not args.allow_packing_without_flash:
        raise RuntimeError(
            "packing=True with non-flash attention can cause cross-sample contamination in TRL. "
            "Set packing=false or use a supported flash attention implementation. "
            "Override with --allow_packing_without_flash only for custom attention you verified."
        )

    training_args = SFTConfig(**sft)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() in {"", "none", "null", "false"}:
        resume = None
    elif isinstance(resume, str) and resume.lower() == "true":
        resume = True

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    Path(training_args.output_dir, "run_config.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] saved final model/tokenizer to {training_args.output_dir}")


if __name__ == "__main__":
    main()
