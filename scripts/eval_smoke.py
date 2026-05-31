#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Allow imports of local custom architectures such as `multiscreen_transformers`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from multiscreen_transformers import register_multiscreen_auto_classes
except ImportError:
    register_multiscreen_auto_classes = None

from cache_utils import apply_hf_cache_env, make_cache_paths


def choose_text_column(dataset, requested: str) -> str:
    columns = list(getattr(dataset, "column_names", []) or [])
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"text_column={requested!r} not found. Available columns: {columns}")
        return requested
    for name in ("text", "story", "content"):
        if name in columns:
            return name
    return columns[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--dataset_name", default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--split", default="validation[:512]")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    cache_paths = make_cache_paths(None, cache_dir=args.cache_dir)
    apply_hf_cache_env(cache_paths)
    if register_multiscreen_auto_classes is not None:
        register_multiscreen_auto_classes()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, cache_dir=str(cache_paths.tokenizer_cache_dir) if cache_paths.tokenizer_cache_dir else None)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, cache_dir=str(cache_paths.model_cache_dir) if cache_paths.model_cache_dir else None)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split, cache_dir=str(cache_paths.datasets_cache_dir) if cache_paths.datasets_cache_dir else None)
    col = choose_text_column(ds, args.text_column)

    losses = []
    with torch.no_grad():
        for row in ds:
            text = str(row[col])
            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            if enc.input_ids.shape[1] < 2:
                continue
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            losses.append(float(out.loss.detach().cpu()))
    mean_loss = sum(losses) / max(len(losses), 1)
    print({"n": len(losses), "loss": mean_loss, "ppl": math.exp(mean_loss) if mean_loss < 20 else float("inf")})


if __name__ == "__main__":
    main()
