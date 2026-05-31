#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, CONFIG_MAPPING


# Allow imports of local custom architectures such as `multiscreen_transformers`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from multiscreen_transformers import register_multiscreen_auto_classes
except ImportError:
    register_multiscreen_auto_classes = None

from cache_utils import apply_hf_cache_env, make_cache_paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    cache_paths = make_cache_paths(None, cache_dir=args.cache_dir)
    apply_hf_cache_env(cache_paths)
    if register_multiscreen_auto_classes is not None:
        register_multiscreen_auto_classes()

    cfg_path = Path(args.config_path)
    config = AutoConfig.from_pretrained(str(cfg_path), trust_remote_code=args.trust_remote_code, cache_dir=str(cache_paths.model_cache_dir) if cache_paths.model_cache_dir else None)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({"class": model.__class__.__name__, "params": total, "trainable": trainable, "vocab_size": model.config.vocab_size})


if __name__ == "__main__":
    main()
