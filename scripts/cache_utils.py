from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CACHE_KEYS = (
    "cache_dir",
    "hf_home",
    "hub_cache_dir",
    "datasets_cache_dir",
    "model_cache_dir",
    "tokenizer_cache_dir",
    "modules_cache_dir",
    "assets_cache_dir",
)


@dataclass(frozen=True)
class CachePaths:
    cache_dir: Path | None = None
    hf_home: Path | None = None
    hub_cache_dir: Path | None = None
    datasets_cache_dir: Path | None = None
    model_cache_dir: Path | None = None
    tokenizer_cache_dir: Path | None = None
    modules_cache_dir: Path | None = None
    assets_cache_dir: Path | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {k: (str(getattr(self, k)) if getattr(self, k) is not None else None) for k in CACHE_KEYS}


def _as_path(value: str | os.PathLike[str] | None, *, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    p = Path(s).expanduser()
    if p.is_absolute():
        return p
    if base is not None:
        return base / p
    return Path.cwd() / p


def make_cache_paths(config: Mapping[str, Any] | None = None, **overrides: Any) -> CachePaths:
    """Create consistent HF cache paths.

    Priority:
      1. CLI overrides passed as keyword arguments when non-null
      2. config[...]
      3. defaults derived from cache_dir

    If cache_dir=/x/cache, the defaults are:
      HF_HOME=/x/cache
      HF_HUB_CACHE=/x/cache/hub
      HF_DATASETS_CACHE=/x/cache/datasets
      model/tokenizer cache_dir=/x/cache/models
      HF_MODULES_CACHE=/x/cache/modules
      HF_ASSETS_CACHE=/x/cache/assets
    """
    raw: dict[str, Any] = {k: None for k in CACHE_KEYS}
    if config:
        for k in CACHE_KEYS:
            raw[k] = config.get(k)
    for k, v in overrides.items():
        if k in CACHE_KEYS and v not in (None, "", "none", "null"):
            raw[k] = v

    base = _as_path(raw.get("cache_dir"))

    hf_home = _as_path(raw.get("hf_home"), base=base) or base
    hub = _as_path(raw.get("hub_cache_dir"), base=base) or (base / "hub" if base else None)
    datasets = _as_path(raw.get("datasets_cache_dir"), base=base) or (base / "datasets" if base else None)
    models = _as_path(raw.get("model_cache_dir"), base=base) or (base / "models" if base else None)
    tokenizers = _as_path(raw.get("tokenizer_cache_dir"), base=base) or models
    modules = _as_path(raw.get("modules_cache_dir"), base=base) or (base / "modules" if base else None)
    assets = _as_path(raw.get("assets_cache_dir"), base=base) or (base / "assets" if base else None)

    return CachePaths(
        cache_dir=base,
        hf_home=hf_home,
        hub_cache_dir=hub,
        datasets_cache_dir=datasets,
        model_cache_dir=models,
        tokenizer_cache_dir=tokenizers,
        modules_cache_dir=modules,
        assets_cache_dir=assets,
    )


def apply_hf_cache_env(paths: CachePaths, *, create_dirs: bool = True, verbose: bool = True) -> None:
    mapping = {
        "HF_HOME": paths.hf_home,
        "HF_HUB_CACHE": paths.hub_cache_dir,
        "HF_DATASETS_CACHE": paths.datasets_cache_dir,
        "HF_MODULES_CACHE": paths.modules_cache_dir,
        "HF_ASSETS_CACHE": paths.assets_cache_dir,
        # Kept for older Transformers versions. Newer versions prefer HF_HOME/HF_HUB_CACHE.
        "TRANSFORMERS_CACHE": paths.model_cache_dir,
    }
    for p in mapping.values():
        if create_dirs and p is not None:
            p.mkdir(parents=True, exist_ok=True)
    for key, p in mapping.items():
        if p is not None:
            os.environ[key] = str(p)

    if verbose:
        print("[info] Hugging Face cache directories:")
        for k, v in paths.as_dict().items():
            print(f"[info]   {k}={v}")
