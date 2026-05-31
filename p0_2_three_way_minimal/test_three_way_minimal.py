"""P0-2 minimal three-way comparison for Multiscreen implementations.

Compares, on tiny shapes:

  1. dieOD/multiscreen-pytorch original reference implementation
     package API: ``from multiscreen import MultiscreenConfig, MultiscreenModel``
  2. the local Hugging Face Transformers port
     package API: ``multiscreen_transformers``
  3. the local paper-math oracle
     module API: ``paper_math_oracle.PaperMultiscreenForCausalLM``

The test is deliberately small and exactness-oriented.  It does not test
training throughput, long-context behavior, attention masks, or paper-scale
quality.  It is the next step after P0-1: verify that the current HF port is
not merely oracle-consistent, but also still equivalent to the unofficial
PyTorch reference from which it was ported.

Expected command:

    python test_three_way_minimal.py \
      --reference-root /path/to/multiscreen-pytorch \
      --hf-root /path/to/multiscreen_tinystories_sft \
      --oracle-root /path/to/multiscreen_oracle

All paths are optional if the corresponding packages are already importable.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import sys
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class ShapeCase:
    name: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    key_dim: int
    value_dim: int
    max_seq_len: int
    mipe_threshold: float


DEFAULT_CASES: tuple[ShapeCase, ...] = (
    ShapeCase(
        name="minimal_one_layer_one_head",
        vocab_size=23,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        key_dim=2,
        value_dim=3,
        max_seq_len=12,
        mipe_threshold=8.0,
    ),
    ShapeCase(
        name="two_layers_two_heads_key_value_mismatch",
        vocab_size=29,
        hidden_dim=12,
        num_layers=2,
        num_heads=2,
        key_dim=4,
        value_dim=5,
        max_seq_len=16,
        mipe_threshold=8.0,
    ),
    ShapeCase(
        name="position_mod_boundary",
        vocab_size=31,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        key_dim=3,
        value_dim=4,
        max_seq_len=6,
        mipe_threshold=5.0,
    ),
)

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        default=None,
        help="Path to local dieOD/multiscreen-pytorch repo root. Optional if `multiscreen` is already importable.",
    )
    parser.add_argument(
        "--hf-root",
        default=None,
        help="Path to local HF port root containing `multiscreen_transformers`. Optional if already importable.",
    )
    parser.add_argument(
        "--oracle-root",
        default=here,
        help="Path containing `paper_math_oracle.py`. Defaults to this script's directory.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--quick", action="store_true", help="Run only the first two shape cases and sampled splits.")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--no-layer-hooks", action="store_true", help="Disable per-layer hidden-state hook comparisons.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def prepend_path(path: Optional[str]) -> None:
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def import_modules(args: argparse.Namespace) -> dict[str, Any]:
    prepend_path(args.reference_root)
    prepend_path(args.hf_root)
    prepend_path(args.oracle_root)

    try:
        ref_pkg = importlib.import_module("multiscreen")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Could not import original reference package `multiscreen`. "
            "Clone/install https://github.com/dieOD/multiscreen-pytorch or pass --reference-root."
        ) from exc
    try:
        hf_cfg_mod = importlib.import_module("multiscreen_transformers.configuration_multiscreen")
        hf_model_mod = importlib.import_module("multiscreen_transformers.modeling_multiscreen")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Could not import local HF port `multiscreen_transformers`. "
            "Pass --hf-root or add the HF implementation root to PYTHONPATH."
        ) from exc
    try:
        oracle_mod = importlib.import_module("paper_math_oracle")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Could not import `paper_math_oracle`. Pass --oracle-root or add the oracle directory to PYTHONPATH."
        ) from exc

    return {
        "RefConfig": getattr(ref_pkg, "MultiscreenConfig"),
        "RefModel": getattr(ref_pkg, "MultiscreenModel"),
        "HFConfig": getattr(hf_cfg_mod, "MultiscreenConfig"),
        "HFModel": getattr(hf_model_mod, "MultiscreenForCausalLM"),
        "convert_original_state_dict_for_causal_lm": getattr(
            hf_model_mod,
            "convert_original_state_dict_for_causal_lm",
        ),
        "PaperConfig": getattr(oracle_mod, "PaperMultiscreenConfig"),
        "PaperModel": getattr(oracle_mod, "PaperMultiscreenForCausalLM"),
    }


def default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype in (torch.bfloat16, torch.float16):
        return 3e-2, 3e-2
    return 1e-5, 1e-5


def compute_mode_for_dtype(dtype: torch.dtype) -> str:
    # Original reference computes MiPE/Softmask in incoming tensor dtype.  Match
    # that in the HF port for exact reference comparison.
    return "reference"


def make_ref_config(RefConfig: type, case: ShapeCase) -> Any:
    return RefConfig(
        vocab_size=case.vocab_size,
        hidden_dim=case.hidden_dim,
        num_layers=case.num_layers,
        num_heads=case.num_heads,
        key_dim=case.key_dim,
        value_dim=case.value_dim,
        max_seq_len=case.max_seq_len,
        mipe_threshold=case.mipe_threshold,
        gradient_checkpointing=False,
    )


def make_hf_config(HFConfig: type, case: ShapeCase, *, dtype: torch.dtype) -> Any:
    return HFConfig(
        vocab_size=case.vocab_size,
        hidden_size=case.hidden_dim,
        num_hidden_layers=case.num_layers,
        num_attention_heads=case.num_heads,
        key_dim=case.key_dim,
        value_dim=case.value_dim,
        max_position_embeddings=case.max_seq_len,
        mipe_threshold=case.mipe_threshold,
        use_cache=True,
        zero_pad_hidden_states=False,
        strict_position_ids=True,
        strict_cache_positions=True,
        mipe_compute_dtype=compute_mode_for_dtype(dtype),
        softmask_compute_dtype=compute_mode_for_dtype(dtype),
    )


def make_paper_config(PaperConfig: type, hf_config: Any) -> Any:
    return PaperConfig.from_hf_config(
        hf_config,
        # The original reference has the same post-max-position modulo branch as
        # the HF port.  For literal paper tests use the oracle default instead.
        position_rule="hf_mod_after_max_position",
        strict_cache_positions=True,
        # Match the original reference's low-precision MiPE/Softmask rounding.
        # Without this, bf16 full cases that exceed max_seq_len can differ in
        # cached K/V even though CPU fp32 and stable-oracle P0-1 tests pass.
        mipe_compute_dtype=getattr(hf_config, "mipe_compute_dtype", "reference"),
        softmask_compute_dtype=getattr(hf_config, "softmask_compute_dtype", "reference"),
    )


def build_three_models(
    modules: dict[str, Any],
    case: ShapeCase,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module, Any]:
    torch.manual_seed(seed)
    ref_cfg = make_ref_config(modules["RefConfig"], case)
    ref_model = modules["RefModel"](ref_cfg).eval()

    hf_cfg = make_hf_config(modules["HFConfig"], case, dtype=dtype)
    hf_model = modules["HFModel"](hf_cfg).eval()
    converted = modules["convert_original_state_dict_for_causal_lm"](ref_model.state_dict())
    missing, unexpected = hf_model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict conversion mismatch for case={case.name}: missing={missing}, unexpected={unexpected}"
        )

    paper_cfg = make_paper_config(modules["PaperConfig"], hf_cfg)
    oracle = modules["PaperModel"](paper_cfg).eval()
    oracle.copy_from_hf_model(hf_model, hf_uses_inverse_sr=True)

    ref_model.to(device=device, dtype=dtype)
    hf_model.to(device=device, dtype=dtype)
    oracle.to(device=device, dtype=dtype)
    return ref_model, hf_model, oracle, hf_cfg


def rand_ids(*, vocab_size: int, shape: tuple[int, int], seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randint(0, vocab_size, shape, generator=gen, dtype=torch.long).to(device)


def external_next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.new_zeros(())
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
    )


def assert_close(name: str, a: torch.Tensor, b: torch.Tensor, *, ctx: str, rtol: float, atol: float) -> None:
    try:
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)
    except TypeError:  # for older PyTorch
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    except AssertionError as exc:
        with torch.no_grad():
            aa = a.detach().float()
            bb = b.detach().float()
            max_abs = (aa - bb).abs().max().item() if aa.numel() and bb.numel() else float("nan")
            ref_abs = bb.abs().max().item() if bb.numel() else float("nan")
        raise AssertionError(
            f"{name} mismatch in {ctx}; max_abs_diff={max_abs:.6g}, max_ref_abs={ref_abs:.6g}\n{exc}"
        ) from exc


def assert_cache_close(
    name: str,
    cache_a: Sequence[tuple[torch.Tensor, torch.Tensor]],
    cache_b: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    ctx: str,
    rtol: float,
    atol: float,
) -> None:
    if len(cache_a) != len(cache_b):
        raise AssertionError(f"{name} layer-count mismatch in {ctx}: {len(cache_a)} vs {len(cache_b)}")
    for layer_idx, ((ak, av), (bk, bv)) in enumerate(zip(cache_a, cache_b)):
        assert_close(f"{name}[{layer_idx}].K", ak, bk, ctx=ctx, rtol=rtol, atol=atol)
        assert_close(f"{name}[{layer_idx}].V", av, bv, ctx=ctx, rtol=rtol, atol=atol)


@contextmanager
def capture_layer_outputs(model: nn.Module, layer_attr_path: str, *, enabled: bool):
    outputs: list[torch.Tensor] = []
    handles: list[Any] = []
    if enabled:
        root: Any = model
        for attr in layer_attr_path.split("."):
            root = getattr(root, attr)
        for layer in root:
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, _outputs: list[torch.Tensor] = outputs) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                _outputs.append(tensor.detach())
            handles.append(layer.register_forward_hook(hook))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def reference_forward(ref_model: nn.Module, input_ids: torch.Tensor, *, start_pos: int = 0, kv_caches: Any = None) -> tuple[torch.Tensor, Any]:
    # The original API uses kv_caches and always returns caches in eval mode.
    return ref_model(input_ids, start_pos=start_pos, kv_caches=kv_caches)


def compare_prefill(
    *,
    ref_model: nn.Module,
    hf_model: nn.Module,
    oracle: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    ctx: str,
    rtol: float,
    atol: float,
    check_layers: bool,
) -> tuple[Any, Any, Any]:
    with torch.no_grad():
        with capture_layer_outputs(ref_model, "layers", enabled=check_layers) as ref_layers:
            ref_logits, ref_cache = reference_forward(ref_model, input_ids)
        with capture_layer_outputs(hf_model, "multiscreen.layers", enabled=check_layers) as hf_layers:
            hf_out = hf_model(input_ids=input_ids, use_cache=True, return_dict=True)
        with capture_layer_outputs(oracle, "layers", enabled=check_layers) as oracle_layers:
            oracle_out = oracle(input_ids, use_cache=True)

    assert_close("ref vs hf logits", ref_logits, hf_out.logits, ctx=ctx, rtol=rtol, atol=atol)
    assert_close("ref vs oracle logits", ref_logits, oracle_out.logits, ctx=ctx, rtol=rtol, atol=atol)
    assert_close("hf vs oracle logits", hf_out.logits, oracle_out.logits, ctx=ctx, rtol=rtol, atol=atol)

    ref_loss = external_next_token_loss(ref_logits, labels)
    hf_loss = external_next_token_loss(hf_out.logits, labels)
    oracle_loss = external_next_token_loss(oracle_out.logits, labels)
    assert_close("external ref vs hf CE", ref_loss, hf_loss, ctx=ctx, rtol=rtol, atol=atol)
    assert_close("external ref vs oracle CE", ref_loss, oracle_loss, ctx=ctx, rtol=rtol, atol=atol)

    assert_cache_close("ref vs hf cache", ref_cache, hf_out.past_key_values, ctx=ctx, rtol=rtol, atol=atol)
    assert_cache_close("ref vs oracle cache", ref_cache, oracle_out.past_key_values, ctx=ctx, rtol=rtol, atol=atol)

    if check_layers:
        if not (len(ref_layers) == len(hf_layers) == len(oracle_layers)):
            raise AssertionError(
                f"layer hook count mismatch in {ctx}: ref={len(ref_layers)}, hf={len(hf_layers)}, oracle={len(oracle_layers)}"
            )
        for i, (r, h, o) in enumerate(zip(ref_layers, hf_layers, oracle_layers)):
            assert_close(f"layer[{i}] ref vs hf hidden", r, h, ctx=ctx, rtol=rtol, atol=atol)
            assert_close(f"layer[{i}] ref vs oracle hidden", r, o, ctx=ctx, rtol=rtol, atol=atol)

    return ref_cache, hf_out.past_key_values, oracle_out.past_key_values


def unique_splits(seq_len: int, *, quick: bool) -> list[int]:
    if seq_len <= 1:
        return []
    raw = (1, seq_len // 2, seq_len - 1) if quick else range(1, seq_len)
    out: list[int] = []
    seen: set[int] = set()
    for split in raw:
        split = int(split)
        if 1 <= split <= seq_len - 1 and split not in seen:
            seen.add(split)
            out.append(split)
    return out


def compare_cache_split(
    *,
    ref_model: nn.Module,
    hf_model: nn.Module,
    oracle: nn.Module,
    input_ids: torch.Tensor,
    full_ref_logits: torch.Tensor,
    full_hf_logits: torch.Tensor,
    full_oracle_logits: torch.Tensor,
    split: int,
    ctx: str,
    rtol: float,
    atol: float,
) -> None:
    prefix = input_ids[:, :split]
    suffix = input_ids[:, split:]
    with torch.no_grad():
        ref_prefix_logits, ref_prefix_cache = reference_forward(ref_model, prefix)
        ref_suffix_logits, ref_suffix_cache = reference_forward(
            ref_model,
            suffix,
            start_pos=split,
            kv_caches=ref_prefix_cache,
        )
        hf_prefix = hf_model(input_ids=prefix, use_cache=True, return_dict=True)
        hf_suffix = hf_model(
            input_ids=suffix,
            past_key_values=hf_prefix.past_key_values,
            start_pos=split,
            use_cache=True,
            return_dict=True,
        )
        oracle_prefix = oracle(prefix, use_cache=True)
        oracle_suffix = oracle(
            suffix,
            past_key_values=oracle_prefix.past_key_values,
            start_pos=split,
            use_cache=True,
        )

    assert_close("prefix ref vs hf logits", ref_prefix_logits, hf_prefix.logits, ctx=ctx, rtol=rtol, atol=atol)
    assert_close("prefix ref vs oracle logits", ref_prefix_logits, oracle_prefix.logits, ctx=ctx, rtol=rtol, atol=atol)
    assert_cache_close("prefix ref vs hf cache", ref_prefix_cache, hf_prefix.past_key_values, ctx=ctx, rtol=rtol, atol=atol)
    assert_cache_close("prefix ref vs oracle cache", ref_prefix_cache, oracle_prefix.past_key_values, ctx=ctx, rtol=rtol, atol=atol)

    assert_close("suffix ref vs hf logits", ref_suffix_logits, hf_suffix.logits, ctx=ctx, rtol=rtol, atol=atol)
    assert_close("suffix ref vs oracle logits", ref_suffix_logits, oracle_suffix.logits, ctx=ctx, rtol=rtol, atol=atol)
    assert_cache_close("suffix ref vs hf cache", ref_suffix_cache, hf_suffix.past_key_values, ctx=ctx, rtol=rtol, atol=atol)
    assert_cache_close("suffix ref vs oracle cache", ref_suffix_cache, oracle_suffix.past_key_values, ctx=ctx, rtol=rtol, atol=atol)

    assert_close("ref cached suffix vs full suffix", ref_suffix_logits, full_ref_logits[:, split:, :], ctx=ctx, rtol=rtol, atol=atol)
    assert_close("hf cached suffix vs full suffix", hf_suffix.logits, full_hf_logits[:, split:, :], ctx=ctx, rtol=rtol, atol=atol)
    assert_close("oracle cached suffix vs full suffix", oracle_suffix.logits, full_oracle_logits[:, split:, :], ctx=ctx, rtol=rtol, atol=atol)


def run_case(
    modules: dict[str, Any],
    case: ShapeCase,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    rtol: float,
    atol: float,
    quick: bool,
    check_layers: bool,
    verbose: bool,
) -> Counter:
    stats: Counter = Counter()
    ref_model, hf_model, oracle, hf_cfg = build_three_models(
        modules,
        case,
        seed=seed,
        dtype=dtype,
        device=device,
    )

    batch_sizes = (1, 2) if quick else (1, 2, 3)
    seq_lens = (2, 5, 8) if quick else (2, 3, 5, 8, case.max_seq_len + 2)
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            input_ids = rand_ids(
                vocab_size=case.vocab_size,
                shape=(batch_size, seq_len),
                seed=seed + 100 * batch_size + 7 * seq_len,
                device=device,
            )
            labels = rand_ids(
                vocab_size=case.vocab_size,
                shape=(batch_size, seq_len),
                seed=seed + 101 * batch_size + 11 * seq_len,
                device=device,
            )
            ctx = f"case={case.name} B={batch_size} T={seq_len}"
            ref_cache, hf_cache, oracle_cache = compare_prefill(
                ref_model=ref_model,
                hf_model=hf_model,
                oracle=oracle,
                input_ids=input_ids,
                labels=labels,
                ctx=ctx,
                rtol=rtol,
                atol=atol,
                check_layers=check_layers,
            )
            stats["prefill_three_way"] += 1

            with torch.no_grad():
                full_ref_logits, _ = reference_forward(ref_model, input_ids)
                full_hf = hf_model(input_ids=input_ids, use_cache=True, return_dict=True)
                full_oracle = oracle(input_ids, use_cache=True)
            for split in unique_splits(seq_len, quick=quick):
                compare_cache_split(
                    ref_model=ref_model,
                    hf_model=hf_model,
                    oracle=oracle,
                    input_ids=input_ids,
                    full_ref_logits=full_ref_logits,
                    full_hf_logits=full_hf.logits,
                    full_oracle_logits=full_oracle.logits,
                    split=split,
                    ctx=ctx + f" split={split}",
                    rtol=rtol,
                    atol=atol,
                )
                stats["cache_split_three_way"] += 1
            if verbose:
                print(f"[ok] {ctx}")
    return stats


def main() -> None:
    args = parse_args()
    dtype = DTYPES[args.dtype]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    rtol_default, atol_default = default_tolerances(dtype)
    rtol = rtol_default if args.rtol is None else args.rtol
    atol = atol_default if args.atol is None else args.atol

    modules = import_modules(args)
    cases = DEFAULT_CASES[:2] if args.quick else DEFAULT_CASES
    stats: Counter = Counter()

    print(
        "Running P0-2 three-way minimal comparison "
        f"device={device} dtype={dtype} rtol={rtol:g} atol={atol:g} quick={args.quick} "
        f"layer_hooks={not args.no_layer_hooks}"
    )
    for idx, case in enumerate(cases):
        stats.update(
            run_case(
                modules,
                case,
                seed=args.seed + 1000 * idx,
                dtype=dtype,
                device=device,
                rtol=rtol,
                atol=atol,
                quick=args.quick,
                check_layers=not args.no_layer_hooks,
                verbose=args.verbose,
            )
        )

    print("\nAll P0-2 three-way minimal comparisons passed.")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
