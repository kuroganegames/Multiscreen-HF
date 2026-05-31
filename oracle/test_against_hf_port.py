"""Shape/cache/padding sweep tests against the current Hugging Face Multiscreen port.

Run this in an environment where the HF implementation is importable, e.g.:

    PYTHONPATH=/path/to/multiscreen_tinystories_sft:/path/to/multiscreen_oracle \
      python /path/to/multiscreen_oracle/test_against_hf_port.py

The test deliberately stays at tiny tensor sizes.  Its purpose is not speed; it is
P0 correctness: catch formula, mask, cache, transposition, and parameterization
drift between:

  - paper_math_oracle.PaperMultiscreenForCausalLM
  - multiscreen_transformers.modeling_multiscreen.MultiscreenForCausalLM

Notes:
  * The HF port parameterizes Trim with inv_r = exp(sr) + 1.
    The paper oracle parameterizes Trim with r = sigmoid(s_r).
    copy_from_hf_model(..., hf_uses_inverse_sr=True) maps s_r_paper = -sr_hf.
  * For this HF comparison only, the oracle uses
    position_rule="hf_mod_after_max_position" so that the HF port's long-position
    modulo branch is also tested.  Literal paper checks should use
    position_rule="paper".
  * The position/cache contract is intentionally strict: cached suffix calls must
    use start_pos == past_len, and no-cache full-context calls must start at 0.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from typing import Iterable, Optional

import torch

from paper_math_oracle import PaperMultiscreenConfig, PaperMultiscreenForCausalLM
from multiscreen_transformers.configuration_multiscreen import MultiscreenConfig
from multiscreen_transformers.modeling_multiscreen import MultiscreenForCausalLM


@dataclasses.dataclass(frozen=True)
class ShapeCase:
    name: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    key_dim: int
    value_dim: int
    max_position_embeddings: int
    mipe_threshold: float = 8.0


DEFAULT_SHAPE_CASES: tuple[ShapeCase, ...] = (
    ShapeCase(
        name="one_layer_one_head_min_key",
        vocab_size=23,
        hidden_size=6,
        num_hidden_layers=1,
        num_attention_heads=1,
        key_dim=2,
        value_dim=3,
        max_position_embeddings=16,
        mipe_threshold=8.0,
    ),
    ShapeCase(
        name="two_layers_two_heads_key_value_mismatch",
        vocab_size=29,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        key_dim=4,
        value_dim=3,
        max_position_embeddings=16,
        mipe_threshold=8.0,
    ),
    ShapeCase(
        name="three_layers_three_heads_non_square_dims",
        vocab_size=31,
        hidden_size=12,
        num_hidden_layers=3,
        num_attention_heads=3,
        key_dim=5,
        value_dim=4,
        max_position_embeddings=20,
        mipe_threshold=7.0,
    ),
    ShapeCase(
        name="four_heads_position_boundary",
        vocab_size=37,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        key_dim=3,
        value_dim=5,
        # Some sweeps intentionally exceed this to verify HF's modulo branch.
        max_position_embeddings=6,
        mipe_threshold=5.0,
    ),
)


DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda, cuda:0")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=sorted(DTYPES.keys()),
        help="Parameter/activation dtype. float32 is recommended for strict P0 checks.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller subset. Useful for rapid local sanity checks.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    if dtype == torch.float16:
        return 3e-2, 3e-2
    return 1e-5, 1e-5


def compute_mode_for_dtype(dtype: torch.dtype) -> str:
    # The oracle computes MiPE/Softmask in fp32 for bf16/fp16 inputs.  Match that
    # path in the HF port for non-fp32 sweeps.  For fp32, "reference" and "fp32"
    # are equivalent, and "reference" keeps the original tiny test semantics.
    return "reference" if dtype == torch.float32 else "fp32"


def build_hf_config(case: ShapeCase, *, dtype: torch.dtype) -> MultiscreenConfig:
    compute_mode = compute_mode_for_dtype(dtype)
    return MultiscreenConfig(
        vocab_size=case.vocab_size,
        hidden_size=case.hidden_size,
        num_hidden_layers=case.num_hidden_layers,
        num_attention_heads=case.num_attention_heads,
        key_dim=case.key_dim,
        value_dim=case.value_dim,
        max_position_embeddings=case.max_position_embeddings,
        mipe_threshold=case.mipe_threshold,
        use_cache=True,
        zero_pad_hidden_states=False,
        strict_position_ids=True,
        mipe_compute_dtype=compute_mode,
        softmask_compute_dtype=compute_mode,
    )


def build_oracle_config(hf_cfg: MultiscreenConfig) -> PaperMultiscreenConfig:
    return PaperMultiscreenConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        key_dim=hf_cfg.key_dim,
        value_dim=hf_cfg.value_dim,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        mipe_threshold=hf_cfg.mipe_threshold,
        # This test targets exact HF-port equivalence, including the HF long-
        # position modulo branch.  Use "paper" for literal paper-only tests.
        position_rule="hf_mod_after_max_position",
    )


def make_models(
    case: ShapeCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[MultiscreenForCausalLM, PaperMultiscreenForCausalLM, MultiscreenConfig]:
    torch.manual_seed(seed)
    hf_cfg = build_hf_config(case, dtype=dtype)
    hf = MultiscreenForCausalLM(hf_cfg).eval()
    oracle = PaperMultiscreenForCausalLM(build_oracle_config(hf_cfg)).eval()
    oracle.copy_from_hf_model(hf, hf_uses_inverse_sr=True)
    hf.to(device=device, dtype=dtype)
    oracle.to(device=device, dtype=dtype)
    return hf, oracle, hf_cfg


def randint_tensor(
    *,
    low: int,
    high: int,
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randint(low, high, shape, generator=gen, dtype=torch.long).to(device=device)


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    ctx: str,
    rtol: float,
    atol: float,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    except TypeError:  # Older PyTorch fallback; tests should avoid NaN except loss edge cases.
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        with torch.no_grad():
            a = actual.detach().float()
            e = expected.detach().float()
            max_abs = (a - e).abs().max().item() if a.numel() and e.numel() else float("nan")
            max_ref = e.abs().max().item() if e.numel() else float("nan")
        raise AssertionError(
            f"{name} mismatch in {ctx}; max_abs_diff={max_abs:.6g}, max_expected_abs={max_ref:.6g}\n{exc}"
        ) from exc


def assert_optional_loss_close(
    oracle_loss: Optional[torch.Tensor],
    hf_loss: Optional[torch.Tensor],
    *,
    ctx: str,
    rtol: float,
    atol: float,
) -> None:
    if oracle_loss is None and hf_loss is None:
        return
    if oracle_loss is None or hf_loss is None:
        raise AssertionError(f"loss presence mismatch in {ctx}: oracle={oracle_loss}, hf={hf_loss}")
    assert_close("loss", oracle_loss, hf_loss, ctx=ctx, rtol=rtol, atol=atol)


def assert_cache_close(
    oracle_cache: object,
    hf_cache: object,
    *,
    ctx: str,
    rtol: float,
    atol: float,
) -> None:
    if oracle_cache is None and hf_cache is None:
        return
    if oracle_cache is None or hf_cache is None:
        raise AssertionError(f"cache presence mismatch in {ctx}")
    if len(oracle_cache) != len(hf_cache):
        raise AssertionError(f"cache layer count mismatch in {ctx}: {len(oracle_cache)} vs {len(hf_cache)}")
    for layer_idx, ((ok, ov), (hk, hv)) in enumerate(zip(oracle_cache, hf_cache)):
        assert_close(f"cache[{layer_idx}].K", ok, hk, ctx=ctx, rtol=rtol, atol=atol)
        assert_close(f"cache[{layer_idx}].V", ov, hv, ctx=ctx, rtol=rtol, atol=atol)


def compare_forward(
    *,
    hf: MultiscreenForCausalLM,
    oracle: PaperMultiscreenForCausalLM,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    use_cache: bool,
    ctx: str,
    rtol: float,
    atol: float,
    labels_are_shifted: Optional[bool] = None,
    logits_to_keep: int = 0,
) -> tuple[object, object]:
    with torch.no_grad():
        hf_out = hf(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
            labels_are_shifted=labels_are_shifted,
            logits_to_keep=logits_to_keep,
        )
        oracle_out = oracle(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=use_cache,
            labels_are_shifted=labels_are_shifted,
            logits_to_keep=logits_to_keep,
        )
    assert_close("logits", oracle_out.logits, hf_out.logits, ctx=ctx, rtol=rtol, atol=atol)
    assert_optional_loss_close(oracle_out.loss, hf_out.loss, ctx=ctx, rtol=rtol, atol=atol)
    if use_cache:
        assert_cache_close(oracle_out.past_key_values, hf_out.past_key_values, ctx=ctx, rtol=rtol, atol=atol)
    return hf_out, oracle_out


def unique_ints(values: Iterable[int], *, min_value: int, max_value: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        value = int(value)
        if value < min_value or value > max_value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def run_shape_sweep(
    *,
    cases: tuple[ShapeCase, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
    quick: bool,
    verbose: bool,
) -> Counter:
    stats: Counter = Counter()
    batch_sizes = (1, 2) if quick else (1, 2, 3)
    seq_lens = (1, 2, 7) if quick else (1, 2, 3, 7, 10)

    for case_idx, case in enumerate(cases if not quick else cases[:2]):
        hf, oracle, hf_cfg = make_models(case, dtype=dtype, device=device, seed=seed + 1000 * case_idx)
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                input_ids = randint_tensor(
                    low=0,
                    high=hf_cfg.vocab_size,
                    shape=(batch_size, seq_len),
                    seed=seed + 11 * batch_size + 101 * seq_len + 10000 * case_idx,
                    device=device,
                )
                labels = randint_tensor(
                    low=0,
                    high=hf_cfg.vocab_size,
                    shape=(batch_size, seq_len),
                    seed=seed + 17 * batch_size + 103 * seq_len + 10000 * case_idx,
                    device=device,
                )
                ctx = f"shape_sweep case={case.name} B={batch_size} T={seq_len}"
                compare_forward(
                    hf=hf,
                    oracle=oracle,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=None,
                    use_cache=False,
                    ctx=ctx,
                    rtol=rtol,
                    atol=atol,
                )
                stats["shape_forward_loss"] += 1

                # Also verify the packed-label-style loss path.
                compare_forward(
                    hf=hf,
                    oracle=oracle,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=None,
                    use_cache=False,
                    ctx=ctx + " labels_are_shifted=True",
                    rtol=rtol,
                    atol=atol,
                    labels_are_shifted=True,
                )
                stats["shape_shifted_loss"] += 1

                # Verify logits_to_keep when no labels are provided.
                keep_values = unique_ints((1, 2, seq_len), min_value=1, max_value=seq_len)
                for keep in keep_values:
                    compare_forward(
                        hf=hf,
                        oracle=oracle,
                        input_ids=input_ids,
                        labels=None,
                        attention_mask=None,
                        use_cache=False,
                        ctx=ctx + f" logits_to_keep={keep}",
                        rtol=rtol,
                        atol=atol,
                        logits_to_keep=keep,
                    )
                    stats["shape_logits_to_keep"] += 1
                if verbose:
                    print(f"[ok] {ctx}")
    return stats


def run_cache_split_sweep(
    *,
    cases: tuple[ShapeCase, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
    quick: bool,
    verbose: bool,
) -> Counter:
    stats: Counter = Counter()
    batch_sizes = (1, 2) if not quick else (1,)
    seq_lens = (2, 3, 7, 10) if not quick else (3, 7)

    for case_idx, case in enumerate(cases if not quick else cases[:2]):
        hf, oracle, hf_cfg = make_models(case, dtype=dtype, device=device, seed=seed + 2000 * case_idx)
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                input_ids = randint_tensor(
                    low=0,
                    high=hf_cfg.vocab_size,
                    shape=(batch_size, seq_len),
                    seed=seed + 19 * batch_size + 109 * seq_len + 10000 * case_idx,
                    device=device,
                )
                labels = randint_tensor(
                    low=0,
                    high=hf_cfg.vocab_size,
                    shape=(batch_size, seq_len),
                    seed=seed + 23 * batch_size + 113 * seq_len + 10000 * case_idx,
                    device=device,
                )

                with torch.no_grad():
                    hf_full = hf(input_ids=input_ids, labels=labels, use_cache=False, return_dict=True)
                    oracle_full = oracle(input_ids, labels=labels, use_cache=False)
                ctx_full = f"cache_split full case={case.name} B={batch_size} T={seq_len}"
                assert_close("full logits", oracle_full.logits, hf_full.logits, ctx=ctx_full, rtol=rtol, atol=atol)
                assert_optional_loss_close(oracle_full.loss, hf_full.loss, ctx=ctx_full, rtol=rtol, atol=atol)

                split_points = unique_ints(
                    (1, seq_len // 2, seq_len - 1),
                    min_value=1,
                    max_value=seq_len - 1,
                )
                if not quick:
                    split_points = unique_ints(range(1, seq_len), min_value=1, max_value=seq_len - 1)

                for split in split_points:
                    prefix = input_ids[:, :split]
                    suffix = input_ids[:, split:]
                    ctx = f"cache_split case={case.name} B={batch_size} T={seq_len} split={split}"
                    with torch.no_grad():
                        hf_prefix = hf(input_ids=prefix, use_cache=True, return_dict=True)
                        oracle_prefix = oracle(prefix, use_cache=True)
                        hf_suffix = hf(
                            input_ids=suffix,
                            past_key_values=hf_prefix.past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                        oracle_suffix = oracle(
                            suffix,
                            past_key_values=oracle_prefix.past_key_values,
                            use_cache=True,
                        )

                    assert_close("prefix logits", oracle_prefix.logits, hf_prefix.logits, ctx=ctx, rtol=rtol, atol=atol)
                    assert_cache_close(oracle_prefix.past_key_values, hf_prefix.past_key_values, ctx=ctx + " prefix", rtol=rtol, atol=atol)
                    assert_close("suffix logits", oracle_suffix.logits, hf_suffix.logits, ctx=ctx, rtol=rtol, atol=atol)
                    assert_cache_close(oracle_suffix.past_key_values, hf_suffix.past_key_values, ctx=ctx + " suffix", rtol=rtol, atol=atol)
                    assert_close(
                        "cached suffix vs full suffix",
                        hf_suffix.logits,
                        hf_full.logits[:, split:, :],
                        ctx=ctx + " HF self-consistency",
                        rtol=rtol,
                        atol=atol,
                    )
                    assert_close(
                        "oracle cached suffix vs full suffix",
                        oracle_suffix.logits,
                        oracle_full.logits[:, split:, :],
                        ctx=ctx + " oracle self-consistency",
                        rtol=rtol,
                        atol=atol,
                    )
                    stats["cache_split"] += 1
                if verbose:
                    print(f"[ok] cache split case={case.name} B={batch_size} T={seq_len}")
    return stats


def make_attention_masks(batch_size: int, seq_len: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    ones = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    masks: dict[str, torch.Tensor] = {"all_ones": ones}
    if seq_len < 2:
        return masks

    # Right padding: each row keeps a different valid prefix, with at least two
    # valid tokens so shifted CE has a non-empty denominator.
    right = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for row in range(batch_size):
        valid = max(2, seq_len - (row % min(seq_len, 4)))
        right[row, :valid] = 1
    masks["right_padding"] = right

    # Left padding: each row keeps a different valid suffix.
    left = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for row in range(batch_size):
        valid = max(2, seq_len - ((row + 1) % min(seq_len, 4)))
        left[row, seq_len - valid :] = 1
    masks["left_padding"] = left

    # Non-contiguous mask.  This is not a typical packed-dataset pattern, but it
    # stress-tests that key/query masks are applied identically.
    if seq_len >= 4:
        sparse = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        sparse[:, 1::3] = 0
        sparse[:, -2:] = 1  # guarantee at least one valid shifted pair
        masks["sparse_noncontiguous"] = sparse

    return masks


def run_padding_mask_sweep(
    *,
    cases: tuple[ShapeCase, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
    quick: bool,
    verbose: bool,
) -> Counter:
    stats: Counter = Counter()
    batch_sizes = (1, 3) if not quick else (2,)
    seq_lens = (3, 7, 10) if not quick else (7,)

    for case_idx, case in enumerate(cases if not quick else cases[:2]):
        hf, oracle, hf_cfg = make_models(case, dtype=dtype, device=device, seed=seed + 3000 * case_idx)
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                input_ids = randint_tensor(
                    low=0,
                    high=hf_cfg.vocab_size,
                    shape=(batch_size, seq_len),
                    seed=seed + 29 * batch_size + 127 * seq_len + 10000 * case_idx,
                    device=device,
                )
                labels = randint_tensor(
                    low=0,
                    high=hf_cfg.vocab_size,
                    shape=(batch_size, seq_len),
                    seed=seed + 31 * batch_size + 131 * seq_len + 10000 * case_idx,
                    device=device,
                )
                masks = make_attention_masks(batch_size, seq_len, device=device)
                for mask_name, mask in masks.items():
                    ctx = f"padding_full case={case.name} B={batch_size} T={seq_len} mask={mask_name}"
                    with torch.no_grad():
                        hf_full = hf(
                            input_ids=input_ids,
                            attention_mask=mask,
                            labels=labels,
                            use_cache=False,
                            return_dict=True,
                        )
                        oracle_full = oracle(input_ids, attention_mask=mask, labels=labels, use_cache=False)
                    assert_close("full masked logits", oracle_full.logits, hf_full.logits, ctx=ctx, rtol=rtol, atol=atol)
                    assert_optional_loss_close(oracle_full.loss, hf_full.loss, ctx=ctx, rtol=rtol, atol=atol)
                    stats["padding_full"] += 1

                    split_points = unique_ints((1, seq_len // 2, seq_len - 1), min_value=1, max_value=seq_len - 1)
                    for split in split_points:
                        ctx_split = f"padding_cache case={case.name} B={batch_size} T={seq_len} split={split} mask={mask_name}"
                        with torch.no_grad():
                            hf_prefix = hf(
                                input_ids=input_ids[:, :split],
                                attention_mask=mask[:, :split],
                                use_cache=True,
                                return_dict=True,
                            )
                            oracle_prefix = oracle(
                                input_ids[:, :split],
                                attention_mask=mask[:, :split],
                                use_cache=True,
                            )
                            hf_suffix = hf(
                                input_ids=input_ids[:, split:],
                                attention_mask=mask,
                                past_key_values=hf_prefix.past_key_values,
                                use_cache=True,
                                return_dict=True,
                            )
                            oracle_suffix = oracle(
                                input_ids[:, split:],
                                attention_mask=mask,
                                past_key_values=oracle_prefix.past_key_values,
                                use_cache=True,
                            )
                        assert_close("masked prefix logits", oracle_prefix.logits, hf_prefix.logits, ctx=ctx_split, rtol=rtol, atol=atol)
                        assert_cache_close(oracle_prefix.past_key_values, hf_prefix.past_key_values, ctx=ctx_split + " prefix", rtol=rtol, atol=atol)
                        assert_close("masked suffix logits", oracle_suffix.logits, hf_suffix.logits, ctx=ctx_split, rtol=rtol, atol=atol)
                        assert_cache_close(oracle_suffix.past_key_values, hf_suffix.past_key_values, ctx=ctx_split + " suffix", rtol=rtol, atol=atol)
                        assert_close(
                            "masked cached suffix vs full suffix",
                            hf_suffix.logits,
                            hf_full.logits[:, split:, :],
                            ctx=ctx_split + " HF self-consistency",
                            rtol=rtol,
                            atol=atol,
                        )
                        assert_close(
                            "oracle masked cached suffix vs full suffix",
                            oracle_suffix.logits,
                            oracle_full.logits[:, split:, :],
                            ctx=ctx_split + " oracle self-consistency",
                            rtol=rtol,
                            atol=atol,
                        )
                        stats["padding_cache"] += 1
                if verbose:
                    print(f"[ok] padding case={case.name} B={batch_size} T={seq_len}")
    return stats


def expect_value_error(fn, *, ctx: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError in {ctx}")


def run_zero_relevance_sweep(
    *,
    cases: tuple[ShapeCase, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
    quick: bool,
    verbose: bool,
) -> Counter:
    stats: Counter = Counter()
    selected = cases[:1] if quick else cases[:2]
    for case_idx, case in enumerate(selected):
        hf, oracle, hf_cfg = make_models(case, dtype=dtype, device=device, seed=seed + 4000 * case_idx)
        batch_size, seq_len = (2, 5)
        input_ids = randint_tensor(
            low=0,
            high=hf_cfg.vocab_size,
            shape=(batch_size, seq_len),
            seed=seed + 40000 + case_idx,
            device=device,
        )
        zero_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        ctx = f"zero_relevance case={case.name} B={batch_size} T={seq_len}"
        with torch.no_grad():
            hf_out = hf(input_ids=input_ids, attention_mask=zero_mask, use_cache=False, return_dict=True)
            oracle_out = oracle(input_ids, attention_mask=zero_mask, use_cache=False)
        if not torch.isfinite(hf_out.logits).all():
            raise AssertionError(f"HF logits contain non-finite values in {ctx}")
        if not torch.isfinite(oracle_out.logits).all():
            raise AssertionError(f"oracle logits contain non-finite values in {ctx}")
        assert_close("zero-relevance logits", oracle_out.logits, hf_out.logits, ctx=ctx, rtol=rtol, atol=atol)
        stats["zero_relevance"] += 1
        if verbose:
            print(f"[ok] {ctx}")
    return stats


def run_position_contract_sweep(
    *,
    cases: tuple[ShapeCase, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
    quick: bool,
    verbose: bool,
) -> Counter:
    stats: Counter = Counter()
    selected = cases[:1] if quick else cases[:2]
    for case_idx, case in enumerate(selected):
        hf, oracle, hf_cfg = make_models(case, dtype=dtype, device=device, seed=seed + 5000 * case_idx)
        batch_size, seq_len = (2, 5)
        input_ids = randint_tensor(
            low=0,
            high=hf_cfg.vocab_size,
            shape=(batch_size, seq_len),
            seed=seed + 50000 + case_idx,
            device=device,
        )
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        ctx = f"position_ids_zero case={case.name}"
        with torch.no_grad():
            hf_pos = hf(input_ids=input_ids, position_ids=position_ids, use_cache=False, return_dict=True)
            hf_base = hf(input_ids=input_ids, use_cache=False, return_dict=True)
            oracle_base = oracle(input_ids, start_pos=0, use_cache=False)
        assert_close("HF position_ids arange vs no position_ids", hf_pos.logits, hf_base.logits, ctx=ctx, rtol=rtol, atol=atol)
        assert_close("oracle vs HF position_ids arange", oracle_base.logits, hf_pos.logits, ctx=ctx, rtol=rtol, atol=atol)
        stats["position_ids_zero"] += 1

        bad_position_ids = torch.arange(1, seq_len + 1, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        expect_value_error(
            lambda: hf(input_ids=input_ids, position_ids=bad_position_ids, use_cache=False, return_dict=True),
            ctx=f"HF offset position_ids no-cache case={case.name}",
        )
        expect_value_error(
            lambda: oracle(input_ids, start_pos=1, use_cache=False),
            ctx=f"oracle offset start_pos no-cache case={case.name}",
        )
        stats["position_contract_negative_no_cache"] += 1

        split = 3
        with torch.no_grad():
            hf_prefix = hf(input_ids=input_ids[:, :split], use_cache=True, return_dict=True)
            oracle_prefix = oracle(input_ids[:, :split], use_cache=True)
        expect_value_error(
            lambda: hf(
                input_ids=input_ids[:, split:],
                past_key_values=hf_prefix.past_key_values,
                start_pos=0,
                use_cache=True,
                return_dict=True,
            ),
            ctx=f"HF cached start_pos mismatch case={case.name}",
        )
        expect_value_error(
            lambda: oracle(
                input_ids[:, split:],
                past_key_values=oracle_prefix.past_key_values,
                start_pos=0,
                use_cache=True,
            ),
            ctx=f"oracle cached start_pos mismatch case={case.name}",
        )
        stats["position_contract_negative_cache"] += 1
        if verbose:
            print(f"[ok] position contract case={case.name}")
    return stats


def main() -> None:
    args = parse_args()
    dtype = DTYPES[args.dtype]
    device = torch.device(args.device)
    default_rtol, default_atol = default_tolerances(dtype)
    rtol = default_rtol if args.rtol is None else args.rtol
    atol = default_atol if args.atol is None else args.atol

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    print(
        "Running HF-port equivalence sweeps "
        f"device={device} dtype={dtype} rtol={rtol:g} atol={atol:g} quick={args.quick}"
    )
    cases = DEFAULT_SHAPE_CASES
    stats = Counter()
    stats.update(
        run_shape_sweep(
            cases=cases,
            dtype=dtype,
            device=device,
            seed=args.seed,
            rtol=rtol,
            atol=atol,
            quick=args.quick,
            verbose=args.verbose,
        )
    )
    stats.update(
        run_cache_split_sweep(
            cases=cases,
            dtype=dtype,
            device=device,
            seed=args.seed,
            rtol=rtol,
            atol=atol,
            quick=args.quick,
            verbose=args.verbose,
        )
    )
    stats.update(
        run_padding_mask_sweep(
            cases=cases,
            dtype=dtype,
            device=device,
            seed=args.seed,
            rtol=rtol,
            atol=atol,
            quick=args.quick,
            verbose=args.verbose,
        )
    )
    stats.update(
        run_zero_relevance_sweep(
            cases=cases,
            dtype=dtype,
            device=device,
            seed=args.seed,
            rtol=rtol,
            atol=atol,
            quick=args.quick,
            verbose=args.verbose,
        )
    )
    stats.update(
        run_position_contract_sweep(
            cases=cases,
            dtype=dtype,
            device=device,
            seed=args.seed,
            rtol=rtol,
            atol=atol,
            quick=args.quick,
            verbose=args.verbose,
        )
    )

    print("\nAll HF-port equivalence sweeps passed.")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
