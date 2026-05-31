# Multiscreen `paper_math_oracle`

This is a deliberately explicit PyTorch implementation of the Multiscreen paper equations.
It is intended for tiny-shape correctness tests against the Hugging Face port, not for training speed.

## Files

- `paper_math_oracle.py` - standalone oracle implementation.
- `test_paper_math_oracle_smoke.py` - CPU smoke tests that only require PyTorch.

## Basic usage

```python
import torch
from paper_math_oracle import PaperMultiscreenConfig, PaperMultiscreenForCausalLM

cfg = PaperMultiscreenConfig(
    vocab_size=128,
    hidden_size=8,
    num_hidden_layers=2,
    num_attention_heads=2,
    key_dim=4,
    value_dim=3,
    max_position_embeddings=16,
)
model = PaperMultiscreenForCausalLM(cfg)
input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
out = model(input_ids, return_aux=True)
print(out.logits.shape)
```

## Copying weights from the current HF port

```python
oracle = PaperMultiscreenForCausalLM(cfg).eval()
oracle.copy_from_hf_model(hf_model, hf_uses_inverse_sr=True)

with torch.no_grad():
    hf_logits = hf_model(input_ids, use_cache=False).logits
    oracle_logits = oracle(input_ids).logits
    print((hf_logits - oracle_logits).abs().max())
```

The current HF port uses `inv_r = exp(sr) + 1` for Trim.  The paper uses `r = sigmoid(s_r)`.
These match when `s_r_paper = -s_r_hf`, so `copy_from_hf_model(..., hf_uses_inverse_sr=True)`
converts the scalar automatically.

For literal paper checks, use the default `position_rule="paper"` and sequence lengths within
`max_position_embeddings`.  To reproduce the current HF port's long-position modulo branch, set
`position_rule="hf_mod_after_max_position"` in the config.

## Extended HF-port equivalence sweep

`test_against_hf_port.py` now runs three P0 sweeps:

- shape sweep: batch/sequence/model-shape variations, shifted-label loss, and `logits_to_keep`
- cache split sweep: every/sampled prefix-suffix split, cache tensor equality, and cached-vs-full self-consistency
- padding mask sweep: all-ones, right padding, left padding, and sparse non-contiguous masks, including cached decoding with full-length masks

Run from an environment where the HF port is importable:

```bash
PYTHONPATH=/path/to/multiscreen_tinystories_sft:/path/to/multiscreen_oracle \
python /path/to/multiscreen_oracle/test_against_hf_port.py
```

For a faster smoke check:

```bash
PYTHONPATH=/path/to/multiscreen_tinystories_sft:/path/to/multiscreen_oracle \
python /path/to/multiscreen_oracle/test_against_hf_port.py --quick
```

The HF comparison uses `position_rule="hf_mod_after_max_position"` in the oracle, so it also tests the current HF port's long-position modulo branch. Literal paper-only tests should keep the oracle default `position_rule="paper"`.

## Review-response update notes

This version incorporates the external review recommendations:

- `PaperMultiscreenConfig.strict_cache_positions=True` by default.
  - No-cache/full-context calls reject `start_pos != 0`.
  - Cached suffix calls reject `start_pos != past_len`.
- `unit_normalize()` and `tanh_norm()` use dtype-safe epsilons so fp16 zero-norm inputs do not underflow the configured eps to zero.
- `test_formula_units.py` now includes:
  - literal paper-mode MiPE test with `T > max_position_embeddings`
  - MiPE hand-calculation test with `start_pos > 0` and extra dimensions
  - strengthened Softmask boundary test for `rel == -w`, `rel == 0`, future tokens, fractional windows, and multi-head broadcasting
  - fp16 zero-norm stability test
  - negative cache-position contract test
- `test_paper_math_oracle_selfcheck.py` now includes explicit extreme-`sr` conversion and cache-position negative tests.
- `test_against_hf_port.py` now includes zero-relevance stability and strict position/cache contract sweeps.

`position_rule="paper"` remains the default and should be used for literal paper checks.
`position_rule="hf_mod_after_max_position"` is only for current-HF-port equivalence tests.
