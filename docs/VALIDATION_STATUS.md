# Validation Status

This document records the validation state of the current Multiscreen HF implementation. It is intended to be more detailed than the README and to make the current quality boundary explicit.

For project-wide handoff information and recommended next steps, see [HANDOFF.md](HANDOFF.md).

For compact validation run summaries, see [validation_results/VALIDATION_LOG_INDEX.md](validation_results/VALIDATION_LOG_INDEX.md).

For future validation logging rules, see [LOGGING_POLICY.md](LOGGING_POLICY.md).

## Status summary

```text
P0-1: paper_math_oracle vs HF implementation
  Status: complete

P0-2: unofficial PyTorch reference vs HF implementation vs paper_math_oracle
  Status: complete

P0-3: Psi=8/16 TinyStories smoke training
  Status: complete

P0-4: GPT-2 vocab + context 4096 short pretraining smoke
  Status: harness/config/docs prepared; qualifying CUDA bf16 execution pending
```

The current implementation can be considered a **P0-qualified research implementation** based on completed P0-1/P0-2/P0-3 gates. P0-4 is an additional pending gate and must not be reported as complete from static config validation or a reduced diagnostic.

## Scope of “P0-qualified”

P0-qualified means the implementation has passed small-shape math/caching tests and a short TinyStories training smoke test. It does **not** mean the implementation reproduces the paper at scale, is optimized, or is production-ready.

Confirmed:

- formula-level oracle checks
- HF-vs-oracle forward/loss/cache/mask sweeps
- three-way comparison with the unofficial PyTorch reference
- CPU fp32 and CUDA bf16 sweeps
- DynamicCache-compatible greedy `generate()` smoke path
- Psi=8/16 short TinyStories bf16 training stability
- P0-4 script/config static validation for Psi=8 and Psi=16

Not confirmed:

- P0-4 qualifying GPT-2-vocab, context-4096 CUDA bf16 training
- 28M/286M/1.3B paper-scale training
- long-context retrieval benchmarks at paper settings
- runtime efficiency relative to Transformer baselines
- windowed or Triton kernels
- PEFT, LoRA, QLoRA, Unsloth, vLLM, SGLang
- production generation features beyond the smoke path

## P0-1: paper oracle vs HF implementation

### Purpose

Verify that the HF implementation agrees with a dense, equation-oriented `paper_math_oracle` implementation on tiny shapes.

### Covered behavior

- forward logits
- next-token loss
- `labels_are_shifted=True`
- `logits_to_keep`
- shape variations
- cache split
- cached suffix vs full suffix
- padding mask behavior
- right padding
- left padding
- sparse non-contiguous mask
- zero relevance path
- `position_ids = arange(0, T)`
- rejection of unsupported no-cache offset positions
- rejection of inconsistent cache `start_pos`
- fp32, bf16 quick, fp16 quick

### Recorded pass summary

CPU fp32 quick:

```text
cache_split: 10
padding_cache: 24
padding_full: 8
position_contract_negative_cache: 1
position_contract_negative_no_cache: 1
position_ids_zero: 1
shape_forward_loss: 12
shape_logits_to_keep: 24
shape_shifted_loss: 12
zero_relevance: 1
```

CPU fp32 full:

```text
cache_split: 144
padding_cache: 240
padding_full: 88
position_contract_negative_cache: 2
position_contract_negative_no_cache: 2
position_ids_zero: 2
shape_forward_loss: 60
shape_logits_to_keep: 144
shape_shifted_loss: 60
zero_relevance: 2
```

CUDA bf16 full:

```text
cache_split: 144
padding_cache: 240
padding_full: 88
position_contract_negative_cache: 2
position_contract_negative_no_cache: 2
position_ids_zero: 2
shape_forward_loss: 60
shape_logits_to_keep: 144
shape_shifted_loss: 60
zero_relevance: 2
```

CUDA fp16 quick:

```text
cache_split: 10
padding_cache: 24
padding_full: 8
position_contract_negative_cache: 1
position_contract_negative_no_cache: 1
position_ids_zero: 1
shape_forward_loss: 12
shape_logits_to_keep: 24
shape_shifted_loss: 12
zero_relevance: 1
```

### Key implementation notes

The oracle has two compute modes for MiPE/Softmask auxiliary scalar math:

```python
mipe_compute_dtype="fp32"
softmask_compute_dtype="fp32"
```

This is the stable paper/oracle default.

For low-precision reference compatibility:

```python
mipe_compute_dtype="reference"
softmask_compute_dtype="reference"
```

P0-2 uses the reference-compatible mode to match the unofficial PyTorch implementation in CUDA bf16 full sweeps.

## P0-2: three-way comparison

### Purpose

Verify that the three implementations agree:

```text
dieOD/multiscreen-pytorch
== HF multiscreen_transformers port
== paper_math_oracle
```

### Covered behavior

- prefill logits
- external next-token CE loss
- KV cache tensors
- per-layer hidden states via hooks
- prefix/suffix cache split
- cached suffix logits vs full-forward suffix logits
- max-position modulo branch behavior used by the reference/HF implementations

P0-2 intentionally does not test padding masks because the reference implementation API does not expose `attention_mask`. Padding/mask behavior remains covered by P0-1.

### Recorded pass summary

CPU fp32 quick:

```text
prefill_three_way: 12
cache_split_three_way: 28
```

CPU fp32 full:

```text
prefill_three_way: 45
cache_split_three_way: 237
```

CUDA bf16 quick:

```text
prefill_three_way: 12
cache_split_three_way: 28
```

CUDA bf16 full:

```text
prefill_three_way: 45
cache_split_three_way: 237
```

### Low-precision note

A CUDA bf16 full mismatch was initially observed in `cache[0].K` at a long-position MiPE modulo boundary. The cause was comparison-mode mismatch: the oracle was using stable fp32 auxiliary MiPE/Softmask math, while the reference implementation performed that scalar math in bf16. The oracle now supports `reference` compute mode, and P0-2 sets this mode for low-precision three-way comparisons. After this update, CUDA bf16 full passes.

## P0-3: TinyStories Psi=8/16 smoke training

### Purpose

Verify that the implementation can run short TinyStories training in bf16 for both Psi=8 and Psi=16, and that checkpoint/generation/cache paths remain functional after training.

### Command shape

```bash
python scripts/p0_3_tinystories_stability.py \
  --tokenizer-path tokenizers/tinystories_spm768 \
  --cache-dir /path/to/hf_cache \
  --device cuda:0 \
  --amp-dtype bf16 \
  --seq-len 128 \
  --batch-size 4 \
  --steps-per-psi 8:40,16:25 \
  --output-dir outputs/p0_3_tinystories_stability_dynamic_cache_patch
```

### Recorded results

Psi=8:

```text
params: 966,850
steps: 40
seq_len: 128
batch_size: 4
amp_dtype: bf16
initial_probe_loss: 8.215893
final_probe_loss: 4.312645
abs_loss_drop: 3.903248
rel_loss_drop: 47.5085%
save_load_logits_max_abs: 0
cache_split_logits_max_abs: 0
```

Psi=16:

```text
params: 14,877,442
steps: 25
seq_len: 128
batch_size: 4
amp_dtype: bf16
initial_probe_loss: 15.899660
final_probe_loss: 5.928024
abs_loss_drop: 9.971636
rel_loss_drop: 62.7160%
save_load_logits_max_abs: 0
cache_split_logits_max_abs: 0
```

Detailed JSON: [validation_results/p0_3_results.json](validation_results/p0_3_results.json).

### Confirmed after training

- losses stayed finite
- gradient norms stayed finite
- probe loss decreased
- `save_pretrained` / `from_pretrained` preserved logits
- `generate()` worked with cache enabled
- cached suffix logits matched full forward suffix logits after loading

## DynamicCache compatibility

The original HF port assumed legacy tuple/list `past_key_values`. Current Transformers generation can pass `DynamicCache`. The implementation now normalizes empty `DynamicCache` to no-cache prefill behavior and converts non-empty cache objects to legacy form where possible.

Validated paths:

- P0-1 quick after DynamicCache patch
- P0-3 `generate(use_cache=True)` after training
- post-load manual cache split in P0-3

Still not validated:

- beam search
- sampling processors beyond greedy smoke
- streamers
- assisted generation
- distributed generation / `synced_gpus`

## P0-4: GPT-2 vocabulary + context 4096 smoke

### Current status

```text
Harness: prepared
Psi=8 config: prepared and statically validated
Psi=16 config: prepared and statically validated
Qualifying CUDA bf16 run: pending
Recorded result: none yet
```

Primary files:

```text
scripts/p0_4_gpt2_context4096_smoke.py
configs/p0_4_multiscreen_psi8_gpt2_ctx4096/
configs/p0_4_multiscreen_psi16_gpt2_ctx4096/
docs/P0_4_PLAN.md
docs/P0_4_RESULTS_TEMPLATE.md
```

### Intended coverage

- model construction from checked-in `MultiscreenConfig`
- GPT-2 tokenizer loading with exact vocabulary size 50,257
- packed text dataset
- sequence length 4,096 forward/backward
- finite train loss
- finite gradient norm
- short-run probe-loss decrease
- `save_pretrained` / `from_pretrained`
- loaded-logit comparison
- greedy `generate(use_cache=True)`
- manual cache split vs full-forward suffix
- stepwise `metrics.jsonl`
- `summary.json`
- generated `P0-4_COMPLETE.md` only for a qualifying run

### Qualification rule

The harness writes `P0-4_COMPLETE.md` only if the run uses:

```text
GPT-2 vocab size: 50,257
context length: 4,096
device: CUDA
AMP dtype: bf16
optimizer steps: at least 50
```

A reduced-context, CPU, other-dtype, or shorter run can pass diagnostic checks but writes `P0-4_DIAGNOSTIC_COMPLETE.md` and does not complete P0-4.

See [P0_4_PLAN.md](P0_4_PLAN.md) for execution order and [P0_4_RESULTS_TEMPLATE.md](P0_4_RESULTS_TEMPLATE.md) for the review record.

## Recommended next validation step

Execute P0-4 on a suitable CUDA bf16 system in this order:

```text
1. rerun P0-1/P0-2 quick checks
2. optional Psi=8 context-1024 diagnostic
3. qualifying Psi=8 context-4096 run
4. review memory, loss, reload, generation, and cache artifacts
5. qualifying Psi=16 context-4096 run only after Psi=8 passes
```

P1 ecosystem work should not be described as validated merely because the P0-4 harness exists.
