# Validation Status

This document records the validation state of the current Multiscreen HF implementation. It is intended to be more detailed than the README and to make the current quality boundary explicit.

For project-wide handoff information and recommended next steps, see [HANDOFF.md](HANDOFF.md).

For compact validation run summaries, see [validation_results/VALIDATION_LOG_INDEX.md](validation_results/VALIDATION_LOG_INDEX.md).

For future validation logging rules, see [LOGGING_POLICY.md](LOGGING_POLICY.md).

For the current Stage 5 gate, see [LEVEL1_CORE_REQUALIFICATION_PLAN.md](LEVEL1_CORE_REQUALIFICATION_PLAN.md).

## Status summary

```text
P0-1: paper_math_oracle vs HF implementation
  Status: complete

P0-2: unofficial PyTorch reference vs HF implementation vs paper_math_oracle
  Status: complete

P0-3: Psi=8/16 TinyStories smoke training
  Status: complete

P0-4: GPT-2 vocab + context 4096 short pretraining smoke
  Status: complete; reviewed qualifying CUDA bf16 Psi=8/Psi=16 execution passed

P0.5-C1: architecture / initialization / all-scale contract
  Status: complete; focused PR #9 reviewed and merged

P0.5-C2: long-position / MiPE / cache semantics
  Status: complete; focused PR #10 merged; corrective PR #11 merged

P1-preflight B: gradient-checkpointing API modernization
  Status: complete; focused PR #12 reviewed and merged

P0.5-C3: paper-training-contract smoke
  Status: complete; focused PR #13 reviewed and merged

final Level 1 requalification and evidence
  Status: current Stage 5; pending and not validated
```

The current implementation can be considered a **P0-qualified research implementation through P0-4**. P0-4 was accepted only from reviewed qualifying Psi=8/Psi=16 CUDA bf16 artifacts; static config validation and reduced diagnostics remain non-qualifying substitutes.

## Scope of “P0-qualified”

P0-qualified means the implementation has passed small-shape math/caching tests, a short TinyStories training smoke test, and the recorded GPT-2-vocabulary context-4096 short-run qualification. It does **not** mean the implementation reproduces the paper at scale, is optimized, or is production-ready.

Confirmed:

- formula-level oracle checks
- HF-vs-oracle forward/loss/cache/mask sweeps
- three-way comparison with the unofficial PyTorch reference
- CPU fp32 and CUDA bf16 sweeps
- DynamicCache-compatible greedy `generate()` smoke path
- Psi=8/16 short TinyStories bf16 training stability
- P0-4 script/config static validation for Psi=8 and Psi=16
- P0-4 qualifying GPT-2-vocabulary, context-4096 CUDA bf16 Psi=8/Psi=16 training

Not confirmed:

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
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
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
- P0-4 qualifying Psi=8/Psi=16 greedy generation and manual cache split

Still not validated:

- beam search
- sampling processors beyond greedy smoke
- streamers
- assisted generation
- distributed generation / `synced_gpus`

## P0-4: GPT-2 vocabulary + context 4096 smoke

### Current status

```text
Harness: prepared and exercised
Psi=8 config: statically validated and qualifying run passed
Psi=16 config: statically validated and qualifying run passed
Qualifying CUDA bf16 execution: complete
Recorded result: passed
```

Primary files:

```text
scripts/p0_4_gpt2_context4096_smoke.py
configs/p0_4_multiscreen_psi8_gpt2_ctx4096/
configs/p0_4_multiscreen_psi16_gpt2_ctx4096/
docs/P0_4_PLAN.md
docs/validation_results/P0_4_SUMMARY.md
docs/validation_results/P0_4_SUMMARY.json
```

### Recorded results

Both accepted runs used GPT-2 vocabulary 50,257, sequence length 4,096, CUDA bf16, microbatch 1, gradient accumulation 8, gradient checkpointing, and 50 optimizer steps.

| Metric | Psi=8 | Psi=16 |
|---|---:|---:|
| parameters | 4,134,146 | 27,546,626 |
| initial probe loss | 11.140747 | 15.799321 |
| final probe loss | 4.675382 | 3.495601 |
| relative loss drop | 58.0335% | 77.8750% |
| max finite grad norm | 5.393857 | 23.194632 |
| training elapsed seconds | 107.6805 | 425.8058 |
| peak allocated bytes | 3,156,709,888 | 6,622,802,944 |
| peak reserved bytes | 4,525,654,016 | 9,130,999,808 |
| loaded-logits max abs | 0 | 0 |
| cache-split max abs | 0 | 0.125, within configured atol/rtol |
| prompt / generated length | 4 / 12 | 4 / 12 |
| `qualification.qualified` | `true` | `true` |

The environment used Python 3.12.11, PyTorch 2.7.1+cu128, Transformers 4.57.6, CUDA 12.8, and an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.

Every event in both 57-event metrics streams was reviewed. All 50 train losses and gradient norms per run were finite, optimizer steps were contiguous, memory peaks stabilized after step 2, probe loss met the configured threshold, saved models and tokenizers reloaded, greedy generation appended tokens, and manual cache splits passed the configured tolerances. Both qualifying markers were present and failure artifacts absent. The Psi=16 cache value passed the combined `atol=0.03, rtol=0.03` predicate; it is not an absolute-only threshold.

Full ignored artifacts remain local. Their SHA-256 hashes, exact sanitized commands, package versions, and detailed metrics are recorded in [P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md) and [P0_4_SUMMARY.json](validation_results/P0_4_SUMMARY.json).

### Qualification rule

P0-4 still counts only when all strict conditions are met:

```text
GPT-2 vocab size: 50,257
context length: 4,096
device: CUDA
AMP dtype: bf16
microbatch: 1
optimizer steps: at least 50
finite loss and gradient norms
configured probe-loss decrease
save/load and tokenizer reload
loaded logits within tolerance
greedy generate(use_cache=True)
manual cache split within tolerance
complete summary, metrics, and qualifying marker artifacts
```

A reduced-context, CPU, other-dtype, or shorter run can pass diagnostic checks but remains non-qualifying. The accepted runtime and memory values are feasibility diagnostics only, not evidence of long-context efficiency.

## Staged Level 1 Core

P0.5-C1 is accepted from merged PR #9. Its meta-only all-scale architecture,
initialization, tied-embedding, config, and packed-text contracts are recorded
in [P0_5_C1_SUMMARY.md](validation_results/P0_5_C1_SUMMARY.md).

P0.5-C2 was accepted from merged PR #10, and its separate CUDA-autocast
cache-dtype prediction correction was merged as PR #11. C2 adds explicit
serialized MiPE modes:

```text
paper_absolute
reference_mod_after_wrap_boundary
```

Legacy configs resolve to the reference rule with an inclusive boundary from
their existing `max_position_embeddings`; paper behavior is explicit. Stable
long-boundary correctness and incoming-dtype vendored compatibility are tested
as separate numerical lanes. Strict scalar contiguous position/cache schemas,
Transformers 4.x/5.x DynamicCache normalization, full/cache/chunk agreement,
and greedy boundary-crossing generation are covered by the focused suite.

See [ADR-0001](adr/ADR-0001-mipe-position-semantics.md),
[P0_5_C2_PLAN.md](P0_5_C2_PLAN.md), and
[P0_5_C2_SUMMARY.md](validation_results/P0_5_C2_SUMMARY.md). These results do
not demonstrate dense 131K feasibility, retrieval quality, efficiency, or a
P1 model/ecosystem capability.

P1-preflight B replaces the deprecated checkpointing hook with the supported
Transformers runtime API, preserves an explicit non-reentrant default, and
invokes the function installed by Transformers. Its focused tests passed under
exact Transformers 4.57.6 and 5.14.1 lanes without the old-format or
missing-input-gradient warnings. Deterministic logits/loss/gradient agreement,
custom function injection, finite optimization, transient serialization,
save/reload, and greedy generation contracts passed.

Full P0-1/P0-2 CPU fp32 and CUDA bf16 regressions also passed. A checkpointed
CUDA bf16 TinyStories Psi=8/Psi=16 smoke and a reduced Psi=8 context-1024 P0-4
checkpointed diagnostic completed all postchecks. The latter is explicitly
non-qualifying and does not replace the accepted historical P0-4 result. See
[P1_PREFLIGHT_B_PLAN.md](P1_PREFLIGHT_B_PLAN.md) and
[P1_PREFLIGHT_B_SUMMARY.md](validation_results/P1_PREFLIGHT_B_SUMMARY.md).
This Stage 3 result was accepted by merged PR #12.

P0.5-C3 was the focused Stage 4. It encodes the paper's tokenizer,
data-stream, optimizer, scheduler, weight-decay, and no-gradient-clipping
recipe as executable contracts; see [P0_5_C3_PLAN.md](P0_5_C3_PLAN.md).

The exact contract and scheduler boundaries passed. The pinned GPT-2 tokenizer
and third-party SlimPajama-family reupload test shard reproduced the recorded
asset, row, token-stream, accounting, and chunk identities. Focused tests passed
under Transformers 4.57.6 and 5.14.1. Full P0-1/P0-2 CPU fp32 and CUDA bf16
regressions also passed.

Psi=8 and Psi=16 context-4096 CUDA bf16 operational diagnostics passed, as did
one bounded exact-0.0625 peak-learning-rate update for each order. All values
and updates were finite, clipping remained disabled, completion markers were
present, and failure artifacts were absent. See
[P0_5_C3_SUMMARY.md](validation_results/P0_5_C3_SUMMARY.md) and
[P0_5_C3_EVIDENCE_ARCHIVE.json](validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json).

The focused Stage 4 implementation and result were reviewed and accepted by
merged PR #13. Its existing evidence descriptor still truthfully records that
exact/private retention is blocked, sanitized evidence is verified but
unpublished, and an explicit evidence reviewer was not supplied; the merge does
not rewrite those historical retention facts.

## Next validation boundary

P0-4, P0.5-C1, P0.5-C2, P1-preflight B, and P0.5-C3 are complete. Final Level
1 core requalification is the current Stage 5 under
[LEVEL1_CORE_REQUALIFICATION_PLAN.md](LEVEL1_CORE_REQUALIFICATION_PLAN.md).
It remains pending and has no accepted qualifying Stage 5 evidence; therefore
the five-stage Level 1 Core program is not complete. No P1 ecosystem capability
is validated yet.
