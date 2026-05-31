# Development Handoff

This is the primary handoff document for resuming development of this repository after the P0 validation phase.

The short version: this repository contains a **P0-qualified research implementation** of Multiscreen for Hugging Face Transformers. It has passed formula-level, reference-equivalence, cache/generation, and short TinyStories training checks. It is suitable as a correctness-first research baseline, but it is not yet a paper-scale, optimized, or production-serving implementation.

## Quick restart checklist

Use this checklist after a fresh clone or after switching machines.

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle

python oracle/test_against_hf_port.py --quick
python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --quick
```

If both quick checks pass, the repository is in the expected P0-qualified state. For full reproduction commands, see [TESTING.md](TESTING.md). For the detailed validation record, see [VALIDATION_STATUS.md](VALIDATION_STATUS.md).

## 1. Current project state

### Status

```text
Current status: P0-qualified research baseline
Suggested baseline tag: p0-qualified-v0
Primary implementation: multiscreen_transformers/modeling_multiscreen.py
Primary validation record: docs/VALIDATION_STATUS.md
Primary handoff document: docs/HANDOFF.md
```

### Completed milestones

| Milestone | Status | Meaning |
|---|---:|---|
| P0-1 | Complete | `paper_math_oracle` and HF implementation agree on small-shape formula/cache/mask tests. |
| P0-2 | Complete | Unofficial PyTorch reference, HF implementation, and oracle agree in CPU fp32 and CUDA bf16 full sweeps. |
| P0-3 | Complete | Ψ=8/16 TinyStories bf16 smoke training passed, including save/load and cache-enabled generation. |

### Current confidence boundary

You can currently trust this repository for:

```text
- small-shape Multiscreen forward correctness checks
- HF AutoModel-compatible loading after registration
- cache split behavior under validated conditions
- DynamicCache-compatible greedy generate smoke path
- TinyStories Ψ=8/16 bf16 smoke training
- future research experiments that need a correctness-first HF baseline
```

Do **not** yet claim:

```text
- paper-scale performance reproduction
- runtime efficiency advantage
- long-context retrieval benchmark reproduction
- production serving compatibility
- Triton/windowed-kernel performance
- PEFT/LoRA/Unsloth integration readiness
```

## 2. Repository map

```text
multiscreen_transformers/
  configuration_multiscreen.py   HF config, Ψ scaling, validation options
  modeling_multiscreen.py        Main HF CausalLM implementation; current P0-qualified model core
  data.py                        Dataset/preprocessing helpers
  compile_utils.py               Compile-related helper utilities

oracle/
  paper_math_oracle.py           Dense equation-oriented reference implementation
  test_against_hf_port.py        P0-1 HF-vs-oracle sweep
  test_formula_units.py          Formula-level unit tests
  test_paper_math_oracle_*.py    Oracle smoke/self-check tests

p0_2_three_way_minimal/
  test_three_way_minimal.py      P0-2 reference-vs-HF-vs-oracle comparison

third_party/multiscreen-pytorch/
  Vendored unofficial reference implementation used for P0-2

scripts/
  train_tokenizer_spm.py         Creates the 768-token TinyStories tokenizer
  p0_3_tinystories_stability.py P0-3 smoke training harness
  train_pretrain_sft.py          Larger TRL/SFT-style training entry point
  eval_smoke.py                  Smoke evaluation helper
  count_params.py                Parameter counting helper
  cache_utils.py                 Cache-related helper

configs/
  Tiny/debug/P0 training configs

tokenizers/tinystories_spm768/
  768-vocab TinyStories tokenizer used for P0-3

docs/
  VALIDATION_STATUS.md           Detailed validation record
  TESTING.md                     Reproduction commands
  KNOWN_LIMITATIONS.md           Explicit non-goals / unvalidated scope
  HANDOFF.md                     This handoff document
  validation_results/            Recorded P0-3 result files
```

## 3. Key design decisions

### 3.1 HF implementation is the development baseline

`multiscreen_transformers/modeling_multiscreen.py` is now the baseline implementation. It should be treated as the source to extend for future work.

It is validated against two independent references:

```text
paper_math_oracle
unofficial dieOD/multiscreen-pytorch reference
```

### 3.2 `paper_math_oracle` is dense and correctness-oriented

The oracle is intentionally slow. It constructs dense `T x T` relevance matrices and should be used only for tiny correctness tests. It is not a speed or long-context reference.

### 3.3 `sr` parameterization has two equivalent forms

Paper form:

```python
r = sigmoid(s_r)
alpha = clamp(1 - (1 - sim) / r, min=0) ** 2
```

HF/reference inverse-width form:

```python
inv_r = exp(sr) + 1
alpha = clamp(1 - inv_r * (1 - sim), min=0) ** 2
```

Conversion:

```text
s_r_paper = -s_r_hf
```

This relation is validated in oracle unit tests and P0 comparisons.

### 3.4 Oracle compute modes

The oracle supports two MiPE/Softmask auxiliary compute modes:

```python
mipe_compute_dtype="fp32"
softmask_compute_dtype="fp32"
```

This is the stable paper/oracle default.

For reference-compatibility in low precision:

```python
mipe_compute_dtype="reference"
softmask_compute_dtype="reference"
```

P0-2 uses `reference` mode to match the unofficial PyTorch reference in CUDA bf16 full sweeps. P0-1 should generally use the stable `fp32` mode.

### 3.5 Position handling

The paper oracle default is:

```python
position_rule="paper"
```

The HF/reference-compatible path also supports:

```python
position_rule="hf_mod_after_max_position"
```

Use the paper default for equation checks. Use the HF/reference mode for compatibility tests involving the vendored implementation.

### 3.6 DynamicCache support

The original HF port assumed legacy tuple/list `past_key_values`. Current Transformers generation can pass `DynamicCache` objects. The current implementation handles this by normalizing empty `DynamicCache` to prefill/no-cache behavior and converting non-empty cache objects to legacy form when possible.

Validated DynamicCache paths:

```text
- P0-1 quick after patch
- P0-3 generate(use_cache=True)
- post-load manual cache split in P0-3
```

Not broadly validated yet:

```text
- beam search
- do_sample=True with full logits processors
- streamers
- assisted generation
- distributed/synced generation
```

### 3.7 Dense implementation is not a speed claim

The current HF path is a dense PyTorch implementation. It is useful for validation and smoke training, not for evaluating the paper's speed claims. Windowed/fused/Triton screening kernels are future work.

## 4. Validation completed

The canonical validation record is [VALIDATION_STATUS.md](VALIDATION_STATUS.md). This section summarizes the gates.

### 4.1 P0-1: oracle vs HF

Purpose: verify the HF implementation against a dense paper-math oracle.

Passed:

```text
- CPU fp32 quick
- CPU fp32 full
- CUDA bf16 full
- CUDA fp16 quick
```

Covered:

```text
- logits
- loss
- labels_are_shifted
- logits_to_keep
- shape sweep
- cache split
- padding masks
- zero relevance
- position contract checks
```

### 4.2 P0-2: three-way comparison

Purpose: verify:

```text
dieOD/multiscreen-pytorch
== HF multiscreen_transformers
== paper_math_oracle
```

Passed:

```text
CPU fp32 quick:
  prefill_three_way: 12
  cache_split_three_way: 28

CPU fp32 full:
  prefill_three_way: 45
  cache_split_three_way: 237

CUDA bf16 quick:
  prefill_three_way: 12
  cache_split_three_way: 28

CUDA bf16 full:
  prefill_three_way: 45
  cache_split_three_way: 237
```

Covered:

```text
- prefill logits
- external CE loss
- KV cache tensors
- layer hook outputs
- prefix/suffix cache split
- cached suffix vs full-forward suffix
- max-position modulo branch compatibility
```

Padding masks are not covered by P0-2 because the reference implementation does not expose an attention-mask API. Padding behavior is covered by P0-1.

### 4.3 P0-3: TinyStories smoke training

Purpose: verify short-run training stability and checkpoint/generation behavior for Ψ=8 and Ψ=16.

Results:

```text
Ψ=8:
  params: 966,850
  initial_probe_loss: 8.215893
  final_probe_loss: 4.312645
  abs_loss_drop: 3.903248
  rel_loss_drop: 47.5085%

Ψ=16:
  params: 14,877,442
  initial_probe_loss: 15.899660
  final_probe_loss: 5.928024
  abs_loss_drop: 9.971636
  rel_loss_drop: 62.7160%
```

Confirmed:

```text
- finite loss
- finite gradient norms
- bf16 autocast training
- save_pretrained / from_pretrained
- post-load logits equality
- manual cache split equality
- generate(use_cache=True)
```

Recorded result files:

```text
docs/validation_results/p0_3_results.json
docs/validation_results/P0-3_COMPLETE.md
```

## 5. What is safe to assume now

Safe assumptions for the next developer:

```text
- `modeling_multiscreen.py` is P0-qualified as a research baseline.
- Forward math is consistent with the paper oracle on small shapes.
- The HF implementation is consistent with the vendored unofficial reference on small shapes.
- Basic cache behavior is consistent across oracle, HF, and reference.
- DynamicCache-compatible greedy generation works in smoke tests.
- Short TinyStories bf16 training works for Ψ=8 and Ψ=16.
```

Do **not** assume:

```text
- paper-scale performance reproduction
- long-context retrieval performance
- speed advantage over Transformer baselines
- efficient long-context memory use
- PEFT/LoRA/Unsloth compatibility
- production generation compatibility
- serving compatibility with vLLM/SGLang
```

## 6. Recommended next step: P0-4

Recommended next validation gate:

```text
P0-4: GPT-2 vocab + context 4096 short pretraining smoke test
```

Purpose:

```text
- move beyond TinyStories 768-vocab smoke setting
- test larger vocab and more realistic embedding size
- test longer context and memory behavior
- verify bf16 stability under a more realistic sequence length
```

Suggested minimal shape:

```text
model: Multiscreen Ψ=8 first, then Ψ=16 if Ψ=8 passes
vocab: GPT-2 tokenizer, 50,257 tokens
context: 1024 first, then 4096
steps: short smoke, e.g. 50-200 depending on runtime
batch: start with microbatch 1 and grad accumulation
amp: bf16
metrics: finite loss, loss decrease, save/load, generate/cache smoke, peak memory
```

Caution:

```text
- Do not begin with Ψ=32.
- Do not use this dense implementation to evaluate speed claims.
- Watch memory carefully; dense screening is still O(T^2) in the current implementation.
```

## 7. Alternative next step: P1 ecosystem work

If prioritizing ecosystem integration instead of P0-4, suggested P1 tasks are:

```text
P1-1: PEFT/LoRA adapter compatibility
P1-2: QLoRA/bitsandbytes compatibility check
P1-3: Unsloth loader/wrapper prototype
P1-4: generation compatibility matrix
P1-5: torch.compile smoke checks
```

Recommended LoRA target modules:

```text
q_proj
k_proj
v_proj
g_proj
o_proj
```

`g_proj` should be included because the gate path is likely important for language-modeling quality.

## 8. Immediate commands for a fresh checkout

After cloning:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle
```

Run P0-1 smoke:

```bash
python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
python oracle/test_against_hf_port.py --quick
```

Run P0-2 quick:

```bash
python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --quick
```

Run P0-3 smoke if TinyStories/cache are available:

```bash
python scripts/p0_3_tinystories_stability.py \
  --tokenizer-path tokenizers/tinystories_spm768 \
  --cache-dir /path/to/hf_cache \
  --device cuda:0 \
  --amp-dtype bf16 \
  --seq-len 128 \
  --batch-size 4 \
  --steps-per-psi 8:40,16:25 \
  --output-dir outputs/p0_3_tinystories_stability
```

## 9. Files to preserve exactly

These files are the most important baseline files and should be treated as checkpointed P0 artifacts:

```text
multiscreen_transformers/modeling_multiscreen.py
multiscreen_transformers/configuration_multiscreen.py
oracle/paper_math_oracle.py
oracle/test_against_hf_port.py
p0_2_three_way_minimal/test_three_way_minimal.py
scripts/p0_3_tinystories_stability.py
docs/VALIDATION_STATUS.md
docs/HANDOFF.md
```

If any of these change, rerun at least:

```bash
python oracle/test_against_hf_port.py --quick
python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --quick
```

For changes to cache/generation or `modeling_multiscreen.py`, also rerun a P0-3 quick smoke.

## 10. GitHub publishing notes

Do include:

```text
- source code
- oracle tests
- P0-2 tests
- docs/ validation records
- tokenizer if redistribution is acceptable for your use case
- third_party notices and original license files
```

Do not include:

```text
- outputs/
- checkpoint weights from smoke training
- local cache directories
- __pycache__/
- .git/ from vendored third-party repos
- local absolute-path logs
```

Third-party/data caveat:

```text
- `third_party/multiscreen-pytorch` retains its Apache-2.0 license.
- The TinyStories-derived tokenizer is included for reproducibility; check the dataset's own license/terms before redistribution in contexts where that matters.
```

## 11. Suggested tag

After uploading to GitHub, create a tag such as:

```bash
git tag p0-qualified-v0
git push origin p0-qualified-v0
```

Suggested release note:

```text
P0-qualified unofficial HF Multiscreen implementation:
- paper oracle equivalence
- three-way reference equivalence
- DynamicCache-compatible generation smoke
- TinyStories Ψ=8/16 bf16 smoke training
```
