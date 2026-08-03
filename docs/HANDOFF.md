# Development Handoff

This is the primary development-restart document for `kuroganegames/Multiscreen-HF`.

The repository contains a **P0-qualified research implementation** of Multiscreen for Hugging Face Transformers. P0-1, P0-2, P0-3, and P0-4 are complete. Reviewed local CUDA bf16 context-4096 evidence is recorded for both Psi=8 and Psi=16.

The current phase is therefore:

```text
selection of the next focused validation gate; no P1 gate has been selected or validated
```

For ordinary continuation after a local clone, read the root [`AGENTS.md`](../AGENTS.md), [VALIDATION_STATUS.md](VALIDATION_STATUS.md), and the accepted [P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md). Use [CODEX_P0_4_HANDOFF.md](CODEX_P0_4_HANDOFF.md) only for an intentional P0-4 reproduction or requalification.

For repository hygiene checks, see [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md).

## 1. Current project state

### Milestones

| Milestone | Status | Meaning |
|---|---:|---|
| P0-1 | Complete | `paper_math_oracle` and the HF implementation agree on validated small-shape formula, loss, mask, position, and cache behavior. |
| P0-2 | Complete | The vendored unofficial PyTorch reference, HF implementation, and oracle agree in recorded CPU fp32 and CUDA bf16 sweeps. |
| P0-3 | Complete | Psi=8/16 TinyStories bf16 smoke training passed, including finite loss/gradients, save/load, cache split, and greedy generation. |
| P0-4 harness | Merged | GPT-2 vocab/context-4096 harness, Psi=8/16 configs, plan, result template, static checks, and tiny CPU integration diagnostic were merged in PR #3. |
| P0-4 qualifying execution | Complete | Reviewed CUDA bf16 context-4096, 50-step Psi=8 and Psi=16 runs passed; see [P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md). |
| P1 ecosystem work | Not started as validated work | PEFT/LoRA, QLoRA, Unsloth, generation matrix, compile, and serving remain future work. |

### Baseline identity

```text
Current status: P0-qualified research baseline through P0-4
Primary implementation: multiscreen_transformers/modeling_multiscreen.py
Primary config: multiscreen_transformers/configuration_multiscreen.py
Primary equation oracle: oracle/paper_math_oracle.py
Primary validation record: docs/VALIDATION_STATUS.md
Current execution plan: docs/P0_4_PLAN.md
Codex local handoff: docs/CODEX_P0_4_HANDOFF.md
Repository instructions for Codex: AGENTS.md
```

P0-4 is complete from reviewed qualifying Psi=8/Psi=16 CUDA bf16 artifacts. Static config validation, CI CPU diagnostics, and reduced-context runs remain non-qualifying substitutes.

## 2. First ten minutes after a fresh clone

Clone and inspect the checkout:

```bash
git clone https://github.com/kuroganegames/Multiscreen-HF.git
cd Multiscreen-HF

git status --short --branch
git log -1 --oneline
```

Install the local package and declared dependencies:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle
```

Run the minimum P0 baseline checks:

```bash
python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
python oracle/test_against_hf_port.py --quick

python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --quick
```

Run the P0-4 static preflight:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --validate-config-only

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --validate-config-only
```

If these checks pass, the checkout matches the expected merged baseline at the CPU/static level. The accepted P0-4 GPU evidence is recorded separately; a fresh clone has not itself reproduced those CUDA runs.

## 3. Codex continuation

Start Codex from the repository root so it reads `AGENTS.md`:

```bash
codex
```

If the `/goal` command is not available:

```bash
codex features enable goals
codex
```

[CODEX_P0_4_HANDOFF.md](CODEX_P0_4_HANDOFF.md) is retained as a strict P0-4 reproduction/requalification prompt. P0-4 is no longer the pending next task; select a new focused gate explicitly rather than rerunning it by default.

Do not start Codex from a parent directory and assume repository instructions were loaded. Confirm the working directory is the Git root.

## 4. Repository map

```text
AGENTS.md
  Persistent repository-level instructions for Codex.

multiscreen_transformers/
  configuration_multiscreen.py   HF config, Psi scaling, validation options
  modeling_multiscreen.py        P0-qualified HF CausalLM implementation
  data.py                        Packed dataset helper
  compile_utils.py               Compile environment helpers

oracle/
  paper_math_oracle.py           Dense equation-oriented reference
  test_against_hf_port.py        P0-1 HF-vs-oracle sweep
  test_formula_units.py          Formula-level tests
  test_paper_math_oracle_*.py    Oracle self-check and smoke tests

p0_2_three_way_minimal/
  test_three_way_minimal.py      P0-2 reference-vs-HF-vs-oracle comparison

third_party/multiscreen-pytorch/
  Vendored dieOD/multiscreen-pytorch reference used by P0-2

scripts/
  p0_3_tinystories_stability.py          P0-3 training harness
  p0_4_gpt2_context4096_smoke.py         P0-4 training/qualification harness
  train_pretrain_sft.py                  Larger TRL/SFT-style entry point
  train_tokenizer_spm.py                 TinyStories 768-vocab tokenizer creation
  eval_smoke.py, count_params.py, cache_utils.py

configs/
  p0_4_multiscreen_psi8_gpt2_ctx4096/
  p0_4_multiscreen_psi16_gpt2_ctx4096/
  P0-3 and earlier debug/training configs

tokenizers/tinystories_spm768/
  Committed tokenizer used for P0-3 reproducibility

docs/
  HANDOFF.md                    Main development restart guide
  CODEX_P0_4_HANDOFF.md         Clone-to-Codex goal workflow and full prompt
  VALIDATION_STATUS.md          Canonical validation boundary
  TESTING.md                    Reproduction commands
  KNOWN_LIMITATIONS.md          Explicit unvalidated scope
  P0_4_PLAN.md                  P0-4 execution and failure triage
  P0_4_RESULTS_TEMPLATE.md      Human-readable result template
  LOGGING_POLICY.md             Result logging and sanitization policy
  REPOSITORY_AUDIT.md           Hygiene and handoff-readiness audit
  RELEASE_CHECKLIST.md          Tag/release checklist
  validation_results/           Accepted compact validation summaries
```

## 5. Key design contracts

### 5.1 HF implementation is the extension baseline

`multiscreen_transformers/modeling_multiscreen.py` is the current development baseline. It is validated against:

```text
paper_math_oracle
dieOD/multiscreen-pytorch
```

This validation is strong for the recorded small-shape and smoke conditions, not for paper-scale training or optimized serving.

### 5.2 The oracle is dense and correctness-oriented

`oracle/paper_math_oracle.py` deliberately follows the equations with dense tensors. It is intended for tiny correctness comparisons. It must not be used as a speed or long-context implementation reference.

### 5.3 Trim parameterization

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

Required conversion:

```text
s_r_paper = -s_r_hf
```

Do not alter this mapping without a focused mathematical and three-way validation update.

### 5.4 Oracle compute modes

Stable paper/oracle checks:

```python
mipe_compute_dtype="fp32"
softmask_compute_dtype="fp32"
```

Low-precision reference compatibility:

```python
mipe_compute_dtype="reference"
softmask_compute_dtype="reference"
```

P0-2 uses reference-compatible behavior when matching the vendored low-precision implementation. P0-1 normally emphasizes stable fp32 auxiliary math.

### 5.5 Position handling

Literal paper checks use:

```python
position_rule="paper"
```

HF/reference compatibility can use:

```python
position_rule="hf_mod_after_max_position"
```

The HF public API only supports the validated scalar contiguous position/cache contract. Arbitrary batch-specific offsets are not silently supported.

### 5.6 DynamicCache boundary

The HF implementation normalizes empty Transformers cache objects for prefill and converts compatible non-empty cache objects to the internal legacy tuple form.

Validated:

```text
- P0-1 cache comparisons under covered conditions
- P0-3 greedy generate(use_cache=True)
- P0-3 post-load manual cache split
- P0-4 tiny CPU integration diagnostic in CI
- P0-4 qualifying Psi=8/Psi=16 greedy generation and manual cache split
```

Still not broadly validated:

```text
- beam search
- broad do_sample/logits-processor combinations
- variable-length batch generation
- streamers
- assisted generation
- distributed/synced generation
```

### 5.7 Dense path is not an efficiency claim

The current HF screening path remains dense and quadratic in sequence length. P0-4 records time and CUDA memory only to diagnose feasibility and stability. Do not report those numbers as evidence for the paper's long-context efficiency claims.

## 6. Completed validation

The detailed counts and commands are in [VALIDATION_STATUS.md](VALIDATION_STATUS.md) and [TESTING.md](TESTING.md).

### P0-1

Recorded passes include CPU fp32 quick/full, CUDA bf16 full, and CUDA fp16 quick across:

```text
- logits and loss
- shifted-label loss
- logits_to_keep
- shape sweeps
- cache split and cached suffix
- padding and sparse masks
- zero-relevance stability
- position/cache negative contracts
```

### P0-2

Recorded passes include CPU fp32 and CUDA bf16 quick/full comparisons across:

```text
- prefill logits
- external CE loss
- KV cache tensors
- per-layer hook outputs
- prefix/suffix cache split
- cached suffix vs full-forward suffix
- long-position modulo compatibility branch
```

P0-2 does not cover padding masks because the vendored reference lacks an attention-mask API. P0-1 covers mask behavior against the oracle.

### P0-3

Recorded smoke results:

```text
Psi=8
  params: 966,850
  steps: 40
  seq_len: 128
  initial_probe_loss: 8.215893
  final_probe_loss: 4.312645
  relative drop: 47.5085%

Psi=16
  params: 14,877,442
  steps: 25
  seq_len: 128
  initial_probe_loss: 15.899660
  final_probe_loss: 5.928024
  relative drop: 62.7160%
```

Both runs recorded finite gradients, exact save/load logits under the test conditions, exact manual cache-split logits under the test conditions, and cache-enabled greedy generation.

## 7. P0-4 recorded result

P0-4 passed on the source commit recorded in [P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md) and [P0_4_SUMMARY.json](validation_results/P0_4_SUMMARY.json). Both runs used the checked-in defaults without weakening the gate:

```text
GPT-2 vocabulary: 50,257
sequence length: 4,096
device: CUDA
AMP dtype: bf16
microbatch / gradient accumulation: 1 / 8
optimizer steps: 50
```

| Metric | Psi=8 | Psi=16 |
|---|---:|---:|
| parameters | 4,134,146 | 27,546,626 |
| probe loss | 11.140747 → 4.675382 | 15.799321 → 3.495601 |
| relative drop | 58.0335% | 77.8750% |
| max finite grad norm | 5.393857 | 23.194632 |
| peak CUDA allocated | 3,156,709,888 bytes | 6,622,802,944 bytes |
| peak CUDA reserved | 4,525,654,016 bytes | 9,130,999,808 bytes |
| loaded-logits max abs | 0 | 0 |
| cache-split max abs | 0 | 0.125, within configured atol/rtol |
| prompt / generated length | 4 / 12 | 4 / 12 |
| `qualification.qualified` | `true` | `true` |

Every event in both 57-event metrics streams was reviewed. All train losses and gradient norms were finite; memory peaks stabilized after step 2; save/load, tokenizer reload, generation, and cache checks passed; qualifying markers were present and failure artifacts absent. The Psi=16 cache result passed the configured combined `atol=0.03, rtol=0.03` predicate and must not be interpreted as an absolute-only threshold.

The run environment used Python 3.12.11, PyTorch 2.7.1+cu128, Transformers 4.57.6, CUDA 12.8, and an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition. These results confirm the recorded short dense-reference smoke only. Runtime and memory are feasibility diagnostics, not evidence of the paper's efficiency claims.

The original execution order and strict failure rules remain in [P0_4_PLAN.md](P0_4_PLAN.md) for reproduction. A reduced-context, CPU, non-bf16, or shorter run remains diagnostic and cannot replace the accepted evidence.

## 8. Testing policy for future changes

Minimum after documentation or experiment-harness changes:

```bash
export PYTHONPATH=$PWD:$PWD/oracle

python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
python oracle/test_against_hf_port.py --quick

python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --quick
```

If model/config/oracle/cache/generation/state-dict behavior changes, run the strongest relevant P0-1/P0-2 comparisons, including CUDA bf16 where available, and add a focused regression test.

A model-core change during P0-4 should not be mixed casually into a result-recording PR. First establish a focused diagnosis and the required requalification scope.

## 9. What is safe to assume

Safe:

```text
- the HF math matches the oracle on validated small shapes
- the HF implementation matches the vendored reference on validated shapes
- validated cache splits are consistent across the three implementations
- greedy DynamicCache-compatible generation works in smoke conditions
- short TinyStories bf16 training works for Psi=8 and Psi=16
- the P0-4 harness has passed static and tiny CPU integration checks
- the recorded Psi=8/Psi=16 CUDA bf16 context-4096 short-run qualification passed
```

Not safe:

```text
- paper-scale or paper-quality benchmark reproduction
- efficient long-context memory or runtime
- PEFT/LoRA/QLoRA/Unsloth compatibility
- torch.compile stability at scale
- broad generation compatibility
- vLLM/SGLang serving compatibility
- production readiness
```

## 10. Result logging

Accepted compact summaries belong under:

```text
docs/validation_results/
```

For P0-4, use the template and create sanitized Markdown and JSON summaries. Preserve exact commands, environment versions, GPU identity/memory, qualification flags, losses, gradients, peak memory, reload/cache errors, and generation status.

Do not commit:

```text
outputs/
checkpoints/
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
Hugging Face caches
wandb/
large raw terminal logs
local absolute-path reports
```

See [LOGGING_POLICY.md](LOGGING_POLICY.md) for the canonical policy.

## 11. Files requiring special care

```text
multiscreen_transformers/modeling_multiscreen.py
multiscreen_transformers/configuration_multiscreen.py
oracle/paper_math_oracle.py
oracle/test_against_hf_port.py
p0_2_three_way_minimal/test_three_way_minimal.py
scripts/p0_3_tinystories_stability.py
scripts/p0_4_gpt2_context4096_smoke.py
docs/VALIDATION_STATUS.md
docs/HANDOFF.md
AGENTS.md
```

Changing these files changes either the baseline, its validation interpretation, or the agent instructions used for future development.

## 12. After P0-4

P0-4 is accurately recorded. A subsequent task may now select one focused P1 validation gate, but none of the following candidates is validated yet:

```text
P1-1: PEFT/LoRA smoke
P1-2: QLoRA/bitsandbytes smoke
P1-3: Unsloth loader/wrapper prototype
P1-4: generation compatibility matrix
P1-5: torch.compile smoke
```

Recommended initial LoRA targets remain:

```text
q_proj
k_proj
v_proj
g_proj
o_proj
```

`g_proj` should remain included in the first adapter experiment unless evidence justifies a narrower ablation.

Do not combine all P1 tasks into one long-running goal. Choose one verifiable gate at a time.

## 13. Final handoff report format

```text
変更ファイル:
  - ...

追加ファイル:
  - ...

実行テスト:
  - ...

結果:
  - ...

未確認:
  - ...

次にやるべきこと:
  - ...
```

Always distinguish executed evidence from planned work. A static preflight, CI CPU diagnostic, or reduced local run must never be summarized as a qualifying GPU result.
