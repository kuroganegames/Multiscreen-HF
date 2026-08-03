# Development Handoff

This is the primary development-restart document for `kuroganegames/Multiscreen-HF`.

The repository contains a **P0-qualified research implementation** of Multiscreen for Hugging Face Transformers. P0-1, P0-2, and P0-3 are complete. PR #3 added and exercised the P0-4 harness, but the qualifying local CUDA bf16 context-4096 runs remain pending.

The current phase is therefore:

```text
P0-4 execution, evidence review, and validation-record update
```

For Codex-based continuation after a local clone, read the root [`AGENTS.md`](../AGENTS.md) and [`CODEX_P0_4_HANDOFF.md`](CODEX_P0_4_HANDOFF.md).

For repository hygiene checks, see [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md).

## 1. Current project state

### Milestones

| Milestone | Status | Meaning |
|---|---:|---|
| P0-1 | Complete | `paper_math_oracle` and the HF implementation agree on validated small-shape formula, loss, mask, position, and cache behavior. |
| P0-2 | Complete | The vendored unofficial PyTorch reference, HF implementation, and oracle agree in recorded CPU fp32 and CUDA bf16 sweeps. |
| P0-3 | Complete | Psi=8/16 TinyStories bf16 smoke training passed, including finite loss/gradients, save/load, cache split, and greedy generation. |
| P0-4 harness | Merged | GPT-2 vocab/context-4096 harness, Psi=8/16 configs, plan, result template, static checks, and tiny CPU integration diagnostic were merged in PR #3. |
| P0-4 qualifying execution | Pending | CUDA bf16, context 4096, at least 50 optimizer steps, Psi=8 first and Psi=16 second. |
| P1 ecosystem work | Not started as validated work | PEFT/LoRA, QLoRA, Unsloth, generation matrix, compile, and serving remain future work. |

### Baseline identity

```text
Current status: P0-qualified research baseline based on P0-1/P0-2/P0-3
Primary implementation: multiscreen_transformers/modeling_multiscreen.py
Primary config: multiscreen_transformers/configuration_multiscreen.py
Primary equation oracle: oracle/paper_math_oracle.py
Primary validation record: docs/VALIDATION_STATUS.md
Current execution plan: docs/P0_4_PLAN.md
Codex local handoff: docs/CODEX_P0_4_HANDOFF.md
Repository instructions for Codex: AGENTS.md
```

P0-4 is an additional pending gate. Static config validation, CI CPU diagnostics, or reduced-context runs do not complete it.

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

If these checks pass, the checkout matches the expected merged baseline at the CPU/static level. It still has not reproduced the qualifying P0-4 CUDA runs.

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

Then use the complete prompt in [CODEX_P0_4_HANDOFF.md](CODEX_P0_4_HANDOFF.md). That goal is intentionally limited to P0-4 qualification and evidence recording. It defines both a successful completion state and a blocked-with-evidence state without weakening the gate.

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

## 7. P0-4 current state

PR #3 added:

```text
scripts/p0_4_gpt2_context4096_smoke.py
configs/p0_4_multiscreen_psi8_gpt2_ctx4096/
configs/p0_4_multiscreen_psi16_gpt2_ctx4096/
docs/P0_4_PLAN.md
docs/P0_4_RESULTS_TEMPLATE.md
CI static preflight for both configs
CI tiny Psi=8 CPU end-to-end diagnostic
```

The CI diagnostic exercised tokenizer loading, packed data, model construction, forward/backward, finite loss/gradient, save/load, generation, cache checks, and correct diagnostic-note classification. It did not use CUDA bf16 context 4096 and therefore is not a qualifying P0-4 result.

A qualifying run requires:

```text
GPT-2 vocabulary: exactly 50,257
sequence length: exactly 4,096
device: CUDA
AMP dtype: bf16
optimizer steps: at least 50
finite train loss and gradient norms
probe-loss decrease
save/load logits within configured tolerance
greedy generate(use_cache=True)
manual cache split within configured tolerance
metrics.jsonl, summary.json, P0-4_COMPLETE.md
```

A reduced-context, CPU, non-bf16, or shorter run writes `P0-4_DIAGNOSTIC_COMPLETE.md` and must remain diagnostic.

Execution order:

```text
1. baseline quick tests
2. both config preflights
3. optional Psi=8 reduced diagnostic
4. qualifying Psi=8
5. artifact and memory review
6. qualifying Psi=16
7. compact sanitized result records
8. status-document updates and final regressions
```

If local hardware cannot complete an unweakened run, preserve evidence and record `partial`, `failed`, or `blocked`. Do not silently change the acceptance criteria.

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
```

Not safe:

```text
- P0-4 CUDA bf16 context-4096 qualification has passed
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

Only after the P0-4 outcome is accurately recorded should the project select a P1 task. Current candidates are:

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
