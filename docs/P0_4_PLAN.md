# P0-4 Plan: GPT-2 Vocabulary + Context 4096 Smoke

## Status

Implementation and reproducibility harness: **merged in PR #3**.

Static Psi=8/Psi=16 preflight and tiny Psi=8 CPU end-to-end diagnostic: **passed in GitHub Actions**.

Qualifying CUDA bf16 execution: **pending**.

For a complete local Codex `/goal` workflow, see [CODEX_P0_4_HANDOFF.md](CODEX_P0_4_HANDOFF.md). Codex also reads the repository rules in [`AGENTS.md`](../AGENTS.md) when launched from the repository root.

P0-4 must not be marked complete in `docs/VALIDATION_STATUS.md` until qualifying runs have produced the required artifacts and their metrics have been reviewed. Static validation, CI CPU diagnostics, and reduced-context runs remain non-qualifying.

## Objective

P0-4 moves the P0-qualified Multiscreen baseline beyond the 768-token TinyStories tokenizer and 128-token smoke setting. It checks that the dense Hugging Face implementation remains trainable with:

```text
GPT-2 tokenizer vocabulary: 50,257
context length:              4,096
microbatch:                  1
AMP dtype:                   bf16
optimizer steps:             at least 50
model order:                 Psi=8 first, then Psi=16
```

This is a correctness and stability gate. It is not a throughput benchmark, a long-context efficiency claim, or paper-scale pretraining.

## Files

```text
scripts/p0_4_gpt2_context4096_smoke.py
configs/p0_4_multiscreen_psi8_gpt2_ctx4096/config.json
configs/p0_4_multiscreen_psi8_gpt2_ctx4096/run.json
configs/p0_4_multiscreen_psi16_gpt2_ctx4096/config.json
configs/p0_4_multiscreen_psi16_gpt2_ctx4096/run.json
docs/P0_4_PLAN.md
docs/P0_4_RESULTS_TEMPLATE.md
docs/CODEX_P0_4_HANDOFF.md
AGENTS.md
```

The P0 model core, paper oracle, state-dict conversion, and cache implementation were intentionally unchanged by PR #3.

## Acceptance criteria

A run writes `P0-4_COMPLETE.md` only when all of the following are true:

```text
- tokenizer vocabulary is exactly 50,257
- sequence length is exactly 4,096
- device is CUDA
- AMP dtype is bf16
- at least 50 optimizer steps completed
- packed input batch has shape [microbatch, 4096]
- train losses remain finite
- gradient norms remain finite
- probe loss decreases by the configured absolute or relative threshold
- save_pretrained / from_pretrained succeeds
- loaded logits pass configured atol/rtol comparison
- generate(use_cache=True) appends tokens
- manual cache split logits match the full-forward suffix
- metrics.jsonl and summary.json are written
```

A reduced-context, CPU, non-bf16, or shorter run can still exercise the code path, but it writes `P0-4_DIAGNOSTIC_COMPLETE.md` and does not qualify P0-4.

For the project-level P0-4 completion record, review both intended model sizes. A Psi=8 pass does not imply a Psi=16 pass. If Psi=16 cannot complete on the available system, record the overall state as partial or blocked rather than complete.

## Execution order

### 1. Restore the P0 baseline

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
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

Do not proceed to qualifying training if the baseline quick suite fails. Separate environment failures from repository regressions before changing code.

### 2. Validate the checked-in P0-4 configs without downloading data

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --validate-config-only

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --validate-config-only
```

### 3. Optional reduced diagnostic

This is useful for catching environment, tokenizer, data, save/load, and generation issues before allocating a full 4096-token run. It is not a P0-4 pass.

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --seq-len 1024 \
  --steps 2 \
  --gradient-accumulation-steps 1 \
  --output-dir outputs/p0_4_psi8_ctx1024_diagnostic
```

Verify that the reduced run writes `P0-4_DIAGNOSTIC_COMPLETE.md` and does not write `P0-4_COMPLETE.md`.

### 4. Qualifying Psi=8 run

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096
```

Review `summary.json`, every event in `metrics.jsonl`, and `P0-4_COMPLETE.md` before proceeding. Confirm `qualification.qualified=true` rather than relying only on the filename.

### 5. Qualifying Psi=16 run

Run only after Psi=8 passes and memory headroom is understood.

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

Apply the same artifact and qualification review independently.

## Output contract

Each run writes below its configured `outputs/` directory:

```text
metrics.jsonl
summary.json
checkpoint/
P0-4_COMPLETE.md                 qualifying run only
P0-4_DIAGNOSTIC_COMPLETE.md      non-qualifying passed run only
failure.json                     failed run only
P0-4_FAILED.md                   failed run only
```

Checkpoints and `outputs/` must remain untracked.

## Memory and runtime caution

The current implementation materializes dense screening tensors and remains quadratic in sequence length. At bf16, a single similarity tensor has this lower-bound size before masks, activations, gradients, optimizer state, and framework overhead:

```text
Psi=8:  1 x 8  x 4096 x 4096 x 2 bytes = 0.250 GiB per layer invocation
Psi=16: 1 x 16 x 4096 x 4096 x 2 bytes = 0.500 GiB per layer invocation
```

The actual peak is substantially larger. The script records CUDA allocated/reserved peaks and a clearly labeled similarity-tensor lower bound. These figures diagnose feasibility; they must not be reported as an efficiency benchmark.

Before retrying an OOM, check for unrelated GPU processes and confirm that microbatch 1 and gradient checkpointing are active. Do not weaken the qualifying context, dtype, Psi, or minimum-step rule.

## Failure triage

For out-of-memory failures:

1. Preserve `failure.json` and the last `metrics.jsonl` event locally.
2. Confirm gradient checkpointing is enabled and microbatch is 1.
3. Check available GPU memory and unrelated processes.
4. Run the 1024 diagnostic to separate correctness failures from 4096 memory pressure.
5. Do not reduce the qualifying sequence length or bf16 requirement and still call the result P0-4 complete.
6. Do not move to Psi=16 until Psi=8 has a reviewed qualifying result.

For cache or reload mismatches, rerun P0-1/P0-2 quick checks and identify whether the failure is in the harness, environment, or baseline before changing the model core.

If evidence indicates a model-core, oracle, position, mask, cache, generation, or state-dict defect, create a focused diagnosis and a separate correctness change with the required stronger P0 reruns. Do not hide such a change inside a result-recording PR.

## Recording the result

Use `docs/P0_4_RESULTS_TEMPLATE.md`. Retain full outputs locally and commit only compact sanitized summaries.

Suggested files:

```text
docs/validation_results/P0_4_SUMMARY.md
docs/validation_results/P0_4_SUMMARY.json
```

Record SHA-256 for each accepted `summary.json` and `metrics.jsonl`, plus:

```text
- exact command
- commit SHA
- Python/package versions
- GPU model and total memory
- qualification conditions
- losses and loss drop
- maximum finite gradient norm
- peak allocated/reserved CUDA memory
- save/load logits error
- cache-split logits error
- generation status
- pass, partial, failed, or blocked verdict
```

Remove secrets, unnecessary host/user identifiers, cache paths, local absolute paths, and checkpoint paths from committed summaries.

After reviewed execution, update according to evidence:

```text
README.md
docs/HANDOFF.md
docs/VALIDATION_STATUS.md
docs/TESTING.md
docs/KNOWN_LIMITATIONS.md
docs/validation_results/VALIDATION_LOG_INDEX.md
```

Do not replace `pending` with `complete` based only on static config validation, CI CPU diagnostics, or a reduced local run.
