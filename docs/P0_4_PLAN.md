# P0-4 Plan: GPT-2 Vocabulary + Context 4096 Smoke

## Status

Implementation and reproducibility harness: **prepared**.

Qualifying CUDA bf16 execution: **pending**.

P0-4 must not be marked complete in `docs/VALIDATION_STATUS.md` until a qualifying run has produced `P0-4_COMPLETE.md` and its metrics have been reviewed.

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
```

The P0 model core, paper oracle, state-dict conversion, and cache implementation are intentionally unchanged by this task.

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

### 4. Qualifying Psi=8 run

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096
```

Review `summary.json`, every event in `metrics.jsonl`, and `P0-4_COMPLETE.md` before proceeding.

### 5. Qualifying Psi=16 run

Run only after Psi=8 passes and memory headroom is understood.

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

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

## Failure triage

For out-of-memory failures:

1. Preserve `failure.json` and the last `metrics.jsonl` event.
2. Confirm gradient checkpointing is enabled and microbatch is 1.
3. Run the 1024 diagnostic to separate correctness failures from 4096 memory pressure.
4. Do not reduce the qualifying sequence length or bf16 requirement and still call the result P0-4 complete.
5. Do not move to Psi=16 until Psi=8 has a reviewed qualifying result.

For cache or reload mismatches, rerun P0-1/P0-2 quick checks and then the relevant full or CUDA bf16 comparison before changing the model core.

## Recording the result

Use `docs/P0_4_RESULTS_TEMPLATE.md`. After both intended runs have been reviewed, update:

```text
README.md
docs/HANDOFF.md
docs/VALIDATION_STATUS.md
docs/TESTING.md
docs/KNOWN_LIMITATIONS.md
docs/validation_results/VALIDATION_LOG_INDEX.md
```

Do not replace “pending” with “complete” based only on static config validation or a reduced diagnostic.
