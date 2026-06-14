# P0-4 Plan: GPT-2 vocab + context 4096 smoke

## Goal

P0-4 is the next validation gate after the P0-qualified baseline. It moves beyond the TinyStories 768-token tokenizer smoke setting while still remaining a short, correctness-first research smoke.

The gate checks that the current dense Hugging Face-compatible Multiscreen implementation can run with:

- GPT-2 tokenizer / `vocab_size=50257`
- context length 4096
- microbatch size 1 with gradient accumulation
- bf16 autocast training
- short-run finite loss and finite gradient norms
- short-run probe loss decrease
- checkpoint save/load
- cache-enabled greedy generation
- manual cache split equality on a post-check slice

This gate does not test paper-scale pretraining, long-context retrieval quality, long-context efficiency, or production generation compatibility.

## Added scaffold

```text
scripts/p0_4_gpt2_context4096_smoke.py
configs/p0_4_multiscreen_psi8_gpt2_ctx4096/
configs/p0_4_multiscreen_psi16_gpt2_ctx4096/
docs/P0_4_PLAN.md
docs/P0_4_RESULTS_TEMPLATE.md
```

## Recommended execution order

Run from the repository root after the standard setup:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle
```

First, confirm the existing P0 baseline quick checks:

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

Then run the smallest P0-4 diagnostic first. This is not a pass condition for P0-4, but it catches configuration/tokenizer/dataset errors cheaply:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --psi-values 8 \
  --steps-per-psi 8:3 \
  --seq-len 1024 \
  --microbatch-size 1 \
  --grad-accum-steps 1 \
  --amp-dtype bf16 \
  --synthetic-text \
  --output-dir outputs/p0_4_debug_psi8_ctx1024
```

After the diagnostic path works, run the actual Ψ=8 context-4096 smoke:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --psi-values 8 \
  --steps-per-psi 8:50 \
  --seq-len 4096 \
  --microbatch-size 1 \
  --grad-accum-steps 8 \
  --amp-dtype bf16 \
  --output-dir outputs/p0_4_gpt2_ctx4096_psi8
```

Only after Ψ=8 passes, run Ψ=16:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --psi-values 16 \
  --steps-per-psi 16:25 \
  --seq-len 4096 \
  --microbatch-size 1 \
  --grad-accum-steps 8 \
  --amp-dtype bf16 \
  --gradient-checkpointing \
  --output-dir outputs/p0_4_gpt2_ctx4096_psi16
```

If one GPU has enough memory and Ψ=8 and Ψ=16 both pass independently, a combined run may be recorded:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --psi-values 8,16 \
  --steps-per-psi 8:50,16:25 \
  --seq-len 4096 \
  --microbatch-size 1 \
  --grad-accum-steps 8 \
  --amp-dtype bf16 \
  --gradient-checkpointing \
  --output-dir outputs/p0_4_gpt2_ctx4096_combined
```

## Expected output files

Each run writes under `--output-dir`:

```text
metrics.jsonl
run_config.json
p0_4_results.json
P0-4_COMPLETE.md
psi8/p0_4_metrics.json
psi8/config.json
psi8/model.safetensors
psi8/tokenizer files
```

`P0-4_COMPLETE.md` is generated only after all requested Ψ runs finish successfully.

Do not commit `outputs/`, checkpoints, tokenizer cache directories, or local dataset cache directories.

## Pass criteria

A P0-4 pass requires all of the following for the target Ψ value:

- model construction succeeds with GPT-2 vocab size
- tokenizer load succeeds and uses GPT-2-compatible EOS/PAD IDs
- packed text dataset creates at least one context-4096 training batch
- `seq_len=4096` forward/backward training runs for the requested steps
- every recorded training loss is finite
- every recorded gradient norm is finite
- final probe loss is lower than initial probe loss, unless an explicitly documented diagnostic run disables the loss-drop check
- `save_pretrained` and `from_pretrained` reload succeeds
- loaded logits match original logits within the recorded tolerances on the post-check slice
- `generate(use_cache=True)` appends tokens
- manual prefix-cache suffix logits match the full-forward suffix on the post-check slice
- `metrics.jsonl`, `p0_4_results.json`, and `P0-4_COMPLETE.md` are saved

## Failure triage

If the run fails before training starts, first check tokenizer cache access and dataset loading. Passing `--text-file` or `--synthetic-text` isolates model/config issues from Hub or dataset issues.

If the run fails with CUDA OOM, keep Ψ=8, reduce `--grad-accum-steps` only if necessary, try `--gradient-checkpointing`, and reduce `--postcheck-tokens` before changing the core `--seq-len 4096` gate. A `seq_len < 4096` run can be recorded as a diagnostic only, not as a P0-4 pass.

If loss does not decrease, rerun with more steps or a larger `--train-probe-every` frequency. Do not mark P0-4 complete from a run that used `--no-loss-drop-check` unless the result is explicitly labeled as a diagnostic failure investigation.
