# P0-4 Results Template

Use this template to record a completed GPT-2 vocab + context-4096 smoke run. Do not mark P0-4 complete until the script has generated `P0-4_COMPLETE.md` and the existing P0 quick checks still pass.

## Environment

```text
Date:
Commit:
Branch:
Host/GPU:
Python:
PyTorch:
Transformers:
CUDA:
Tokenizer source:
Dataset source:
```

## Commands

### Existing P0 quick checks

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

### P0-4 smoke command

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

## Result summary

```text
Overall result: PASS / FAIL / DIAGNOSTIC ONLY
Output dir:
metrics.jsonl present: yes/no
p0_4_results.json present: yes/no
P0-4_COMPLETE.md present: yes/no
```

## Ψ=8 metrics

```text
params:
steps:
seq_len:
microbatch_size:
grad_accum_steps:
amp_dtype:
initial_probe_loss:
final_probe_loss:
abs_loss_drop:
rel_loss_drop:
train_loss_first:
train_loss_last:
grad_norm_max:
save_load_logits_max_abs:
cache_split_logits_max_abs:
generate prompt_len:
generate generated_len:
cuda_peak_allocated_gib:
cuda_peak_reserved_gib:
checkpoint_dir:
```

## Ψ=16 metrics

Run Ψ=16 only after Ψ=8 passes.

```text
params:
steps:
seq_len:
microbatch_size:
grad_accum_steps:
amp_dtype:
initial_probe_loss:
final_probe_loss:
abs_loss_drop:
rel_loss_drop:
train_loss_first:
train_loss_last:
grad_norm_max:
save_load_logits_max_abs:
cache_split_logits_max_abs:
generate prompt_len:
generate generated_len:
cuda_peak_allocated_gib:
cuda_peak_reserved_gib:
checkpoint_dir:
```

## Existing P0 quick-check results after P0-4 scaffold/run

```text
oracle/test_formula_units.py: PASS/FAIL
oracle/test_paper_math_oracle_selfcheck.py: PASS/FAIL
oracle/test_paper_math_oracle_smoke.py: PASS/FAIL
oracle/test_against_hf_port.py --quick: PASS/FAIL
p0_2_three_way_minimal/test_three_way_minimal.py --quick: PASS/FAIL
```

## Notes and anomalies

```text
OOMs:
Non-finite losses:
Non-finite grad norms:
Loss-drop failures:
Save/load failures:
Generation/cache failures:
Deviations from default args:
```

## Scope statement

This result confirms only the P0-4 short smoke path. It does not confirm paper-scale pretraining, long-context efficiency, custom kernels, PEFT/LoRA/QLoRA, Unsloth, serving stacks, or production generation compatibility.
