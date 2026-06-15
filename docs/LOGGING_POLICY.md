# Validation Logging Policy

This document defines what validation logs should be kept in the repository and how new validation runs should be recorded.

The repository is a research artifact. Raw terminal logs can be useful during debugging, but the repository should keep **compact, human-readable and machine-readable summaries** rather than large logs or checkpoints.

## Goals

The logging policy should make it possible to answer these questions after a fresh clone:

```text
1. Which validation gates have passed?
2. Which exact checks were run?
3. Which dtype/device combinations were covered?
4. What counts/results were observed?
5. Which artifacts/scripts produced those results?
6. What remains unvalidated?
```

## Files to keep

Validation summaries should live under:

```text
docs/validation_results/
```

Recommended files:

```text
P0_1_SUMMARY.md
P0_1_SUMMARY.json
P0_2_SUMMARY.md
P0_2_SUMMARY.json
P0_3_SUMMARY.md
P0_3_SUMMARY.json
VALIDATION_LOG_INDEX.md
```

Markdown files are for humans. JSON files are for future automation and scripts.

## What to record for each run

Each validation run should record at least:

```text
validation gate name
status: passed / failed / partial
command or script name
device
amp / dtype
quick or full
key counts
important metrics
known caveats
```

For training smoke tests, also record:

```text
model size
steps
seq_len
batch_size
dataset
tokenizer
initial loss
final loss
absolute loss drop
relative loss drop
save/load status
generation/cache status
```

## What not to commit

Do not commit:

```text
outputs/
checkpoints/
*.safetensors
*.bin
*.pt
*.pth
wandb/
large raw terminal logs
cache directories
__pycache__/
*.pyc
```

If raw logs are useful, keep them outside the repository or attach them to a release/issue only when needed.

## When to update logs

Update the validation summaries whenever any of these files change:

```text
multiscreen_transformers/modeling_multiscreen.py
multiscreen_transformers/configuration_multiscreen.py
oracle/paper_math_oracle.py
oracle/test_against_hf_port.py
p0_2_three_way_minimal/test_three_way_minimal.py
scripts/p0_3_tinystories_stability.py
```

Minimum rerun policy:

```text
modeling/config/oracle change:
  rerun P0-1 quick and P0-2 quick

cache/generation change:
  rerun P0-1 quick and a P0-3 quick smoke

training script change:
  rerun P0-3 quick or full depending on the change

P0-qualified release/tag:
  rerun P0-1 CPU fp32 full, P0-2 CPU fp32 full, and at least CUDA bf16 quick/full if available
```
