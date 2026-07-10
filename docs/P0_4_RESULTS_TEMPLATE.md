# P0-4 GPT-2 Vocabulary + Context 4096 Results

> Template only. Replace placeholders from generated `summary.json`, `metrics.jsonl`, and `P0-4_COMPLETE.md`. Do not mark P0-4 complete unless the qualification block reports `qualified: true`.

## Result

```text
Status: pending | passed | failed
Run date (UTC): <YYYY-MM-DD>
Commit SHA: <sha>
Reviewer: <name or handle>
```

## Qualification

| Condition | Psi=8 | Psi=16 |
|---|---:|---:|
| GPT-2 vocab = 50,257 | pending | pending |
| context = 4,096 | pending | pending |
| CUDA device | pending | pending |
| bf16 AMP | pending | pending |
| optimizer steps >= 50 | pending | pending |
| generated `P0-4_COMPLETE.md` | pending | pending |

## Environment

```text
Python: <version>
PyTorch: <version>
Transformers: <version>
Datasets: <version>
CUDA runtime: <version>
GPU: <name>
GPU memory: <bytes or GiB>
bf16 supported: <true/false>
```

## Commands

### P0 baseline quick checks

```bash
<commands and exit status>
```

### Psi=8

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096
```

### Psi=16

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

## Data and tokenizer

| Field | Psi=8 | Psi=16 |
|---|---:|---:|
| tokenizer class | `<value>` | `<value>` |
| tokenizer vocab size | `<value>` | `<value>` |
| dataset source | `<value>` | `<value>` |
| train split | `<value>` | `<value>` |
| texts loaded | `<value>` | `<value>` |
| packed chunks | `<value>` | `<value>` |
| max train tokens | `<value>` | `<value>` |

## Model and training metrics

| Metric | Psi=8 | Psi=16 |
|---|---:|---:|
| parameters | `<value>` | `<value>` |
| hidden size | `<value>` | `<value>` |
| layers | `<value>` | `<value>` |
| heads | `<value>` | `<value>` |
| sequence length | `<value>` | `<value>` |
| microbatch size | `<value>` | `<value>` |
| gradient accumulation | `<value>` | `<value>` |
| effective batch tokens | `<value>` | `<value>` |
| optimizer steps | `<value>` | `<value>` |
| learning rate | `<value>` | `<value>` |
| initial probe loss | `<value>` | `<value>` |
| final probe loss | `<value>` | `<value>` |
| absolute loss drop | `<value>` | `<value>` |
| relative loss drop | `<value>` | `<value>` |
| first train loss | `<value>` | `<value>` |
| last train loss | `<value>` | `<value>` |
| minimum train loss | `<value>` | `<value>` |
| maximum finite grad norm | `<value>` | `<value>` |
| elapsed seconds | `<value>` | `<value>` |
| peak CUDA allocated bytes | `<value>` | `<value>` |
| peak CUDA reserved bytes | `<value>` | `<value>` |

## Correctness checks

| Check | Psi=8 | Psi=16 |
|---|---:|---:|
| finite train loss | pending | pending |
| finite gradient norm | pending | pending |
| probe loss decrease | pending | pending |
| save_pretrained | pending | pending |
| from_pretrained | pending | pending |
| loaded logits max abs | `<value>` | `<value>` |
| loaded logits tolerance | `<atol/rtol>` | `<atol/rtol>` |
| generate(use_cache=True) | pending | pending |
| generated length | `<value>` | `<value>` |
| cache split logits max abs | `<value>` | `<value>` |
| cache tolerance | `<atol/rtol>` | `<atol/rtol>` |

## Stepwise behavior

Summarize notable events from `metrics.jsonl`, including any transient loss spike, memory growth, fallback from fused AdamW, or retry. Preserve the original JSONL as the machine-readable record.

```text
Psi=8 notes:
- <note>

Psi=16 notes:
- <note>
```

## Output artifacts

```text
Psi=8 output directory: <path>
Psi=8 summary.json SHA-256: <hash>
Psi=8 metrics.jsonl SHA-256: <hash>
Psi=8 checkpoint retained externally: <yes/no/location>

Psi=16 output directory: <path>
Psi=16 summary.json SHA-256: <hash>
Psi=16 metrics.jsonl SHA-256: <hash>
Psi=16 checkpoint retained externally: <yes/no/location>
```

Do not commit `checkpoint/` or the full `outputs/` directory. Commit only compact, sanitized summaries under `docs/validation_results/` when the result is accepted.

## Verdict

```text
Psi=8: pass | fail | not run
Psi=16: pass | fail | not run
P0-4 overall: complete | partial | failed | pending
```

## Scope and limitations

This result confirms only the short dense-reference smoke conditions recorded above. It does not establish:

```text
- paper-scale pretraining or benchmark reproduction
- long-context runtime or memory efficiency
- long-context retrieval quality
- Triton/windowed kernel behavior
- PEFT/LoRA/QLoRA/Unsloth readiness
- broad or production generation compatibility
- vLLM/SGLang serving compatibility
```

## Follow-up

```text
- <issue or next experiment>
- <documentation update>
- <whether P1-1 PEFT/LoRA may begin>
```
