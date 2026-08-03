# P0-4 Summary: GPT-2 vocabulary + context-4096 smoke

## Status

```text
passed
```

Both intended model orders passed the unweakened P0-4 qualification. The full
ignored outputs were reviewed event by event; only this compact sanitized record
is committed.

## Provenance

```text
tested source commit: 3d734d74e04ce6a320fb31cf3d8241f823ff43fa
source branch: validation/p0-4-local-cuda
run date (UTC): 2026-08-03
Psi=8 completion: 2026-08-03T15:59:10.372865Z
Psi=16 completion: 2026-08-03T16:07:54.047702Z
device: cuda:0
GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
GPU total memory reported by PyTorch: 101,973,491,712 bytes
GPU compute capability: 12.0
NVIDIA driver: 595.71.05
CUDA runtime reported by PyTorch: 12.8
bf16 supported: true
```

Software:

```text
Python: 3.12.11
PyTorch: 2.7.1+cu128
Transformers: 4.57.6
Datasets: 5.0.1
Tokenizers: 0.22.0
Safetensors: 0.5.3
NumPy: 1.26.4
Accelerate: 1.6.0
TRL: 1.9.2
SentencePiece: 0.2.0
```

## Qualification

| Condition | Psi=8 | Psi=16 |
|---|---:|---:|
| GPT-2 tokenizer vocabulary = 50,257 | passed | passed |
| context = 4,096 | passed | passed |
| CUDA device | passed | passed |
| bf16 AMP | passed | passed |
| microbatch = 1 | passed | passed |
| optimizer steps >= 50 | passed (50) | passed (50) |
| gradient checkpointing active | passed | passed |
| all train losses finite | passed | passed |
| all gradient norms finite | passed | passed |
| configured probe-loss decrease | passed | passed |
| save/load and loaded logits | passed | passed |
| tokenizer reload | passed | passed |
| `generate(use_cache=True)` appends tokens | passed | passed |
| manual cache split within configured tolerance | passed | passed |
| `metrics.jsonl`, `summary.json`, `P0-4_COMPLETE.md` | present | present |
| failure/diagnostic marker absent | passed | passed |
| `qualification.qualified` | `true` | `true` |

The qualification flags were not accepted on their own. Review also verified
microbatch size, gradient checkpointing, every one of the 50 step events,
loss/gradient finiteness, the probe threshold, correctness checks, marker files,
and the absence of failure artifacts.

## Data and training configuration

| Field | Psi=8 | Psi=16 |
|---|---:|---:|
| tokenizer | GPT-2 fast tokenizer | GPT-2 fast tokenizer |
| dataset | `roneneldan/TinyStories` | `roneneldan/TinyStories` |
| split | `train[:20000]` | `train[:20000]` |
| texts loaded | 20,000 | 20,000 |
| packed chunks | 128 | 128 |
| maximum packed tokens | 524,416 | 524,416 |
| parameters | 4,134,146 | 27,546,626 |
| hidden size | 64 | 256 |
| layers / heads | 8 / 8 | 16 / 16 |
| key / value dimension | 16 / 64 | 16 / 64 |
| sequence length | 4,096 | 4,096 |
| microbatch | 1 | 1 |
| gradient accumulation | 8 | 8 |
| effective batch tokens | 32,768 | 32,768 |
| optimizer steps | 50 | 50 |
| learning rate | 0.0006 | 0.0006 |
| weight decay | 0 | 0 |
| AMP dtype | bf16 | bf16 |

## Results

| Metric | Psi=8 | Psi=16 |
|---|---:|---:|
| initial probe loss | 11.14074707 | 15.79932117 |
| final probe loss | 4.67538166 | 3.49560118 |
| absolute probe-loss drop | 6.46536541 | 12.30372000 |
| relative probe-loss drop | 58.0335% | 77.8750% |
| first train loss | 11.18622065 | 15.67019737 |
| last train loss | 5.87040472 | 4.47259158 |
| minimum train loss | 4.75179291 | 3.60795999 |
| maximum finite gradient norm | 5.39385700 | 23.19463158 |
| training elapsed seconds | 107.6805 | 425.8058 |
| peak CUDA allocated bytes | 3,156,709,888 | 6,622,802,944 |
| peak CUDA reserved bytes | 4,525,654,016 | 9,130,999,808 |
| loaded-logits maximum absolute error | 0 | 0 |
| reload tolerance (`atol`, `rtol`) | 1e-5, 1e-5 | 1e-5, 1e-5 |
| cache-split maximum absolute error | 0 | 0.125 |
| cache tolerance (`atol`, `rtol`) | 0.03, 0.03 | 0.03, 0.03 |
| prompt / generated length | 4 / 12 | 4 / 12 |

The Psi=16 cache comparison passed the configured combined absolute/relative
predicate. A separate review reproduction found zero violating elements; at the
maximum-absolute-difference element, the permitted error was 0.52125, and the
largest `difference - permitted_error` over all elements was -0.0239368. The
0.125 maximum absolute value must therefore not be compared to `atol` alone.

## Event review

Each accepted `metrics.jsonl` contained exactly 57 events:

```text
run_start
preflight_complete
train_step x 50, numbered 1 through 50
training_complete
save_reload_check
cache_split_check
generation_check
run_complete
```

All recorded train and microbatch losses and gradient norms were finite. Peak
allocated and reserved memory stabilized after step 2 for both runs. Periodic
probe-replay steps produced the expected lower-loss sawtooth; there was no
unbounded spike or memory growth. Fused AdamW did not fall back. Transformers
4.57.6 emitted a deprecation warning for the model's legacy gradient-checkpointing
hook, but checkpointed training completed and this evidence-only validation did
not change model core.

An optional Psi=8 context-1024, two-step CUDA bf16 diagnostic ran first. It
passed its correctness checks, wrote `P0-4_DIAGNOSTIC_COMPLETE.md` rather than
`P0-4_COMPLETE.md`, and reported `qualification.qualified=false`; it is not
counted as qualifying evidence.

## Commands

The local absolute interpreter path is intentionally omitted. The arguments
below are otherwise the executed reproducibility commands.

```bash
export PYTHONPATH=$PWD:$PWD/oracle

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

The harness saves tokenizer files but does not reload them itself. The following
post-run reviewer check was therefore executed against both ignored checkpoints:

```bash
python -c "from transformers import AutoTokenizer; paths=['outputs/p0_4_multiscreen_psi8_gpt2_ctx4096/checkpoint','outputs/p0_4_multiscreen_psi16_gpt2_ctx4096/checkpoint']; values=[(AutoTokenizer.from_pretrained(p, local_files_only=True, use_fast=True).__class__.__name__, len(AutoTokenizer.from_pretrained(p, local_files_only=True, use_fast=True))) for p in paths]; print(values)"
```

It returned `GPT2TokenizerFast` with vocabulary length 50,257 for both
checkpoints.

Before qualification, the formula-unit, oracle self-check, oracle smoke,
P0-1 quick, and P0-2 quick regressions passed. P0-1 recorded 94 checks; P0-2
recorded 12 prefill and 28 cache-split comparisons. Python syntax and both
P0-4 static config preflights also passed.

## Evidence hashes

```text
Psi=8 summary.json:
  ee866111911ad289a5d01303f53164bfb3065084030dc4f9267f9b29723ca4b5
Psi=8 metrics.jsonl:
  97b14ce7888cdbe14b19bc5dba859c63f02f4571e866edbb0c39b1ebe0e985ad

Psi=16 summary.json:
  679f17d8de1cf6fc5324ea79b05b14afbcdcb0986d0101b51ea492ec814b406f
Psi=16 metrics.jsonl:
  28d2fe326c3d12ee92bfd1a747658751d5f7ce962a97b0781ad86369b0942269
```

The raw summaries, metrics, generated notes, and checkpoints remain ignored and
local. They contain machine-local paths and are not committed. The source SHA
above identifies the code under test; the later evidence-documentation commit is
intentionally a different commit.

## Interpretation and limits

P0-4 confirms short-run feasibility and stability for the checked-in Psi=8 and
Psi=16 dense models at GPT-2 vocabulary size and context 4096 on the recorded
CUDA bf16 environment. It does not establish paper-scale training, retrieval
quality, long-context efficiency, cross-hardware reproducibility, optimized
kernels, P1 adapter compatibility, broad generation behavior, serving readiness,
or production readiness. Runtime and memory figures above are feasibility
diagnostics only.
