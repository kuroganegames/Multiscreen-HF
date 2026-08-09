# P1-preflight B Gradient-Checkpointing Summary

## Verdict

```text
Local gate result: passed
Acceptance state: REVIEW_REQUIRED
Next stage: not started
```

The supported Transformers gradient-checkpointing contract, exact two-release
compatibility matrix, strong P0 regressions, and checkpointed CUDA smokes
passed on 2026-08-09. This local result is not accepted until its focused draft
pull request is reviewed and merged. It does not validate P0.5-C3, final Level 1
requalification, PEFT, or any P1 model/ecosystem capability.

## Provenance

```text
C1 prerequisite: accepted; PR #9 merged
C2 prerequisite: accepted; PR #10 merged
C2 CUDA-autocast correction: PR #11 merged
Stage 3 base / PR #11 merge: 0c83be6b4b043f4b965df4528534f24e9d5ab4f1
branch: agent/p1-preflight-b-gradient-checkpointing
base relation at branch creation: HEAD == origin/main
base worktree: clean
base porcelain bytes: 0
base porcelain SHA-256: e3b0c44298fc1c149afbf4f8996fb92427ae41e4649b934ca495991b7852b855
source/API audit date: 2026-08-09
final local validation date: 2026-08-09
```

The dirty Stage 3 worktree held during PR #11 remained unchanged while a fresh
branch was created from merged `main`. Before and after transfer, its tracked
diff SHA-256 was
`4fe4a6f635d1272cb5d7c5d984bb22910852beddccd23c33b4eeae22c0fc800b`
and its untracked focused-test SHA-256 was
`c3e6da0d4c35709420da45fdc521e6dec53ca406c121d05742978bb77d345201`.
The model-only patch ID matched after transfer, while the PR #11 oracle and C2
test files remained byte-unmodified relative to the new base.

## API result

The model no longer overrides `_set_gradient_checkpointing` with the legacy
`module`/`value` signature. It inherits the Transformers implementation that
propagates `enable` and an installed function to modules exposing the runtime
boolean. `gradient_checkpointing_enable()` copies the caller kwargs, supplies
`use_reentrant=False` only when omitted, and delegates installation to the
supported parent API.

The checkpointed layer loop invokes `self._gradient_checkpointing_func`. The
runtime boolean starts false, so a true flag cannot exist without an installed
callable. Config opt-in is consumed by Transformers `post_init`; runtime harness
opt-in is explicit and not serialized. State dicts and saved config JSON contain
no function object, `use_reentrant`, or checkpoint kwargs.

The focused executable suite directly proves:

```text
- supports_gradient_checkpointing remains true;
- inherited hook has no legacy `value` parameter;
- enable/disable propagates to the Multiscreen runtime module;
- disable prevents an injected function from being invoked;
- arbitrary `preserve_rng_state=False` and non-reentrant default are installed;
- caller kwargs remain unmodified;
- custom checkpoint function is called once per layer;
- a first checkpoint input with requires_grad=False still produces finite
  layer gradients without the reentrant missing-input-gradient warning;
- checkpointed/plain logits, loss, and all parameter gradients agree;
- forward, loss, gradients, parameters, and optimizer step remain finite;
- save/reload logits and greedy generated token IDs match exactly;
- transient runtime objects are absent after reload;
- the committed P0-3 tokenizer has identical vocab/special IDs and causal
  model inputs in the 4.57.6 fallback and 5.14.1 native loader.
```

No old-format deprecation warning or
`None of the inputs have requires_grad=True` warning was observed in either
exact compatibility lane.

## Compatibility environments

No package was installed, upgraded, or removed after resume. The active Conda
base environment was not mutated. Two existing isolated Stage 3 environments
were reused:

```text
recorded lane:
  Python 3.12.11
  PyTorch 2.7.1+cu128
  Transformers 4.57.6
  tokenizers 0.22.0
  safetensors 0.5.3
  NumPy 1.26.4

current lane:
  Python 3.12.10
  PyTorch 2.8.0+cu128
  Transformers 5.14.1
  tokenizers 0.22.2
  safetensors 0.8.0
  NumPy 2.3.2

CUDA validation:
  CUDA runtime reported by PyTorch: 12.8
  GPU 0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
  driver: 595.71.05
```

Official release records showed 5.14.1 as the current Transformers release on
the audit date. They also showed 4.57.0 as yanked for setup/installation
problems. Active dependency declarations and the lower-bound CI job therefore
use non-yanked 4.57.6; the compatibility matrix pins exactly 4.57.6 and 5.14.1.

## Test results

```text
focused gradient-checkpointing contract:
  Transformers 4.57.6: 7/7 passed
  Transformers 5.14.1: 7/7 passed

all tests/test_*.py under 4.57.6 with CUDA available:
  103/103 passed

C2 focused contract under 4.57.6:
  CPU-only: 24 passed, 1 expected CUDA skip
  CUDA bf16 available: 25/25 passed

C1 focused contracts:
  architecture: 5/5 passed
  initialization: 3/3 passed
  packed text: 5/5 passed
  checked manifest: exact match

P1-preflight A evidence tooling:
  58/58 passed with Python -S

formula/oracle support:
  formula units: passed
  paper oracle self-check: passed
  paper oracle smoke: passed

P0-1 full CPU fp32: all 744 sweep checks passed
P0-1 full CUDA bf16: all 744 sweep checks passed
P0-2 full CPU fp32: all 282 three-way comparisons passed
P0-2 full CUDA bf16: all 282 three-way comparisons passed

repository syntax, JSON, workflow YAML, changed Markdown local links, git diff, and
artifact/privacy/symlink/size hygiene: passed
```

A full-tree Markdown link scan also reported two pre-existing relative-link
issues in `APPLY_REPOSITORY_AUDIT.md` and
`repo_audit_supplement/APPLY_REPOSITORY_AUDIT.md`. Neither file is changed by
this gate; all changed Markdown files passed the focused link scan.

P0-1 CUDA bf16 uses stable fp32 auxiliary MiPE/Softmask math. P0-2 CUDA
bf16 deliberately uses incoming-dtype reference arithmetic to match the
vendored implementation; it is compatibility evidence, not long-position
causal-correctness evidence.

## P0-3 checkpointed smoke

The recorded 4.57.6 lane ran offline TinyStories checkpointed CUDA bf16 smokes
with sequence length 128, batch size 4, stable fp32 auxiliary math, and explicit
`use_reentrant=False`:

| Psi | Steps | Initial probe loss | Final probe loss | Relative drop | Max grad norm | Save/reload max abs | Cache max abs | Generation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 40 | 8.215893 | 4.312126 | 47.5148% | 10.255060 | 0 | 0 | 4 -> 16 tokens |
| 16 | 25 | 15.899660 | 5.927410 | 62.7199% | 25.480322 | 0 | 0 | 4 -> 16 tokens |

All train losses and gradient norms were finite. Both Psi runs completed
save/load, manual cache split, and greedy cached generation postchecks.

The first fresh-tree attempt completed Psi=8 training, save/reload, and cache
split but exposed that the 4.57.6 tokenizer fallback emitted unused
`token_type_ids` during generation. The fallback was narrowed to the same
`input_ids`/`attention_mask` causal contract as the 5.14.1 native loader, a
focused regression was added, and the complete Psi=8/Psi=16 smoke was rerun
from the beginning in a new output directory. Only the final passing rerun is
counted above.

## Reduced P0-4 checkpointed diagnostic

The recorded lane ran Psi=8 with GPT-2 vocabulary 50,257, CUDA bf16, context
1024, microbatch 1, gradient accumulation 1, four optimizer steps, and explicit
non-reentrant checkpointing:

```text
status: diagnostic_passed
initial probe loss: 11.143433
final probe loss: 10.727448
relative probe drop: 3.7330%
train loss first/last: 11.191833 / 10.908966
max grad norm: 5.849789
save/reload logits max abs: 0
cache-split logits max abs: 0.0078125
generation: 4 -> 12 tokens
peak allocated bytes: 603,976,192
```

The marker is `P0-4_DIAGNOSTIC_COMPLETE.md`, not `P0-4_COMPLETE.md`.
Qualification is false because context is 1024 rather than 4096 and the run has
four rather than at least 50 optimizer steps. This is Stage 3 training-path
evidence only and does not replace the accepted historical P0-4 result.

## Evidence and limitations

This compact summary and deterministic tests are the committed Stage 3
evidence. Raw logs, failed-attempt diagnostics, smoke outputs, checkpoints,
tokenizer copies, and model weights remain outside Git in ephemeral local
storage. No public asset or durable archive was created, and no acceptance
reviewer is inferred before the draft PR is reviewed.

Historical P0-4 retention remains partial/blocked exactly as recorded. The
accepted historical metrics and descriptor were not rewritten.

Transformers 5.14.1 installs an input-embedding gradient hook for checkpointed
`input_ids`, while Multiscreen performs normalized lookup through
`F.embedding`. The focused no-grad-input test establishes the supported
non-reentrant full-model path; it does not establish frozen-base PEFT behavior.
PEFT/LoRA remains a later, separate gate. Broad generation, paper-scale
training, retrieval quality, long-context efficiency, compile, serving, and
production readiness are also unvalidated.
