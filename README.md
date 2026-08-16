# Multiscreen HF

> \[!WARNING]
> Directed by GPT-5.6 Pro, developed by codex and prayed by me.

Unofficial Hugging Face Transformers-compatible implementation of the **Multiscreen** architecture, with a paper-math oracle and P0 validation tests.

This repository is a research artifact. It is not an official implementation of the Multiscreen paper, and it does not claim paper-scale performance reproduction.

Current status:

> **P0-qualified research implementation through P0-4.** Reviewed local CUDA bf16 GPT-2-vocabulary, context-4096 qualifying runs passed for both Psi=8 and Psi=16. This remains a correctness/stability smoke result, not a paper-scale or efficiency result.

Historical evidence-infrastructure gate:

> **P1-preflight A: validation provenance and evidence retention v1 — partial.** The schema, standard-library provenance/packaging/verification tools, synthetic tests, and policy are implemented. All four retained P0-4 summary/metrics files matched their committed hashes; both completion markers were found and hashed for the new descriptor; and a sanitized archive verified locally. Durable exact/private retention is blocked because `MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not configured, and acceptance review remains pending because no explicit reviewer was supplied. P0-4 remains complete; no P1 model/ecosystem capability is validated.

Staged Level 1 Core status: **Level 1 — Core mathematical Hugging Face
implementation: complete.** C1 was merged as PR #9, C2 as PR #10, the separate
C2 CUDA-autocast cache-dtype correction as PR #11, Stage 3 as PR #12, and Stage
4 as PR #13. The final Stage 5 matrix, human review, private retention,
sanitization, and offline verification passed on tested source
`b224ca1a127ee18fc5fd4b00a5df639401d60679`. The focused Stage 5 result was
reviewed and accepted as merged PR #14 (merge commit
`50af4f8e26b97f3bb0b97fc0bf6d0480a5d0fe06`). This result does not validate
paper-scale reproduction, retrieval benchmarks, optimized long-context
efficiency, distributed training, or any P1 model/ecosystem capability.

Stage E HF contract hardening requalification is **complete locally; draft PR
pending**. The fixed 53-command matrix passed on clean tested source
`0d59083ddbd78619ca29bf9af730999834272a1a` (implementation baseline
`bf8cc34cb6aa16ffeec1f609166db5efae79e9df`) with two exact Transformers
environment records, 117 focused tests in each of Transformers 4.57.6 and
5.14.1, full P0-1/P0-2 CPU fp32 and CUDA bf16 regressions, and fresh
checkpointed CUDA bf16 P0-3 and strict P0-4 Psi=8/Psi=16 runs. Codex reviewed
all 53 lossless logs and all 179 raw events. Exact evidence is retained and
verified privately outside Git; the separately built sanitized archive is
rescanned and verified but remains unpublished. Evidence commit
`4fd704f805ea634c66d2c4c26dded425c819a51d` records the canonical result. This
does not validate paper-scale training, retrieval, optimized long-context
efficiency, broad generation compatibility, distributed training, or a P1
model/ecosystem capability.

## Start here

- Development restart: [docs/HANDOFF.md](docs/HANDOFF.md)
- Historical P1-preflight A design: [docs/P1_PREFLIGHT_A_PLAN.md](docs/P1_PREFLIGHT_A_PLAN.md)
- Historical P1-preflight A Codex Goal: [docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md](docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md)
- Repository instructions for Codex: [AGENTS.md](AGENTS.md)
- Detailed validation boundary: [docs/VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md)
- Reproduction commands: [docs/TESTING.md](docs/TESTING.md)
- Accepted P0-4 evidence: [docs/validation_results/P0_4_SUMMARY.md](docs/validation_results/P0_4_SUMMARY.md)
- P0-4 reproduction plan: [docs/P0_4_PLAN.md](docs/P0_4_PLAN.md)
- Historical P0-4 Codex handoff: [docs/CODEX_P0_4_HANDOFF.md](docs/CODEX_P0_4_HANDOFF.md)
- Known limitations: [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md)
- Validation log index: [docs/validation_results/VALIDATION_LOG_INDEX.md](docs/validation_results/VALIDATION_LOG_INDEX.md)
- Logging policy: [docs/LOGGING_POLICY.md](docs/LOGGING_POLICY.md)
- Evidence archive policy: [docs/EVIDENCE_ARCHIVE_POLICY.md](docs/EVIDENCE_ARCHIVE_POLICY.md)
- P0-4 evidence-retention descriptor: [docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json](docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json)
- Repository audit: [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md)
- C2 semantic decision: [docs/adr/ADR-0001-mipe-position-semantics.md](docs/adr/ADR-0001-mipe-position-semantics.md)
- C2 plan: [docs/P0_5_C2_PLAN.md](docs/P0_5_C2_PLAN.md)
- C2 accepted result: [docs/validation_results/P0_5_C2_SUMMARY.md](docs/validation_results/P0_5_C2_SUMMARY.md)
- Stage 3 plan: [docs/P1_PREFLIGHT_B_PLAN.md](docs/P1_PREFLIGHT_B_PLAN.md)
- Stage 3 accepted result: [docs/validation_results/P1_PREFLIGHT_B_SUMMARY.md](docs/validation_results/P1_PREFLIGHT_B_SUMMARY.md)
- Stage 4 plan: [docs/P0_5_C3_PLAN.md](docs/P0_5_C3_PLAN.md)
- Stage 4 accepted result: [docs/validation_results/P0_5_C3_SUMMARY.md](docs/validation_results/P0_5_C3_SUMMARY.md)
- Stage 4 historical evidence descriptor: [docs/validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json](docs/validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json)
- Stage 4 external-retention closure: [docs/validation_results/P0_5_C3_EVIDENCE_CLOSURE.json](docs/validation_results/P0_5_C3_EVIDENCE_CLOSURE.json)
- Stage 4 closure verification: [exact/private](docs/validation_results/P0_5_C3_EXACT_VERIFICATION.json) and [sanitized](docs/validation_results/P0_5_C3_SANITIZED_VERIFICATION.json)
- Stage 5 plan: [docs/LEVEL1_CORE_REQUALIFICATION_PLAN.md](docs/LEVEL1_CORE_REQUALIFICATION_PLAN.md)
- Stage 5 reviewed result: [docs/validation_results/LEVEL1_CORE_SUMMARY.md](docs/validation_results/LEVEL1_CORE_SUMMARY.md) and [JSON](docs/validation_results/LEVEL1_CORE_SUMMARY.json)
- Stage 5 complete evidence descriptor: [docs/validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json](docs/validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json)
- Stage 5 verification reports: [exact/private](docs/validation_results/LEVEL1_CORE_EXACT_VERIFICATION.json) and [sanitized](docs/validation_results/LEVEL1_CORE_SANITIZED_VERIFICATION.json)
- Stage E plan: [docs/HF_CONTRACT_HARDENING_PLAN.md](docs/HF_CONTRACT_HARDENING_PLAN.md)
- Stage E reviewed result: [Markdown](docs/validation_results/HF_CONTRACT_HARDENING_SUMMARY.md) and [JSON](docs/validation_results/HF_CONTRACT_HARDENING_SUMMARY.json)
- Stage E complete evidence descriptor: [docs/validation_results/HF_CONTRACT_HARDENING_EVIDENCE_ARCHIVE.json](docs/validation_results/HF_CONTRACT_HARDENING_EVIDENCE_ARCHIVE.json)
- Stage E verification reports: [exact/private](docs/validation_results/HF_CONTRACT_HARDENING_EXACT_VERIFICATION.json) and [sanitized](docs/validation_results/HF_CONTRACT_HARDENING_SANITIZED_VERIFICATION.json)

## Current Level 1 stage

P1-preflight B was accepted by merged PR #12. P0.5-C3 separately encodes
the paper's tokenizer, data-stream, optimizer, scheduler, and
no-gradient-clipping recipe as executable contracts. The paper contract and
the repository's operational choices are recorded separately in the
[Stage 4 plan](docs/P0_5_C3_PLAN.md).

The exact contract, pinned-data lane, two-version focused tests, full P0
regressions, and Psi=8/Psi=16 CUDA bf16 operational and peak-exposure
diagnostics passed on tested commit
`8fa5dbf13530c942b2c9e5f03a572bd0cd5ca74f`; see the
[Stage 4 result](docs/validation_results/P0_5_C3_SUMMARY.md). The focused Stage
4 PR was reviewed and merged as PR #13, accepting P0.5-C3. Its historical
evidence descriptor still truthfully records the packaging-time partial state;
PR acceptance does not rewrite that record. A later
[external-retention closure](docs/validation_results/P0_5_C3_EVIDENCE_CLOSURE.json)
retains and verifies the exact/private archive and reverifies the sanitized
archive. Codex reviewed all 26 source artifacts and all 8 optimizer-step raw
events; acceptance review is recorded and overall evidence status is complete.

Final Level 1 requalification passed locally under the
[Stage 5 plan](docs/LEVEL1_CORE_REQUALIFICATION_PLAN.md). The reviewed result
and complete evidence closure are recorded in the
[Level 1 summary](docs/validation_results/LEVEL1_CORE_SUMMARY.md) and
[descriptor](docs/validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json). The
focused result was reviewed and accepted as merged PR #14. Stage 4 CUDA runs
remain reduced project diagnostics, while Stage 5 remains a correctness and
short-run qualification result rather than paper-scale reproduction, quality,
or efficiency evidence.

The later Stage E requalification validates the seven post-Level-1 hardening
resolutions together, including the hardened eight-condition P0-4 predicate.
Its reviewed local result and complete evidence closure are recorded in the
[Stage E summary](docs/validation_results/HF_CONTRACT_HARDENING_SUMMARY.md) and
[descriptor](docs/validation_results/HF_CONTRACT_HARDENING_EVIDENCE_ARCHIVE.json).
The result has not yet been published as a draft PR and does not broaden the
project's P0/Level 1 claim into a P1 capability.

## What is included

```text
AGENTS.md                      Codex project instructions and validation rules
multiscreen_transformers/     HF-compatible model, config, and data code
scripts/                      tokenizer, training, P0-3, and P0-4 harnesses
configs/                      Tiny/debug/P0-3/P0-4 configs
oracle/                       paper_math_oracle and HF-vs-oracle tests
p0_2_three_way_minimal/        three-way comparison against the reference
third_party/multiscreen-pytorch/
                              vendored dieOD reference used by P0-2
tokenizers/tinystories_spm768/
                              committed tokenizer used by P0-3
docs/                         handoff, validation, testing, and result policy
```

The vendored reference retains its Apache-2.0 license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Install

The user's development environments are managed with Conda. Preserve an already suitable active environment; create an isolated environment when needed rather than broadly upgrading or replacing existing package state. `uv` may be used as a scoped installation helper against an explicit environment.

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle
```

Do not install globally, modify the Conda base environment, or run broad upgrades merely to start this repository.

## Minimal usage

```python
import torch
from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM

config = MultiscreenConfig.from_psi(
    8,
    vocab_size=768,
    max_seq_len=128,
    key_dim=16,
    value_dim=64,
    mipe_position_mode="paper_absolute",
)
model = MultiscreenForCausalLM(config).eval()
input_ids = torch.randint(0, config.vocab_size, (1, 16))

with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)

print(out.logits.shape)
```

For AutoClass loading in the same process:

```python
from multiscreen_transformers import register_multiscreen_auto_classes

register_multiscreen_auto_classes()
```

## Minimum baseline checks

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

Full CPU and CUDA comparison commands are in [docs/TESTING.md](docs/TESTING.md).

The evidence tooling is standard-library-only and can be checked without site
packages or network access:

```bash
python -S -m py_compile \
  scripts/validation_evidence_common.py \
  scripts/collect_validation_provenance.py \
  scripts/package_validation_evidence.py \
  scripts/verify_validation_evidence.py \
  tests/test_validation_evidence*.py

python -S -m unittest discover \
  -s tests \
  -p 'test_validation_evidence*.py' \
  -v
```

The collector hashes the exact stdout from
`git status --porcelain=v1 --untracked-files=all --ignore-submodules=none` and,
when applicable, records privacy-safe recursive submodule state without raw
paths. A review record requires an explicit reviewer and method, a full review
commit, and an explicit raw-events-reviewed boolean.

The verifier accepts exactly one canonical gzip member, enforces normalized
USTAR member boundaries and padding, rescans every sanitized member including
control metadata, and binds a supplied descriptor to the validation gate,
tested-source commit, and complete source-artifact metadata.

## Validation status

### P0-1: complete

The dense paper-math oracle and HF implementation match under the recorded small-shape forward, loss, mask, position, and cache sweeps.

### P0-2: complete

The vendored unofficial PyTorch reference, HF implementation, and oracle match under the recorded CPU fp32 and CUDA bf16 three-way comparisons.

### P0-3: complete

Psi=8 and Psi=16 TinyStories bf16 smoke training passed, including:

```text
finite loss
finite gradient norms
probe-loss decrease
save_pretrained / from_pretrained
greedy generate(use_cache=True)
manual cache split vs full-forward suffix
```

Recorded metrics are in [docs/validation_results/p0_3_results.json](docs/validation_results/p0_3_results.json).

### P0-4: complete

Reviewed local runs passed the strict GPT-2-vocabulary, context-4096 CUDA bf16 gate for both intended model orders:

| Metric | Psi=8 | Psi=16 |
|---|---:|---:|
| parameters | 4,134,146 | 27,546,626 |
| optimizer steps | 50 | 50 |
| probe loss | 11.140747 → 4.675382 | 15.799321 → 3.495601 |
| peak CUDA allocated | 3,156,709,888 bytes | 6,622,802,944 bytes |
| reload max abs | 0 | 0 |
| cache max abs | 0 | 0.125, within configured atol/rtol |
| `qualification.qualified` | `true` | `true` |

Both runs recorded finite losses and gradients, configured probe-loss decrease, save/load, tokenizer reload, greedy generation with cache, and manual cache-split agreement within tolerance. The compact reviewed evidence and raw-artifact hashes are in [P0_4_SUMMARY.md](docs/validation_results/P0_4_SUMMARY.md) and [P0_4_SUMMARY.json](docs/validation_results/P0_4_SUMMARY.json).

For a future reproduction, qualification requires GPT-2 vocabulary 50,257,
sequence length 4,096, CUDA bf16, microbatch 1, at least 50 actually completed
optimizer steps, runtime gradient checkpointing enabled, and the supported
non-reentrant checkpointing path (`use_reentrant=False`). Gradient accumulation
is not a qualification condition.

The qualifying reproduction commands remain:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096
```

Review and accept the new Psi=8 artifacts and memory headroom before running Psi=16.

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

A CPU, reduced-context, different-dtype, microbatch-other-than-1, fewer-step,
checkpointing-disabled, or reentrant-checkpointing run remains diagnostic and
must not be reported as a P0-4 pass.

## P1-preflight A

P1-preflight A now provides:

```text
explicit reviewer provenance
clean/dirty worktree provenance
truthful historical not-recorded fields
deterministic exact/private evidence packaging
separate sanitized/shareable packaging
offline archive verification and tamper detection
long-term off-repository evidence retention descriptors
```

The current P0-4 backfill is deliberately partial. All four Psi=8/Psi=16
summary and metrics files matched their committed hashes. Both completion markers
were found and hashed for the new descriptor, and the sanitized archive passed offline verification. No exact/private archive
was created because no external archive directory was configured; no reviewer
was inferred in the absence of explicit input; and no public asset was
published. The compact state is recorded in
[P0_4_EVIDENCE_ARCHIVE.json](docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json).

After retrieving the ignored sanitized archive by its logical locator, verify
it without extraction or network access:

```bash
SANITIZED_ARCHIVE=/path/to/retrieved/validation-evidence-sanitized-p0-4-v1-r2.tar.gz

python -S scripts/verify_validation_evidence.py \
  --archive "$SANITIZED_ARCHIVE" \
  --expected-sha256 d58a4c9ecf28f20a135f4ba2ce95c5a532a04ea92f36e5b54d893400ae4c62fd \
  --evidence-document docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json \
  --schema schemas/validation_evidence_v1.schema.json \
  --json
```

The gate must not change model behavior or accepted P0 metrics. Exact evidence
must remain private and outside Git; only a separately sanitized, verified
archive may be published when explicitly configured. See
[docs/EVIDENCE_ARCHIVE_POLICY.md](docs/EVIDENCE_ARCHIVE_POLICY.md).

The original design is retained in
[docs/P1_PREFLIGHT_A_PLAN.md](docs/P1_PREFLIGHT_A_PLAN.md); use the current
[handoff](docs/HANDOFF.md) for operational continuation.

## Local Codex continuation

After cloning, start Codex from the repository root:

```bash
codex
```

Codex reads [AGENTS.md](AGENTS.md) before working. Use
[docs/HANDOFF.md](docs/HANDOFF.md) as the current operational entrypoint. The
[original P1-preflight A Goal](docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md) is retained
as execution history; do not paste it as a fresh full-implementation request.

To complete the two currently blocked requirements, provide an explicit
reviewer and durable archive directory before resuming the partial gate:

```bash
export MULTISCREEN_EVIDENCE_REVIEWERS=kuroganegames
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/outside/the/repository
```

If a fresh checkout does not contain the original ignored P0-4 outputs:

```bash
export MULTISCREEN_P0_4_RAW_ROOT=/absolute/path/to/the/original/P0-4/raw-output-root
```

If `/goal` is unavailable:

```bash
codex features enable goals
codex
```

## Known limitations

Not yet validated:

- P1-preflight A acceptance review and durable exact/private retention
- paper-scale pretraining or paper-quality reproduction
- long-context retrieval at paper settings
- long-context runtime or memory efficiency
- distributed training
- custom Triton/windowed kernels
- PEFT/LoRA/QLoRA or Unsloth
- torch.compile stability at scale
- broad generation compatibility
- vLLM/SGLang serving
- production readiness

The current HF path is a dense PyTorch correctness baseline. Do not use it to substantiate the paper's speed claims.

## License

The repository is provided under Apache-2.0. The vendored reference under `third_party/multiscreen-pytorch/` retains its original Apache-2.0 licensing.
