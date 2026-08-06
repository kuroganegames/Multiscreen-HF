# Level 1 Plan: Core Mathematical Hugging Face Implementation

## Status

```text
Program: Level 1 — Core mathematical HF implementation
State: selected and designed; staged implementation pending
P0-1 through P0-4: complete
P1-preflight A tooling: merged
P1-preflight A historical P0-4 retention descriptor: partial
P1 model/ecosystem capabilities: none validated
```

P1-preflight A supplied the versioned evidence schema, provenance collector,
deterministic exact/sanitized packager, offline verifier, security fixtures, and
archive policy. The historical P0-4 descriptor remains truthful: its source
hashes and sanitized archive were verified, while exact/private retention and an
explicit acceptance review were not recorded as complete. That retention status
does not reopen the accepted P0-4 model result.

This plan defines the work required before this repository may describe the
dense correctness-first implementation as a **Level 1 core mathematical Hugging
Face implementation**. It does not claim paper-scale reproduction, optimized
long-context execution, or P1 ecosystem compatibility.

The Codex entry point and ready-to-paste Goal are in
[CODEX_LEVEL1_CORE_HF_HANDOFF.md](CODEX_LEVEL1_CORE_HF_HANDOFF.md).

## Completion definition

Level 1 is complete only when reviewed evidence establishes all of the following:

```text
1. paper architecture, scaling, parameter counts, parameter shapes, and
   initialization contracts are machine-checked;
2. paper-absolute and reference-compatible long-position/MiPE semantics are
   explicit, documented, and tested across the training-context boundary;
3. gradient checkpointing uses the supported Transformers API without the
   legacy-hook deprecation warning while preserving non-reentrant behavior;
4. the paper optimizer/data/scheduler contract has an operational short smoke
   path with clearly bounded claims;
5. the resulting implementation passes the strongest required P0 regression
   and CUDA bf16 requalification suite;
6. the final evidence has explicit reviewer provenance, clean-worktree
   provenance, an externally retained exact/private archive, and a separately
   verified sanitized archive.
```

Level 1 completion still does not establish:

```text
- SlimPajama 2^38-token paper-scale pretraining;
- the paper's scaling curves or benchmark quality;
- PG-19, ABCDigits, passkey, or lost-in-the-middle reproduction;
- 131,072-token dense execution feasibility;
- Triton/window-skipping efficiency;
- inference-time screening-window expansion quality/latency;
- multi-GPU/FSDP/DeepSpeed training;
- PEFT/LoRA, QLoRA, Unsloth, torch.compile, serving, or broad generation.
```

## Program structure and review protocol

This is a multi-PR program. Do not combine all changes in one PR and do not use
stacked PRs by default. Each stage starts from the latest reviewed and merged
`main`.

```text
PR 1: P0.5-C1 — architecture / initialization / all-scale contract
PR 2: P0.5-C2 — long-position / MiPE / cache semantics
PR 3: P1-preflight B — gradient-checkpointing API modernization
PR 4: P0.5-C3 — paper-training-contract smoke
PR 5: Level 1 final P0 core requalification and evidence
```

At the end of each stage Codex must:

```text
- run the stage acceptance suite and required P0 regressions;
- inspect the complete diff and repository hygiene;
- commit only the focused stage;
- push a branch and open a draft PR when permitted;
- report REVIEW_REQUIRED;
- stop before starting the next stage;
- never merge the PR automatically.
```

After the user reviews and merges the PR, update `main`, verify the merge, and
resume the same Goal for the next stage.

## Source hierarchy

Use sources in this order and record exact versions/commits:

```text
1. Multiscreen paper v3:
   https://arxiv.org/abs/2604.01178
2. paper source/TeX when accessible;
3. oracle/paper_math_oracle.py for explicit paper-mode equations;
4. current HF implementation and config;
5. vendored dieOD/multiscreen-pytorch reference;
6. other third-party implementations only as supplemental evidence;
7. official Transformers source/documentation for API compatibility.
```

A third-party implementation may explain historical behavior but must not
silently override an explicit paper equation. When sources disagree, preserve
the disagreement in an ADR and implement explicit modes rather than hiding it.

# Stage P0.5-C1

## Architecture, initialization, and all-scale contract

### Objective

Convert the paper's architecture table and initialization table into executable,
non-flaky contracts.

### Paper scaling contract

```text
N_L = N_H = Psi
d_E = Psi^2
d_K = 16
d_V = 64
w_th = 256
vocabulary = 50,257 for the paper configurations
```

Expected paper parameter counts:

| Psi | Total parameters | Non-embedding parameters |
|---:|---:|---:|
| 8 | 4,134,146 | 917,698 |
| 16 | 27,546,626 | 14,680,834 |
| 32 | 286,347,266 | 234,884,098 |
| 48 | 1,304,884,226 | 1,189,092,098 |
| 64 | 3,963,961,346 | 3,758,108,674 |

The test must independently derive the counts from named parameter shapes. Do
not make the expected table depend on the implementation's own estimate helper.

### Initialization contract

```text
W_Q, W_K ~ N(0, 0.1 / sqrt(d_K))
W_V      ~ N(0, 0.1 / sqrt(d_V))
W_G      ~ N(0, 0.1)
W_O      ~ N(0, 0.1 / sqrt(d_E))
W_E      ~ N(0, 0.1 / sqrt(d_E))
s_w      = headwise linspace(0, log(w_th))
s_r      = 0
s_O      = log(1 / sqrt(N_H * N_L))
s_E      = 0
s_F      = log(sqrt(d_E))
```

Exact scalar/vector initializers must be checked exactly. Random projection
initializers must be tested non-flakily by intercepting or auditing requested
initializer parameters, plus a broad statistical sanity check with fixed seeds.
Do not use a fragile single-small-tensor sample-standard-deviation threshold as
the sole proof.

### Required coverage

```text
- Psi 8/16/32/48/64 config construction;
- exact architecture dimensions;
- exact total and non-embedding parameter counts;
- meta-device or equivalent allocation-safe shape construction for large scales;
- state_dict key and shape manifest;
- config aliases and serialization round trip;
- tied normalized embedding identity;
- absence of a separate trainable lm_head parameter;
- exact scalar/vector initialization;
- projection/embedding initializer contract;
- PackedTextDataset golden sequence:
  document EOS boundaries, continuous stream, seq_len+1 chunking,
  one-token shift, max-token boundary, no dropped/duplicated tokens.
```

### Expected outputs

```text
tests/test_paper_architecture_contract.py
tests/test_paper_initialization_contract.py
tests/test_packed_text_contract.py
scripts/generate_paper_scale_manifest.py
docs/P0_5_C1_PLAN.md
docs/validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json
focused CI coverage
```

The exact layout may change after repository inspection, but all acceptance
properties must remain covered.

### Stage acceptance

```text
- all five paper configurations satisfy the independent table;
- large-scale checks do not allocate full 1.3B/4B real weights;
- initializers satisfy exact requested rules;
- config/save-load manifests are deterministic;
- data-packing golden tests pass;
- existing P0 quick tests pass;
- no numerical model behavior is changed merely to make the manifest pass.
```

# Stage P0.5-C2

## Long-position, MiPE, and cache semantics

### Objective

Resolve and make explicit the difference between the paper's absolute-position
MiPE equation and the current reference-compatible modulo-after-max-position
branch.

### Known starting point

The paper equation uses absolute position `i`:

```text
angle(i, w) = pi * i * gamma(w) / w
```

The current HF implementation uses:

```text
effective_position = position                         before max position
effective_position = position mod w                   at/after max position
```

The oracle already exposes separate paper and HF/reference position rules. The
HF public config does not yet expose an explicit position-mode field, and
`max_position_embeddings` currently also acts as the modulo transition
boundary.

### Required source audit and ADR

Before code changes, create an ADR that records:

```text
- exact paper v3 equation and explanatory text;
- paper source/TeX findings;
- current oracle behavior;
- current HF behavior;
- vendored reference behavior;
- any available author/official implementation evidence;
- third-party implementation findings;
- whether any source specifies the transition boundary;
- compatibility and migration consequences.
```

If the available sources do not resolve author intent, say so. Do not guess.

### Required design properties

If paper and reference behavior differ, represent both explicitly. Names may
change after review, but semantics must be unambiguous, for example:

```text
paper_absolute
reference_mod_after_max_position
```

Requirements:

```text
- no silent change to existing serialized checkpoints/configs;
- a tested migration/default policy for configs lacking the new field;
- paper-mode configs can be constructed deliberately;
- reference-compatible P0-2 behavior remains available;
- max supported positions, training context, and reference wrap boundary are
  not conflated without an explicit documented reason;
- config clone/serialization/AutoClass round trips preserve the mode;
- unsupported arbitrary batch-specific offsets still fail loudly.
```

Do not force a particular public default merely to finish the gate. The ADR and
backward-compatibility requirements control the final choice.

### Position test matrix

At minimum:

```text
positions:
  0, 1, 255, 256, 257,
  4095, 4096, 4097,
  8191, 8192,
  131071

window regimes:
  below 256
  approximately 256
  equal to 256
  above 256
  fractional learned windows
```

Required comparisons:

```text
- HF paper mode vs paper oracle;
- HF reference mode vs reference-compatible oracle and vendored reference;
- fp32 stable auxiliary math;
- reference low-precision auxiliary math where relevant;
- full forward vs cached suffix;
- multi-chunk decode;
- boundaries crossing 4096 and 8192;
- strict invalid-position/cache cases.
```

Required cache scenarios include at least:

```text
prefix 4080 + suffix 32
prefix 4096 + suffix 16
prefix 8192 + suffix 16
multiple one-token decode steps crossing 4096
multiple uneven suffix chunks crossing 4096
```

Use tiny shapes and direct position/oracle tests where dense long sequences
would be wasteful. A 131,072-token dense full forward is not required.

### Expected outputs

```text
docs/adr/ADR-XXXX-mipe-position-semantics.md
focused position/MiPE/cache tests
configuration and oracle/HF updates justified by the ADR
docs/P0_5_C2_PLAN.md
compact reviewed result record
```

### Stage acceptance

```text
- paper and reference semantics are explicit;
- no unresolved implicit transition remains;
- position/config migration is tested;
- paper-mode equation matches the oracle at all matrix positions;
- reference mode retains P0-2 compatibility;
- cache/full equivalence passes across the context boundary;
- strongest required P0-1/P0-2 CPU and CUDA comparisons pass;
- ambiguity is reported as SPECIFICATION_DECISION_REQUIRED rather than hidden.
```

# Stage P1-preflight B

## Gradient-checkpointing API modernization

### Objective

Remove the legacy Transformers gradient-checkpointing hook warning while
preserving the validated non-reentrant training behavior.

### Known starting point

The current hook accepts old `module`/`value` parameters together with the new
API parameters. Transformers 4.57.6 identifies this as an old-format hook and
warns that checkpointing kwargs are ignored. The forward path directly calls
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` instead of the
checkpoint function installed by Transformers.

### Required behavior

```text
- use the supported Transformers checkpointing contract;
- enable/disable propagates to the correct Multiscreen module;
- forward uses the installed `_gradient_checkpointing_func` or supported
  equivalent;
- preserve explicit `use_reentrant=False` behavior;
- no old-format deprecation warning;
- no missing-input-gradient warning;
- finite forward/backward and optimizer step;
- checkpointed and non-checkpointed logits/loss/gradients agree within justified
  tolerances under deterministic small shapes;
- custom checkpoint-function injection is demonstrably honored;
- save/load does not serialize transient function objects.
```

### Compatibility lanes

Use isolated environments when needed and record exact versions:

```text
recorded P0-4 lane:
  Transformers 4.57.6

current supported lane:
  the current version selected and pinned during implementation
```

Do not broadly upgrade the user's active Conda environment. Create isolated
Conda/venv lanes for the matrix when appropriate.

### Required revalidation

This changes the model training path. Run:

```text
- focused gradient-checkpointing tests;
- P0-1 CPU fp32 quick/full as required by the diff;
- P0-2 CPU fp32 quick/full as required by the diff;
- CUDA bf16 P0 comparisons where available;
- P0-3 short checkpointed smoke;
- a P0-4 reduced checkpointed diagnostic;
```

Full P0-4 requalification is deferred to the final Level 1 stage unless evidence
shows it is needed immediately.

# Stage P0.5-C3

## Paper-training-contract smoke

### Objective

Verify that the paper's data, optimizer, scheduler, and no-clipping path is
implemented and operational without claiming paper-scale reproduction.

### Paper contract to encode

```text
tokenizer vocabulary: GPT-2 50,257
dataset family: SlimPajama
document handling: EOS-concatenated continuous stream
sequence length: 4096
global batch in paper: 2^22 tokens
optimizer: AdamW
betas: (0.9, 0.95)
weight decay: 0
warmup: 4096 optimizer steps
post-warmup schedule: constant
Multiscreen peak learning rate: 2^-4 = 0.0625
gradient clipping: disabled
```

### Test separation

Do not pretend a short workstation smoke reproduces the paper's global batch or
training duration. Separate:

```text
A. exact contract/unit tests:
   config, optimizer, no clipping, and scheduler values at
   steps 0, 1, 4095, 4096, and 4097;

B. deterministic data test:
   pinned SlimPajama source revision/fingerprint, EOS concatenation, packing,
   and exact token accounting;

C. short operational CUDA bf16 smoke:
   feasible accumulation, short warmup transition, no clipping, and finite
   update behavior;

D. peak-LR exposure diagnostic:
   explicitly bounded short exposure to LR 0.0625, without a required quality or
   loss-decrease claim.
```

The full 4096-step warmup is tested mathematically; the short smoke may use a
reduced warmup only to exercise the transition, provided its output is labeled
as such.

### Execution order

```text
1. Psi=8 operational smoke and artifact review;
2. Psi=16 only after Psi=8 passes and memory headroom is understood;
3. preserve failures and do not introduce clipping to force a pass.
```

### Stage acceptance

```text
- paper recipe is represented exactly in a checked-in config/manifest;
- dataset revision/fingerprint is recorded;
- token packing is exact;
- optimizer/scheduler/no-clipping unit contract passes;
- short CUDA bf16 Psi=8 and Psi=16 paths complete with finite loss, gradients,
  and parameter updates under the explicitly documented diagnostic schedule;
- LR 0.0625 exposure result is recorded honestly;
- no paper-scale, superiority, or benchmark-quality claim is made;
- required P0 regressions pass.
```

# Final Level 1 requalification

## Objective

Requalify the merged result of C1, C2, preflight B, and C3, package reviewed
evidence with P1-preflight A tooling, and update the canonical validation
boundary.

## Minimum regression matrix

Run the exact commands documented in the then-current `docs/TESTING.md`. At
minimum the matrix must include:

```text
- formula-unit and oracle self/smoke tests;
- P0-1 CPU fp32 full;
- P0-1 CUDA bf16 full;
- P0-2 CPU fp32 full;
- P0-2 CUDA bf16 full;
- C1 architecture/init/all-scale suite;
- C2 paper/reference position and cache-boundary suite;
- gradient-checkpointing compatibility matrix;
- C3 paper-training-contract unit and CUDA smoke suite;
- P0-3 checkpointed Psi=8/16 smoke at the required strength;
- P0-4 qualifying CUDA bf16 context-4096 Psi=8 then Psi=16;
- save/load, tokenizer reload, generation, and cache checks;
- repository-wide syntax, JSON, Markdown-link, diff, and artifact hygiene.
```

If a model/config/oracle/cache behavior changed, old P0 evidence remains a
historical record for its tested commit; it must not be relabeled as evidence
for the new commit.

## Evidence acceptance

Before the final run, require explicit inputs:

```bash
export MULTISCREEN_EVIDENCE_REVIEWERS=<explicit-reviewer-list>
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/outside/the-repository
```

Record:

```text
- clean worktree before execution;
- exact tested commit and branch;
- environment and compatibility-lane versions;
- exact commands and run outputs;
- explicit reviewer, method, review commit, and raw-events-reviewed boolean;
- clean worktree after the evidence commit;
- exact/private archive stored outside Git and verified;
- separately sanitized archive and verification report;
- compact descriptor and human/machine summaries in Git.
```

Do not fabricate historical P0-4 provenance or change its partial retention
descriptor. Create a new Level 1 evidence descriptor for the new tested commit.

## Final status language

When all requirements pass, the repository may state:

```text
Level 1 — Core mathematical Hugging Face implementation: complete

The dense correctness-first implementation has reviewed contracts for paper
architecture/initialization/scaling, explicit paper/reference long-position
semantics, supported gradient checkpointing, a bounded paper-training-contract
smoke, and full P0 core requalification.
```

It must immediately retain the limitations that this is not paper-scale,
optimized, retrieval-benchmark, distributed-training, or P1 ecosystem
validation.

After the final evidence PR is reviewed and merged, an immutable tag such as
`p0-core-qualified-dense-v1` may be created from the merge commit. Never move an
existing tag and never tag an unmerged evidence branch.

## Development-environment contract

Throughout the program:

```text
- preserve the Conda-managed development environment;
- inspect the active environment before installation;
- use it when suitable;
- create isolated Conda/venv environments for compatibility matrices when
  useful;
- use uv only as a scoped helper against an explicit target environment;
- never install globally or modify/delete Conda base;
- never run broad upgrades or unrelated lockfile rewrites;
- record package versions before and after every environment change;
- remove or ignore temporary environments and never commit them.
```

## Program terminal states

### COMPLETE

All five PR stages are reviewed and merged, the final evidence is accepted and
externally retained, and canonical documents state Level 1 complete with proper
limitations.

### REVIEW_REQUIRED

A stage PR is open and awaiting user review/merge. This is the normal stopping
state between stages.

### SPECIFICATION_DECISION_REQUIRED

C2 source evidence is insufficient to choose a public compatibility/default
policy. Preserve the source audit and ADR options; do not guess or proceed to a
semantic code change.

### PARTIAL/BLOCKED WITH EVIDENCE

A reproducible environment, hardware, source ambiguity, failing core contract,
reviewer, or external-retention blocker prevents completion. Preserve compact
reproduction evidence and do not weaken the gate.
