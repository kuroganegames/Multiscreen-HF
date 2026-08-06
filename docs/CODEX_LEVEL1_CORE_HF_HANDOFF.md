# Codex Goal Handoff: Level 1 Core Mathematical HF Implementation

This document is the local-Codex entry point for completing the dense
correctness-first Multiscreen Hugging Face core before any P1 model/ecosystem
gate.

The authoritative staged design is
[LEVEL1_CORE_HF_PLAN.md](LEVEL1_CORE_HF_PLAN.md). Read that document rather than
trying to infer the program from an old P0-4 prompt.

## Current phase

```text
Repository: kuroganegames/Multiscreen-HF
P0-1: complete
P0-2: complete
P0-3: complete
P0-4: complete
P1-preflight A evidence tooling: merged
Historical P0-4 retention descriptor: partial
Level 1 core-completion program: selected; implementation pending
P1 model/ecosystem capabilities: none validated
```

P1-preflight A's merged infrastructure is available for new evidence. Its
historical P0-4 descriptor remains truthful: source hashes and the sanitized
archive were verified, while exact/private retention and explicit acceptance
review were not completed in the original handoff. Do not rewrite that history.

## Program shape

The Goal is durable across five sequential PRs:

```text
1. P0.5-C1 architecture / initialization / all-scale contract
2. P0.5-C2 long-position / MiPE / cache semantics
3. P1-preflight B gradient-checkpointing API modernization
4. P0.5-C3 paper-training-contract smoke
5. final Level 1 P0 core requalification and evidence
```

At the end of each stage, Codex opens a draft PR and stops with
`REVIEW_REQUIRED`. The user reviews and merges it, updates `main`, and resumes
the same Goal. Do not create stacked semantic PRs and never auto-merge.

## Before starting Codex

Run from the Git root so `AGENTS.md` is loaded:

```bash
git clone https://github.com/kuroganegames/Multiscreen-HF.git
cd Multiscreen-HF
codex
```

For an existing clone:

```bash
git switch main
git fetch --prune
git pull --ff-only
git status --short --branch
codex
```

If Goal mode is unavailable:

```bash
codex features enable goals
codex
```

## Evidence inputs

These are mandatory before the **final** requalification stage and may also be
used for intermediate accepted results:

```bash
export MULTISCREEN_EVIDENCE_REVIEWERS=kuroganegames
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/outside/the-repository
```

The reviewer value must be explicit. GitHub authentication is not reviewer
evidence. The exact/private archive directory must be user-controlled, durable,
and outside the checkout. Exact raw evidence must not be uploaded publicly.

## Development environment

Python environments are managed with Conda. `uv` is installed as a scoped
installation helper. The Goal may create isolated Conda or other virtual
environments for compatibility matrices, but must not damage the existing
environment, mutate Conda base, install globally, perform broad upgrades, or
rewrite unrelated lockfiles.

## Ready-to-paste `/goal` prompt

Paste the complete block below into Codex from the repository root.

```text
/goal Complete the staged Level 1 Core mathematical Hugging Face implementation program for the current `kuroganegames/Multiscreen-HF` checkout. Work through the five explicitly defined stages, but never combine the stages into one pull request. At the end of each stage, open a focused draft PR, report REVIEW_REQUIRED, and stop until the user reviews and merges it and resumes this Goal. Do not begin PEFT/LoRA, QLoRA, Unsloth, serving, Triton optimization, broad generation, or any P1 model/ecosystem capability during this Goal.

You are the continuing engineer and research-validation agent for the current checkout.

==================================================
1. PROJECT STATE
==================================================

This repository is an unofficial correctness-first Hugging Face Transformers-compatible implementation of Multiscreen.

Accepted model-validation state:

P0-1:
  paper_math_oracle vs HF small-shape formula/loss/mask/position/cache
  status: complete

P0-2:
  dieOD/multiscreen-pytorch vs HF vs paper_math_oracle three-way comparison
  status: complete

P0-3:
  Psi=8/16 TinyStories bf16 smoke training
  status: complete

P0-4:
  GPT-2 vocab 50,257 + context 4096 CUDA bf16 short training
  Psi=8 and Psi=16
  status: complete

Evidence infrastructure:

P1-preflight A:
  schema, provenance collector, exact/sanitized deterministic packager,
  offline verifier, archive policy, CI security fixtures
  implementation: merged

Historical P0-4 evidence-retention descriptor:
  source hashes verified
  completion markers hashed
  sanitized archive verified
  exact/private retention: blocked in the recorded handoff
  acceptance reviewer: not recorded in the recorded handoff
  status: partial

Do not alter the accepted P0-4 metrics or fabricate historical provenance. The partial retention descriptor does not reopen the accepted P0-4 model result.

Selected program:

Level 1 — Core mathematical Hugging Face implementation
  status: implementation pending

No P1 model/ecosystem capability is validated.

==================================================
2. PRIMARY OBJECTIVE
==================================================

Complete these stages in order, each in a separate reviewed and merged PR:

Stage 1 — P0.5-C1:
  architecture / initialization / all-scale contract

Stage 2 — P0.5-C2:
  long-position / MiPE / cache semantics

Stage 3 — P1-preflight B:
  gradient-checkpointing API modernization

Stage 4 — P0.5-C3:
  paper-training-contract smoke

Stage 5 — final Level 1 requalification:
  strongest required P0 regression, CUDA bf16 qualification, reviewed evidence,
  and canonical status update

The full design and acceptance criteria are authoritative in:

  docs/LEVEL1_CORE_HF_PLAN.md

Do not weaken that plan silently. If repository inspection justifies a change, document the proposed change, its evidence, and its effect on acceptance before implementing it.

==================================================
3. SUCCESS BOUNDARY
==================================================

Level 1 is COMPLETE only when all five stage PRs have been reviewed and merged and the final accepted evidence establishes:

- exact paper architecture/scaling/parameter-count contracts;
- exact initialization contracts;
- allocation-safe shape/state manifests for Psi=8/16/32/48/64;
- explicit paper-absolute and reference-compatible MiPE/position semantics;
- full/cache equivalence across the 4096 boundary;
- supported Transformers gradient-checkpointing API without the legacy warning;
- preserved use_reentrant=False behavior;
- operational bounded paper-data/optimizer/scheduler/no-clipping smoke;
- strongest required P0 regression and P0-4 Psi=8/16 requalification;
- explicit reviewer and clean-worktree provenance;
- externally retained exact/private archive;
- separately verified sanitized archive;
- accurate canonical documentation.

Level 1 does NOT validate:

- paper-scale 2^38-token training;
- paper scaling curves or benchmark quality;
- PG-19, ABCDigits, passkey, lost-in-the-middle, or 131K behavior;
- Triton/window skipping or efficiency;
- SWE quality/latency;
- distributed training;
- PEFT/LoRA, QLoRA, Unsloth, compile, serving, or broad generation.

==================================================
4. MANDATORY READING AND SOURCE AUDIT
==================================================

Before changing anything, read these repository files in order:

- AGENTS.md
- README.md
- docs/HANDOFF.md
- docs/LEVEL1_CORE_HF_PLAN.md
- docs/VALIDATION_STATUS.md
- docs/TESTING.md
- docs/KNOWN_LIMITATIONS.md
- docs/LOGGING_POLICY.md
- docs/EVIDENCE_ARCHIVE_POLICY.md
- docs/REPOSITORY_AUDIT.md
- docs/RELEASE_CHECKLIST.md
- docs/validation_results/VALIDATION_LOG_INDEX.md
- docs/validation_results/P0_4_SUMMARY.md
- docs/validation_results/P0_4_SUMMARY.json
- docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json
- schemas/validation_evidence_v1.schema.json
- scripts/collect_validation_provenance.py
- scripts/package_validation_evidence.py
- scripts/verify_validation_evidence.py
- multiscreen_transformers/configuration_multiscreen.py
- multiscreen_transformers/modeling_multiscreen.py
- multiscreen_transformers/data.py
- oracle/paper_math_oracle.py
- oracle/test_against_hf_port.py
- p0_2_three_way_minimal/test_three_way_minimal.py
- third_party/multiscreen-pytorch source relevant to config, MiPE, cache, and initialization

Read the paper and record the exact version:

- https://arxiv.org/abs/2604.01178
- use v3 unless the repository or user explicitly selects another version
- inspect the paper source/TeX when accessible

For gradient checkpointing, inspect official Transformers source/documentation for:

- the recorded P0-4 version, Transformers 4.57.6;
- the current supported version selected during Stage 3.

For every external source, record URL, version/tag/commit, access date, and the exact claim it supports. Do not replace paper text with third-party assumptions.

Initial report must summarize:

- current accepted validation boundary;
- P1-preflight A's implemented tooling and remaining historical-retention limitations;
- Level 1 completion definition;
- the five-PR stage order;
- current paper/reference/HF MiPE difference;
- current gradient-checkpointing warning cause;
- environment and evidence constraints;
- any source ambiguity already visible.

==================================================
5. NON-NEGOTIABLE CORRECTNESS CONTRACTS
==================================================

1. Trim parameterization

Paper:
  r = sigmoid(s_r)

HF/reference:
  inv_r = exp(sr) + 1

Required conversion:
  s_r_paper = -s_r_hf

2. Oracle compute modes

Stable paper/oracle:
  mipe_compute_dtype="fp32"
  softmask_compute_dtype="fp32"

Reference-compatible low precision:
  mipe_compute_dtype="reference"
  softmask_compute_dtype="reference"

Do not merge their meanings.

3. Oracle purpose

oracle/paper_math_oracle.py is a dense equation-oriented correctness oracle. It is not a speed, memory, production, or long-context implementation.

4. Tied normalized embeddings

tie_word_embeddings=True is architectural. Logits use the normalized input embedding and learned s_F. Do not create an independent trainable lm_head.

5. DynamicCache and position contract

Preserve validated greedy cache behavior and the scalar contiguous prefix/start_pos contract. Do not silently accept arbitrary batch-specific offsets.

6. Dense path boundary

The HF screening implementation is dense and quadratic. Runtime and memory are feasibility diagnostics, never paper-efficiency evidence.

7. Source disagreement

The paper equation is primary for paper mode. The vendored reference is primary only for reference-compatibility mode. If source intent is ambiguous, create an ADR and stop with SPECIFICATION_DECISION_REQUIRED rather than guessing.

8. Accepted evidence truth

- do not rewrite accepted historical metrics;
- do not fabricate reviewer identity;
- do not infer a clean historical worktree from a commit SHA;
- preserve unknown historical facts as explicit not_recorded/null;
- exact raw evidence stays private and outside Git;
- sanitized evidence is a distinct artifact;
- old evidence applies only to its tested commit.

9. Development environment

The current development environment must not be broken.

- Python environments are managed with Conda;
- uv is available only as a scoped installation helper;
- inspect the active environment before installing anything;
- use the active environment when suitable;
- isolated Conda/venv environments may be created when useful;
- never delete or broadly mutate an existing environment or Conda base;
- never install globally;
- never run conda update --all, unconstrained pip install -U, broad uv sync, or equivalent broad changes;
- when using uv, target an explicit environment and avoid unrelated lockfile changes;
- record package versions before and after every environment change;
- temporary environments must remain outside Git and be cleaned or documented.

10. No forced pass

Do not relax tolerances, reduce a qualifying context, enable gradient clipping, change a source-defined mode, or alter acceptance criteria merely to obtain a pass.

==================================================
6. ALLOWED CHANGE SCOPE
==================================================

Allowed across the program only when justified by the current stage:

- focused model/config/oracle changes required by C1/C2/preflight B;
- focused data/config/scheduler/harness work required by C3;
- focused tests, manifests, ADRs, plans, result templates, evidence descriptors, and CI lanes;
- evidence tooling integration without weakening its security contract;
- isolated virtual-environment creation;
- exact dependency pins/constraints for compatibility lanes;
- canonical documentation updates after reviewed evidence.

Not allowed:

- PEFT/LoRA, QLoRA, Unsloth;
- Triton/windowed optimization;
- serving or broad generation features;
- unrelated refactors;
- changing the vendored reference without an explicit vendoring update and separate justification;
- committing outputs, caches, environments, checkpoints, raw archives, weights, secrets, private paths, or large logs;
- mixing more than one stage's semantic changes in a PR;
- automatically merging any PR.

If a stage exposes a separate defect, create the smallest reproduction, record the affected contract and required requalification, and either fix it within that stage's direct scope or stop with PARTIAL/BLOCKED WITH EVIDENCE. Do not opportunistically expand scope.

==================================================
7. GIT AND STAGED-PR PROTOCOL
==================================================

At program start and before every stage:

- switch to main;
- fetch --prune;
- pull --ff-only;
- verify the expected previous stage PR is merged;
- record HEAD, latest commit, branch, remote, and exact status porcelain;
- require a clean tracked/untracked worktree before creating the stage branch;
- do not discard pre-existing user changes.

Suggested branches:

Stage 1:
  validation/p0-5-c1-architecture-init-scale

Stage 2:
  validation/p0-5-c2-mipe-position-cache

Stage 3:
  compat/p1-preflight-b-gradient-checkpointing

Stage 4:
  validation/p0-5-c3-paper-training-contract

Stage 5:
  validation/level1-core-requalification

At the end of every stage:

- inspect git diff and git diff --check;
- run the complete stage acceptance suite;
- stage only in-scope files;
- commit intentionally;
- push the stage branch;
- open a draft PR with exact tests, evidence, limitations, and requalification scope;
- do not merge;
- report REVIEW_REQUIRED and stop.

Do not start Stage N+1 from an unmerged Stage N branch. After the user merges, update main and resume.

If push/PR permissions are unavailable, leave a clean local commit and print exact push/PR commands, then stop at REVIEW_REQUIRED.

==================================================
8. CHECKPOINT 0 — REPOSITORY AND ENVIRONMENT AUDIT
==================================================

Collect without exposing secrets:

Repository:
- current branch and HEAD;
- origin/main relation;
- worktree status hash/count/clean state using the evidence collector;
- remotes with credentials removed;
- disk space;
- gh authentication/push ability without credentials.

Environment:
- CONDA_DEFAULT_ENV and CONDA_PREFIX;
- Python executable/version;
- pip, conda, and uv versions;
- torch, transformers, datasets, tokenizers, safetensors, numpy, accelerate, trl, sentencepiece;
- CUDA availability/runtime;
- GPU name, capability, memory, bf16 support;
- nvidia-smi summary and unrelated GPU processes;
- relevant cache variables and free space.

Before installation, capture package versions. Use the existing Conda environment if suitable. If compatibility isolation is needed, create a named isolated environment and record its exact creation command. Do not broadly upgrade the active environment.

Install only repository-declared requirements needed for the current stage. Record before/after changes.

Run the initial baseline:

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

Also run syntax checks for current Python sources and the P1-preflight A evidence-tooling unit suite.

Do not begin Stage 1 if the accepted baseline fails. Diagnose environment versus repository regression. Fix only a narrowly proven environment issue; otherwise stop with PARTIAL/BLOCKED WITH EVIDENCE.

==================================================
9. STAGE 1 — P0.5-C1 ARCHITECTURE / INITIALIZATION / ALL-SCALE
==================================================

Read the exact C1 contract in docs/LEVEL1_CORE_HF_PLAN.md.

Implement independent tests for:

Paper architecture:
  N_L=N_H=Psi
  d_E=Psi^2
  d_K=16
  d_V=64
  w_th=256

Paper counts:
  Psi=8:  total 4,134,146; non-embedding 917,698
  Psi=16: total 27,546,626; non-embedding 14,680,834
  Psi=32: total 286,347,266; non-embedding 234,884,098
  Psi=48: total 1,304,884,226; non-embedding 1,189,092,098
  Psi=64: total 3,963,961,346; non-embedding 3,758,108,674

Paper initializers:
  WQ/WK std=0.1/sqrt(dK)
  WV std=0.1/sqrt(dV)
  WG std=0.1
  WO/WE std=0.1/sqrt(dE)
  sw=linspace(0,log(wth))
  sr=0
  sO=log(1/sqrt(NH*NL))
  sE=0
  sF=log(sqrt(dE))

Requirements:

- derive expected counts independently from named shapes;
- do not call the implementation estimate to define expected results;
- use meta-device or allocation-safe construction for large scales;
- produce deterministic state_dict key/shape manifests;
- test config aliases, from_psi, clone, save/load, and AutoClass metadata;
- test normalized tied input/output embedding identity and absence of separate lm_head Parameter;
- test exact scalar/vector initialization;
- verify random initializer arguments non-flakily by interception/audit and fixed-seed sanity checks;
- add deterministic PackedTextDataset golden tests for EOS concatenation, seq_len+1 chunks, shifts, token limits, and no token loss/duplication.

Do not change numerical behavior simply to make an expected count pass. If the paper table and current implementation disagree, record the exact shape-level delta and stop with PARTIAL/BLOCKED unless a focused correction is clearly justified.

Expected focused outputs include tests, an allocation-safe paper-scale manifest generator, C1 plan/result documentation, and CI coverage. Follow repository style.

Required validation:

- all focused C1 tests;
- syntax/JSON/Markdown/hygiene;
- minimum P0 quick baseline;
- stronger tests if production code changes.

Create a draft PR, report REVIEW_REQUIRED, and stop. Do not start C2 until the user confirms the C1 PR is merged and resumes the Goal.

==================================================
10. STAGE 2 — P0.5-C2 LONG-POSITION / MiPE / CACHE SEMANTICS
==================================================

After resume, update clean main and verify C1 merge.

First create a source-audit ADR before semantic code changes. Compare:

- paper v3 equation and TeX;
- paper oracle paper mode;
- paper oracle hf_mod_after_max_position mode;
- current HF implementation;
- vendored reference;
- any accessible author/official implementation;
- supplemental third-party implementations.

Known starting discrepancy:

Paper:
  angle = pi * absolute_position * gamma(w) / w

Current HF/reference path:
  at/after max_position_embeddings, effective_position = position mod w

Do not silently choose a winner. The ADR must state what is proven, what is inferred, what remains unknown, and backward-compatibility implications.

If both semantics are needed, expose explicit unambiguous modes, preserve old serialized config behavior through a tested migration policy, and make deliberate paper-mode construction possible. Do not conflate training context, max supported position, and reference wrap boundary without an explicit documented rationale.

Minimum position matrix:
  0,1,255,256,257,4095,4096,4097,8191,8192,131071

Minimum window matrix:
  below 256, around 256, equal 256, above 256, fractional learned windows

Minimum comparisons:

- HF paper mode vs paper oracle;
- HF reference mode vs reference-compatible oracle and vendored reference;
- stable fp32 auxiliary math;
- reference low-precision auxiliary math;
- config serialization and migration;
- full vs cached suffix;
- multi-chunk decode;
- strict invalid-offset/cache cases.

Cache scenarios:

- prefix 4080 + suffix 32;
- prefix 4096 + suffix 16;
- prefix 8192 + suffix 16;
- one-token decode crossing 4096;
- uneven multiple suffix chunks crossing 4096.

Use tiny direct position tests rather than allocating a dense 131K forward. Preserve the scalar contiguous start_pos contract.

Run the strongest relevant P0-1/P0-2 CPU and CUDA comparisons because this stage may change config, oracle, position, and cache semantics.

If sources do not support a safe compatibility/default decision, open a source-audit/ADR PR only, report SPECIFICATION_DECISION_REQUIRED, and stop. Do not guess.

Otherwise open the focused draft C2 PR, report REVIEW_REQUIRED, and stop until merge/resume.

==================================================
11. STAGE 3 — P1-PREFLIGHT B GRADIENT CHECKPOINTING
==================================================

After resume, update clean main and verify C2 merge.

Known starting point:

- `_set_gradient_checkpointing` accepts old `module`/`value` arguments;
- Transformers 4.57.6 classifies it as old-format and warns;
- the forward path directly calls checkpoint(..., use_reentrant=False);
- the installed Transformers checkpoint function/kwargs are not honored.

Modernize to the supported API while preserving non-reentrant behavior.

Required tests:

- supports_gradient_checkpointing remains true;
- enable and disable propagate correctly;
- the installed checkpoint function is actually invoked;
- use_reentrant=False is preserved explicitly;
- old-format deprecation warning is absent;
- no missing-input-gradient warning;
- deterministic checkpointed/non-checkpointed logits, loss, and gradients agree within justified tolerances;
- finite backward and optimizer step;
- custom checkpoint function injection is honored;
- transient function objects are not serialized;
- save/load and generation behavior remain correct where applicable.

Compatibility matrix:

- Transformers 4.57.6 in an isolated recorded lane;
- a separately pinned current supported Transformers lane.

Do not mutate the user's active Conda environment broadly. Prefer isolated environments and record exact commands/package versions.

Because this changes model training behavior, run focused tests plus the strongest required P0-1/P0-2 comparisons, CUDA bf16 where available, P0-3 checkpointed smoke, and a reduced P0-4 checkpointed diagnostic. Full P0-4 requalification occurs in Stage 5 unless evidence requires it earlier.

Open a focused draft PR, report REVIEW_REQUIRED, and stop until merge/resume.

==================================================
12. STAGE 4 — P0.5-C3 PAPER-TRAINING-CONTRACT SMOKE
==================================================

After resume, update clean main and verify Stage 3 merge.

Encode and test the paper contract:

- GPT-2 vocabulary 50,257;
- SlimPajama family with pinned revision/fingerprint;
- EOS-concatenated continuous token stream;
- sequence length 4096;
- AdamW betas (0.9,0.95);
- weight decay 0;
- 4096 optimizer-step warmup then constant;
- peak Multiscreen LR 0.0625;
- gradient clipping disabled;
- paper global batch is 2^22 tokens, but the workstation smoke must not claim to reproduce it.

Separate four kinds of evidence:

A. exact config/optimizer/scheduler unit contract;
B. deterministic pinned data and token-accounting contract;
C. feasible short CUDA bf16 operational smoke with a clearly labeled reduced warmup;
D. bounded peak-LR=0.0625 exposure diagnostic.

Scheduler unit checks must include steps 0,1,4095,4096,4097. The short smoke may use a reduced warmup only to exercise the transition and must record that it is diagnostic. Do not require a paper-quality loss decrease from peak-LR exposure; require finite values, a valid update, and honest reporting.

Run Psi=8 first. Review artifacts and memory before Psi=16. Do not add clipping or lower the defined peak LR and still call the peak-exposure check passed. Preserve failures.

Use P1-preflight A evidence tooling. Exact outputs remain ignored/private; compact summaries and descriptors may be committed after sanitization and review.

Open a focused draft PR, report REVIEW_REQUIRED, and stop until merge/resume.

==================================================
13. STAGE 5 — FINAL LEVEL 1 REQUALIFICATION
==================================================

After resume, update clean main and verify all four prior stage PRs are merged.

Before execution require:

  MULTISCREEN_EVIDENCE_REVIEWERS
  MULTISCREEN_EVIDENCE_ARCHIVE_DIR

If either is absent, perform implementation/testing that does not require it, but the final accepted evidence cannot be COMPLETE. Do not infer reviewer identity from GitHub login.

Create branch:
  validation/level1-core-requalification

Record clean worktree provenance before the run and exact tested commit/environment.

Run the then-current exact commands in docs/TESTING.md. At minimum include:

- formula units;
- oracle self-check and smoke;
- P0-1 CPU fp32 full;
- P0-1 CUDA bf16 full;
- P0-2 CPU fp32 full;
- P0-2 CUDA bf16 full;
- C1 architecture/init/all-scale/data suite;
- C2 paper/reference position and cache-boundary suite;
- Stage 3 gradient-checkpoint compatibility matrix;
- C3 recipe unit/data/CUDA smoke suite;
- P0-3 checkpointed Psi=8/16 smoke at documented strength;
- P0-4 strict qualifying CUDA bf16 context-4096 Psi=8 then Psi=16;
- save/load, tokenizer reload, generation, and cache checks;
- syntax, JSON, Markdown links, diff, security, and repository hygiene.

Do not reuse old P0-4 metrics as evidence for the new tested commit. They remain historical comparison data only.

Review every raw event and create:

- human-readable Level 1 summary;
- machine-readable Level 1 summary;
- evidence archive descriptor;
- exact/private archive outside Git;
- separately sanitized archive;
- offline verification report;
- explicit reviewer/method/review-commit/raw-events-reviewed record;
- clean post-commit worktree provenance.

The exact archive must be verified at its durable external location. The sanitized archive may be public only if explicitly requested and verified. Never publish exact raw evidence.

Update canonical documents only from reviewed evidence:

- README.md;
- AGENTS.md;
- docs/HANDOFF.md;
- docs/VALIDATION_STATUS.md;
- docs/TESTING.md;
- docs/KNOWN_LIMITATIONS.md;
- docs/LOGGING_POLICY.md if necessary;
- docs/RELEASE_CHECKLIST.md;
- docs/validation_results/VALIDATION_LOG_INDEX.md.

Final accepted language may state:

  Level 1 — Core mathematical Hugging Face implementation: complete

It must immediately state the exclusions: no paper-scale reproduction, no retrieval benchmark, no optimized long-context efficiency, no distributed training, and no P1 ecosystem capability.

Open the final draft evidence PR, report REVIEW_REQUIRED, and stop. Do not create the final immutable tag before merge.

After the user reviews and merges the final PR and resumes the Goal:

- verify the merge commit and CI;
- verify canonical documents and evidence descriptor;
- confirm clean main;
- propose, but do not create without explicit user instruction, immutable tag `p0-core-qualified-dense-v1`;
- report COMPLETE.

==================================================
14. EVIDENCE AND RETENTION RULES
==================================================

Use the merged P1-preflight A tools and schema. Do not weaken archive canonicalization, allowlist, path/type/link rejection, checksum coverage, sanitization, descriptor binding, or offline verification.

For each stage record:

- implementation base and tested commit;
- branch;
- clean worktree before edits/run;
- environment versions;
- exact commands and exit codes;
- test counts and key metrics;
- warnings/fallbacks;
- reviewer status;
- artifact hashes;
- stage verdict and limitations.

Intermediate stage evidence may be compact. The final Level 1 evidence must have explicit acceptance review and external exact/private retention.

Do not alter docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json to pretend its historical blockers were completed unless the missing exact archive and explicit review are actually supplied and verified through a separate truthful update.

==================================================
15. PROGRESS REPORTING
==================================================

At each checkpoint report:

Checkpoint:
  program stage and current subtask

Commands executed:
  exact commands and exit status

Development environment:
  Conda/venv identity and package changes

Files changed:
  tracked, untracked, ignored artifacts

Tests and results:
  pass/fail, counts, tolerances, key metrics

Source/decision status:
  paper/reference/API evidence and unresolved ambiguity

Evidence status:
  provenance, exact/private, sanitized, reviewer

Current blocker or risk:
  none, environment, hardware, source ambiguity, core defect

Next checkpoint:
  next action within the current stage

Do not ask for permission before ordinary in-scope actions. Stop only at a defined review/decision/blocked state or when safety/credentials/hardware require user action.

==================================================
16. TERMINAL AND PAUSE STATES
==================================================

REVIEW_REQUIRED

A focused stage PR is open and all stage-local work is complete. Report branch, commit, PR, tests, evidence, and remaining limitations. Do not start the next stage until merge and user resume.

SPECIFICATION_DECISION_REQUIRED

Source evidence does not support a safe C2 public mode/default/migration decision. Commit the audit/ADR/options if useful, open a draft PR, and stop without guessing.

PARTIAL/BLOCKED WITH EVIDENCE

A reproducible core failure, unavailable required hardware, incompatible environment, absent reviewer/archive storage, failed retention verification, or other non-weakenable blocker remains. Preserve compact evidence and state the smallest supported next action.

COMPLETE

All five stage PRs are reviewed and merged, final evidence is accepted and externally retained, main is clean, canonical documents state Level 1 complete with correct limitations, and no P1 model/ecosystem claim was introduced.

==================================================
17. FINAL RESPONSE FORMAT
==================================================

Level 1 program status:
  - current stage:
  - terminal/pause state:

Merged stage PRs:
  - P0.5-C1:
  - P0.5-C2:
  - P1-preflight B:
  - P0.5-C3:
  - final requalification:

変更ファイル:
  - ...

追加ファイル:
  - ...

開発環境:
  - active/isolated environments
  - before/after versions

実行テスト:
  - command: ...
    result: ...

Core contracts:
  - architecture/init:
  - MiPE/position/cache:
  - gradient checkpointing:
  - paper-training contract:

Evidence:
  - tested commit:
  - reviewer:
  - start/end worktree:
  - exact/private archive:
  - sanitized archive:
  - descriptor/verifier:

結果:
  - Level 1 verdict:
  - claims supported:
  - claims not supported:

未確認・制限:
  - ...

作成したPR:
  - ...

次にやるべきこと:
  - ...

==================================================
18. FINAL PRINCIPLE
==================================================

The objective is not to make every test green by weakening the contract. The objective is to establish exactly what the paper, oracle, reference implementation, HF implementation, and supported Transformers API require; implement explicit compatible semantics; and preserve reviewed evidence. A documented ambiguity or reproducible failure is a valid blocked result. Hidden assumptions are not.
```

## Expected stopping behavior

The first run of this Goal should finish **only Stage P0.5-C1**, open its draft
PR, and stop at `REVIEW_REQUIRED`. After the user merges the PR:

```bash
git switch main
git fetch --prune
git pull --ff-only
```

Then return to Codex and resume:

```text
/goal resume
```

Repeat this review/merge/resume cycle for C2, preflight B, C3, and final
requalification.
