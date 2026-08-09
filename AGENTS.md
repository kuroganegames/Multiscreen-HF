# AGENTS.md

## Project identity

This repository is an **unofficial Hugging Face Transformers-compatible implementation of Multiscreen**. Treat it as a correctness-first research artifact, not an official implementation, production serving stack, or paper-scale reproduction.

Current validation state:

```text
P0-1: paper_math_oracle vs HF implementation
  complete

P0-2: dieOD/multiscreen-pytorch vs HF implementation vs paper_math_oracle
  complete

P0-3: Psi=8/16 TinyStories bf16 smoke training
  complete

P0-4: GPT-2 vocab 50,257 + context 4096 short pretraining smoke
  complete
  qualifying local CUDA bf16 Psi=8/Psi=16 runs passed
  reviewed compact evidence is recorded under docs/validation_results/

P1-preflight A: validation provenance / evidence retention v1
  infrastructure implemented; status partial/blocked
  four P0-4 summary/metrics files matched committed hashes
  both completion markers were found and hashed for the new descriptor
  sanitized archive verified locally
  exact/private retention blocked: MULTISCREEN_EVIDENCE_ARCHIVE_DIR unset
  acceptance review pending: no explicit reviewer supplied
  no public asset

P0.5-C1: architecture / initialization / all-scale contract
  accepted and merged in PR #9

P0.5-C2: long-position / MiPE / cache semantics
  accepted and merged in PR #10
  CUDA-autocast cache-dtype correction merged in PR #11

P1-preflight B: gradient-checkpointing API modernization
  accepted and merged in PR #12

P0.5-C3: paper-training-contract smoke
  accepted and merged in PR #13
  local contract, pinned-data, CUDA, regression, and evidence checks passed
  tested source: 8fa5dbf13530c942b2c9e5f03a572bd0cd5ca74f
  sanitized archive verified; exact/private retention and explicit evidence
  review remain pending in the historical descriptor

final Level 1 requalification and evidence
  current focused Stage 5
  plan: docs/LEVEL1_CORE_REQUALIFICATION_PLAN.md
  pending; no Stage 5 qualifying evidence has been accepted
  five-stage Level 1 Core program not complete

P1 model/ecosystem capabilities
  none validated
```

P1-preflight A is evidence infrastructure. It does not validate a new model capability and must not be combined with gradient-checkpointing modernization, P0.5 core work, or PEFT/LoRA.

## Read before changing anything

Read these files in order:

```text
README.md
docs/HANDOFF.md
docs/VALIDATION_STATUS.md
docs/TESTING.md
docs/KNOWN_LIMITATIONS.md
docs/P0_5_C2_PLAN.md
docs/LEVEL1_CORE_REQUALIFICATION_PLAN.md
docs/adr/ADR-0001-mipe-position-semantics.md
docs/P0_5_C3_PLAN.md
docs/P1_PREFLIGHT_B_PLAN.md
docs/P1_PREFLIGHT_A_PLAN.md
docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md
docs/LOGGING_POLICY.md
docs/EVIDENCE_ARCHIVE_POLICY.md
docs/REPOSITORY_AUDIT.md
docs/RELEASE_CHECKLIST.md
docs/validation_results/VALIDATION_LOG_INDEX.md
docs/validation_results/P0_5_C2_SUMMARY.md
docs/validation_results/P1_PREFLIGHT_B_SUMMARY.md
docs/validation_results/P0_4_SUMMARY.md
docs/validation_results/P0_4_SUMMARY.json
docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json
docs/validation_results/P0_5_C3_SUMMARY.md
docs/validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json
```

The P0-4 plan and Codex handoff are retained reproduction/history documents; they do not mean P0-4 is pending:

```text
docs/P0_4_PLAN.md
docs/P0_4_RESULTS_TEMPLATE.md
docs/CODEX_P0_4_HANDOFF.md
```

For model or oracle changes in later gates, also inspect:

```text
multiscreen_transformers/configuration_multiscreen.py
multiscreen_transformers/modeling_multiscreen.py
oracle/paper_math_oracle.py
oracle/test_against_hf_port.py
p0_2_three_way_minimal/test_three_way_minimal.py
```

## Non-negotiable correctness contracts

Do not break these contracts:

1. Paper Trim parameterization and HF/reference parameterization are equivalent only under:

   ```text
   s_r_paper = -s_r_hf
   ```

2. `oracle/paper_math_oracle.py` is a dense equation-oriented correctness oracle. It is not a speed reference.

3. Oracle compute modes have different purposes:

   ```text
   stable paper/oracle checks:
     mipe_compute_dtype="fp32"
     softmask_compute_dtype="fp32"

   low-precision reference compatibility:
     mipe_compute_dtype="reference"
     softmask_compute_dtype="reference"
   ```

4. The current HF implementation contains DynamicCache compatibility logic. Greedy `generate(use_cache=True)` is smoke-tested; broad generation compatibility is not.

5. The current screening implementation is dense and quadratic in sequence length. Never present runtime or memory results as evidence of the paper's efficiency claims.

6. `tie_word_embeddings=True` is part of the architecture contract. Logits use normalized tied embeddings.

7. Accepted evidence must remain truthful:

   - do not alter accepted P0 metrics to fit a new schema;
   - do not fabricate reviewer identity;
   - do not infer historical clean-worktree state from a commit SHA;
   - preserve unknown historical provenance as explicit `not_recorded` data;
   - never upload exact raw evidence to a public location;
   - do not commit raw archives, outputs, checkpoints, model weights, secrets, or private absolute paths.

8. Preserve the development environment:

   - Python environments are managed with Conda;
   - `uv` is available as a scoped installation helper, not as a replacement for Conda;
   - inspect the active environment before installing anything;
   - use the current Conda environment when suitable;
   - create a separate Conda or other isolated environment when isolation is useful;
   - never delete or broadly mutate an existing environment or the Conda base environment;
   - never install globally;
   - do not run `conda update --all`, unconstrained `pip install -U`, broad `uv sync`, or equivalent changes;
   - when using `uv`, target an explicit environment and do not rewrite unrelated lockfiles;
   - record package versions before and after any environment change;
   - prefer Python standard-library implementations for evidence tooling.

Creating an isolated virtual environment is allowed during P1-preflight A.

## Git workflow

- Start from an up-to-date `main` and a clean working tree.
- Do not develop directly on `main`.
- Create a focused branch for each validation or implementation step.
- Keep changes scoped. Separate model-core changes from documentation, evidence, or experiment changes.
- Do not rewrite or discard user changes.
- Do not commit checkpoints, `outputs/`, caches, raw archives, raw large logs, or generated model weights.
- Before opening a PR, inspect `git diff`, `git diff --check`, `git status`, and the exact files staged.
- Do not merge the final PR automatically.

## Testing policy

Always run the tests relevant to the files changed.

Minimum baseline after a fresh clone or documentation/evidence-tooling change:

```bash
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

For P1-preflight A evidence tooling, run the standard-library-only syntax and
focused fixture suite:

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

The focused suite must cover schema and archive fixtures, deterministic
packaging, tampering, sanitization, and offline verification.

The verifier contract requires exactly one canonical gzip member, canonical
normalized USTAR headers, member boundaries, and padding, and an independent
rescan of every sanitized member including control metadata. When a descriptor
is supplied, it must match the validation gate, tested-source commit, and full
source-artifact set and metadata.

If any of the following change in a later gate, rerun P0-1 and P0-2 at the strongest feasible level, including CUDA bf16 where available:

```text
multiscreen_transformers/modeling_multiscreen.py
multiscreen_transformers/configuration_multiscreen.py
oracle/paper_math_oracle.py
cache/generation handling
state_dict conversion
position or mask behavior
```

P1-preflight A must not change those files. A model-core diff is a scope violation, not a reason to expand the gate.

## P1-preflight A scope (historical gate-specific contract)

The implemented infrastructure follows [docs/P1_PREFLIGHT_A_PLAN.md](docs/P1_PREFLIGHT_A_PLAN.md), and the Codex Goal is in [docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md](docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md).

The in-scope implementation includes:

```text
scripts/collect_validation_provenance.py
scripts/package_validation_evidence.py
scripts/verify_validation_evidence.py
schemas/validation_evidence_v1.schema.json
focused evidence-tooling tests
docs/EVIDENCE_ARCHIVE_POLICY.md
docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json
policy, handoff, index, release-checklist, CI, and ignore updates
isolated virtual-environment creation when useful
```

Expected out-of-scope work includes:

```text
model/config/oracle/cache/generation/state-dict behavior
P0 training harness or config changes
new P0-4 GPU training
P1-preflight B
P0.5-C1/C2/C3
PEFT/LoRA/QLoRA/Unsloth
broad generation, compile, serving, or Triton work
```

## Provenance and retention rules

Separate three concepts:

```text
original validation-run provenance
evidence-packaging/handoff provenance
acceptance review
```

For the historical P0-4 run, facts not captured during execution must remain explicit `null`/`not_recorded_in_original_run`. Do not retroactively claim a clean worktree or an original-run reviewer.

Recorded worktree provenance hashes the exact stdout bytes from
`git status --porcelain=v1 --untracked-files=all --ignore-submodules=none`.
When applicable, it also records privacy-safe state/count/hash summaries from
`git submodule status --recursive`; shareable output never includes raw paths.

Reviewer identity for the new evidence handoff must come from explicit input, for example:

```bash
export MULTISCREEN_EVIDENCE_REVIEWERS=kuroganegames
```

The authenticated GitHub login is not automatically a reviewer.

A recorded review also requires a non-empty explicit method, a full 40- or
64-character hexadecimal review commit, and an explicitly supplied
`raw-events-reviewed` boolean.

Exact raw evidence must be written outside the repository to an explicitly configured user-controlled location:

```bash
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/outside/the/repository
```

When a fresh checkout does not contain the original ignored outputs:

```bash
export MULTISCREEN_P0_4_RAW_ROOT=/absolute/path/to/the/original/P0-4/raw-output-root
```

Exact archives stay private. A separately sanitized archive may be published only when explicitly configured and verified. Never publish the exact archive to a public GitHub release.

Current P0-4 retention status is partial: all four summary/metrics files matched
their committed hashes, both completion markers were found and hashed for the
new descriptor, and the sanitized archive verified locally. Exact/private
retention is blocked because `MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not
configured; acceptance review is pending because no explicit reviewer was
supplied; and no public asset exists. P0-4 remains complete, and no P1
model/ecosystem capability is validated by this infrastructure.

## P0-4 qualification and reproduction rules

P0-4 remains complete from accepted evidence. A future reproduction is qualifying only when all strict conditions are met:

```text
GPT-2 tokenizer vocabulary = 50,257
sequence length = 4,096
device = CUDA
AMP dtype = bf16
optimizer steps >= 50
finite train losses and gradient norms
configured probe-loss decrease
save/load and tokenizer reload
generation with cache
manual cache-split comparison
summary/metrics and completion marker
failure artifacts absent
```

A CPU, shorter-context, different-dtype, or fewer-step run is diagnostic only. Static validation or CI diagnostics do not replace the accepted CUDA evidence.

## Future validation strategy

The staged Level 1 core sequence currently stands at:

```text
P0.5-C1       accepted and merged in PR #9
P0.5-C2       accepted and merged in PR #10
C2 correction accepted and merged in PR #11
P1-preflight B accepted and merged in PR #12
P0.5-C3       accepted and merged in PR #13
final P0 core requalification current Stage 5; pending and not validated
P1-1 PEFT/LoRA smoke remains outside the Level 1 core program
```

P0.5-C3 was accepted by reviewed and merged PR #13. Proceed with final P0 core
requalification only as the separate Stage 5 defined in
`docs/LEVEL1_CORE_REQUALIFICATION_PLAN.md`. Do not describe Stage 5 or the full
Level 1 Core program as validated or complete until its focused implementation,
test contract, evidence, and status update are reviewed.

## Validation records

Compact accepted summaries and archive descriptors belong in:

```text
docs/validation_results/
```

Follow `docs/LOGGING_POLICY.md` and `docs/EVIDENCE_ARCHIVE_POLICY.md`. The
current compact P0-4 retention state is recorded in
`docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json`; its partial/blocked
retention status is independent of the accepted P0-4 result. Exact raw archives
remain outside Git. Public artifacts must be sanitized and independently verified.

Never commit:

```text
outputs/
checkpoint directories
raw evidence archives
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
local Hugging Face caches
wandb/
large raw terminal logs
private absolute-path reports
```

## Reporting format

At each checkpoint, report:

```text
Checkpoint:
Commands executed:
Environment:
Files changed:
Tests and results:
Evidence status:
Current blocker or risk:
Next checkpoint:
```

At completion, report:

```text
変更ファイル:
追加ファイル:
開発環境:
実行テスト:
Provenance:
Evidence retention:
結果:
未確認・制限:
作成したPR:
次にやるべきこと:
```

Be precise about what was executed and what is historical, inferred, unavailable, staged, private, sanitized, or publicly retained.
