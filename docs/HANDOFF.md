# Development Handoff

This is the primary development-restart document for `kuroganegames/Multiscreen-HF`.

The repository contains a **P0-qualified correctness-first research implementation** of Multiscreen for Hugging Face Transformers. P0-1, P0-2, P0-3, and P0-4 are complete. Reviewed CUDA bf16 GPT-2-vocabulary, context-4096 evidence is recorded for Psi=8 and Psi=16.

The current evidence-infrastructure gate and status are:

```text
P1-preflight A: Validation provenance / evidence retention v1 — PARTIAL/BLOCKED WITH EVIDENCE
```

This is an infrastructure gate. It does not validate a new model capability and must not be combined with P1-preflight B, P0.5 core work, or PEFT/LoRA.

The infrastructure is implemented and tested. All four retained P0-4
summary/metrics files matched their committed hashes; both completion markers
were found and hashed for the new descriptor; and a sanitized archive verified
locally. Durable exact/private retention is blocked because
`MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not configured; acceptance review is
pending because no explicit reviewer was supplied; and no public asset exists.
P0-4 remains complete and no P1 model/ecosystem capability is validated.

The current staged Level 1 Core state is:

```text
P0.5-C1: accepted; focused PR #9 merged
P0.5-C2: accepted; focused PR #10 merged; correction PR #11 merged
P1-preflight B: accepted; focused PR #12 merged
P0.5-C3 / Stage 4: local validation passed; focused draft PR REVIEW_REQUIRED
```

Start with:

- C2 semantic decision: [ADR-0001](adr/ADR-0001-mipe-position-semantics.md)
- C2 plan: [P0_5_C2_PLAN.md](P0_5_C2_PLAN.md)
- C2 accepted result: [P0_5_C2_SUMMARY.md](validation_results/P0_5_C2_SUMMARY.md)
- Stage 3 plan: [P1_PREFLIGHT_B_PLAN.md](P1_PREFLIGHT_B_PLAN.md)
- Stage 3 accepted result: [P1_PREFLIGHT_B_SUMMARY.md](validation_results/P1_PREFLIGHT_B_SUMMARY.md)
- Stage 4 plan: [P0_5_C3_PLAN.md](P0_5_C3_PLAN.md)
- Stage 4 local result: [P0_5_C3_SUMMARY.md](validation_results/P0_5_C3_SUMMARY.md)
- Stage 4 evidence: [P0_5_C3_EVIDENCE_ARCHIVE.json](validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json)
- repository instructions: [`AGENTS.md`](../AGENTS.md)
- gate design: [P1_PREFLIGHT_A_PLAN.md](P1_PREFLIGHT_A_PLAN.md)
- Codex Goal handoff: [CODEX_P1_PREFLIGHT_A_HANDOFF.md](CODEX_P1_PREFLIGHT_A_HANDOFF.md)
- canonical validation boundary: [VALIDATION_STATUS.md](VALIDATION_STATUS.md)
- accepted P0-4 evidence: [P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md) and [P0_4_SUMMARY.json](validation_results/P0_4_SUMMARY.json)
- current logging policy: [LOGGING_POLICY.md](LOGGING_POLICY.md)
- evidence archive policy: [EVIDENCE_ARCHIVE_POLICY.md](EVIDENCE_ARCHIVE_POLICY.md)
- P0-4 retention descriptor: [P0_4_EVIDENCE_ARCHIVE.json](validation_results/P0_4_EVIDENCE_ARCHIVE.json)
- repository audit: [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md)

Use [CODEX_P0_4_HANDOFF.md](CODEX_P0_4_HANDOFF.md) only for an intentional P0-4 reproduction or requalification.

## 1. Current project state

| Milestone | Status | Meaning |
|---|---:|---|
| P0-1 | Complete | Oracle and HF small-shape formula, loss, mask, position, and cache checks passed under the recorded conditions. |
| P0-2 | Complete | Vendored unofficial reference, HF implementation, and oracle three-way comparisons passed under recorded CPU fp32 and CUDA bf16 conditions. |
| P0-3 | Complete | Psi=8/16 TinyStories bf16 smoke training passed, including finite loss/gradients, save/load, cache split, and greedy generation. |
| P0-4 | Complete | Psi=8/16 GPT-2-vocabulary, context-4096 CUDA bf16 qualification passed and compact reviewed evidence was committed. |
| P1-preflight A | Partial/blocked | Tooling, schema, policy, source-hash audit, and sanitized verification are complete; explicit review and external exact/private retention remain blocked. |
| P0.5-C1 | Complete | Architecture, initialization, all-scale meta shapes, config, tied embeddings, and packed-text contracts were reviewed and merged as PR #9. |
| P0.5-C2 | Complete | Dual MiPE modes and long-boundary contracts were merged as PR #10; the CUDA-autocast cache-dtype correction was merged as PR #11. |
| P1-preflight B | Complete | Supported non-reentrant API, exact 4.57.6/5.14.1 matrix, strong P0 regressions, and checkpointed CUDA smokes were accepted by merged PR #12. |
| P0.5-C3 | Review required | Local validation passed; acceptance remains pending. |
| P1 ecosystem capabilities | None validated | PEFT/LoRA, QLoRA, Unsloth, generation matrix, compile, and serving remain future gates. |

### Baseline identity

```text
Current baseline: P0-qualified research implementation through P0-4
P0.5-C1 merge / C2 base: ec805c1ba60c55ea4beb3ad68e0a00c0d718e909
P0.5-C2 merge: 54aa96ee914e9f8ac7b58e6d176d4e3fbeb0ae27
C2 correction / Stage 3 base: 0c83be6b4b043f4b965df4528534f24e9d5ab4f1
Stage 3 merge / Stage 4 base: a2d43517c45dc39855db81b9286c4abf190a2c14
Current staged gate: P0.5-C3 — local validation passed; REVIEW_REQUIRED
P1-preflight A implementation base (origin/main at branch creation): 34cbecd25bb38a6f92125071b1c08e42d71008f9
Primary implementation: multiscreen_transformers/modeling_multiscreen.py
Primary config: multiscreen_transformers/configuration_multiscreen.py
Primary equation oracle: oracle/paper_math_oracle.py
Canonical validation status: docs/VALIDATION_STATUS.md
Accepted P0-4 summary: docs/validation_results/P0_4_SUMMARY.{md,json}
Selected gate design: docs/P0_5_C3_PLAN.md
Stage 4 tested source: 8fa5dbf13530c942b2c9e5f03a572bd0cd5ca74f
Stage 4 local result: docs/validation_results/P0_5_C3_SUMMARY.{md,json}
Stage 4 evidence descriptor: docs/validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json
Selected Codex Goal: docs/CODEX_LEVEL1_CORE_HF_HANDOFF.md
Evidence archive policy: docs/EVIDENCE_ARCHIVE_POLICY.md
P0-4 retention descriptor: docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json
```

P0-4 remains complete from its accepted evidence. The current descriptor
truthfully records that all four summary/metrics files matched their committed
hashes, both completion markers were found and hashed for the new descriptor,
and a sanitized archive verified locally. Exact/private retention is blocked,
acceptance review is pending, and no public asset exists. P1-preflight A remains
partial and must not rewrite the original metrics, imply a new model validation
result, or validate any P1 capability.

P0.5-C3 is the current focused Stage 4. Its plan distinguishes the paper's
stated training recipe from derived arithmetic and repository operational
choices, then separates exact contract checks from bounded workstation
diagnostics.

The exact contract, pinned data identity and accounting, Transformers
4.57.6/5.14.1 focused matrix, full P0-1/P0-2 CPU fp32 and CUDA bf16
regressions, and four Psi=8/Psi=16 CUDA bf16 diagnostics passed on tested
commit `8fa5dbf13530c942b2c9e5f03a572bd0cd5ca74f`. The sanitized evidence
archive verified locally. Exact/private retention is blocked because
`MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not configured, no public asset
exists, and no explicit reviewer was supplied. This local result is
`REVIEW_REQUIRED`, not accepted.

## 2. First ten minutes after a fresh clone

```bash
git clone https://github.com/kuroganegames/Multiscreen-HF.git
cd Multiscreen-HF

git status --short --branch
git rev-parse HEAD
git log -1 --oneline
git remote -v
```

Confirm that `main` is current and the working tree is clean before creating a focused branch.

### Development environment

Python development environments are managed with Conda. `uv` is installed and may be used as a scoped installation helper.

Do not replace or broadly mutate the current environment by default.

```text
- inspect `CONDA_DEFAULT_ENV`, `CONDA_PREFIX`, Python, pip, and package versions;
- use the active Conda environment when suitable;
- create a separate Conda or other isolated virtual environment when isolation is useful;
- do not modify or delete Conda base;
- do not install globally;
- do not run broad upgrades such as `conda update --all` or unconstrained `pip install -U`;
- when using uv, target an explicit environment and avoid rewriting unrelated lockfiles;
- record before/after versions for any package change.
```

Creating an isolated environment is explicitly allowed for P1-preflight A.

Install only what is needed:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle
```

If the active environment already has the required dependencies, do not reinstall or upgrade them speculatively.

## 3. Codex continuation

Start Codex from the Git root so it reads `AGENTS.md`:

```bash
codex
```

If `/goal` is unavailable:

```bash
codex features enable goals
codex
```

To complete the current partial gate, configure an explicit reviewer and a
user-controlled archive directory outside the repository:

```bash
export MULTISCREEN_EVIDENCE_REVIEWERS=kuroganegames
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/to/durable/evidence-storage
```

A fresh clone normally does not contain ignored P0-4 raw outputs. Point to the original retained files when needed:

```bash
export MULTISCREEN_P0_4_RAW_ROOT=/absolute/path/to/the/original/P0-4/raw-output-root
```

Optional sanitized public publication must be explicit:

```bash
export MULTISCREEN_EVIDENCE_PUBLIC_RELEASE_TAG=p0-4-qualified-v0
```

Never publish the exact raw archive to a public release.

The original implementation prompt is retained in
[CODEX_P1_PREFLIGHT_A_HANDOFF.md](CODEX_P1_PREFLIGHT_A_HANDOFF.md). Do not
replay it; use this handoff to complete only the blockers recorded above.

## 4. P1-preflight A purpose

The gate addresses three evidence-review follow-ups:

```text
1. reviewer and review-method provenance;
2. clean/dirty worktree provenance;
3. long-term raw evidence retention with separate exact/private and sanitized/shareable archives.
```

Implemented infrastructure:

```text
scripts/collect_validation_provenance.py
scripts/package_validation_evidence.py
scripts/verify_validation_evidence.py
schemas/validation_evidence_v1.schema.json
focused evidence-tooling tests
docs/EVIDENCE_ARCHIVE_POLICY.md
docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json
policy, index, release-checklist, CI, ignore, agent, README, and handoff updates
```

No model, oracle, cache, generation, position, state-dict, tokenizer, dataset, training-harness, or P0 config source should change.

## 5. Evidence truth rules

Keep these concepts separate:

```text
original validation-run provenance
evidence-packaging/handoff provenance
acceptance review
```

For the historical P0-4 run, a commit SHA does not prove a clean worktree. If clean state or original-run reviewer identity was not recorded at execution time, store:

```text
value: null
status: not_recorded_in_original_run
```

Do not convert unknown into `false`, and never guess `true`.

For the current evidence handoff, record:

```text
starting commit and branch
SHA-256 and byte count of exact `git status --porcelain=v1 --untracked-files=all --ignore-submodules=none` stdout
privacy-safe recursive submodule state/hash/count when applicable
staged/unstaged/untracked state
clean state before tracked edits
final commit
clean state after commit
explicit reviewer(s)
review method/time/full commit/raw-events-reviewed boolean
archive creation and verification times
```

Reviewer identity must come from explicit CLI/environment input. The authenticated GitHub login is not automatically the reviewer.

The collector rejects a recorded review unless the method is non-empty, the
review commit is a full 40- or 64-character hexadecimal object ID, and
`--raw-events-reviewed` is explicitly `true` or `false`.

## 6. Exact and sanitized evidence

### Exact/private archive

Preserve original bytes, source hashes, sizes, completion markers, and machine-readable raw events. Store it outside Git in the configured retention location.

Do not include checkpoints or model weights by default. If checkpoint retention is desired, handle it as a separate private asset and record only a manifest in the validation descriptor.

### Sanitized/shareable archive

Create a separate archive after scanning and redacting:

```text
secrets and tokens
usernames and unnecessary hostnames
local absolute paths
cache paths
private archive paths
credential-bearing remotes
```

Preserve useful versions, GPU identity, metrics, relative repository paths, commands normalized to `python`, hashes, and verdicts.

The exact archive must never be published publicly. The sanitized archive may be published only when explicitly configured and verification passes.

## 7. Required tooling behavior

The design is detailed in [P1_PREFLIGHT_A_PLAN.md](P1_PREFLIGHT_A_PLAN.md). At minimum:

```text
- versioned JSON schema;
- explicit reviewer input;
- Git worktree collection at start/end;
- deterministic allowlist-based archive packaging;
- source-hash verification;
- symlink/path-traversal/device-file rejection;
- deterministic manifest and SHA256SUMS;
- offline verifier;
- tamper detection;
- sanitization report;
- truthful historical not-recorded representation;
- compact archive descriptor under docs/validation_results/.
```

Prefer Python standard-library code and synthetic CI fixtures. CI must not require private P0-4 artifacts, GPU, Hub access, external storage, or release credentials.

The verifier accepts exactly one canonical gzip member, reconstructs canonical
normalized USTAR headers/boundaries/padding, and rescans every sanitized
member—including `MANIFEST.json`, `SHA256SUMS`, and the sanitization report.
A supplied descriptor is bound to the archive, validation gate, tested-source
commit, and complete source-artifact set and metadata.

### Standard-library tooling and recovery

The evidence tools and synthetic security fixtures require only the Python
standard library:

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

Create a recorded review only with an explicit reviewer, non-empty method,
full 40- or 64-character commit, and explicit raw-events-reviewed boolean:

```bash
python -S scripts/collect_validation_provenance.py \
  --reviewer "$MULTISCREEN_EVIDENCE_REVIEWERS" \
  --review-method "raw-event and archive review" \
  --review-commit "$(git rev-parse HEAD)" \
  --raw-events-reviewed true \
  --json
```

Given an allowlisted package-input JSON and the original raw root, create new
non-overwriting exact and sanitized archives:

```bash
PACKAGE_INPUT=/path/to/allowlisted-p0-4-package-input.json
SANITIZED_STAGING=/path/to/ignored/sanitized-staging

python -S scripts/package_validation_evidence.py \
  --input "$PACKAGE_INPUT" \
  --root "raw=${MULTISCREEN_P0_4_RAW_ROOT:?must be configured}" \
  --mode both \
  --exact-output "${MULTISCREEN_EVIDENCE_ARCHIVE_DIR:?must be configured}/validation-evidence-exact-p0-4-v1.tar.gz" \
  --sanitized-output "$SANITIZED_STAGING/validation-evidence-sanitized-p0-4-v1-r2.tar.gz" \
  --repository-root . \
  --json
```

For a restore drill, retrieve the current sanitized archive by its logical
locator and verify it before any extraction:

```bash
SANITIZED_ARCHIVE=/path/to/retrieved/validation-evidence-sanitized-p0-4-v1-r2.tar.gz

python -S scripts/verify_validation_evidence.py \
  --archive "$SANITIZED_ARCHIVE" \
  --expected-sha256 d58a4c9ecf28f20a135f4ba2ce95c5a532a04ea92f36e5b54d893400ae4c62fd \
  --evidence-document docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json \
  --schema schemas/validation_evidence_v1.schema.json \
  --json
```

The exact/private recovery command becomes valid only after the external
archive is created, the descriptor is updated with its hashes, and an
explicit reviewer completes acceptance. Never publish that exact archive.

## 8. Baseline tests

Before and after tracked changes:

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

Also run focused syntax, schema, fixture, deterministic-package, sanitization, and tamper tests.

A model-core diff is a scope violation. Do not turn P1-preflight A into a P0 requalification task.

## 9. P0-4 raw evidence audit

Read the expected raw SHA-256 values from:

```text
docs/validation_results/P0_4_SUMMARY.json
```

Locate the retained Psi=8/16 files only through the configured raw root or expected ignored output directories. At minimum verify:

```text
Psi=8 summary.json and metrics.jsonl
Psi=8 P0-4_COMPLETE.md
Psi=16 summary.json and metrics.jsonl
Psi=16 P0-4_COMPLETE.md
```

Do not regenerate substitute evidence if the originals are unavailable or hashes mismatch. Implement generic tooling, preserve the accepted P0-4 status, and report P1-preflight A as partial/blocked.

For this handoff, both Psi=8/16 `summary.json` and `metrics.jsonl` files
matched the four committed hashes. Both `P0-4_COMPLETE.md` markers were found
and hashed for the new descriptor. The sanitized archive verified locally.
Exact/private retention remains blocked because
`MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not configured; acceptance review is
pending because no explicit reviewer was supplied; and no public asset exists.
These retention gaps do not reopen the accepted P0-4 result.

## 10. Completion boundary

P1-preflight A is complete only when:

```text
schema/provenance/packaging/verifier/tests are implemented;
reviewer input is explicit;
current handoff worktree provenance is recorded;
historical unknown provenance remains truthful;
all four P0-4 summary/metrics files match their committed hashes;
both completion markers are found and hashed for the descriptor;
exact archive is stored in configured external retention storage;
sanitized archive is produced and verified;
compact archive descriptor and policy updates are committed;
quick P0 regression and repository hygiene pass;
a focused draft PR is opened or exact push/PR commands are provided;
no model behavior or P1 capability is changed.
```

If raw files, reviewer input, retention storage, sanitization, or verification are unavailable, use `PARTIAL/BLOCKED WITH EVIDENCE` rather than weakening the requirements.

## 11. Core contracts retained from P0

### Trim parameterization

```text
s_r_paper = -s_r_hf
```

### Oracle compute modes

```text
stable paper/oracle:
  mipe_compute_dtype="fp32"
  softmask_compute_dtype="fp32"

low-precision reference compatibility:
  mipe_compute_dtype="reference"
  softmask_compute_dtype="reference"
```

### Dense implementation boundary

The oracle and current HF screening path are correctness implementations, not evidence of paper-scale long-context efficiency.

### DynamicCache and tied embedding

Preserve `tie_word_embeddings=True`. C2 makes cache/position behavior explicit:
paper-absolute and reference-compatible MiPE are separate serialized modes,
legacy configs retain the reference rule, and cache calls require a complete
zero-based contiguous prefix. See ADR-0001. Broad generation remains outside
the validated boundary.

## 12. Accepted P0-4 record

The accepted record is in [P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md) and [P0_4_SUMMARY.json](validation_results/P0_4_SUMMARY.json).

Both accepted runs used:

```text
GPT-2 vocabulary: 50,257
sequence length: 4,096
CUDA bf16
microbatch / gradient accumulation: 1 / 8
optimizer steps: 50
```

The recorded result includes finite losses and gradients, probe-loss decrease, save/load, tokenizer reload, greedy generation with cache, manual cache-split comparison, completion markers, and raw summary/metrics SHA-256 hashes.

P1-preflight A may add a separate evidence-archive descriptor. It must not change the accepted training metrics.

## 13. Planned sequence after this gate

```text
P0.5-C1  architecture / initialization / all-scale contract — merged
P0.5-C2  long-position / MiPE / cache semantics — merged
P1-preflight B  gradient-checkpointing API modernization — merged as PR #12
P0.5-C3  paper-training-contract smoke — local validation passed; REVIEW_REQUIRED
final P0 core requalification
P1-1  PEFT/LoRA smoke
```

Do not begin final P0 core requalification until P0.5-C3 is reviewed and
merged. No later gate is validated merely because an earlier implementation or
infrastructure gate passes.

## 14. Final report format

```text
変更ファイル:
  - ...

追加ファイル:
  - ...

開発環境:
  - active Conda environment:
  - isolated environment created:
  - uv usage:
  - package changes:

実行テスト:
  - ...

Provenance:
  - reviewer:
  - original-run worktree status:
  - handoff worktree state:

Evidence retention:
  - raw files and hash verification:
  - exact/private archive:
  - sanitized archive:
  - archive verification:

結果:
  - COMPLETE or PARTIAL/BLOCKED WITH EVIDENCE

未確認・制限:
  - ...

作成したPR:
  - ...

次にやるべきこと:
  - ...
```

Always distinguish historical evidence, current evidence handoff, planned work, private storage, sanitized artifacts, and publicly retained artifacts.
