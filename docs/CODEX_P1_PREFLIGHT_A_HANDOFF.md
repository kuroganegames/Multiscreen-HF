# Codex Goal Handoff: P1-preflight A Validation Provenance and Evidence Retention v1

> **Retained original execution handoff.** The infrastructure in this Goal has
> been implemented. P1-preflight A is currently partial/blocked pending explicit
> acceptance review and durable external exact/private retention. Use
> [HANDOFF.md](HANDOFF.md) for current operational continuation; do not paste the
> full original Goal below as a fresh implementation request.

This was the local-Codex entry point for implementing the selected gate after the accepted P0-4 evidence PR.

## Phase recorded when this handoff was created

```text
Repository: kuroganegames/Multiscreen-HF
P0-1: complete
P0-2: complete
P0-3: complete
P0-4: complete
P1-preflight A: selected; implementation pending
P1-preflight B: not started
P0.5-C1/C2/C3: not started
P1 model/ecosystem capabilities: none validated
```

P1-preflight A is evidence infrastructure. It must not change Multiscreen model behavior or mark PEFT/LoRA or any other P1 capability as validated.

The design and acceptance criteria are in [P1_PREFLIGHT_A_PLAN.md](P1_PREFLIGHT_A_PLAN.md).

## Before starting Codex

Run Codex from the repository root so it loads `AGENTS.md`.

```bash
git clone https://github.com/kuroganegames/Multiscreen-HF.git
cd Multiscreen-HF

codex
```

If Goal mode is not available:

```bash
codex features enable goals
codex
```

Goal mode is intended for a durable outcome with explicit success criteria. Keep this task as one focused gate rather than combining it with P1-preflight B or LoRA work.

## Required local configuration

The reviewer identity and durable archive location should be provided explicitly before starting the Goal.

```bash
export MULTISCREEN_EVIDENCE_REVIEWERS=kuroganegames
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/to/user-controlled/durable/evidence-storage
```

When the Codex checkout is not the original P0-4 working directory, also point to the retained raw outputs:

```bash
export MULTISCREEN_P0_4_RAW_ROOT=/absolute/path/to/the/original/P0-4/raw-output-root
```

Optional publication of the **sanitized** archive may be requested explicitly:

```bash
export MULTISCREEN_EVIDENCE_PUBLIC_RELEASE_TAG=p0-4-qualified-v0
```

Do not set the public release variable unless publishing a sanitized archive is intended. The exact raw archive must remain private and must never be uploaded to a public release.

## Development environment

The user's Python development environments are managed with Conda. `uv` is installed but is used as an installation helper, not as permission to replace Conda or broadly rewrite package state.

Codex must preserve the current environment. It may use the active Conda environment when suitable or create an isolated environment when isolation is useful. It must not modify or delete the Conda base environment, perform broad upgrades, install packages globally, or rewrite unrelated lock/configuration files.

## Original ready-to-paste `/goal` prompt (retained history)

Paste the complete block below into Codex from the repository root.

```text
/goal Implement P1-preflight A: Validation provenance and evidence retention v1 for the current `kuroganegames/Multiscreen-HF` checkout. Continue through ordinary repository inspection, environment setup, implementation, testing, evidence packaging, documentation, commit, and PR preparation until one of the explicitly defined terminal states is reached. Do not begin P1-preflight B, P0.5 core work, PEFT/LoRA, or any model-capability gate during this Goal.

You are the continuing engineer and research-validation agent for the current repository checkout.

==================================================
1. PROJECT STATE
==================================================

This repository is an unofficial Hugging Face Transformers-compatible Multiscreen research implementation.

Accepted validation state:

P0-1:
  paper_math_oracle vs HF small-shape formula/loss/mask/position/cache validation
  status: complete

P0-2:
  dieOD/multiscreen-pytorch vs HF vs paper_math_oracle three-way validation
  status: complete

P0-3:
  Psi=8/16 TinyStories bf16 smoke training
  status: complete

P0-4:
  GPT-2 vocab 50,257 + context 4096 short CUDA bf16 training smoke
  Psi=8 and Psi=16 accepted
  status: complete
  compact reviewed evidence is under docs/validation_results/

Selected current gate:

P1-preflight A:
  validation provenance / evidence retention v1
  status: implementation pending

This gate validates evidence infrastructure only. It does not validate a new model capability and must not change the accepted P0 metrics or verdict.

The following are explicitly outside this Goal:

- P1-preflight B gradient-checkpointing modernization;
- P0.5-C1 architecture/initialization/all-scale validation;
- P0.5-C2 long-position/MiPE/cache semantics;
- P0.5-C3 paper-training-contract smoke;
- PEFT/LoRA, QLoRA, Unsloth;
- broad generation, torch.compile, serving, Triton/windowed kernels;
- new P0-4 GPU training or paper-scale reproduction.

==================================================
2. PRIMARY OBJECTIVE
==================================================

Create a versioned, tested, offline-verifiable evidence system that:

1. records reviewer identity and review method explicitly;
2. records clean/dirty Git worktree provenance without retroactive fabrication;
3. preserves exact raw validation artifacts outside the Git repository;
4. creates a separately sanitized archive suitable for sharing;
5. verifies file hashes, manifests, archive integrity, sanitization, and tamper resistance;
6. backfills a truthful P0-4 evidence-archive descriptor without changing accepted P0-4 metrics;
7. updates repository policy, handoff, release, index, and agent instructions;
8. leaves a focused committed branch and draft PR, but does not merge it.

Prefer Python standard-library implementations. Avoid adding runtime dependencies unless repository evidence shows they are necessary.

==================================================
3. SUCCESS BOUNDARY
==================================================

P1-preflight A COMPLETE means evidence infrastructure and the P0-4 archive handoff are complete. It does not mean any P1 model/ecosystem feature is validated.

A COMPLETE result requires:

- schema v1 implemented and documented;
- provenance collector implemented and tested;
- exact/private and sanitized packaging implemented and tested;
- offline verifier and tamper detection implemented and tested;
- reviewer fields populated from explicit input;
- historical unknown provenance represented as not-recorded, not guessed;
- P0-4 raw summary/metrics artifacts found and matched to committed SHA-256 values;
- exact archive written to the configured external retention directory;
- sanitized archive produced and verified;
- compact archive descriptor committed without private absolute paths;
- documentation and validation index updated;
- quick P0 regression and repository hygiene checks passed;
- focused branch committed and draft PR opened, or exact push/PR commands printed when authentication is unavailable.

If any retention-critical requirement cannot be met, use PARTIAL/BLOCKED WITH EVIDENCE rather than weakening the gate.

==================================================
4. MANDATORY READING
==================================================

Before editing anything, read these files in order:

- AGENTS.md
- README.md
- docs/HANDOFF.md
- docs/VALIDATION_STATUS.md
- docs/TESTING.md
- docs/KNOWN_LIMITATIONS.md
- docs/P1_PREFLIGHT_A_PLAN.md
- docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md
- docs/LOGGING_POLICY.md
- docs/REPOSITORY_AUDIT.md
- docs/RELEASE_CHECKLIST.md
- docs/validation_results/VALIDATION_LOG_INDEX.md
- docs/validation_results/P0_4_SUMMARY.md
- docs/validation_results/P0_4_SUMMARY.json
- .gitignore
- .github/workflows/p0-smoke.yml

Inspect, but do not modify during this Goal:

- multiscreen_transformers/configuration_multiscreen.py
- multiscreen_transformers/modeling_multiscreen.py
- multiscreen_transformers/data.py
- oracle/paper_math_oracle.py
- oracle/test_against_hf_port.py
- p0_2_three_way_minimal/test_three_way_minimal.py
- scripts/p0_3_tinystories_stability.py
- scripts/p0_4_gpt2_context4096_smoke.py
- both P0-4 config directories

In the first checkpoint report, summarize:

- the accepted validation boundary;
- what P1-preflight A changes and does not change;
- current logging/retention limitations;
- the distinction between original run provenance and later evidence-handoff provenance;
- the configured reviewer and archive location;
- any missing raw P0-4 artifacts.

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

Stable paper/oracle checks:
  mipe_compute_dtype="fp32"
  softmask_compute_dtype="fp32"

Reference-compatible low precision:
  mipe_compute_dtype="reference"
  softmask_compute_dtype="reference"

3. paper_math_oracle

`oracle/paper_math_oracle.py` is a dense equation-oriented correctness oracle. It is not a speed or long-context efficiency reference.

4. Tied normalized embedding

Preserve `tie_word_embeddings=True` and the current normalized tied-logit contract.

5. DynamicCache and position/cache semantics

Do not change cache, generation, mask, position, or state-dict behavior in this Goal.

6. Dense implementation boundary

Do not describe P0-4 runtime or memory as paper-efficiency evidence.

7. Evidence integrity

- do not alter accepted P0-4 metrics or verdict;
- do not generate substitute raw files when originals are missing;
- do not claim reviewer identity without explicit input;
- do not infer historical clean worktree state from a commit SHA;
- do not upload exact raw evidence to a public location;
- do not commit raw archives, outputs, checkpoints, model weights, secrets, or private paths.

8. Development environment

The current development environment must not be broken.

Python environments are managed with Conda. `uv` is installed and may be used only as a scoped installation helper.

- inspect the active environment before installing anything;
- use the existing Conda environment when suitable;
- if isolation is useful, create a separate Conda or other isolated virtual environment;
- never delete or broadly mutate the existing environment;
- never modify the Conda base environment;
- never install globally;
- do not run `conda update --all`, unconstrained `pip install -U`, `uv sync`, or equivalent broad changes unless a repository requirement explicitly demands it and the change is isolated;
- when using `uv`, target an explicit environment and do not rewrite unrelated lockfiles;
- record package versions before and after any environment change;
- prefer standard-library code so verification remains portable.

==================================================
6. ALLOWED CHANGE SCOPE
==================================================

This Goal may change, when justified:

- scripts/collect_validation_provenance.py
- scripts/package_validation_evidence.py
- scripts/verify_validation_evidence.py
- schemas/validation_evidence_v1.schema.json
- focused evidence-tooling tests
- docs/EVIDENCE_ARCHIVE_POLICY.md
- docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json
- docs/validation_results/VALIDATION_LOG_INDEX.md
- docs/LOGGING_POLICY.md
- docs/RELEASE_CHECKLIST.md
- docs/HANDOFF.md
- AGENTS.md
- README.md
- .gitignore
- a focused CI step using synthetic fixtures
- isolated virtual-environment creation outside tracked repository content

A dependency or pyproject change is allowed only if standard-library implementation is demonstrably insufficient. Justify it explicitly and keep it optional/dev-only.

Do not change:

- model/config/oracle semantics;
- cache, generation, position, mask, or state-dict code;
- P0-3/P0-4 training harness behavior;
- P0-4 configs or qualification tolerances;
- accepted P0 summary metrics;
- third-party reference code;
- package version merely for this infrastructure gate.

If a model-core diff appears, stop and remove it from this branch. Do not expand this Goal into a model requalification task.

==================================================
7. GIT AND CHANGE-SAFETY RULES
==================================================

Start by running:

- git status --short --branch
- git rev-parse HEAD
- git log -1 --oneline
- git remote -v
- git diff --quiet
- git diff --cached --quiet

Expected starting state:

- up-to-date main;
- accepted P0-4 evidence merge present;
- clean working tree;
- no unrelated user changes.

Create a focused branch before tracked edits:

  infra/p1-preflight-a-evidence-v1

Rules:

- never commit directly to main;
- never discard or overwrite existing user work;
- do not use `git reset --hard` as cleanup;
- stage explicit paths, not unrelated files;
- do not commit ignored evidence archives or raw outputs;
- inspect `git diff`, `git diff --check`, staged diff, and status before commit;
- do not auto-merge the PR;
- do not create or move a release tag unless explicitly configured;
- do not expose credentials in command output.

Record P1-preflight A handoff provenance separately from historical P0-4 run provenance.

==================================================
8. EVIDENCE TRUTH MODEL
==================================================

Use separate fields for:

A. Original validation-run provenance
B. Evidence-packaging/handoff provenance
C. Acceptance review

For the historical P0-4 run, do not retroactively set these booleans unless primary evidence exists:

- run_worktree_clean_at_start
- run_worktree_clean_at_end
- original_run_reviewer

Represent missing historical facts structurally, for example:

  value: null
  status: not_recorded_in_original_run

Do not encode unknown values as false, and do not encode guessed values as true.

For the current P1-preflight A operation, record:

- starting base commit;
- working branch;
- exact `git status --porcelain=v1` SHA-256 before edits;
- whether staged, unstaged, and untracked changes existed;
- clean state before edits;
- final commit SHA;
- clean state after commit;
- archive creation and verification timestamps.

Reviewer identity must come from explicit input:

- `MULTISCREEN_EVIDENCE_REVIEWERS`, or
- a required `--reviewer` CLI argument.

The authenticated GitHub login may be reported separately but must not silently become a reviewer.

For P0-4, distinguish:

- original-run reviewer status;
- later acceptance/evidence reviewers;
- raw-events-reviewed boolean;
- review method;
- review timestamp;
- review commit.

==================================================
9. CHECKPOINT 1: REPOSITORY AND ENVIRONMENT AUDIT
==================================================

Collect and report without exposing secrets:

Repository:
- branch and HEAD;
- latest commit;
- status porcelain;
- staged/unstaged/untracked state;
- remote identity without embedded credentials;
- free disk space.

Conda/Python:
- `CONDA_DEFAULT_ENV` and `CONDA_PREFIX` when present;
- `conda --version` and `conda info --envs`;
- Python executable and version;
- pip version;
- `uv --version` when present;
- relevant package versions before any install.

Storage/configuration:
- `MULTISCREEN_EVIDENCE_REVIEWERS`;
- `MULTISCREEN_EVIDENCE_ARCHIVE_DIR`;
- `MULTISCREEN_P0_4_RAW_ROOT`;
- optional public release tag;
- whether configured paths are outside the repository;
- whether they are writable;
- available free space.

GitHub:
- `gh auth status` when `gh` exists;
- authenticated login without credentials;
- push permission when discoverable.

Environment decision:

- If the active Conda environment already supports repository tests and standard-library tooling, preserve and use it.
- If isolation is needed, create a focused environment such as `multiscreen-p1-preflight-a`; do not alter base or remove existing environments.
- If using `uv`, target the selected environment explicitly.
- Do not install an additional package merely for convenience when the standard library suffices.

Report any environment change before proceeding.

==================================================
10. CHECKPOINT 2: BASELINE AND RAW-EVIDENCE AUDIT
==================================================

Run the minimum baseline before implementing tooling:

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

Audit committed P0-4 evidence:

- read `docs/validation_results/P0_4_SUMMARY.json`;
- enumerate expected Psi=8/Psi=16 raw summary and metrics hashes;
- confirm tested source commit and evidence commit distinction;
- confirm current summary limitations remain unchanged.

Locate raw evidence in this order:

1. `MULTISCREEN_P0_4_RAW_ROOT` when set;
2. expected ignored P0-4 output directories in the current checkout;
3. no other broad filesystem search unless explicitly safe and scoped.

Expected run evidence includes at least:

- Psi=8 `summary.json`;
- Psi=8 `metrics.jsonl`;
- Psi=8 `P0-4_COMPLETE.md`;
- Psi=16 `summary.json`;
- Psi=16 `metrics.jsonl`;
- Psi=16 `P0-4_COMPLETE.md`.

Verify raw `summary.json` and `metrics.jsonl` SHA-256 values against the committed P0-4 summary before packaging.

Do not rerun P0-4 to replace missing evidence.

If raw artifacts are missing or hashes mismatch:

- continue implementing and testing generic tooling;
- record the exact missing/mismatched files;
- do not create a P0-4 archive descriptor claiming complete retention;
- finish in PARTIAL/BLOCKED WITH EVIDENCE unless the originals are recovered.

==================================================
11. CHECKPOINT 3: SCHEMA V1 DESIGN
==================================================

Implement a versioned JSON schema at:

  schemas/validation_evidence_v1.schema.json

The schema must distinguish at least:

- schema version;
- gate and validation status;
- tested source commit/branch;
- original run provenance;
- evidence-handoff provenance;
- reviewers and review method;
- source artifacts with logical names, size, SHA-256, and classification;
- exact/private archive descriptor;
- sanitized archive descriptor;
- sanitization report;
- verification report;
- retention status;
- limitations.

Use structured status for unknown historical values. Required examples:

- recorded;
- not_recorded_in_original_run;
- not_applicable;
- pending;
- verified.

Worktree fields must distinguish:

- clean boolean when recorded;
- staged changes;
- unstaged changes;
- untracked count;
- SHA-256 of exact porcelain bytes;
- collection timestamp.

Reviewer entries should include:

- GitHub handle or explicit identifier;
- role;
- review method;
- reviewed-at UTC;
- review commit;
- raw-events-reviewed boolean.

Keep the schema portable and validate it without network access.

==================================================
12. CHECKPOINT 4: IMPLEMENT PROVENANCE AND ARCHIVE TOOLS
==================================================

Expected scripts:

1. `scripts/collect_validation_provenance.py`

Required behavior:

- collect repository/branch/HEAD information;
- collect exact porcelain bytes and SHA-256;
- classify staged, unstaged, and untracked state;
- collect reviewer input explicitly;
- redact credentials from remote URLs;
- emit deterministic JSON;
- support `--output` and `--json`/machine-readable operation;
- return nonzero on invalid required input.

2. `scripts/package_validation_evidence.py`

Required behavior:

- accept an explicit input manifest or allowlisted artifact list;
- never blindly archive an entire output directory;
- reject symlinks, device files, sockets, path traversal, and files outside allowed roots;
- verify source hashes before packaging;
- create an exact/private archive with unchanged source bytes;
- create a separately sanitized archive;
- normalize archive metadata for deterministic output;
- create `MANIFEST.json`, `SHA256SUMS`, and a sanitization report;
- exclude checkpoints and model weights by default;
- never write the exact archive inside the Git repository;
- support dry-run mode.

3. `scripts/verify_validation_evidence.py`

Required behavior:

- work fully offline;
- verify archive SHA-256;
- verify manifest and every member hash/size;
- reject unexpected members, duplicate paths, symlinks, and unsafe paths;
- verify schema conformance;
- verify sanitization assertions;
- detect single-byte tampering;
- produce human and machine-readable reports;
- return nonzero on any failure.

Prefer deterministic `.tar.gz` created with the Python standard library. Normalize member ordering, timestamps, uid/gid, uname/gname, and gzip timestamp while preserving exact source bytes in the private archive.

==================================================
13. CHECKPOINT 5: SANITIZATION CONTRACT
==================================================

Sanitization must be explicit, reviewable, and fail closed.

Scan at least:

- local absolute paths;
- home directory paths;
- usernames;
- unnecessary hostnames;
- cache directories;
- private archive paths;
- GitHub/Hugging Face/API tokens;
- common secret assignment forms;
- remote URLs containing credentials.

Preserve useful non-secret evidence such as:

- package versions;
- CUDA version;
- GPU model and memory;
- relative repository paths;
- commands with interpreter normalized to `python`;
- validation metrics;
- hashes and qualification verdicts.

The sanitization report must list:

- rules applied;
- replacements/redactions;
- files scanned;
- unresolved findings;
- final pass/fail status.

Do not publicize a sanitized archive when unresolved high-confidence findings remain.

==================================================
14. CHECKPOINT 6: TESTING
==================================================

Use synthetic fixtures and standard-library `unittest` unless repository convention clearly supports another existing framework.

Minimum tests:

- valid schema fixture accepted;
- missing required provenance rejected;
- explicit historical `not_recorded` accepted;
- reviewer parsing for one and multiple handles;
- clean Git repository classified correctly;
- staged-only, unstaged-only, and untracked states classified correctly;
- deterministic archive hash across repeated packaging;
- exact source bytes preserved;
- source hash mismatch rejected;
- archive tampering detected;
- symlink rejected;
- absolute/out-of-root input rejected;
- path traversal member rejected;
- duplicate/unexpected member rejected;
- sensitive fixture values removed from sanitized output;
- private exact archive retains raw bytes;
- verifier operates without network;
- failure exit codes are stable.

Add a focused CI step using only synthetic fixtures. It must not require P0-4 raw artifacts, GPU, Hub access, release credentials, or an external archive directory.

Run:

- `python -m py_compile` on new scripts and tests;
- evidence-tooling unit tests;
- schema fixture tests;
- deterministic packaging twice and compare hashes;
- deliberate tamper test;
- sanitization negative and positive tests.

==================================================
15. CHECKPOINT 7: PACKAGE THE ACCEPTED P0-4 EVIDENCE
==================================================

Only after generic tooling passes:

1. verify the original P0-4 raw summary/metrics hashes;
2. create an exact/private archive in `MULTISCREEN_EVIDENCE_ARCHIVE_DIR`;
3. create a sanitized archive in an ignored staging directory and, when configured, copy/upload it to the intended public location;
4. run the offline verifier against both archives;
5. record archive SHA-256, size, manifest SHA-256, source-file hashes, creation time, verification time, and storage class;
6. create `docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json`.

The committed descriptor must not contain private absolute paths. Use a logical storage locator, archive filename, storage class, optional public release asset identifier, and hashes.

The exact archive must not contain checkpoints by default and must never be uploaded publicly.

If `MULTISCREEN_EVIDENCE_PUBLIC_RELEASE_TAG` is set:

- verify `gh` authentication;
- publish only the sanitized archive;
- do not create or move an unrelated tag;
- do not overwrite an existing asset silently;
- record the release tag and asset identifier;
- if external publication requires a decision not already encoded by the environment variable, stop before publishing and report the exact command.

==================================================
16. CHECKPOINT 8: DOCUMENTATION AND POLICY UPDATE
==================================================

Create or update:

- docs/EVIDENCE_ARCHIVE_POLICY.md;
- docs/LOGGING_POLICY.md;
- docs/RELEASE_CHECKLIST.md;
- docs/validation_results/VALIDATION_LOG_INDEX.md;
- docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json;
- docs/HANDOFF.md;
- AGENTS.md;
- README.md;
- .gitignore as needed.

Document:

- exact/private vs sanitized/public evidence;
- reviewer and worktree provenance rules;
- historical unknown-field policy;
- archive allowlist and exclusions;
- sanitization requirements;
- deterministic packaging and verification;
- retention status values;
- recovery/verification commands;
- deletion and supersession rules;
- no-public-raw-evidence rule;
- no-checkpoint-in-default-archive rule.

Do not mark P1-preflight B, P0.5-C1/C2/C3, or P1-1 complete.

P0-4 remains complete based on its accepted run evidence; this gate adds retention/provenance metadata and does not rewrite the original metrics.

==================================================
17. CHECKPOINT 9: FINAL REGRESSION AND HYGIENE
==================================================

Run the quick P0 baseline again after tracked changes:

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

Also verify:

- all new Python files compile;
- all JSON and JSON-schema files parse;
- Markdown links resolve locally;
- evidence-tooling unit tests pass;
- actual P0-4 archive verification passes when raw evidence is available;
- `git diff --check` passes;
- no raw archive, output, checkpoint, model weight, cache, pyc, or private path is tracked;
- no model/config/oracle/cache/generation/training-harness source changed;
- working tree is clean after the final commit.

==================================================
18. CHECKPOINT 10: COMMIT AND PR
==================================================

Use a focused commit message, for example:

  Add validation evidence retention v1

Before commit:

- review every changed file;
- stage explicit paths;
- inspect staged diff;
- confirm archive files remain ignored/untracked;
- confirm no secret or private absolute path appears in the diff.

Push the focused branch and open a draft PR when authenticated.

Suggested PR title:

  Add P1-preflight A evidence retention v1

PR body must include:

- why the gate is needed;
- schema/tooling/policy changes;
- reviewer and worktree semantics;
- P0-4 source-hash verification result;
- exact/private archive retention status;
- sanitized archive status;
- tests and quick P0 regression;
- environment use and whether an isolated Conda environment was created;
- remaining limitations;
- confirmation that no model behavior or P1 capability changed.

Do not merge the PR.

==================================================
19. PROGRESS REPORTING
==================================================

At each checkpoint, report:

Checkpoint:
  - current checkpoint and status

Commands executed:
  - exact command
  - exit code

Environment:
  - active Conda environment
  - whether isolation was created
  - package changes, if any

Files changed:
  - tracked files
  - ignored/private artifacts

Tests and results:
  - pass/fail
  - counts and important outputs

Evidence:
  - source artifacts found
  - hash verification
  - exact archive status
  - sanitized archive status
  - reviewer/worktree provenance status

Current blocker or risk:
  - none / missing raw artifacts / hash mismatch / storage / sanitization / permissions

Next checkpoint:
  - next action

Do not repeatedly ask for permission for ordinary in-scope actions. Continue automatically when safe. Stop before destructive environment changes, public publication not explicitly configured, secret handling, or scope expansion into model code.

==================================================
20. TERMINAL STATE A: COMPLETE
==================================================

Use COMPLETE only when all are true:

- tooling and schema implemented;
- synthetic tests pass;
- quick P0 regression passes;
- explicit reviewer input recorded;
- current handoff clean-worktree provenance recorded;
- historical unknown provenance remains truthful;
- raw P0-4 artifacts found;
- committed hashes match raw artifacts;
- exact archive stored outside the repository in the configured retention directory;
- sanitized archive produced and verified;
- archive descriptor and policy committed;
- final repository hygiene passes;
- branch committed;
- draft PR opened or exact push/PR commands provided;
- no model behavior changed;
- no P1 capability marked validated.

==================================================
21. TERMINAL STATE B: PARTIAL/BLOCKED WITH EVIDENCE
==================================================

Use this state when generic tooling is completed as far as possible but any retention-critical requirement remains unresolved, including:

- raw P0-4 artifacts missing;
- raw hash mismatch;
- reviewer input absent;
- external archive directory absent or unwritable;
- sanitization unresolved;
- archive verification failure;
- permission/authentication blocker;
- unrelated dirty worktree that cannot safely be separated.

In this state:

- do not mark P1-preflight A complete;
- do not change P0-4 accepted status;
- preserve local staging evidence outside Git;
- commit only truthful infrastructure/documentation changes when useful;
- record exact missing files, hashes, commands, or storage action;
- provide the smallest supported next step;
- open a draft partial-infrastructure PR when appropriate, but do not claim retention complete.

==================================================
22. FINAL RESPONSE FORMAT
==================================================

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
  - command: ...
    result: ...

Provenance:
  - tested source commit:
  - implementation base commit:
  - reviewer(s):
  - original-run worktree status:
  - handoff worktree clean before/after:

Evidence retention:
  - raw source artifacts:
  - source hash verification:
  - exact/private archive:
  - sanitized archive:
  - archive verification:
  - public asset, if any:

結果:
  - terminal state: COMPLETE or PARTIAL/BLOCKED WITH EVIDENCE
  - P1-preflight A status:
  - P0-4 status:
  - branch:
  - commit:

未確認・制限:
  - ...

作成したPR:
  - ...

次にやるべきこと:
  - ...

==================================================
23. FINAL PRINCIPLE
==================================================

The goal is not to produce an archive-shaped file. The goal is to create evidence that remains truthful, reviewable, tamper-detectable, privacy-safe, and recoverable after a fresh clone.

Never fabricate missing historical provenance. Never expose exact raw evidence publicly. Never weaken retention or verification conditions merely to report COMPLETE.
```

## Expected result

The Goal should finish with one focused evidence-infrastructure PR. It should not contain model-core changes, P0 requalification, gradient-checkpointing changes, or LoRA work.

If the original P0-4 raw files or a durable external archive destination are unavailable, a truthful partial result is preferable to inventing or regenerating evidence.
