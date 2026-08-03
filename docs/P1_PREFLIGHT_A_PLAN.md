# P1-preflight A Plan: Validation Provenance and Evidence Retention v1

## Status

```text
Gate: P1-preflight A
State: selected and designed; implementation pending
Model capability validated by this gate: none
```

This gate improves the trustworthiness, reproducibility, and retention of validation evidence before any P1 ecosystem capability is attempted. It does not modify Multiscreen mathematics, model behavior, training behavior, or the accepted P0-1 through P0-4 verdicts.

The Codex execution handoff and ready-to-paste Goal are in [CODEX_P1_PREFLIGHT_A_HANDOFF.md](CODEX_P1_PREFLIGHT_A_HANDOFF.md).

## Why this gate comes next

P0-4 has accepted compact Markdown and JSON summaries plus SHA-256 hashes of the original `summary.json` and `metrics.jsonl` files. The original run record did not formally capture every provenance field required for future gates, and the repository policy does not yet define a complete long-term raw-evidence archive format.

P1-preflight A addresses three review follow-ups:

```text
1. record reviewer identity and review method;
2. record clean/dirty worktree state without retroactive claims;
3. retain exact raw evidence outside the Git repository and produce a verified sanitized archive.
```

## Non-goals

Do not use this gate to begin or validate:

```text
- gradient-checkpointing modernization (P1-preflight B)
- architecture/initialization/all-scale validation (P0.5-C1)
- long-position/MiPE semantics work (P0.5-C2)
- paper-training-contract smoke (P0.5-C3)
- PEFT/LoRA, QLoRA, Unsloth, generation matrix, compile, or serving
- new P0-4 training runs
- paper-scale or efficiency reproduction
```

## Development-environment contract

The current Python development environment is managed with Conda. `uv` is installed and may be used as an installation helper, but it is not permission to replace the environment-management strategy or rewrite unrelated dependency state.

The implementation must obey these rules:

```text
- do not break, delete, or broadly upgrade the active Conda environment;
- do not modify the Conda base environment or install packages globally;
- do not run `conda update --all`, unconstrained `pip install -U`, or equivalent broad upgrades;
- inspect the active environment before installing anything;
- prefer the current environment when it already satisfies the task;
- when isolation is useful, create a separate Conda environment or other isolated virtual environment;
- using `uv` is allowed only against an explicit target environment and for scoped installs;
- record environment changes and before/after versions;
- prefer Python standard-library implementations so that evidence verification remains portable.
```

Creating an isolated virtual environment is explicitly in scope for this gate.

## Intended implementation files

The exact layout may change if repository inspection justifies it, but the expected result is:

```text
scripts/collect_validation_provenance.py
scripts/package_validation_evidence.py
scripts/verify_validation_evidence.py
schemas/validation_evidence_v1.schema.json
tests/test_validation_evidence.py
docs/EVIDENCE_ARCHIVE_POLICY.md
docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json
```

Likely supporting updates:

```text
.gitignore
.github/workflows/p0-smoke.yml
AGENTS.md
README.md
docs/HANDOFF.md
docs/LOGGING_POLICY.md
docs/RELEASE_CHECKLIST.md
docs/validation_results/VALIDATION_LOG_INDEX.md
```

No model, oracle, cache, generation, position, state-dict, tokenizer, dataset, or training-harness source should change in this gate.

## Evidence model

### Exact private archive

The exact archive preserves original bytes and is never committed to the Git repository or uploaded to a public release.

Expected source artifacts for each accepted run include, when available:

```text
summary.json
metrics.jsonl
completion marker
failure or diagnostic marker when applicable
environment/provenance record
exact command record
```

Full checkpoints and model weights are excluded by default. If retained separately, record only their manifest and hashes in the evidence descriptor.

### Sanitized archive

The sanitized archive is suitable for sharing after an explicit scan. It may contain:

```text
sanitized summary
sanitized metrics
sanitized completion note
sanitization report
manifest
SHA256SUMS
verification report
```

It must not contain:

```text
secrets or tokens
usernames or unnecessary hostnames
local absolute paths
cache paths
private storage paths
checkpoints or model weights
symlinks or files outside the allowlist
```

### Determinism and integrity

Prefer a deterministic standard-library archive format such as normalized `.tar.gz`:

```text
- paths sorted;
- fixed archive timestamps and ownership metadata;
- source bytes unchanged in the exact archive;
- every file size and SHA-256 recorded;
- archive SHA-256 and manifest SHA-256 recorded;
- verifier detects tampering, unexpected files, symlinks, and path traversal.
```

## Provenance model

Do not conflate the historical validation run with the later evidence-handoff operation.

### Run provenance

For future runs, record at start and end:

```text
HEAD SHA
branch
worktree clean boolean
staged changes present
unstaged changes present
untracked path count
SHA-256 of exact `git status --porcelain=v1` bytes
submodule state when applicable
timestamp UTC
```

For the already-completed P0-4 run, fields not recorded at execution time must remain explicit `null`/`not_recorded_in_original_run`. Never infer a clean worktree retroactively from a commit SHA.

### Evidence-handoff provenance

For this gate, record separately:

```text
P1-preflight A starting commit
working branch
clean state before tracked edits
clean state after the final commit
archive creation time
archive verification time
review commit
```

## Reviewer model

The evidence descriptor must distinguish:

```text
original-run reviewer
acceptance/evidence reviewer
review method
review timestamp
review commit
whether raw events were reviewed
```

A GitHub login or repository owner must not be silently treated as a reviewer. Reviewer identity should come from an explicit CLI option or environment variable, such as:

```bash
export MULTISCREEN_EVIDENCE_REVIEWERS=kuroganegames
```

If the original P0-4 run reviewer was not recorded, preserve that fact. A later acceptance reviewer may still be recorded truthfully.

## Storage contract

The exact archive must be written outside the repository to a user-controlled retention location. The recommended configuration is:

```bash
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/outside/the/repository
```

For a fresh clone that does not contain the ignored P0-4 outputs, point the tooling at the original raw artifacts:

```bash
export MULTISCREEN_P0_4_RAW_ROOT=/absolute/path/to/the/original/P0-4/outputs
```

Optional public publication of the sanitized archive must be opt-in. For example:

```bash
export MULTISCREEN_EVIDENCE_PUBLIC_RELEASE_TAG=p0-4-qualified-v0
```

The exact raw archive must never be uploaded to a public GitHub release. A committed descriptor may record a logical storage class, archive filename, size, hashes, and public asset identifier, but must not expose a private absolute path.

If no external retention directory is configured, the tooling may create a staging archive under an ignored directory, but the gate remains partial rather than claiming long-term retention complete.

## P0-4 backfill rules

The accepted P0-4 metrics and verdict must not be altered. The tooling should verify the original files against the hashes already recorded in:

```text
docs/validation_results/P0_4_SUMMARY.json
```

Expected backfill behavior:

```text
- preserve tested source commit and run metrics;
- verify Psi=8 and Psi=16 raw summary/metrics hashes;
- record the later evidence reviewer separately;
- mark original run worktree cleanliness as not recorded unless primary evidence proves it;
- package exact private evidence and sanitized evidence;
- commit only the compact archive descriptor and policy changes.
```

If the raw files are unavailable or do not match the committed hashes, do not regenerate substitute files and do not claim retention complete.

## Required tests

Use standard-library `unittest` unless repository evidence supports another existing test convention.

Minimum coverage:

```text
- schema accepts a valid fixture;
- schema rejects missing required provenance and artifact fields;
- deterministic packaging produces identical hashes from identical inputs;
- source hash mismatch is rejected;
- archive tampering is detected;
- symlinks are rejected;
- path traversal is rejected;
- unexpected files are rejected;
- exact archive preserves source bytes;
- sanitized archive removes configured sensitive values;
- sanitized archive contains no absolute private paths from the fixture;
- historical `not_recorded` provenance is represented without fabricating booleans;
- reviewer parsing supports one or multiple explicit GitHub handles;
- clean and dirty Git fixtures are classified correctly;
- verifier works offline.
```

CI must use synthetic fixtures only. Do not require private P0-4 raw artifacts or network access in CI.

## Baseline regression

Because this gate should be tooling/documentation only, run the normal quick baseline:

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

Also run syntax, JSON-schema, evidence-tooling unit, tamper, sanitization, and archive-verification tests.

A model-core diff is a scope violation and requires stopping rather than expanding requalification inside this gate.

## Completion criteria

P1-preflight A is complete only when:

```text
- evidence schema v1 is implemented and documented;
- provenance collection is implemented and tested;
- exact and sanitized packaging are implemented and tested;
- offline verification and tamper detection are implemented and tested;
- reviewer and worktree fields are represented truthfully;
- P0-4 raw artifacts are found and match the committed hashes;
- an exact archive is stored in the configured external retention location;
- a sanitized archive is produced and verified;
- a compact P0-4 archive descriptor is committed;
- policies, handoff, index, and release checklist are updated;
- quick P0 regression and repository hygiene checks pass;
- the focused branch is committed and a draft PR is opened or exact push/PR commands are provided;
- no P1 model capability is marked validated.
```

## Partial or blocked result

Use a partial/blocked result when tooling is implemented but any of these remain unresolved:

```text
- original raw P0-4 files are unavailable;
- raw hashes do not match the accepted summary;
- reviewer identity was not explicitly provided;
- no user-controlled external archive directory is available;
- sanitized evidence still contains sensitive data;
- archive verification fails;
- permissions prevent the intended storage or PR action.
```

Preserve the accepted P0-4 validation status. Report the exact missing evidence or storage action and the smallest supported next step.

## Next stages

After this gate is reviewed and merged, the planned core-completion sequence remains:

```text
P0.5-C1  architecture / initialization / all-scale contract
P0.5-C2  long-position / MiPE / cache semantics
P1-preflight B  gradient-checkpointing API modernization
P0.5-C3  paper-training-contract smoke
final P0 core requalification
P1-1  PEFT/LoRA smoke
```
