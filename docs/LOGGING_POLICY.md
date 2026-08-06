# Validation Logging Policy

This document defines what validation logs should be kept in the repository and how new validation runs should be recorded.

The repository is a research artifact. Raw terminal logs can be useful during debugging, but the repository should keep **compact, human-readable and machine-readable summaries** rather than large logs or checkpoints.

## Goals

The logging policy should make it possible to answer these questions after a fresh clone:

```text
1. Which validation gates have passed?
2. Which exact checks were run?
3. Which dtype/device combinations were covered?
4. What counts/results were observed?
5. Which artifacts/scripts produced those results?
6. What remains unvalidated?
7. Which original-run facts were recorded, and which remain unknown?
8. Who reviewed the evidence, by what method, at which commit and time?
9. Was the current handoff worktree clean, and what exact canonical porcelain bytes were hashed?
10. When applicable, what privacy-safe recursive submodule state was recorded?
11. Where are exact/private and sanitized/shareable evidence retained and verified?
```

## Files to keep

Validation summaries should live under:

```text
docs/validation_results/
```

Recommended files:

```text
P0_1_SUMMARY.md
P0_1_SUMMARY.json
P0_2_SUMMARY.md
P0_2_SUMMARY.json
P0_3_SUMMARY.md
P0_3_SUMMARY.json
P0_4_SUMMARY.md
P0_4_SUMMARY.json
VALIDATION_LOG_INDEX.md
```

Markdown files are for humans. JSON files are for future automation and scripts.

## What to record for each run

Each validation run should record at least:

```text
validation gate name
status: passed / failed / partial
command or script name
device
amp / dtype
quick or full
key counts
important metrics
known caveats
```

For training smoke tests, also record:

```text
model size
steps
seq_len
batch_size
dataset
tokenizer
initial loss
final loss
absolute loss drop
relative loss drop
save/load status
generation/cache status
qualification conditions and verdict
peak allocated/reserved CUDA memory when relevant
SHA-256 hashes of retained raw summary and metrics files
```

## What not to commit

Do not commit:

```text
outputs/
checkpoints/
*.safetensors
*.bin
*.pt
*.pth
wandb/
large raw terminal logs
cache directories
__pycache__/
*.pyc
raw evidence archives
sanitized archive staging
private absolute-path reports
```

Exact raw logs and evidence stay outside the repository in explicitly
configured private retention. They must never be attached to a public release
or issue. Only a separately sanitized archive may be published, and only after
offline verification and explicit publication configuration.

The repository keeps compact descriptors under `docs/validation_results/`.
Descriptors contain logical storage locators, filenames, sizes, SHA-256 values,
verification state, and limitations; they never contain private absolute paths.
See [EVIDENCE_ARCHIVE_POLICY.md](EVIDENCE_ARCHIVE_POLICY.md).

Checkpoints and model weights are excluded from the default evidence archive.
If they are retained, they are separate private assets with separate manifests.

## When to update logs

Update the validation summaries whenever any of these files change:

```text
multiscreen_transformers/modeling_multiscreen.py
multiscreen_transformers/configuration_multiscreen.py
oracle/paper_math_oracle.py
oracle/test_against_hf_port.py
p0_2_three_way_minimal/test_three_way_minimal.py
scripts/p0_3_tinystories_stability.py
scripts/p0_4_gpt2_context4096_smoke.py
```

Minimum rerun policy:

```text
modeling/config/oracle change:
  rerun P0-1 quick and P0-2 quick

cache/generation change:
  rerun P0-1 quick and a P0-3 quick smoke

training script change:
  rerun the corresponding P0-3 or P0-4 diagnostic/qualification level depending on the change

P0-qualified release/tag:
  rerun P0-1 CPU fp32 full, P0-2 CPU fp32 full, and at least CUDA bf16 quick/full if available
```

## Provenance required for new runs

Keep these concepts separate:

```text
original validation-run provenance
later evidence-packaging/handoff provenance
acceptance/evidence review
```

At run start and end, record:

```text
HEAD and branch
clean boolean
staged and unstaged change booleans
untracked path count
SHA-256 and byte count of exact git status --porcelain=v1 --untracked-files=all --ignore-submodules=none stdout
privacy-safe recursive git submodule status --recursive state/hash/count when applicable
UTC collection timestamp
```

Do not put raw porcelain or recursive submodule-status bytes in a shareable
record because paths may be private. Keep the exact command, byte count,
SHA-256, record counts, and aggregate submodule state instead. For historical
facts that were not recorded, use a structured
`not_recorded_in_original_run` status with null values. A commit SHA does not
prove a clean worktree.

Reviewer identity must be explicit through `--reviewer` or
`MULTISCREEN_EVIDENCE_REVIEWERS`. A recorded review requires a non-empty
explicit method, a full 40- or 64-character hexadecimal review commit, and an
explicit `raw-events-reviewed` boolean. Also record reviewer role and UTC time.
Never infer a reviewer from a GitHub login, Git configuration, repository
owner, or ambient username.

## Archive records

An allowlist-based package input records each source artifact's logical name,
classification, archive path, size, and accepted SHA-256. Packaging verifies
those hashes before producing:

```text
exact/private archive:
  unchanged source bytes
  user-controlled storage outside Git
  never public

sanitized/shareable archive:
  separate transformed bytes
  fail-closed sanitization report
  independent offline rescan
  public only when explicitly configured
```

`MANIFEST.json` lists payload/report members. `SHA256SUMS` covers every member
except itself, including the manifest. The compact descriptor records the whole
archive SHA-256 and manifest SHA-256.

Offline verification accepts exactly one canonical gzip member and enforces
canonical normalized USTAR headers, member boundaries, and zero padding. For
sanitized archives it independently rescans every member, including control
metadata. When a descriptor is supplied, verification binds it to the archive,
validation gate, tested-source commit, and complete source-artifact metadata.

Retention status is independent of the validation verdict. Missing reviewer
identity, raw files, external exact storage, or sanitization may leave evidence
retention partial while an already accepted validation gate remains passed.
