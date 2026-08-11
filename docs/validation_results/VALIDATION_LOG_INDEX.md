# Validation Log Index

This directory stores compact validation summaries for the P0-qualified
Multiscreen-HF baseline through P0-4 and separately labeled staged Level 1
records with explicit gate-specific acceptance states.

## Current summaries

| Gate | Human summary | Machine-readable summary | Status |
|---|---|---|---|
| P0-1 | [P0_1_SUMMARY.md](P0_1_SUMMARY.md) | [P0_1_SUMMARY.json](P0_1_SUMMARY.json) | passed |
| P0-2 | [P0_2_SUMMARY.md](P0_2_SUMMARY.md) | [P0_2_SUMMARY.json](P0_2_SUMMARY.json) | passed |
| P0-3 | [P0_3_SUMMARY.md](P0_3_SUMMARY.md) | [P0_3_SUMMARY.json](P0_3_SUMMARY.json) | passed |
| P0-4 | [P0_4_SUMMARY.md](P0_4_SUMMARY.md) | [P0_4_SUMMARY.json](P0_4_SUMMARY.json) | passed |

## Staged Level 1 records

| Gate | Plan | Compact result | Machine-readable contract | Acceptance state |
|---|---|---|---|---|
| P0.5-C1 | [P0_5_C1_PLAN.md](../P0_5_C1_PLAN.md) | [P0_5_C1_SUMMARY.md](P0_5_C1_SUMMARY.md) | [P0_5_C1_ARCHITECTURE_MANIFEST.json](P0_5_C1_ARCHITECTURE_MANIFEST.json) | accepted; PR #9 merged |
| P0.5-C2 | [P0_5_C2_PLAN.md](../P0_5_C2_PLAN.md) | [P0_5_C2_SUMMARY.md](P0_5_C2_SUMMARY.md) | focused deterministic tests | accepted; PR #10 and correction PR #11 merged |
| P1-preflight B | [P1_PREFLIGHT_B_PLAN.md](../P1_PREFLIGHT_B_PLAN.md) | [P1_PREFLIGHT_B_SUMMARY.md](P1_PREFLIGHT_B_SUMMARY.md) | exact 4.57.6/5.14.1 focused tests and CUDA smokes | accepted; PR #12 merged |
| P0.5-C3 | [plan][c3-plan] | [summary][c3-summary] | [JSON][c3-json] | accepted; PR #13 merged |
| Final Level 1 requalification | [plan](../LEVEL1_CORE_REQUALIFICATION_PLAN.md) | [summary](LEVEL1_CORE_SUMMARY.md) | [JSON](LEVEL1_CORE_SUMMARY.json) | passed locally; evidence complete; draft PR `REVIEW_REQUIRED` |

This remains an unofficial correctness-first result; the dense quadratic path
is not efficiency evidence, and it does not validate paper-scale reproduction,
retrieval benchmarks, optimized long-context efficiency, distributed training,
or any P1 model/ecosystem capability.

[c3-plan]: ../P0_5_C3_PLAN.md
[c3-summary]: P0_5_C3_SUMMARY.md
[c3-json]: P0_5_C3_SUMMARY.json

C1 was reviewed and merged as focused PR #9. C2 was merged as PR #10 and its
CUDA-autocast cache-dtype correction as PR #11. Stage 3 was reviewed and merged
as PR #12. Stage 4 passed locally and was reviewed and accepted by merged PR
#13. Final Level 1 requalification passed locally on tested source
`b224ca1a127ee18fc5fd4b00a5df639401d60679`, with reviewed and complete
evidence closure. Its focused draft PR remains `REVIEW_REQUIRED` and unmerged.
This remains an unofficial correctness-first result; the dense quadratic path
is not efficiency evidence, and it does not validate paper-scale reproduction,
retrieval benchmarks, optimized long-context efficiency, distributed training,
or any P1 model/ecosystem capability.
These staged records do not change the accepted P0 boundary or historical P0-4
and C3 retention states.

## Evidence retention

| Gate | Archive descriptor | Retention status |
|---|---|---|
| P0-4 | [P0_4_EVIDENCE_ARCHIVE.json](P0_4_EVIDENCE_ARCHIVE.json) | partial/blocked |
| P0.5-C3 | [descriptor](P0_5_C3_EVIDENCE_ARCHIVE.json) | partial; sanitized verified, review pending |
| Level 1 Core | [descriptor](LEVEL1_CORE_EVIDENCE_ARCHIVE.json) | complete; exact/private retained and verified; sanitized verified but unpublished |

The Level 1 descriptor is supported by committed
[exact/private](LEVEL1_CORE_EXACT_VERIFICATION.json) and
[sanitized](LEVEL1_CORE_SANITIZED_VERIFICATION.json) verification reports.
Neither archive is published and no public asset exists.

All four Psi=8/Psi=16 summary and metrics files matched their committed
SHA-256 values. Both completion markers were found and hashed for the new
descriptor, and the sanitized archive verified locally. Exact/private
retention remains blocked because `MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not
configured; acceptance review is pending because no explicit reviewer was
supplied; and no public asset exists.

For P0.5-C3, all 26 source artifacts are represented in a 29-member canonical
sanitized archive. Its archive, manifest, member hashes, `SHA256SUMS`, and
independent sanitization rescan verified with zero replacements or unresolved
findings. Exact/private retention is blocked because the external archive
directory was not configured. The sanitized archive is unpublished local
staging and explicit evidence review is pending. These historical descriptor
facts remain partial and are separate from the implementation/result acceptance
recorded by merged PR #13.

The historical P0-4 and C3 retention states do not reopen their accepted
implementation results. The complete Level 1 retention record does not
validate a P1 model/ecosystem capability. See the
[evidence archive policy](../EVIDENCE_ARCHIVE_POLICY.md) for storage,
verification, and recovery requirements.

## Scope

These summaries record the compact information needed for handoff and future development. They intentionally do not include large raw logs or model checkpoints.

For validation rationale and limitations, see:

```text
../VALIDATION_STATUS.md
../HANDOFF.md
../KNOWN_LIMITATIONS.md
../LOGGING_POLICY.md
```
