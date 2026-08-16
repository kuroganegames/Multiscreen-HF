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
| Final Level 1 requalification | [plan](../LEVEL1_CORE_REQUALIFICATION_PLAN.md) | [summary](LEVEL1_CORE_SUMMARY.md) | [JSON](LEVEL1_CORE_SUMMARY.json) | accepted; evidence complete; PR #14 merged |
| HF contract hardening Stage E | [plan](../HF_CONTRACT_HARDENING_PLAN.md) | [summary](HF_CONTRACT_HARDENING_SUMMARY.md) | [JSON](HF_CONTRACT_HARDENING_SUMMARY.json) | evidence complete; draft PR review/merge pending |

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
evidence closure. Its focused result was reviewed and accepted as merged PR
#14 (merge commit `50af4f8e26b97f3bb0b97fc0bf6d0480a5d0fe06`).

HF contract hardening Stage E passed on tested source
`0d59083ddbd78619ca29bf9af730999834272a1a`, based on implementation baseline
`bf8cc34cb6aa16ffeec1f609166db5efae79e9df`. Its 53 commands and two
environment records passed, with 117 focused tests in each exact Transformers
lane, full P0-1/P0-2 CPU fp32 and CUDA bf16, fresh checkpointed P0-3 Psi=8/16,
and fresh strict P0-4 Psi=8/16. Codex reviewed all 53 lossless logs and 179 raw
events. Evidence commit `4fd704f805ea634c66d2c4c26dded425c819a51d`
records the compact result. The draft PR has not yet been created, reviewed, or
merged.

These staged records do not change the accepted P0 boundary. The historical
P0-4 and original C3 descriptors preserve their recorded states; the later C3
retention closure is a separate post-acceptance record.

## Evidence retention

| Gate | Archive descriptor | Retention status |
|---|---|---|
| P0-4 | [P0_4_EVIDENCE_ARCHIVE.json](P0_4_EVIDENCE_ARCHIVE.json) | partial/blocked |
| P0.5-C3 | [historical descriptor](P0_5_C3_EVIDENCE_ARCHIVE.json) / [closure](P0_5_C3_EVIDENCE_CLOSURE.json) | complete; exact/private retained and verified; review recorded; sanitized verified but unpublished |
| Level 1 Core | [descriptor](LEVEL1_CORE_EVIDENCE_ARCHIVE.json) | complete; exact/private retained and verified; sanitized verified but unpublished |
| HF contract hardening Stage E | [descriptor](HF_CONTRACT_HARDENING_EVIDENCE_ARCHIVE.json) | complete; exact/private and sanitized staging archives retained and verified offline; both unpublished |

The Level 1 descriptor is supported by committed
[exact/private](LEVEL1_CORE_EXACT_VERIFICATION.json) and
[sanitized](LEVEL1_CORE_SANITIZED_VERIFICATION.json) verification reports.
Neither archive is published and no public asset exists.

The Stage E descriptor is supported by committed
[exact/private](HF_CONTRACT_HARDENING_EXACT_VERIFICATION.json) and
[sanitized](HF_CONTRACT_HARDENING_SANITIZED_VERIFICATION.json) verification
reports. Its explicit Codex review covered all 53 lossless logs and 179 raw
events. Both archives are retained and verified offline but unpublished, and
no public asset exists. This complete evidence state does not imply draft PR
review, merge, or a tag.

All four Psi=8/Psi=16 summary and metrics files matched their committed
SHA-256 values. Both completion markers were found and hashed for the new
descriptor, and the sanitized archive verified locally. Exact/private
retention remains blocked because `MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not
configured; acceptance review is pending because no explicit reviewer was
supplied; and no public asset exists.

For P0.5-C3, the historical descriptor preserves the packaging-time partial
state. A later [closure](P0_5_C3_EVIDENCE_CLOSURE.json) retains
and verifies the 28-member exact/private archive and reverifies the unchanged
29-member canonical sanitized archive. All 26 source artifacts, complete
archive hashes, manifests, member hashes, `SHA256SUMS`, canonical framing, and
the independent sanitization rescan verified. The closure is supported by
committed [exact/private](P0_5_C3_EXACT_VERIFICATION.json) and
[sanitized](P0_5_C3_SANITIZED_VERIFICATION.json) verification reports.
Both archives are unpublished, no public asset exists, and explicit evidence
review by Codex covered all 26 source artifacts and all 8 optimizer-step raw
events. Acceptance review is recorded and overall evidence status is complete.
This is separate from the implementation/result acceptance recorded by merged
PR #13.

P0-4's historical partial retention and C3's later retention closure do not
reopen their accepted implementation results. Neither the complete Level 1
retention record nor the Stage E evidence validates paper-scale training,
retrieval, optimized long-context efficiency, distributed training, broad
generation compatibility, or a P1 model/ecosystem capability. See the
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
