# Validation Log Index

This directory stores compact validation summaries for the P0-qualified Multiscreen-HF baseline through P0-4.

## Current summaries

| Gate | Human summary | Machine-readable summary | Status |
|---|---|---|---|
| P0-1 | [P0_1_SUMMARY.md](P0_1_SUMMARY.md) | [P0_1_SUMMARY.json](P0_1_SUMMARY.json) | passed |
| P0-2 | [P0_2_SUMMARY.md](P0_2_SUMMARY.md) | [P0_2_SUMMARY.json](P0_2_SUMMARY.json) | passed |
| P0-3 | [P0_3_SUMMARY.md](P0_3_SUMMARY.md) | [P0_3_SUMMARY.json](P0_3_SUMMARY.json) | passed |
| P0-4 | [P0_4_SUMMARY.md](P0_4_SUMMARY.md) | [P0_4_SUMMARY.json](P0_4_SUMMARY.json) | passed |

## Evidence retention

| Gate | Archive descriptor | Retention status |
|---|---|---|
| P0-4 | [P0_4_EVIDENCE_ARCHIVE.json](P0_4_EVIDENCE_ARCHIVE.json) | partial/blocked |

All four Psi=8/Psi=16 summary and metrics files matched their committed
SHA-256 values. Both completion markers were found and hashed for the new
descriptor, and the sanitized archive verified locally. Exact/private
retention remains blocked because `MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not
configured; acceptance review is pending because no explicit reviewer was
supplied; and no public asset exists.

This retention status does not reopen the accepted P0-4 result and does not
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
