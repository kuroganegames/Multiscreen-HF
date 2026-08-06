# Validation Evidence Archive Policy

This policy defines validation-evidence provenance, packaging, verification,
retention, publication, recovery, and supersession for Multiscreen-HF.

P1-preflight A is evidence infrastructure only. It does not change a model
validation verdict, validate a P1 model capability, or turn the dense reference
implementation into efficiency evidence.

## Truth model

Every evidence descriptor keeps three records separate:

```text
original validation-run provenance
later evidence-packaging/handoff provenance
acceptance/evidence review
```

A tested-source commit does not prove that the original worktree was clean. A
reviewed summary does not identify a reviewer unless that identity was recorded
explicitly. Historical facts that were not captured use
`not_recorded_in_original_run` with null values; unknown is never encoded as
`false` and is never guessed as `true`.

Reviewer identities come only from `--reviewer` or
`MULTISCREEN_EVIDENCE_REVIEWERS`. A recorded review also requires a non-empty
explicit method, a full 40- or 64-character hexadecimal review commit, and an
explicit raw-events-reviewed boolean. Repository ownership, Git configuration,
authenticated GitHub login, and ambient usernames are not reviewer evidence.

Current handoff provenance records the base commit, branch, exact
`git status --porcelain=v1 --untracked-files=all --ignore-submodules=none`
stdout byte count and SHA-256, derived staged/unstaged/untracked state, and
collection time. When applicable, `git submodule status --recursive` is also
hashed and summarized as privacy-safe count/state data. Raw porcelain and
submodule-status bytes are not put in shareable records because paths may be
private.

## Evidence classes

### Exact/private

The exact archive preserves every allowlisted source byte unchanged. It must:

- be written to an explicitly configured user-controlled directory outside the
  Git repository;
- remain private and never be uploaded to a public release;
- exclude checkpoints, model weights, optimizer state, caches, and unrelated
  output-directory contents by default;
- be verified against the hashes accepted in the compact validation summary.

Set the destination explicitly:

```bash
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/outside/the/repository
```

The committed descriptor records a logical storage locator, filename, size,
archive SHA-256, manifest SHA-256, and storage class. It never records the
private absolute destination.

### Sanitized/shareable

The sanitized archive is a separate artifact. It is not a renamed copy of the
exact archive. Text payloads are transformed and then independently rescanned.
It may be published only when verification passes and publication is explicitly
configured.

Sanitization covers at least:

```text
local Unix, macOS, and Windows absolute paths
home, cache, and private-retention paths
usernames and unnecessary hostnames supplied as sensitive values
credential-bearing remote URLs
GitHub, Hugging Face, OpenAI, and bearer tokens
common API-key, token, password, and secret assignments
machine-local interpreter paths
```

Useful versions, CUDA/GPU facts, validation metrics, hashes, relative repository
paths, qualification verdicts, and commands normalized to `python` are retained.
The sanitization report contains rule identifiers and counts, never matched
secret values. Any unresolved high-confidence finding fails closed.

## Allowlist and exclusions

Packaging accepts an explicit JSON manifest and named source roots. Each entry
provides a logical name, source-root name, relative source path, archive path,
classification, and expected SHA-256. The packager never archives a directory
recursively.

It rejects:

- absolute, traversal, non-canonical, non-ASCII, duplicate, or out-of-root paths;
- symlinks in any selected path component, hard links, devices, FIFOs, sockets,
  and other non-regular inputs;
- checkpoint directories and common model/optimizer-weight suffixes;
- source-hash mismatches and output/source overlap;
- an exact/private destination inside the repository;
- an existing output file, so retention assets are not overwritten silently.

Checkpoint retention, if ever required, is a separate private asset with its own
manifest. It is never part of the default evidence archive.

## Deterministic format and checksum coverage

Archives use deterministic `.tar.gz` generation with sorted members, gzip
timestamp zero, empty gzip filename, and normalized tar metadata:

```text
mtime = 0
uid/gid = 0
uname/gname = empty
mode = 0644
regular files only
```

No wall-clock timestamp is stored inside the archive. Creation and verification
times live in the descriptor or external verification report.

`MANIFEST.json` lists every payload member and, for a sanitized archive, the
sanitization report. `SHA256SUMS` covers every archive member except itself,
including `MANIFEST.json`. The compact descriptor covers the whole `.tar.gz`
SHA-256. This avoids checksum self-reference while covering every byte through
an explicit layer.

Determinism is guaranteed for repeated packaging with the same inputs and
runtime. A future runtime or compression-library change must be treated as a
new packaging implementation and verified independently.

## Offline verification

The verifier performs no network access and never extracts archive members. It:

- verifies the expected whole-archive SHA-256 when supplied;
- accepts exactly one canonical gzip member and rejects non-canonical headers,
  trailing bytes, and concatenated members;
- reconstructs normalized USTAR headers and enforces canonical member offsets,
  boundaries, zero member padding, and zero terminal-record padding;
- rejects unsafe, duplicate, unexpected, missing, non-regular, or
  non-deterministically ordered members;
- validates canonical JSON, the manifest, sizes, member SHA-256 values, and
  `SHA256SUMS` coverage;
- independently rescans every sanitized archive member, including
  `MANIFEST.json`, `SHA256SUMS`, and `SANITIZATION_REPORT.json`, rather than
  trusting embedded control metadata;
- validates an optional evidence descriptor against the checked-in v1 schema
  and binds it to the archive identity, validation gate, tested-source commit,
  and complete source-artifact set and per-artifact metadata;
- returns nonzero for malformed input, integrity failure, or I/O failure.

Exit classes are stable:

```text
0  verified/success
2  invalid CLI input, manifest, schema, or descriptor
3  integrity, tamper, path-safety, or sanitization failure
4  operational I/O or runtime failure
```

## Retention states

```text
pending   work has not yet produced or checked the artifact
verified  the required artifact and verification record exist
partial   some evidence classes are verified but completion criteria are unmet
blocked   a required source, reviewer, destination, or permission is unavailable
failed    integrity or sanitization verification failed
```

P1-preflight A is complete only when explicit reviewer provenance, matching raw
P0-4 sources, external exact/private retention, a verified sanitized archive,
the committed compact descriptor, tests, regressions, and Git handoff are all
complete. P0-4 remains accepted independently when retention is partial.

## Recovery, deletion, and supersession

For a restore drill, retrieve the exact archive by its logical locator and
filename, verify its whole-archive SHA-256, run the offline verifier, and compare
its source entries with the compact accepted summary. Do not extract before the
verifier passes.

Exact evidence may be deleted only after a verified, byte-identical
exact/private replacement exists in the configured private retention class and
the descriptor is superseded through review. A sanitized archive never
qualifies as a replacement for exact evidence.
Never overwrite an archive in place. New evidence uses a new filename and a new
descriptor or an explicit `supersedes` record in a future schema version.
Sanitized public assets are immutable; publish a new asset instead of replacing
one silently.

A committed descriptor cannot contain the SHA of the commit that contains
itself. The descriptor therefore leaves the final-commit field structured as
pending until a later reviewed record can refer to a preceding commit. The
actual branch-tip SHA and clean-after-commit state are also recorded in the
post-commit handoff/PR report. Self-referential provenance must never be
fabricated.
