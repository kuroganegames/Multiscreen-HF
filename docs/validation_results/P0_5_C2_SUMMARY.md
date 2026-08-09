# P0.5-C2 Long-Position / MiPE / Cache Summary

## Verdict

```text
Local gate result: passed
Acceptance state: accepted; focused PR #10 merged
Post-merge correction: CUDA-autocast cache dtype; PR #11 merged
```

Initial focused C2 implementation and local validation passed on 2026-08-07.
Independent adversarial review then identified and corrected strict cache,
position, autocast, and evidence gaps; final post-review validation passed on
2026-08-08, and PR #10 was reviewed and merged. This record does not validate
P1-preflight B, C3, final Level 1 requalification, dense 131K execution,
retrieval quality, efficiency, or any P1 ecosystem capability.

## Provenance

```text
C1 prerequisite: merged as PR #9
C1 merge commit / C2 base: ec805c1ba60c55ea4beb3ad68e0a00c0d718e909
branch: validation/p0-5-c2-mipe-position-cache
base relation at branch creation: main == origin/main
base worktree: clean
base porcelain bytes: 0
base porcelain SHA-256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
base provenance timestamp: 2026-08-07T14:09:08Z
source audit date: 2026-08-07
final post-review validation date: 2026-08-08
```

The source hierarchy and semantic decision are recorded in
[ADR-0001](../adr/ADR-0001-mipe-position-semantics.md). Paper v3 is the
authority for literal paper behavior; the pinned unofficial dieOD reference is
the compatibility authority for the historical modulo rule. No public
author/official model implementation was identified during the bounded dated
audit, so unpublished implementation behavior remains unknown.

## Post-merge CUDA-autocast correction

PR #11 was reviewed and merged on 2026-08-09. It corrected one cache-validation
prediction exposed by CUDA bf16 autocast: projection output was bf16, but CUDA
normalization returned fp32 cache tensors. Validation now derives the expected
cache dtype from a zero-element normalization probe on the same device. This
does not cast or alter cache values and does not change the C2 position modes.

The focused correction reran the C2 suite on CPU and CUDA bf16, all repository
unit tests, formula/oracle checks, full P0-1/P0-2 CPU fp32 and CUDA bf16, and
the evidence-tooling suite under Transformers 4.57.6; the CUDA focused suite
also passed under Transformers 5.14.1. Exact commands and counts are recorded
in PR #11. Raw logs remained local and no acceptance reviewer is inferred for
the separate historical P0-4 evidence handoff.

## Semantic result

The HF configuration now serializes exactly two canonical modes:

```text
paper_absolute
reference_mod_after_wrap_boundary
```

`paper_absolute` applies the paper's unwrapped absolute position. The
reference mode wraps per-head positions at and after the separately serialized
positive `mipe_reference_wrap_boundary`. Softmask remains based on absolute
query/key distance in both modes.

Missing C2 fields resolve to the legacy-compatible reference mode and a
boundary equal to the existing `max_position_embeddings`. The four checked-in
P0-3/P0-4 Multiscreen configs remain byte-unmodified legacy fixtures and load
with their historical boundary. Re-serialization emits the resolved canonical
fields. Paper mode requires explicit opt-in; no weights are converted.

The direct paper oracle remains paper-oriented by default, accepts canonical
mode names plus its pre-C2 aliases, and now maps HF configs without silently
changing legacy long-position behavior. Stable auxiliary math is an fp32 floor
that preserves float64; reference auxiliary math continues to follow the
incoming dtype for vendored compatibility.

## Position and cache result

Focused tests cover positions:

```text
0, 1, 255, 256, 257, 4095, 4096, 4097, 8191, 8192, 131071
```

They exercise active, near-threshold, threshold, inactive, and fractional
windows. Paper mode matches the equation oracle; reference mode matches both
the reference-compatible oracle and the pinned vendored implementation.

Cache validation now rejects malformed layer counts, pair arity, ranks,
batch/head/dimension shapes, K/V lengths, cross-layer lengths, devices, and
dtypes before screening math. Position IDs and cache positions must be integer,
nonnegative, batch-shared contiguous ranges consistent with the complete
zero-based prefix cache. Fixed-width integer vectors are compared in int64 with
an overflow guard, so narrow integer wraparound cannot masquerade as a valid
range. Conflicting scalar/vector positions and inconsistent generation suffix
lengths fail explicitly. All-empty preallocated DynamicCache forms are accepted;
partially populated layers are rejected. Populated Transformers 4.x and 5.x
DynamicCache forms are normalized to the validated legacy representation.
Autocast cache validation follows runtime post-normalization dtype while preserving
float64. Unsupported cache layouts remain out of scope.

Independent one-layer oracle full-context-suffix rows, HF one-shot cached
suffixes, and HF cached chunks agree for both modes using allocation-safe,
actual-token-derived complete prefix K/V at:

```text
4080 + 32
4096 + 16
8192 + 16
4094 + four one-token steps
4089 + uneven 3/7/2/11 chunks
```

For each long scenario, the independent equation oracle derives prefix K/V from
the actual prefix token IDs and evaluates the entire suffix. The construction is
also proven equal to a real tiny dense prefill cache. A tiny real model passes
full, cached, oracle, uneven-chunk, and greedy generation checks across a reduced
wrap boundary. Long scenarios allocate only suffix-by-total screening matrices;
they do not run dense 8K or 131K prefill.

## Local environment

```text
Conda environment: base (unchanged)
Python: 3.12.10
PyTorch: 2.8.0+cu128
Transformers: 4.55.0
NumPy: 2.3.2
tokenizers: 0.21.4
safetensors: 0.6.2
CUDA: 12.8; bf16 available
GPU 0: NVIDIA RTX PRO 6000 Blackwell Max-Q, 97,887 MiB
GPU 1: NVIDIA GeForce RTX 5090, 32,607 MiB
package installation or upgrade: none
```

The active Transformers version is below the repository-declared minimum.
The post-review 24-test suite also passed offline in these existing isolated
environments:

```text
Python 3.12.11 / PyTorch 2.7.1+cu128 / Transformers 4.57.6
Python 3.12.10 / PyTorch 2.8.0+cu128 / Transformers 5.9.0
Python 3.12.10 / PyTorch 2.8.0+cu128 / Transformers 5.14.0
```

The 5.9.0 and 5.14.0 lanes each emitted the non-failing Transformers warning
that `use_return_dict` is deprecated in favor of `return_dict`.
At C2 review, the exact CI lower-bound lane (Python 3.10, Torch 2.4.0 CPU,
Transformers 4.57.0) was a draft-PR check rather than a locally reproduced
result. Stage 3 later superseded the active floor with non-yanked 4.57.6; this
historical C2 environment record is not rewritten as if it ran that CI lane.

## Tests recorded locally

Pre-change baseline:

```text
formula units, oracle self-check, and oracle smoke: passed
HF-port CPU fp32 quick: passed
P0-2 three-way CPU fp32 quick: passed
P1-preflight A evidence tooling: 58 passed
tracked Python compile: passed after correcting one command that named two
  nonexistent evidence-test files; no source failure was involved
```

Focused and repository suites:

```text
C2 focused position/config/cache suite: 24 passed
  Transformers 4.55.0 / 4.57.6 / 5.9.0 / 5.14.0: 24 passed each
all tests/test_*.py: 95 passed
C1 architecture/initialization/packed-text subset: 13 passed
P1-preflight A evidence subset: 58 passed
formula units, oracle self-check, and oracle smoke: passed
P0-4 Psi=8 and Psi=16 static config preflight: passed
all repository Python compile: passed
workflow YAML parse: passed
changed/new Markdown links: passed
git diff --check: passed
commit-scope artifact/privacy/size/symlink hygiene: passed
```

Strong model-core regressions:

All four strong runs below were independently rerun on the final post-review
tree on 2026-08-08, after the cache/position/autocast hardening fixes.

```text
P0-1 CPU fp32 full: passed
P0-1 CUDA bf16 full: passed
  cache_split 144; padding_cache 240; padding_full 88;
  position negative 4; position_ids_zero 2; shape forward/loss 60;
  logits_to_keep 144; shifted loss 60; zero relevance 2

P0-2 CPU fp32 full: passed
P0-2 CUDA bf16 full: passed
  prefill_three_way 45; cache_split_three_way 237
```

P0-1 CUDA bf16 uses stable fp32 auxiliary MiPE/Softmask math. P0-2 CUDA
bf16 deliberately uses incoming-dtype reference arithmetic to match the
vendored implementation. The latter is compatibility evidence, not proof of
long-position causal correctness.

## Evidence and limitations

This compact summary and the deterministic tests are the C2 evidence. No C2 raw
logs, archives, checkpoints, weights, private paths, or secrets are retained in Git or this compact record.
The position 131071 check is direct scalar MiPE math, not a dense forward. The
implementation remains dense and quadratic, and this gate provides no evidence
for paper-scale memory, runtime, optimized window skipping, retrieval quality,
or model quality.

Historical P0-4 evidence retention remains partial/blocked exactly as
recorded. Its accepted metrics, configs, descriptor, and raw-retention status
were not rewritten. C2 acceptance was completed by merged PR #10, and the
post-merge dtype correction was separately reviewed and merged as PR #11.
