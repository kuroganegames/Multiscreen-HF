# Level 1 Core Requalification Plan

## Status

```text
Gate: final Level 1 Core mathematical Hugging Face requalification
Stage: 5 of 5
Implementation base: 3282eae7cb97ecfe01753460f6bce63d03e3cf88
Branch: validation/level1-core-requalification
Prior focused PRs: #9, #10, #11, #12, and #13 merged
Tested source: b224ca1a127ee18fc5fd4b00a5df639401d60679
Evidence commit: 843d5ac7825a1b0892316b70fa5c81c8de8f2d79
Acceptance review: recorded
Final validation and evidence closure: passed
Final focused PR: reviewed and merged as PR #14
Merge commit: 50af4f8e26b97f3bb0b97fc0bf6d0480a5d0fe06
```

The reviewed result is recorded in
[LEVEL1_CORE_SUMMARY.md](validation_results/LEVEL1_CORE_SUMMARY.md), and the
complete retention and verification state is recorded in
[LEVEL1_CORE_EVIDENCE_ARCHIVE.json](validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json).

This is the final evidence gate for the five-stage Level 1 Core program. It
requalifies the merged C1, C2, gradient-checkpointing, and C3 contracts on one
clean tested-source commit. It does not add or validate a P1 ecosystem
capability.

The accepted final execution required both explicit inputs below. Any future
reproduction must supply them again:

```text
MULTISCREEN_EVIDENCE_REVIEWERS
MULTISCREEN_EVIDENCE_ARCHIVE_DIR
```

The GitHub login is not reviewer evidence. For a future reproduction, if either
input is absent, static implementation and non-qualifying checks may proceed,
but a new final run, accepted evidence, completion claim, and draft evidence PR
remain blocked.

The executable matrix also requires one explicitly supplied, existing
`HF_CACHE_DIR` that is already warm for every Hub input. The cache is not
inferred from ambient Hugging Face variables and is never committed or named in
shareable reports.

## Objective

Produce one independently reviewable evidence unit for the current merged
implementation. The unit must bind exact commands, environments, raw events,
review, source artifacts, private retention, sanitization, verification, and
the tested commit without reusing historical P0-4 metrics.

Stage 5 may add evidence-only execution and review support. Model,
configuration, oracle, MiPE, cache, state-dict, data, optimizer, and training
semantics remain unchanged unless a separately demonstrated core defect forces
a scope decision.

## Tested-source boundary

Evidence-support changes are committed before the qualifying run. That clean
commit becomes the tested source. The final run records clean worktree
observations at start and end and exact environment identities for:

```text
recorded compatibility lane: Transformers 4.57.6
current compatibility lane:  Transformers 5.14.1
CUDA qualification device:   explicit cuda:0 after CUDA_VISIBLE_DEVICES selection
```

The Stage 5 run root must be newly created, mode `0700`, outside every Git
worktree. P0-3, P0-4, and C3 leaf output paths must not exist before their
commands begin. Checkpoints, tokenizer copies, caches, weights, and raw logs
remain outside Git and are excluded from the default evidence archives.

Every recorded child starts through `/usr/bin/env -i` with a fixed locale,
timezone, offline Hub flags, deterministic Python flags, `PYTHONOPTIMIZE=0`,
and an explicit empty CPU or singleton CUDA device selection. The recorder and
reviewer reject missing, extra, duplicated, or reordered environment entries,
optimized Python, disabled assertions, the wrong compatibility interpreter,
and command-tail drift from the checked matrix.

## Required execution matrix

The exact executable commands are maintained in [TESTING.md](TESTING.md). The
final matrix contains, at minimum:

```text
- formula units, oracle self-check, and oracle smoke;
- P0-1 full CPU fp32 and CUDA bf16;
- P0-2 full CPU fp32 and CUDA bf16;
- C1 architecture, initialization, packed-data, and manifest contracts;
- C2 position, MiPE, config-migration, and cache-boundary contracts;
- gradient-checkpointing focused tests under exact Transformers 4.57.6 and 5.14.1;
- C3 contract tests in both lanes, pinned data, and four fresh CUDA lanes;
- checkpointed P0-3 Psi=8 for 40 steps then Psi=16 for 25 steps;
- strict fresh P0-4 Psi=8 then reviewed Psi=16, each for at least 50 steps;
- model save/load, tokenizer reload, cached generation, and cache split checks;
- evidence tooling, syntax, JSON, workflow YAML, Markdown links, security,
  privacy, symlink, artifact, size, diff, and worktree hygiene.
- one pre-run, path-free offline-cache check proving that the same explicit
  cache serves the default GPT-2 tokenizer, default and pinned TinyStories
  inputs, and the exact pinned C3 GPT-2/SlimPajama assets. For C3, both offline
  flags select the fixed datasets 5.0.1 prepared-cache rehydration route; it
  requires a canonical absolute non-symlink layout while retaining the raw
  shard, full/selected fingerprint, and selected-row identity checks.
```

The command recorder executes argument vectors without a shell, streams and
retains complete combined output, and records UTC bounds, exit status, and log
hashes. A command with a missing, duplicate, nonzero, or malformed record fails
the final review. The reviewer also checks each complete child argument vector;
a passing command name attached to a different executable, test, revision,
tolerance, device, or strength is rejected.

## Raw-event review contract

The Stage 5 reviewer is independent of the training harness pass markers. It
rejects missing, duplicate, extra, non-finite, or out-of-order events.

### P0-3

Run with `--log-every 1`. Review all 40 Psi=8 and 25 Psi=16 training events,
checkpointing with `use_reentrant=False`, finite loss and gradient norm,
required probe-loss decrease, model save/reload, tokenizer reload, cache split,
cached greedy generation, completion marker, and absence of failure artifacts.
The run uses the full pinned TinyStories revision and writes a canonical data
contract that binds the selected ordered texts, dataset fingerprint, exact
packed token stream, packing parameters, and the complete tokenizer identity.
The contract digest is repeated in lossless stdout and both per-Psi metric
records; the reviewer requires every reference and both tokenizer reload
reports to agree.

### P0-4

Each fresh output must contain exactly the ordered run-start, preflight, 50
train-step, training-complete, save/reload, cache, generation, and run-complete
events expected by the current harness. Review actual rather than configured
values for:

```text
Psi=8 or Psi=16 as selected
vocabulary=50,257
context=4,096
CUDA bf16
microbatch=1
gradient accumulation=8
non-reentrant gradient checkpointing enabled
optimizer steps >= 50
finite loss and gradient events
probe-loss decrease
qualified completion marker
no diagnostic or failure marker
```

The separately reloaded checkpoint tokenizer must match the source tokenizer's
complete vocabulary and added-vocabulary mappings, special-token contract, and
deterministic probe encodings. Psi=16 starts only after the complete Psi=8 raw
review passes.

Each qualifying run also writes a canonical `data_contract.json` before
training. It binds the resolved default TinyStories source and fingerprint,
ordered selected-text manifest, exact packed `uint32` token stream and packing
parameters, and the normalized GPT-2 tokenizer identity projection. The
contract digest must occur exactly once in lossless stdout and agree with the
run-start, preflight, summary, and run-complete records. The focused and full
reviewers require the Psi=8 and Psi=16 contracts to be byte-identical and bind
their tokenizer projection to the offline cache preflight and independently
reloaded checkpoint tokenizers.

### C3

Review the four fresh lanes in order: Psi=8 operational, Psi=8 peak exposure,
Psi=16 operational, and Psi=16 peak exposure. Check every metric event,
expected learning-rate schedule, finite loss/gradient/parameter values, actual
updates, clipping disabled, completion markers, and absence of failures.

## Evidence inventory and retention

The explicit package allowlist includes only the fresh Stage 5 records:

```text
provenance and environment records
single-cache offline preflight record
canonical command ledger and command logs
formula/oracle/P0/C1/C2/checkpointing/C3 regression logs
C3 contract/data/CUDA compact outputs and markers
P0-3 compact results, per-step stdout evidence, review, and marker
P0-4 Psi=8/Psi=16 data contracts, summaries, JSONL metrics, reviews, and markers
four tokenizer-reload reports
raw-event review report
Level 1 human and machine summaries
```

Checkpoint directories, model weights, tokenizer files, dataset rows, Hugging
Face caches, and unrelated historical outputs are excluded.

After an explicitly named reviewer has genuinely reviewed every raw event, the
provenance collector records a nonempty review method, the full review commit,
and `raw-events-reviewed=true`. Then:

1. create the exact archive directly in the configured durable private
   directory;
2. create a separate sanitized archive;
3. verify both archives offline at their final locations;
4. bind the descriptor to the gate, tested commit, and complete artifact set;
5. publish neither archive by default, and never publish the exact archive.

Historical P0-4 and C3 descriptors remain unchanged. Their partial retention
cannot satisfy this gate and is not rewritten retroactively.

## Descriptor closure

The evidence schema requires recorded clean pre-edit and post-commit handoff
observations, a recorded evidence commit, verified archives, verified
sanitization, verified retention, and recorded acceptance review before
`evidence_status` can be `complete`.

Because a descriptor cannot contain its own commit ID, closure uses two
evidence commits:

```text
prepare:   validate the fixed machine-review artifact inventory and explicit
           acceptance provenance; write deterministic Level 1 JSON/Markdown
           summaries and an exact package-input allowlist
seal:      package and verify exact/private and sanitized archives at their
           final external locations; write a schema-valid partial descriptor
           and byte-stable descriptor-aware verification reports
commit A: add reviewed summaries and descriptor with self-reference fields pending
observe:  clean worktree at commit A
close:     require commit-A clean provenance, record commit A as the evidence
           commit, reverify unchanged archives/reports, and set evidence complete
commit B: record commit A and its clean observation; set evidence complete
handoff:  report clean commit-B branch tip externally in the PR
```

The package input is a fixed allowlist rather than a directory walk. It excludes
the descriptor, verification reports, archives, checkpoints, weights,
tokenizer copies, caches, and every unreviewed path. Exclusive owner-only writes
and stable report projections avoid archive/descriptor/report self-reference.

## Acceptance boundary

Stage 5 is locally ready for final draft-PR review only when every required
command and raw-event review passes, both archives verify, retention and
reviewer fields are complete, canonical documents are updated from that
evidence, the branch tip is clean, and no forbidden artifact or private path is
tracked.

The reviewed record now states:

```text
Level 1 — Core mathematical Hugging Face implementation: complete
```

The same statement must immediately exclude paper-scale reproduction,
retrieval benchmarks, optimized long-context efficiency, distributed training,
and every P1 model/ecosystem capability. The final PR remains draft and the
stage stops at `REVIEW_REQUIRED`. No immutable tag is created before merge or
without explicit user instruction.

All local readiness conditions above passed for tested source
`b224ca1a127ee18fc5fd4b00a5df639401d60679`. The descriptor is complete, and
the focused result was reviewed and accepted as merged PR #14.
