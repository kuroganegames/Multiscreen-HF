# Hugging Face Contract Hardening — Stage E Requalification Plan

## Status

```text
state: PLANNED — NOT RUN
implementation baseline: bf8cc34cb6aa16ffeec1f609166db5efae79e9df
baseline meaning: merge commit containing reviewed Stages A-D (PR #21)
working branch: validation/hf-contract-hardening-requalification
tested-source commit: not selected
review commit: not recorded
evidence commit: not created
closure tip: not created
```

Stage E execution has not started, no fresh training result exists, and no new
validation or retention claim is made by this plan. A clean tested-source
commit will be selected only after the Stage E evidence-support implementation
and its tests are committed.

At planning time on 2026-08-16, both exact Transformers interpreters and CUDA
devices were discoverable. The following required execution inputs were not
configured:

```text
MULTISCREEN_EVIDENCE_REVIEWERS
MULTISCREEN_EVIDENCE_ARCHIVE_DIR
HF_CACHE_DIR
```

These observations are readiness notes, not acceptance evidence. All
preconditions must be rechecked from the eventual clean tested-source commit.

## Authority and objective

This plan implements Stage E from
[CODEX_HF_CONTRACT_HARDENING_SPEC.md](CODEX_HF_CONTRACT_HARDENING_SPEC.md).
It also follows the retention and review contracts in
[EVIDENCE_ARCHIVE_POLICY.md](EVIDENCE_ARCHIVE_POLICY.md),
[LOGGING_POLICY.md](LOGGING_POLICY.md), and the proven closure pattern in
[LEVEL1_CORE_REQUALIFICATION_PLAN.md](LEVEL1_CORE_REQUALIFICATION_PLAN.md).

The objective is to requalify the integrated post-Level-1 implementation after
Stages A-D and create fresh, reviewed, retained evidence for that exact source
commit. Stage E must demonstrate all seven hardening resolutions together:

1. the output head is a callable hidden-to-vocabulary projection;
2. normalized tied output-head storage remains parameter-free;
3. deep copies have isolated owner, mutation, gradient, and lifecycle state;
4. gradient-checkpointed training with past state fails explicitly;
5. a zero-valid-target loss is a finite graph-connected zero;
6. cached-generation suffix ambiguity never silently drops tokens; and
7. packed-text construction fails fast when no EOS identity can be resolved.

It must also prove that the hardened P0-4 qualification predicate requires
microbatch size one and supported non-reentrant gradient checkpointing in
addition to the historical conditions.

## Non-goals and claim boundary

Stage E does not validate paper-scale training, retrieval quality, optimized
long-context efficiency, distributed training, PEFT/LoRA/QLoRA, compilation,
broad generation compatibility, or serving. The implementation remains an
unofficial correctness-first research artifact with a dense quadratic
screening path.

Stage E will not:

- alter model mathematics, oracle semantics, accepted MiPE/cache semantics, or
  accepted tolerances;
- reuse historical P0-4 or C3 metrics as evidence for the new tested source;
- rewrite or reinterpret any accepted Level 1, P0-4, or P0.5-C3 evidence;
- publish the exact archive;
- publish the sanitized archive without a separate explicit instruction;
- merge its final draft PR or create/move a tag automatically.

## Identity boundaries

The evidence must distinguish these identities without inference:

| Identity | Required value |
| --- | --- |
| Historical Level 1 tested commit | `b224ca1a127ee18fc5fd4b00a5df639401d60679` |
| Stage E implementation baseline | `bf8cc34cb6aa16ffeec1f609166db5efae79e9df` |
| Stage E tested source | Future clean evidence-support/plan commit `T` |
| Review commit | Exactly `T` |
| Evidence commit | Future commit `A` |
| Closure tip | Future commit `B` |

The live Stage E reviewer and builder must verify that every tracked path and
blob under these accepted historical record sets is identical between the
implementation baseline and `T`:

```text
docs/validation_results/LEVEL1_CORE_*
docs/validation_results/P0_4_*
docs/validation_results/P0_5_C3_*
```

Only a path-free aggregate of that comparison belongs in the new compact
review. Historical files are not sources for the Stage E archive.

## Evidence-support design

### Reuse unchanged

Stage E will reuse these existing generic or lossless components without
changing their legacy contracts:

```text
scripts/run_level1_requalification_command.py
scripts/report_level1_environment.py
scripts/check_level1_repository.py
scripts/collect_validation_provenance.py
scripts/package_validation_evidence.py
scripts/verify_validation_evidence.py
scripts/validation_evidence_common.py
schemas/validation_evidence_v1.schema.json
scripts/check_tokenizer_reload.py
```

The recorder's existing format name remains truthful provenance; Stage E will
not relabel the bytes as a new recorder format. It will continue to create a
mode-0700 private run root, reserve each command name once, stream complete
merged output, and bind every command record to the clean tested commit.

### Add a Stage E-specific profile

The existing Level 1 reviewer and builder are deliberately fixed to the old
Level 1 gate, 46-command matrix, C3 lanes, artifact paths, and historical P0-4
qualification schema. They must retain that behavior. Stage E therefore adds
dedicated versioned entry points rather than changing their defaults:

```text
scripts/review_hf_contract_hardening.py
scripts/build_hf_contract_hardening_evidence.py
scripts/check_hf_contract_hardening_offline_cache.py
tests/test_hf_contract_hardening_evidence_review.py
tests/test_hf_contract_hardening_evidence_builder.py
tests/test_hf_contract_hardening_offline_cache.py
```

The new offline-cache preflight must require only the fixed P0-3 and P0-4
inputs needed by Stage E. It must not create a false dependency on the unused
C3 prepared cache. Every legacy Level 1 test remains in the matrix to prove
that adding the Stage E path did not weaken the accepted path.

The new tools use explicit, versioned contracts:

```text
review schema: multiscreen-hf-contract-hardening-raw-evidence-review-v1
summary schema: multiscreen-hf-contract-hardening-summary-v1
validation gate: HF Contract Hardening
evidence gate: Stage E final requalification
archive root: artifacts/hf-contract-hardening/
```

The evidence-support implementation and its focused adversarial fixtures must
be committed before `T` is selected. No support code may change while a run
root is being accepted. If support code changes, preserve the old attempt,
select a new `T`, and start from a new run root.

## Execution preconditions

All of the following are fail-closed prerequisites before the first recorded
command:

```text
clean worktree on validation/hf-contract-hardening-requalification
HEAD equals the selected tested-source commit T
T descends from bf8cc34cb6aa16ffeec1f609166db5efae79e9df
exact Transformers 4.57.6 interpreter
exact Transformers 5.14.1 interpreter
CUDA bf16-capable device selected explicitly
existing complete offline Hugging Face cache
new absent owner-controlled run root outside every Git worktree
existing writable archive directory outside every Git worktree
explicit non-empty reviewer identity
owner-only umask 077
```

The execution interface will require absolute values for:

```bash
TF4576_PYTHON
TF5141_PYTHON
HF_CACHE_DIR
HF_HARDENING_RUN_ROOT
MULTISCREEN_EVIDENCE_ARCHIVE_DIR
MULTISCREEN_EVIDENCE_REVIEWERS
```

No value may be inferred from an authenticated GitHub account. The reviewer
must also supply a non-empty review method, a full 40- or 64-character review
commit, and `raw-events-reviewed=true` after actually completing the review.

## Hermetic execution contract

Every recorded command runs through the existing recorder with a fixed
`/usr/bin/env -i` environment. The environment fixes locale, timezone,
hash seed, offline Hugging Face settings, telemetry/progress settings, Python
bytecode behavior, and explicit CPU or CUDA visibility. `HOME` points inside
the private run root. The exact Python executable and package identities are
recorded separately for Transformers 4.57.6 and 5.14.1.

Stage E fixes both `HF_DATASETS_DISABLE_PROGRESS_BARS=1` and
`HF_HUB_DISABLE_PROGRESS_BARS=1`. The latter suppresses carriage-return
progress output from Hub-controlled Transformers save/load paths so the
recorded focused-test logs remain canonical UTF-8 LF text. This is a Stage
E-only addition; the accepted Level 1 environment contract and reviewer remain
unchanged.

Each command name is single-use. `--require-absent` protects every generated
output subtree. A failed attempt is retained; commands are never rerun into the
same run root, and a new attempt receives both a new root and a complete new
ledger.

## Fixed recorded matrix

The qualification matrix contains exactly 53 command records and two runtime
environment records. The Stage E reviewer must reject missing, duplicate,
extra, reordered where ordered, or commit-mismatched records.

### 1. Preflight, static checks, and evidence tooling — 14 commands

```text
environment-tf4576
environment-tf5141
environment-cuda0
offline-cache-preflight
repository-hygiene
syntax-hardening
level1-evidence-support-tests
hardening-evidence-support-tests
tokenizer-reload-tests-tf4576
tokenizer-reload-tests-tf5141
validation-evidence-tests
json-validation
workflow-yaml
markdown-links
```

`repository-hygiene` is the recorded clean observation for `T`. Syntax uses a
private `PYTHONPYCACHEPREFIX`. The Level 1 and validation-evidence suites run
with their standard-library-only modes where supported.

### 2. Focused HF/model/data contracts — 20 commands

Each of these ten files runs once under Transformers 4.57.6 and once under
5.14.1, as a separate fresh Python process:

| Command stem | Test file | Expected tests per lane | Device |
| --- | --- | ---: | --- |
| `hf-output-head` | `test_hf_output_head_contract.py` | 4 | CPU |
| `training-edge` | `test_training_edge_contract.py` | 10 | CPU |
| `gradient-checkpointing` | `test_gradient_checkpointing_contract.py` | 7 | CUDA visible |
| `p0-4-qualification` | `test_p0_4_qualification_contract.py` | 11 | CPU |
| `generation-input` | `test_generation_input_contract.py` | 14 | CPU |
| `packed-text` | `test_packed_text_contract.py` | 11 | CPU |
| `paper-architecture` | `test_paper_architecture_contract.py` | 5 | CPU |
| `paper-initialization` | `test_paper_initialization_contract.py` | 3 | CPU |
| `mipe-position-cache` | `test_mipe_position_cache_contract.py` | 25 | CUDA visible |
| `paper-training-contract` | `test_paper_training_contract.py` | 27 | CUDA visible |

The command names append `-tf4576` or `-tf5141`. The expected total is 117
tests in each exact lane. Separate processes are required so import ordering,
module-level state, and lifecycle mutation cannot make one contract mask
another. The reviewer parses `Ran N tests` and the terminal `OK`; a zero-test
success, failure, error, or unexpected skip stops the gate. In particular, all
required CUDA assertions must execute rather than skip.

Generic evidence tooling is version-independent and runs once in the
standard-library lane; dependency-facing tokenizer reload fixtures still run
in both exact Transformers lanes.

### 3. Architecture manifest — 1 command

```text
c1-manifest
```

The committed paper-scale architecture manifest must regenerate byte- and
field-equivalently under its check mode.

### 4. Formula and oracle — 3 commands

```text
formula-units
oracle-selfcheck
oracle-smoke
```

Stable oracle checks keep `mipe_compute_dtype="fp32"` and
`softmask_compute_dtype="fp32"` where applicable.

### 5. Full P0 regression — 4 commands

```text
p0-1-cpu-fp32
p0-1-cuda-bf16
p0-2-cpu-fp32
p0-2-cuda-bf16
```

These are full runs. `--quick` and `--no-layer-hooks` are forbidden. Existing
accepted seeds and tolerances remain fixed:

```text
P0-1 seed 1234; CPU fp32 rtol/atol 1e-5; CUDA bf16 rtol/atol 0.03
P0-2 seed 4321; CPU fp32 rtol/atol 1e-5; CUDA bf16 rtol/atol 0.03
```

### 6. Fresh P0-3 — 3 commands

```text
p0-3-checkpointed
p0-3-tokenizer-psi8
p0-3-tokenizer-psi16
```

The accepted bounded contract is retained: fresh checkpointed CUDA bf16
training runs Psi=8 for 40 optimizer steps and Psi=16 for 25 optimizer steps.
The reviewer requires 65 finite loss/gradient events, configured loss decrease,
save/reload, tokenizer reload, cache split, generation, the bound data
contract, the completion marker, and no failure artifact.

### 7. Fresh strict P0-4 — 7 commands

```text
p0-4-psi8-preflight
p0-4-psi8
p0-4-tokenizer-psi8
p0-4-review-psi8
p0-4-psi16-preflight
p0-4-psi16
p0-4-tokenizer-psi16
```

Both static preflight logs must be parsed semantically, including the Stage C
microbatch and checkpointing checks. Each fresh run must produce exactly 57
events in this order:

```text
run_start
preflight_complete
50 x train_step
training_complete
save_reload_check
cache_split_check
generation_check
run_complete
```

The reviewer requires the exact
`multiscreen-p0-4-qualification-v2` object and all eight exact-boolean
conditions to be true:

```text
gpt2_vocab_50257
context_4096
cuda_device
bf16_amp
microbatch_size_1
optimizer_steps_at_least_50
gradient_checkpointing_enabled
gradient_checkpointing_non_reentrant
```

It independently cross-checks the summary, `run_complete`, observed optimizer
step count, microbatch size, runtime checkpointing witness, and
`use_reentrant=False`. It also requires finite losses and gradient norms,
configured loss decrease, save/load, tokenizer reload, manual cache split,
generation, data-contract identity, `P0-4_COMPLETE.md`, and the absence of
`P0-4_DIAGNOSTIC_COMPLETE.md`, `P0-4_FAILED.md`, and `failure.json`.

After the Psi=8 run and tokenizer check, the focused v2 reviewer must pass and
its report must be inspected. Only then may Psi=16 begin. Any failure preserves
the attempt and prevents all later Stage E commands in that root.

### 8. Final hygiene — 1 command

```text
repository-hygiene-final
```

The final recorded repository state must still be clean at `T` before the
full machine review begins.

### Deliberate C3 decision

Stage E runs `test_paper_training_contract.py` in both exact lanes, but does
not create fresh C3 data, operational, or peak-exposure CUDA outputs. The Stage
E specification requires fresh P0-3 and P0-4 training only. Copying the old
Level 1 four-lane C3 diagnostic matrix would add an unrelated training and
cache dependency without strengthening the specified Stage E claim. Accepted
C3 evidence remains historical and immutable.

## Machine and human review

### Focused Psi=8 review

The focused reviewer reads only the fixed Psi=8 prerequisite records and raw
outputs, verifies the v2 qualification contract, and emits a deterministic,
path-free report. It is the hard ordering gate before Psi=16.

### Full machine review

After all 53 commands, the Stage E reviewer must independently:

- verify the run marker, two environment records, and exact 53-command ledger;
- verify tested commit, branch, environment suffixes, CPU/CUDA classification,
  command arguments, order constraints, exit status, and complete log hashes;
- parse all focused unittest counts and outcomes rather than trusting exit code;
- verify the full P0-1/P0-2 commands used no reduced flags;
- review P0-3's 65 and P0-4's 114 raw events, for 179 reviewed raw events total;
- verify all tokenizer reports against their checkpoint identities;
- verify every completion, diagnostic, and failure marker rule;
- verify the implementation-base ancestry and historical evidence immutability;
- reject non-finite JSON values, duplicate keys, symlinks, path escape, private
  paths, credentials, and unknown artifact identities; and
- hash every reviewed source and calculate a deterministic aggregate review
  material hash.

The full reviewer is intentionally outside the 53-command ledger so its report
can bind the already-closed ledger without self-reference.

### Explicit human review

The named reviewer must read all 179 raw events and all 53 lossless command
logs, including both P0-4 summaries and markers. Only after that review may
`collect_validation_provenance.py` record:

```text
explicit reviewer identity
non-empty review method
review commit T
raw-events-reviewed = true
```

The machine review does not substitute for this explicit human acceptance.

## Fixed package inventory

The builder never discovers sources by walking the run root. Its versioned
profile contains an exact allowlist, and adversarial tests pin the set and
count before `T` is selected.

Included classes are:

```text
P0-3 data contract, completion marker, aggregate result, Psi=8/16 metrics,
  lossless stdout, and two tokenizer reports
P0-4 Psi=8/16 data contracts, completion markers, summaries, 57-event streams,
  and tokenizer reports
Psi=8 focused raw review
private run marker, command ledger, environment ledger
53 lossless command logs and 53 command records
two runtime environment records
full machine review and explicit acceptance provenance
new Stage E summary JSON and Markdown
```

Excluded classes are:

```text
checkpoints and model/tokenizer copies
weights and optimizer states
caches and dataset rows
historical Level 1/P0-4/C3 evidence
raw archives
the archive descriptor and verification reports
secrets and private absolute paths
```

The descriptor and verification reports stay outside the archive input to
avoid self-reference.

## Retention and canonical outputs

The exact and sanitized archives are separate deterministic builds from the
same reviewed allowlist:

```text
hf-contract-hardening-${T}.exact.tar.gz
hf-contract-hardening-${T}.sanitized.tar.gz
```

The exact archive is written only to the explicit external private archive
directory. It is never uploaded publicly. The sanitized archive is separately
rescanned and offline-verified and remains unpublished by default.

The only new canonical compact files are:

```text
docs/validation_results/HF_CONTRACT_HARDENING_SUMMARY.md
docs/validation_results/HF_CONTRACT_HARDENING_SUMMARY.json
docs/validation_results/HF_CONTRACT_HARDENING_EVIDENCE_ARCHIVE.json
docs/validation_results/HF_CONTRACT_HARDENING_EXACT_VERIFICATION.json
docs/validation_results/HF_CONTRACT_HARDENING_SANITIZED_VERIFICATION.json
```

Both archives require canonical single-member gzip/USTAR structure, complete
member-boundary and padding validation, independent sanitized-member rescans,
offline verification, and recorded SHA-256 values. The committed descriptor
contains privacy-safe storage locators only, never a local absolute path.

## Two-commit evidence closure

The closure sequence is fixed:

1. Commit the plan and all Stage E evidence-support code and tests.
2. Verify the tree is clean and select that commit as `T`.
3. Run the complete matrix into a new private root without changing Git.
4. Complete the focused Psi=8 and full machine reviews.
5. Complete explicit human review and collect acceptance provenance for `T`.
6. Prepare summaries and a package input from the fixed reviewed allowlist.
7. Build exact and sanitized archives independently; verify both offline.
8. Seal a partial descriptor whose evidence-commit field remains unresolved.
9. Create commit `A` containing the two summaries, partial descriptor, and two
   verification reports.
10. Observe clean commit-`A` provenance.
11. Close the descriptor with commit `A`, update canonical documentation, and
    create commit `B`.
12. Reverify the unchanged archives against the complete descriptor, prove the
    generated verification reports equal the committed bytes, and confirm the
    tip is clean.
13. Push and create the final draft PR. Do not merge or tag it.

The descriptor cannot contain its own closure tip without self-reference;
commit `A` is the recorded evidence commit, while commit `B` is the documented
closure tip.

## Documentation closure

Only after reviewed evidence passes, update as needed:

```text
README.md
AGENTS.md
docs/HANDOFF.md
docs/VALIDATION_STATUS.md
docs/TESTING.md
docs/KNOWN_LIMITATIONS.md
docs/RELEASE_CHECKLIST.md
docs/validation_results/VALIDATION_LOG_INDEX.md
```

The final text must state the eight hardened contracts, preserve all existing
limitations, distinguish historical Level 1 from the new tested source, and
avoid broad HF, generation, performance, or ecosystem claims.

## Stop conditions

Stop and report `PARTIAL/BLOCKED WITH EVIDENCE` without weakening tests if:

- model mathematics, oracle semantics, accepted MiPE/cache semantics, or an
  accepted tolerance would need to change;
- Transformers 4.57.6 and 5.14.1 have an irreconcilable contract conflict;
- the required CUDA bf16 path is unavailable;
- the explicit reviewer, offline cache, or durable private retention location
  is unavailable;
- a previously accepted regression fails reproducibly;
- any environment-destructive operation would be necessary;
- raw evidence is missing, inconsistent, non-finite, overwritten, or not bound
  to `T`; or
- the exact or sanitized archive cannot be independently verified.

A test failure does not authorize a tolerance change, a reduced matrix, reuse
of historical metrics, or an in-place retry.

## Acceptance checklist

Stage E reaches `COMPLETE` only when every item is true:

- [ ] all seven Stage A-D resolutions are present in the clean tested source;
- [ ] all 53 recorded commands and two environment records pass review;
- [ ] 117 focused tests pass independently in Transformers 4.57.6;
- [ ] 117 focused tests pass independently in Transformers 5.14.1;
- [ ] formula/oracle and full P0-1/P0-2 CPU fp32/CUDA bf16 pass;
- [ ] fresh checkpointed CUDA bf16 P0-3 Psi=8/16 passes;
- [ ] fresh strict P0-4 Psi=8 passes the hardened predicate and focused review;
- [ ] fresh strict P0-4 Psi=16 passes the hardened predicate;
- [ ] all 179 raw events and 53 lossless logs receive explicit review;
- [ ] exact/private evidence is retained and verified outside Git;
- [ ] a separately built sanitized archive is rescanned and verified offline;
- [ ] reviewer, review method, review commit, and raw-event review are recorded;
- [ ] the five new canonical evidence files are closed without private data;
- [ ] canonical documentation is current and historical evidence is unchanged;
- [ ] the final branch is clean and a draft PR is created;
- [ ] no automatic merge or tag occurs.

Creating this plan satisfies none of the unchecked acceptance items. The next
implementation checkpoint is the Stage E-specific reviewer, builder, cache
preflight, and adversarial fixture suite; fresh GPU execution remains gated on
the explicit reviewer and retention inputs.
