# Level 1 Core / P0 Release and Handoff Checklist

This checklist is for tagging, handing off, or resuming work from the
Level 1 Core and P0-qualified baselines.

## Current staged gate

```text
final Level 1 requalification: complete locally; draft PR REVIEW_REQUIRED
```

This remains an unofficial correctness-first result; the dense quadratic path
is not efficiency evidence, and it does not validate paper-scale reproduction,
retrieval benchmarks, optimized long-context efficiency, distributed training,
or any P1 model/ecosystem capability.

The accepted Stage 4 records and locally complete Stage 5 records are:

```text
docs/P0_5_C3_PLAN.md
configs/p0_5_c3_paper_training_contract.json
docs/validation_results/P0_5_C3_SUMMARY.{md,json}
docs/validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json
docs/LEVEL1_CORE_REQUALIFICATION_PLAN.md
docs/validation_results/LEVEL1_CORE_SUMMARY.{md,json}
docs/validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json
docs/validation_results/LEVEL1_CORE_{EXACT,SANITIZED}_VERIFICATION.json
```

C1 was reviewed and merged as PR #9. C2 was merged as PR #10, and its separate
CUDA-autocast cache-dtype correction was merged as PR #11. Stage 3 was reviewed
and merged as PR #12. Stage 4 unit, pinned-data, CUDA bf16 operational,
peak-exposure, full regression, and evidence checks passed on tested commit
`8fa5dbf13530c942b2c9e5f03a572bd0cd5ca74f`; the focused Stage 4 result was
reviewed and accepted by merged PR #13. The existing C3 evidence descriptor
still records partial retention: exact/private retention and an explicitly
supplied evidence reviewer remain pending. Those facts are not rewritten by PR
acceptance.

Final Level 1 requalification passed locally under
[LEVEL1_CORE_REQUALIFICATION_PLAN.md](LEVEL1_CORE_REQUALIFICATION_PLAN.md) on
tested source `b224ca1a127ee18fc5fd4b00a5df639401d60679`. Acceptance review is
recorded, the exact/private archive is retained externally and verified, and
the sanitized archive is verified but unpublished. The focused draft PR
remains `REVIEW_REQUIRED` and unmerged. No public asset exists, and no P1
model/ecosystem capability is validated.
This remains an unofficial correctness-first result; the dense quadratic path
is not efficiency evidence, and it does not validate paper-scale reproduction,
retrieval benchmarks, optimized long-context efficiency, distributed training,
or any P1 model/ecosystem capability.

The separate evidence-infrastructure design and historical Codex handoff are:

```text
docs/P1_PREFLIGHT_A_PLAN.md
docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md
```

P1-preflight A remains evidence infrastructure and remains partial/blocked.
Neither the merged staged contracts nor Stage 4 complete historical P0-4
retention or its evidence-archive acceptance-review requirements.

P1-preflight A infrastructure is implemented and tested, but the gate remains
`PARTIAL/BLOCKED WITH EVIDENCE`. All four retained Psi=8/Psi=16 summary and
metrics files matched their committed hashes; both completion markers were
found and hashed for the new descriptor; and the sanitized archive verified
locally. Exact/private retention is blocked because
`MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not configured, acceptance review is
pending because no explicit reviewer was supplied, and no public asset exists.
P0-4 remains complete; no P1 model/ecosystem capability is validated.

See [EVIDENCE_ARCHIVE_POLICY.md](EVIDENCE_ARCHIVE_POLICY.md) and the
[P0-4 retention descriptor](validation_results/P0_4_EVIDENCE_ARCHIVE.json).

## Before handoff or tagging

- [ ] `README.md` links to `AGENTS.md`, `docs/HANDOFF.md`, the historical P1-preflight A records, validation/testing/limitation records, `docs/EVIDENCE_ARCHIVE_POLICY.md`, and the P0-4 and Level 1 evidence descriptors.
- [ ] Root `AGENTS.md` reflects the current development phase, Conda/uv environment-safety contract, evidence-integrity rules, and required test policy.
- [ ] `docs/HANDOFF.md` identifies P1-preflight A as partial/blocked without claiming it is complete.
- [ ] `docs/VALIDATION_STATUS.md` reflects the latest P0 and reviewed Level 1 Core evidence.
- [ ] The Level 1 summary, complete descriptor, and both verification reports bind tested source `b224ca1a127ee18fc5fd4b00a5df639401d60679` without exposing private paths.
- [ ] P0-4 remains complete from the reviewed qualifying evidence and is not reinterpreted from static validation or a diagnostic.
- [ ] `docs/validation_results/` contains sanitized compact result files and descriptors only; no secrets, private absolute paths, or raw archives.
- [ ] `docs/validation_results/VALIDATION_LOG_INDEX.md` links every accepted summary and archive descriptor.
- [ ] A recorded review has an explicit reviewer and non-empty method, a full 40- or 64-character hexadecimal review commit, and an explicit raw-events-reviewed boolean; no ambient identity is inferred.
- [ ] Historical worktree/reviewer facts that were not captured remain `not_recorded`, not guessed.
- [ ] Recorded worktree provenance hashes exact `git status --porcelain=v1 --untracked-files=all --ignore-submodules=none` stdout bytes and records privacy-safe recursive submodule state/hash/count when applicable.
- [ ] Exact raw evidence is retained outside Git in user-controlled storage.
- [ ] Any public evidence archive is a separately sanitized and verified archive.
- [ ] Exact raw evidence is not uploaded to a public GitHub release.
- [ ] Archive filenames, sizes, source hashes, archive hashes, manifest hashes, storage class, and verification status are recorded.
- [ ] Verification rejects extra/concatenated gzip members, enforces canonical normalized USTAR boundaries and padding, rescans every sanitized member and control file, and binds the descriptor to gate, tested-source commit, and source-artifact metadata.
- [ ] Root `LICENSE` copyright line matches this repository.
- [ ] `THIRD_PARTY_NOTICES.md` lists vendored reference code and tokenizer/data caveats.
- [ ] `pyproject.toml` project version and `multiscreen_transformers.__version__` are aligned.
- [ ] `.gitignore` is present and excludes outputs, checkpoints, caches, generated weights, archive staging, and raw evidence archives.
- [ ] No checkpoints, output directories, evidence archives, or private storage paths are committed.
- [ ] No `__pycache__`, `.git` directories under `third_party/`, local cache directories, or environment directories are committed.

## Development environment safety

- [ ] Active Conda environment and Python executable were recorded before installation.
- [ ] Existing Conda/base environments were not deleted or broadly upgraded.
- [ ] No global package installation was performed.
- [ ] Any isolated environment was created separately and documented.
- [ ] `uv`, when used, targeted an explicit environment and did not rewrite unrelated lockfiles.
- [ ] Package changes and before/after versions were recorded.

## Minimum local smoke before pushing

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
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

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --validate-config-only

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --validate-config-only
```

For P1-preflight B, also run the focused contract under both exact compatibility
lanes described in [TESTING.md](TESTING.md):

```bash
python -m unittest discover \
  -s tests \
  -p 'test_gradient_checkpointing_contract.py' \
  -v
```

For P0.5-C3, also run the offline focused contract, then the pinned data lane
and the Psi=8-before-Psi=16 CUDA bf16 operational and exact-peak exposure
commands documented in [TESTING.md](TESTING.md). Do not substitute a local text
fallback, shortened context, clipping, or a lower learning rate and still call
the corresponding Stage 4 lane passed.

For final Level 1 requalification, follow the complete matrix, provenance,
review, and evidence-retention boundary in
[LEVEL1_CORE_REQUALIFICATION_PLAN.md](LEVEL1_CORE_REQUALIFICATION_PLAN.md).
The fresh reviewed Stage 5 record satisfied that boundary locally. Stage 4
acceptance alone remains insufficient for a future Stage 5 reproduction.

For P1-preflight A, also run:

```bash
python -S -m py_compile \
  scripts/validation_evidence_common.py \
  scripts/collect_validation_provenance.py \
  scripts/package_validation_evidence.py \
  scripts/verify_validation_evidence.py \
  tests/test_validation_evidence*.py

python -S -m unittest discover \
  -s tests \
  -p 'test_validation_evidence*.py' \
  -v
```

The globbed suite runs the main evidence tests plus collector, common, and
verifier hardening modules. Its synthetic fixtures cover JSON/schema parsing,
review/provenance validation, deterministic packaging, path/type rejection,
canonical archive structure, tampering, sanitization, and offline verification.

After recovering the ignored sanitized archive from its recorded logical
locator, verify it without extraction or network access:

```bash
SANITIZED_ARCHIVE=/path/to/retrieved/validation-evidence-sanitized-p0-4-v1-r2.tar.gz

python -S scripts/verify_validation_evidence.py \
  --archive "$SANITIZED_ARCHIVE" \
  --expected-sha256 d58a4c9ecf28f20a135f4ba2ce95c5a532a04ea92f36e5b54d893400ae4c62fd \
  --evidence-document docs/validation_results/P0_4_EVIDENCE_ARCHIVE.json \
  --schema schemas/validation_evidence_v1.schema.json \
  --json
```

CI must use synthetic fixtures and must not require private P0-4 artifacts, GPU, Hub access, external storage, or release credentials.

## P0-4 result acceptance

The current P0-4 record was accepted only after verifying the following against the actual local artifacts. Apply the same checklist to any future reproduction:

```text
- Psi=8 qualifying run reviewed
- Psi=16 qualifying run reviewed
- qualification.qualified=true for each accepted run
- GPT-2 vocab 50,257
- context 4096
- CUDA bf16
- optimizer steps >= 50
- finite loss and gradient norms
- configured loss decrease reached
- save/load and tokenizer reload passed
- generate(use_cache=True) passed
- manual cache split comparison passed
- metrics.jsonl and summary.json hashes recorded
- compact Markdown and JSON summaries sanitized and committed
```

If Psi=8 passes but Psi=16 cannot complete on available hardware, record a partial or blocked outcome rather than an overall pass.

## P1-preflight A acceptance

P1-preflight A may be marked complete only when:

```text
- schema v1, provenance collector, packager, verifier, and tests are implemented;
- reviewer input is explicit;
- current evidence-handoff worktree state is recorded before edits and after commit;
- missing historical P0-4 provenance remains explicit not-recorded data;
- all four original P0-4 summary/metrics files match their committed SHA-256 values;
- both completion markers are found and hashed for the archive descriptor;
- an exact/private archive is stored outside the repository in configured retention storage;
- a separately sanitized archive is produced and verified;
- the compact P0-4 archive descriptor contains no private absolute paths;
- policies, handoff, index, ignore rules, and CI are updated;
- quick P0 regression passes;
- no model/config/oracle/cache/generation/training-harness source changed;
- no P1 model capability is marked validated.
```

If raw files, reviewer input, durable storage, sanitization, or verification are unavailable, record `PARTIAL/BLOCKED WITH EVIDENCE` rather than weakening the gate.

## Suggested baseline tag

The completed P0-1 through P0-4 baseline should use a new immutable tag, such as `p0-4-qualified-v0`, only after the evidence PR is reviewed and merged. If an earlier tag exists, do not move it; verify remote tags before choosing a new immutable name. Do not create a tag from an evidence-infrastructure branch.

Do not create a Level 1 tag before the focused Stage 5 PR is reviewed and
merged, and never create one without explicit user instruction.

A sanitized archive may be attached to an explicitly configured release. The exact raw archive must remain private and separately retained.

## Suggested current baseline release note

```text
Level 1 Core mathematical Hugging Face implementation complete locally.
Reviewed validation covers paper-oracle and three-way reference equivalence,
C1 architecture/initialization, C2 MiPE/cache semantics, supported
non-reentrant gradient checkpointing, bounded C3 training-contract
diagnostics, TinyStories Psi=8/16 bf16 smoke training, and fresh qualifying
GPT-2-vocabulary context-4096 CUDA bf16 short runs for Psi=8 and Psi=16.
The focused Stage 5 draft PR remains REVIEW_REQUIRED and unmerged.
P1-preflight A evidence infrastructure is implemented; retention remains
partial because explicit review and external exact/private retention are pending.
Not validated: paper-scale reproduction, retrieval quality, optimized
long-context efficiency, distributed training, Triton/windowed kernels,
PEFT/LoRA/Unsloth, broad generation, or production serving.
```
