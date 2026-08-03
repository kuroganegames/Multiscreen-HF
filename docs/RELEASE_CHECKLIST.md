# P0 Release / Handoff Checklist

This checklist is for tagging, handing off, or resuming work from the P0-qualified baseline.

## Current selected gate

```text
P1-preflight A: Validation provenance / evidence retention v1
```

The design and Codex handoff are:

```text
docs/P1_PREFLIGHT_A_PLAN.md
docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md
```

P1-preflight A is evidence infrastructure. It must not change model behavior or mark any P1 model/ecosystem capability as validated.

## Before handoff or tagging

- [ ] `README.md` links to `AGENTS.md`, `docs/HANDOFF.md`, `docs/P1_PREFLIGHT_A_PLAN.md`, `docs/CODEX_P1_PREFLIGHT_A_HANDOFF.md`, `docs/VALIDATION_STATUS.md`, `docs/TESTING.md`, and `docs/KNOWN_LIMITATIONS.md`.
- [ ] Root `AGENTS.md` reflects the current development phase, Conda/uv environment-safety contract, evidence-integrity rules, and required test policy.
- [ ] `docs/HANDOFF.md` identifies P1-preflight A as the selected gate without claiming it is already complete.
- [ ] `docs/VALIDATION_STATUS.md` reflects the latest accepted P0-1/P0-2/P0-3/P0-4 evidence.
- [ ] P0-4 remains complete from the reviewed qualifying evidence and is not reinterpreted from static validation or a diagnostic.
- [ ] `docs/validation_results/` contains sanitized compact result files and descriptors only; no secrets, private absolute paths, or raw archives.
- [ ] `docs/validation_results/VALIDATION_LOG_INDEX.md` links every accepted summary and archive descriptor.
- [ ] Reviewer identity comes from explicit input rather than authenticated-login inference.
- [ ] Historical worktree/reviewer facts that were not captured remain `not_recorded`, not guessed.
- [ ] Exact raw evidence is retained outside Git in user-controlled storage.
- [ ] Any public evidence archive is a separately sanitized and verified archive.
- [ ] Exact raw evidence is not uploaded to a public GitHub release.
- [ ] Archive filenames, sizes, source hashes, archive hashes, manifest hashes, storage class, and verification status are recorded.
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

For P1-preflight A, also run:

```text
- Python syntax checks for new evidence scripts/tests;
- JSON and JSON-schema parsing;
- evidence-tooling unit tests with synthetic fixtures;
- deterministic packaging repeat test;
- deliberate archive tamper test;
- sanitization positive and negative tests;
- offline archive verification;
- Markdown-link and repository-hygiene checks.
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
- original P0-4 summary/metrics files match their committed SHA-256 values;
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

A sanitized archive may be attached to an explicitly configured release. The exact raw archive must remain private and separately retained.

## Suggested current baseline release note

```text
P0-qualified unofficial HF Multiscreen implementation through P0-4.
Validated: paper-oracle equivalence, three-way reference equivalence,
DynamicCache-compatible greedy generation smoke, TinyStories Psi=8/16 bf16
smoke training, and qualifying GPT-2-vocabulary context-4096 CUDA bf16
short-run training for Psi=8 and Psi=16.
P0-4 runtime and memory are feasibility diagnostics only.
P1-preflight A evidence provenance/retention is selected but not yet complete.
Not validated: paper-scale reproduction, retrieval quality, long-context
efficiency, Triton/windowed kernels, PEFT/LoRA/Unsloth, broad generation,
or production serving.
```
