# Codex Goal Handoff: P0-4 Local Qualification

This document is the local-Codex entry point for continuing development after the merge of PR #3.

## Current phase

```text
Repository: kuroganegames/Multiscreen-HF
Baseline status: P0-qualified research implementation
P0-1: complete
P0-2: complete
P0-3: complete
P0-4 harness/config/docs: merged in PR #3
P0-4 CPU static and tiny end-to-end CI diagnostic: passed
P0-4 qualifying CUDA bf16 context-4096 runs: pending
P1 ecosystem work: not yet validated
```

The immediate goal is not to optimize Multiscreen or begin LoRA work. It is to execute P0-4 honestly on the local CUDA system, preserve evidence, and update the validation record without weakening the gate.

## Clone and enter the repository

```bash
git clone https://github.com/kuroganegames/Multiscreen-HF.git
cd Multiscreen-HF

git status --short --branch
git log -1 --oneline
```

Run Codex from the repository root so the root `AGENTS.md` is loaded:

```bash
codex
```

If `/goal` is not shown in the slash-command list, exit Codex and enable the feature:

```bash
codex features enable goals
codex
```

Useful goal controls:

```text
/goal          show current goal
/goal pause    pause it
/goal resume   resume it
/goal clear    remove it
```

## Before starting the goal

The local environment should provide:

```text
- Python 3.10 or newer
- a bf16-capable CUDA GPU
- enough disk space for local Hugging Face caches and ignored checkpoints
- GitHub authentication if Codex should push a branch or open a PR
- Hugging Face access or a populated local cache for the GPT-2 tokenizer and dataset
```

Codex should discover the actual environment rather than assuming paths. The user may already be inside a Conda or virtual environment; do not replace it unnecessarily.

## Ready-to-paste `/goal` prompt

Paste the complete block below into Codex from the repository root.

```text
/goal Complete the local P0-4 qualification and evidence handoff for this repository without stopping for ordinary setup, test, training, logging, or documentation errors. Stop only when one of the explicitly defined terminal states is reached.

You are the continuing engineer and research-validation agent for the current checkout of `kuroganegames/Multiscreen-HF`.

PROJECT STATE

- This is an unofficial Hugging Face Transformers-compatible Multiscreen research implementation.
- P0-1 is complete: paper_math_oracle and the HF implementation agree on validated small-shape formula, loss, mask, position, and cache behavior.
- P0-2 is complete: dieOD/multiscreen-pytorch, the HF implementation, and paper_math_oracle agree in the recorded CPU fp32 and CUDA bf16 sweeps.
- P0-3 is complete: Psi=8 and Psi=16 TinyStories bf16 smoke training passed, including finite losses/gradients, save/load, greedy generation with cache, and manual cache-split equality.
- PR #3 merged the P0-4 harness, Psi=8/Psi=16 GPT-2-context-4096 configs, plans, result template, static preflight, and a tiny CPU end-to-end CI diagnostic.
- The current development phase is P0-4 qualifying local execution. P0-4 is not complete yet.
- Do not begin P1 LoRA, QLoRA, Unsloth, broad generation, serving, Triton, or performance work during this goal.

PRIMARY OBJECTIVE

Execute and review the qualifying P0-4 CUDA bf16 smoke in this order:

1. restore and verify the merged P0 baseline;
2. validate the P0-4 configs;
3. run an optional reduced Psi=8 diagnostic when useful;
4. run qualifying Psi=8 with GPT-2 vocab 50,257, context 4096, CUDA bf16, microbatch 1, and at least 50 optimizer steps;
5. review its loss, gradient, memory, save/load, generation, and cache artifacts;
6. run qualifying Psi=16 only after Psi=8 passes and local memory headroom is understood;
7. create compact sanitized validation records, update the handoff/status documentation accurately, run final regression checks, and prepare a focused PR.

MANDATORY READING

Before changing anything, read these files in order and summarize their relevant contracts in the progress report:

- `AGENTS.md`
- `README.md`
- `docs/HANDOFF.md`
- `docs/VALIDATION_STATUS.md`
- `docs/TESTING.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/P0_4_PLAN.md`
- `docs/P0_4_RESULTS_TEMPLATE.md`
- `docs/LOGGING_POLICY.md`
- `docs/REPOSITORY_AUDIT.md`
- `docs/RELEASE_CHECKLIST.md`
- `scripts/p0_4_gpt2_context4096_smoke.py`
- both P0-4 `config.json` and `run.json` files

Also inspect the core files, without editing them during the initial qualification path:

- `multiscreen_transformers/configuration_multiscreen.py`
- `multiscreen_transformers/modeling_multiscreen.py`
- `oracle/paper_math_oracle.py`
- `oracle/test_against_hf_port.py`
- `p0_2_three_way_minimal/test_three_way_minimal.py`

GIT AND SAFETY RULES

- Confirm the checkout is on an up-to-date `main` and has a clean working tree.
- Record `git rev-parse HEAD`, `git log -1 --oneline`, and `git status --short --branch`.
- Create a focused working branch such as `validation/p0-4-local-cuda` before editing tracked files.
- Never commit directly to `main`.
- Never discard or overwrite pre-existing user changes.
- Do not commit `outputs/`, checkpoints, model weights, local caches, raw large logs, usernames, secrets, or local absolute paths.
- Do not merge the final PR automatically.
- If `gh auth status` and push permissions are available, push the branch and open a PR. Otherwise leave a clean committed branch and print the exact push/PR commands.

NON-NEGOTIABLE MODEL CONTRACTS

- Preserve `s_r_paper = -s_r_hf`.
- Preserve the distinct `fp32` stable oracle mode and `reference` low-precision compatibility mode.
- Preserve tied normalized embeddings and `tie_word_embeddings=True`.
- Preserve current DynamicCache and scalar-position contracts.
- Treat the dense oracle and dense HF screening path as correctness implementations, not speed references.
- Do not claim long-context efficiency from P0-4 memory or runtime data.

SCOPE OF ALLOWED CHANGES

During this goal you may fix narrowly demonstrated issues in:

- P0-4 experiment harness behavior;
- P0-4 configs;
- environment diagnostics;
- result serialization or sanitization;
- experiment and handoff documentation;
- focused tests for the P0-4 harness.

Do not modify the following merely to force a qualifying run to pass:

- `multiscreen_transformers/modeling_multiscreen.py`;
- `multiscreen_transformers/configuration_multiscreen.py` architecture semantics;
- `oracle/paper_math_oracle.py`;
- cache/generation semantics;
- state_dict conversion;
- P0 acceptance tolerances without evidence and explicit justification.

If evidence indicates a model-core, oracle, cache, position, mask, or state-dict defect, reach the BLOCKED terminal state with a focused diagnosis instead of silently changing the baseline in this goal. State which contract is implicated and which stronger P0 tests would be required for a separate fix.

CHECKPOINT 1: REPOSITORY AND ENVIRONMENT AUDIT

Collect and report, without exposing secrets:

- repository HEAD and branch state;
- Python version and executable path;
- pip/Conda environment identity when available;
- PyTorch, Transformers, Datasets, Tokenizers, Safetensors, NumPy versions;
- `torch.cuda.is_available()`;
- CUDA runtime reported by PyTorch;
- GPU name, compute capability, total memory, bf16 support;
- `nvidia-smi` summary when available;
- free disk space for the repository, cache, and output locations;
- relevant Hugging Face cache environment variables;
- whether GPT-2 tokenizer and TinyStories data are already cached;
- `gh auth status` without printing credentials.

Use the existing environment when suitable. Install the repository and declared dependencies from the repository root:

    python -m pip install -e .
    python -m pip install -r requirements.txt
    export PYTHONPATH=$PWD:$PWD/oracle

Do not upgrade unrelated packages speculatively. If installation changes versions, record before/after versions.

CHECKPOINT 2: BASELINE REGRESSION

Run exactly these minimum checks:

    python oracle/test_formula_units.py
    python oracle/test_paper_math_oracle_selfcheck.py
    python oracle/test_paper_math_oracle_smoke.py
    python oracle/test_against_hf_port.py --quick

    python p0_2_three_way_minimal/test_three_way_minimal.py \
      --reference-root third_party/multiscreen-pytorch \
      --hf-root . \
      --oracle-root oracle \
      --quick

Also run:

    python -m py_compile \
      multiscreen_transformers/*.py \
      oracle/*.py \
      scripts/*.py \
      p0_2_three_way_minimal/*.py

Do not proceed to qualifying training if a baseline check fails. Diagnose environment versus repository failure. Fix only an in-scope harness/environment issue; otherwise enter BLOCKED state with logs and reproduction commands.

CHECKPOINT 3: P0-4 STATIC PREFLIGHT

Run both static preflights:

    python scripts/p0_4_gpt2_context4096_smoke.py \
      --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
      --validate-config-only

    python scripts/p0_4_gpt2_context4096_smoke.py \
      --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
      --validate-config-only

Verify directly that the qualifying defaults encode:

- vocab size 50,257;
- context 4,096;
- microbatch 1;
- bf16;
- at least 50 optimizer steps;
- gradient checkpointing enabled;
- Psi scaling: hidden size = Psi squared, layers = heads = Psi.

CHECKPOINT 4: OPTIONAL PSI=8 REDUCED DIAGNOSTIC

Run a reduced diagnostic before context 4096 when the tokenizer/data cache is cold, the GPU memory margin is uncertain, or end-to-end local behavior has not been exercised. A suggested command is:

    python scripts/p0_4_gpt2_context4096_smoke.py \
      --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
      --seq-len 1024 \
      --steps 2 \
      --gradient-accumulation-steps 1 \
      --output-dir outputs/p0_4_psi8_ctx1024_diagnostic

Confirm it creates `P0-4_DIAGNOSTIC_COMPLETE.md`, not `P0-4_COMPLETE.md`. This run never qualifies P0-4.

CHECKPOINT 5: QUALIFYING PSI=8

Run the unweakened checked-in command:

    python scripts/p0_4_gpt2_context4096_smoke.py \
      --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096

A qualifying Psi=8 result requires all of the following:

- `qualification.qualified` is true in `summary.json`;
- GPT-2 tokenizer vocabulary is exactly 50,257;
- packed sequence length is exactly 4,096;
- device is CUDA;
- AMP dtype is bf16;
- optimizer steps are at least 50;
- every recorded train loss is finite;
- every recorded gradient norm is finite;
- probe loss satisfies the configured decrease criterion;
- save/load succeeds;
- loaded logits satisfy configured atol/rtol;
- `generate(use_cache=True)` appends tokens;
- cache-split suffix logits satisfy configured atol/rtol against full forward;
- `metrics.jsonl`, `summary.json`, and `P0-4_COMPLETE.md` exist;
- no `failure.json` or `P0-4_FAILED.md` remains for the accepted run.

Inspect every JSONL event rather than trusting the completion filename alone. Record peak allocated/reserved CUDA memory and elapsed time as diagnostics only.

If Psi=8 OOMs:

- preserve the failure artifacts and last metrics event;
- verify microbatch 1 and gradient checkpointing are active;
- verify no unrelated GPU process is consuming memory;
- retry only when there is a concrete environmental reason;
- use reduced-context diagnostics to distinguish correctness from capacity;
- never reduce the qualifying context, dtype, Psi, or step requirement and call it a pass.

CHECKPOINT 6: QUALIFYING PSI=16

Run this only after Psi=8 qualifies and its artifacts have been reviewed:

    python scripts/p0_4_gpt2_context4096_smoke.py \
      --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096

Apply the same qualification and artifact checks. Do not interpret a Psi=8 pass as a Psi=16 pass. If local hardware cannot complete Psi=16, record P0-4 overall as partial or blocked according to the documented project decision; do not mark the overall gate complete.

CHECKPOINT 7: RESULT RECORDING

For each attempted qualifying run:

- retain the full ignored output locally;
- compute SHA-256 for `summary.json` and `metrics.jsonl`;
- create a compact, sanitized human-readable summary and machine-readable JSON under `docs/validation_results/`;
- use `docs/P0_4_RESULTS_TEMPLATE.md` as the human-readable structure;
- remove local absolute paths, usernames, hostnames when unnecessary, secrets, cache paths, and checkpoint paths from committed summaries;
- preserve exact commands, package versions, GPU model and memory, qualification conditions, losses, grad norms, peak memory, reload error, cache error, and generation status;
- distinguish `passed`, `partial`, `failed`, and `blocked` precisely.

Suggested accepted filenames:

- `docs/validation_results/P0_4_SUMMARY.md`
- `docs/validation_results/P0_4_SUMMARY.json`

Update `docs/validation_results/VALIDATION_LOG_INDEX.md` to link them.

Update these files only according to actual reviewed evidence:

- `README.md`
- `docs/HANDOFF.md`
- `docs/VALIDATION_STATUS.md`
- `docs/TESTING.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/P0_4_PLAN.md` when execution notes need clarification

Do not erase the historical P0-1/P0-2/P0-3 records.

CHECKPOINT 8: FINAL REGRESSION AND REVIEW

After tracked changes, rerun at least:

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

Run repository hygiene checks relevant to the diff:

- Python syntax;
- JSON parsing;
- local Markdown-link validation;
- no tracked checkpoint/model artifacts;
- no tracked `__pycache__` or `.pyc`;
- package version consistency when touched.

Review `git diff --check`, `git diff`, `git status --short`, and staged files before committing.

PROGRESS REPORTING

At every checkpoint, provide a compact status containing:

- current checkpoint;
- commands executed;
- files changed;
- tests and exact results;
- artifacts produced;
- current blocker or risk;
- next checkpoint.

Do not stop merely to ask whether to run the next already-authorized checkpoint. Continue automatically while the next step remains inside this goal and is safe.

TERMINAL STATES

Reach exactly one terminal state.

A. COMPLETE

- Psi=8 qualifying run passed and was reviewed;
- Psi=16 qualifying run passed and was reviewed;
- compact sanitized P0-4 records are committed;
- project status documents accurately mark P0-4 complete;
- final regression and hygiene checks pass;
- a focused PR is open, or the branch is committed and exact PR commands are provided.

B. PARTIAL/BLOCKED WITH EVIDENCE

Use this only after ordinary setup and diagnostic attempts are exhausted and one of these is true:

- local hardware cannot satisfy the unweakened qualifying condition;
- a reproducible environment incompatibility remains;
- a baseline P0 regression fails outside the allowed scope;
- evidence indicates a model-core/oracle/cache/state-dict defect that requires a separate correctness PR;
- Psi=8 passes but Psi=16 cannot qualify on this system.

For this terminal state:

- preserve ignored failure artifacts locally;
- commit only compact sanitized failure/partial summaries and accurate status docs;
- do not mark P0-4 complete;
- include exact reproduction commands and the smallest supported next action;
- open a focused evidence/diagnosis PR when useful, or provide exact commands.

FINAL RESPONSE FORMAT

変更ファイル:
  - ...

追加ファイル:
  - ...

実行テスト:
  - command: ...
    result: ...

P0-4実測結果:
  - Psi=8: ...
  - Psi=16: ...

結果:
  - terminal state: COMPLETE or PARTIAL/BLOCKED WITH EVIDENCE
  - ...

未確認:
  - ...

作成したコミット/PR:
  - ...

次にやるべきこと:
  - ...
```

## Why this is one goal

The prompt deliberately keeps one objective: qualify and record P0-4. It does not combine unrelated P1 work. The stopping conditions are verifiable, and the failure path preserves evidence rather than weakening the gate.

## Recommended supervision

The goal is designed to continue through routine errors without waiting for input. Check status with `/goal` when desired. Pause it before changing hardware allocation, deleting large local artifacts, changing credentials, or changing the research priority.

After the goal reaches a terminal state, review the resulting diff and evidence before merging any PR.
