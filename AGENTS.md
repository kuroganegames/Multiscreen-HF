# AGENTS.md

## Project identity

This repository is an **unofficial Hugging Face Transformers-compatible implementation of Multiscreen**. Treat it as a correctness-first research artifact, not an official implementation, production serving stack, or paper-scale reproduction.

Current validation state:

```text
P0-1: paper_math_oracle vs HF implementation
  complete

P0-2: dieOD/multiscreen-pytorch vs HF implementation vs paper_math_oracle
  complete

P0-3: Psi=8/16 TinyStories bf16 smoke training
  complete

P0-4: GPT-2 vocab 50,257 + context 4096 short pretraining smoke
  harness/config/docs merged in PR #3
  CPU static and tiny end-to-end diagnostic passed in GitHub Actions
  qualifying local CUDA bf16 Psi=8/Psi=16 execution is pending
```

The current development phase is **P0-4 execution and evidence collection**. Do not describe P0-4 as complete until qualifying CUDA bf16 runs have been reviewed and the repository documentation has been updated from actual artifacts.

## Read before changing anything

Read these files in order:

```text
README.md
docs/HANDOFF.md
docs/VALIDATION_STATUS.md
docs/TESTING.md
docs/KNOWN_LIMITATIONS.md
docs/P0_4_PLAN.md
docs/CODEX_P0_4_HANDOFF.md
docs/LOGGING_POLICY.md
docs/REPOSITORY_AUDIT.md
docs/RELEASE_CHECKLIST.md
```

For model or oracle changes, also inspect:

```text
multiscreen_transformers/configuration_multiscreen.py
multiscreen_transformers/modeling_multiscreen.py
oracle/paper_math_oracle.py
oracle/test_against_hf_port.py
p0_2_three_way_minimal/test_three_way_minimal.py
```

## Non-negotiable correctness contracts

Do not break these contracts:

1. Paper Trim parameterization and HF/reference parameterization are equivalent only under:

   ```text
   s_r_paper = -s_r_hf
   ```

2. `oracle/paper_math_oracle.py` is a dense equation-oriented correctness oracle. It is not a speed reference.

3. Oracle compute modes have different purposes:

   ```text
   stable paper/oracle checks:
     mipe_compute_dtype="fp32"
     softmask_compute_dtype="fp32"

   low-precision reference compatibility:
     mipe_compute_dtype="reference"
     softmask_compute_dtype="reference"
   ```

4. The current HF implementation contains DynamicCache compatibility logic. Greedy `generate(use_cache=True)` is smoke-tested; broad generation compatibility is not.

5. The current screening implementation is dense and quadratic in sequence length. Never present its runtime or memory results as evidence of the paper's efficiency claims.

6. `tie_word_embeddings=True` is part of the architecture contract. Logits use normalized tied embeddings.

## Git workflow

- Start from an up-to-date `main` and a clean working tree.
- Do not develop directly on `main`.
- Create a focused branch for each validation or implementation step.
- Keep changes scoped. Separate model-core changes from documentation-only or experiment-only changes when practical.
- Do not rewrite or discard user changes.
- Do not commit checkpoints, `outputs/`, caches, raw large logs, or generated model weights.
- Before opening a PR, inspect `git diff`, `git status`, and the exact files staged.

## Testing policy

Always run the tests relevant to the files changed.

Minimum baseline after a fresh clone or documentation/experiment-harness change:

```bash
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
```

Also run syntax/config checks for the P0-4 harness:

```bash
python -m py_compile scripts/p0_4_gpt2_context4096_smoke.py

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --validate-config-only

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --validate-config-only
```

If any of the following change, rerun P0-1 and P0-2 at the strongest feasible level, including CUDA bf16 where available:

```text
multiscreen_transformers/modeling_multiscreen.py
multiscreen_transformers/configuration_multiscreen.py
oracle/paper_math_oracle.py
cache/generation handling
state_dict conversion
position or mask behavior
```

If cache/generation behavior changes, add or extend a focused generation/cache test and run a P0-3 or P0-4 diagnostic in addition to the P0 quick checks.

## P0-4 execution rules

P0-4 qualification is intentionally strict. A run is qualifying only when all of these are true:

```text
GPT-2 tokenizer vocabulary = 50,257
sequence length = 4,096
device = CUDA
AMP dtype = bf16
optimizer steps >= 50
finite train losses
finite gradient norms
probe loss decreases by configured threshold
save_pretrained / from_pretrained passes
loaded logits pass configured tolerances
generate(use_cache=True) appends tokens
manual cache split matches full-forward suffix
metrics.jsonl and summary.json are written
```

A CPU, shorter-context, different-dtype, or fewer-step run is diagnostic only, even when all diagnostic checks pass. It must not be used to mark P0-4 complete.

Execution order:

1. Verify environment and baseline quick tests.
2. Run both P0-4 config preflights.
3. Optionally run a reduced Psi=8 diagnostic.
4. Run qualifying Psi=8.
5. Review loss, memory, reload, generation, and cache artifacts.
6. Run qualifying Psi=16 only after Psi=8 passes and memory headroom is understood.
7. Sanitize and record compact results under `docs/validation_results/`.
8. Update status documentation only from reviewed evidence.

Do not silently weaken the qualifying criteria to work around an OOM or environment limitation. Preserve failure artifacts and report the run as blocked or failed.

## Change strategy during P0-4

During the first P0-4 local pass:

- Prefer fixing environment, data, tokenizer, config, logging, or harness issues before touching the model core.
- Do not change `modeling_multiscreen.py`, the paper oracle, cache semantics, or state-dict conversion merely to make a long-context run pass.
- If evidence indicates a model-core defect, stop the qualifying sequence, create a focused diagnosis, explain which P0 contract is implicated, and rerun the required P0 comparison suite after any fix.
- Do not start P1 LoRA/QLoRA/Unsloth work until the P0-4 outcome is explicitly recorded or the user changes priorities.

## Validation records

Compact accepted summaries belong in:

```text
docs/validation_results/
```

Follow `docs/LOGGING_POLICY.md`. Keep machine-readable JSON and a human-readable Markdown summary. Remove local absolute paths, secrets, usernames, cache paths, and large raw logs before committing.

Never commit:

```text
outputs/
checkpoint directories
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
local Hugging Face caches
wandb/
large raw terminal logs
```

## Reporting format

At each checkpoint, report:

```text
Checkpoint:
Commands executed:
Files changed:
Tests and results:
Artifacts produced:
Current blocker or risk:
Next checkpoint:
```

At completion, report:

```text
変更ファイル:
追加ファイル:
実行テスト:
結果:
未確認:
次にやるべきこと:
```

Be precise about what was actually executed. Never infer a GPU pass from CI, static validation, or a diagnostic run.
