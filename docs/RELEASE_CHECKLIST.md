# P0 Release / Handoff Checklist

This checklist is for tagging, handing off, or resuming work from the P0-qualified baseline.

## Before handoff or tagging

- [ ] `README.md` links to `AGENTS.md`, `docs/HANDOFF.md`, `docs/CODEX_P0_4_HANDOFF.md`, `docs/VALIDATION_STATUS.md`, `docs/TESTING.md`, and `docs/KNOWN_LIMITATIONS.md`.
- [ ] Root `AGENTS.md` reflects the current development phase and required test policy.
- [ ] `docs/HANDOFF.md` identifies the actual next gate rather than an already completed task.
- [ ] `docs/VALIDATION_STATUS.md` reflects the latest accepted P0-1/P0-2/P0-3/P0-4 evidence.
- [ ] P0-4 is not marked complete from static validation, a CPU diagnostic, reduced context, non-bf16 execution, or fewer than 50 optimizer steps.
- [ ] `docs/validation_results/` contains sanitized compact result files only; no secrets or unnecessary local absolute paths.
- [ ] `docs/validation_results/VALIDATION_LOG_INDEX.md` links every accepted summary.
- [ ] Root `LICENSE` copyright line matches this repository.
- [ ] `THIRD_PARTY_NOTICES.md` lists vendored reference code and tokenizer/data caveats.
- [ ] `pyproject.toml` project version and `multiscreen_transformers.__version__` are aligned.
- [ ] `.gitignore` is present and still excludes outputs, checkpoints, caches, and generated weights.
- [ ] No checkpoints or output directories are committed.
- [ ] No `__pycache__`, `.git` directories under `third_party/`, or local cache directories are committed.

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

## P0-4 result acceptance

Before changing P0-4 from pending to complete, verify from the actual local artifacts:

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
- save/load logits comparison passed
- generate(use_cache=True) passed
- manual cache split comparison passed
- metrics.jsonl and summary.json hashes recorded
- compact Markdown and JSON summaries sanitized and committed
```

If Psi=8 passes but Psi=16 cannot complete on available hardware, record a partial or blocked outcome rather than an overall pass.

## Suggested baseline tag

The existing P0-1/P0-2/P0-3 baseline can use:

```bash
git tag p0-qualified-v0
git push origin p0-qualified-v0
```

Use a new tag or release name for a future accepted P0-4 state rather than moving an existing tag.

## Suggested current baseline release note

```text
P0-qualified unofficial HF Multiscreen implementation.
Validated: paper oracle equivalence, three-way reference equivalence,
DynamicCache-compatible generation smoke, and TinyStories Psi=8/16 bf16 smoke training.
P0-4 harness merged; qualifying GPT-2-vocab context-4096 CUDA bf16 execution pending.
Not validated: paper-scale reproduction, long-context efficiency, Triton/windowed kernels,
PEFT/LoRA/Unsloth, broad generation, and production serving.
```
