# P0 Release / Handoff Checklist

This checklist is for tagging or resuming work from a P0-qualified state.

## Before tagging

- [ ] `README.md` links to `docs/HANDOFF.md`, `docs/VALIDATION_STATUS.md`, `docs/TESTING.md`, and `docs/KNOWN_LIMITATIONS.md`.
- [ ] `docs/VALIDATION_STATUS.md` reflects the latest P0-1/P0-2/P0-3 results.
- [ ] `docs/validation_results/` contains sanitized result files only; no local absolute paths unless intentional.
- [ ] Root `LICENSE` copyright line matches this repository.
- [ ] `THIRD_PARTY_NOTICES.md` lists vendored reference code and tokenizer caveats.
- [ ] `pyproject.toml` project version and `multiscreen_transformers.__version__` are aligned.
- [ ] `.gitignore` is present.
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
```

## Suggested tag

```bash
git tag p0-qualified-v0
git push origin p0-qualified-v0
```

## Suggested release note

```text
P0-qualified unofficial HF Multiscreen implementation.
Validated: paper oracle equivalence, three-way reference equivalence,
DynamicCache-compatible generation smoke, and TinyStories Ψ=8/16 bf16 smoke training.
Not validated: paper-scale reproduction, long-context efficiency, Triton/windowed kernels,
PEFT/LoRA/Unsloth, and production serving.
```
