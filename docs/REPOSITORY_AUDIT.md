# Repository Audit

This document records a repository-hygiene and handoff-readiness audit for `kuroganegames/Multiscreen-HF`.

It is intended to complement:

- [`README.md`](../README.md)
- [`docs/HANDOFF.md`](HANDOFF.md)
- [`docs/VALIDATION_STATUS.md`](VALIDATION_STATUS.md)
- [`docs/TESTING.md`](TESTING.md)
- [`docs/KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)
- [`docs/LOGGING_POLICY.md`](LOGGING_POLICY.md)
- [`docs/validation_results/VALIDATION_LOG_INDEX.md`](validation_results/VALIDATION_LOG_INDEX.md)

## Audit summary

Status: **P0-qualified repository; suitable for continuing research development.**

The repository contains the expected implementation, oracle, validation tests, reference implementation, tokenizer artifact, validation documentation, and handoff notes for the current P0 baseline.

The repository should be treated as a **research baseline**, not a production-ready or paper-scale reproduction.

## Expected repository contents

The following files and directories are expected to exist.

```text
README.md
LICENSE
NOTICE or THIRD_PARTY_NOTICES.md
pyproject.toml
requirements.txt
.gitignore

multiscreen_transformers/
  __init__.py
  configuration_multiscreen.py
  modeling_multiscreen.py
  data.py
  compile_utils.py

scripts/
  train_tokenizer_spm.py
  train_pretrain_sft.py
  p0_3_tinystories_stability.py
  eval_smoke.py
  count_params.py
  cache_utils.py

configs/
  ...

tokenizers/tinystories_spm768/
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json or equivalent tokenizer metadata
  TOKENIZER_NOTE.txt

oracle/
  paper_math_oracle.py
  test_against_hf_port.py
  test_formula_units.py
  test_paper_math_oracle_selfcheck.py
  test_paper_math_oracle_smoke.py

p0_2_three_way_minimal/
  test_three_way_minimal.py

third_party/multiscreen-pytorch/
  LICENSE
  README.md
  multiscreen/
  tests/
  scripts/

docs/
  HANDOFF.md
  VALIDATION_STATUS.md
  TESTING.md
  KNOWN_LIMITATIONS.md
  LOGGING_POLICY.md
  REPOSITORY_AUDIT.md
  RELEASE_CHECKLIST.md
  ENVIRONMENT_TEMPLATE.md
  validation_results/
    VALIDATION_LOG_INDEX.md
    P0_1_SUMMARY.md
    P0_1_SUMMARY.json
    P0_2_SUMMARY.md
    P0_2_SUMMARY.json
    P0_3_SUMMARY.md
    P0_3_SUMMARY.json
    P0-3_COMPLETE.md
    p0_3_results.json
```

Some files may differ slightly by naming, but the repository should preserve the same information: implementation, oracle, P0-1/P0-2/P0-3 test runners, validation summaries, and handoff documentation.

## Validation state expected in the repository

The repository should document the following completed gates.

### P0-1: paper oracle vs HF implementation

Expected status: **passed**.

Expected coverage:

```text
- CPU fp32 quick
- CPU fp32 full
- CUDA bf16 quick
- CUDA bf16 full
- CUDA fp16 quick
- shape sweep
- cache split sweep
- padding mask sweep
- zero relevance path
- loss / shifted loss / logits_to_keep
- position/cache negative contract tests
```

Expected primary files:

```text
oracle/test_against_hf_port.py
docs/VALIDATION_STATUS.md
docs/validation_results/P0_1_SUMMARY.md
docs/validation_results/P0_1_SUMMARY.json
```

### P0-2: reference vs HF vs oracle

Expected status: **passed**.

Expected coverage:

```text
- CPU fp32 quick
- CPU fp32 full
- CUDA bf16 quick
- CUDA bf16 full
- prefill three-way equality
- cache split three-way equality
- layer hook comparison enabled
- reference-to-HF state_dict mapping
- HF-to-oracle parameter conversion
```

Expected primary files:

```text
p0_2_three_way_minimal/test_three_way_minimal.py
third_party/multiscreen-pytorch/
docs/VALIDATION_STATUS.md
docs/validation_results/P0_2_SUMMARY.md
docs/validation_results/P0_2_SUMMARY.json
```

### P0-3: TinyStories Ψ=8/16 smoke training

Expected status: **passed**.

Expected coverage:

```text
- Ψ=8 bf16 smoke training
- Ψ=16 bf16 smoke training
- finite loss
- finite gradient norm
- probe loss decrease
- save_pretrained / from_pretrained
- loaded logits equality
- manual cache split after load
- generate(use_cache=True)
```

Expected primary files:

```text
scripts/p0_3_tinystories_stability.py
docs/VALIDATION_STATUS.md
docs/validation_results/P0_3_SUMMARY.md
docs/validation_results/P0_3_SUMMARY.json
docs/validation_results/p0_3_results.json
docs/validation_results/P0-3_COMPLETE.md
```

## Documentation readiness

The documentation is considered handoff-ready if the following reading path works:

```text
README.md
  -> docs/HANDOFF.md
  -> docs/VALIDATION_STATUS.md
  -> docs/TESTING.md
  -> docs/KNOWN_LIMITATIONS.md
  -> docs/LOGGING_POLICY.md
  -> docs/validation_results/VALIDATION_LOG_INDEX.md
```

Each document should have a clear role:

| Document | Role |
|---|---|
| `README.md` | Public project overview and quick start |
| `docs/HANDOFF.md` | Main development-restart guide |
| `docs/VALIDATION_STATUS.md` | Detailed P0 validation state |
| `docs/TESTING.md` | Reproducibility commands |
| `docs/KNOWN_LIMITATIONS.md` | Boundaries and non-goals |
| `docs/LOGGING_POLICY.md` | Future validation-log policy |
| `docs/validation_results/VALIDATION_LOG_INDEX.md` | Compact log index |
| `docs/REPOSITORY_AUDIT.md` | Repository hygiene and handoff-readiness audit |

## Hygiene checks

Run these before tagging or before handing off to another development session.

### 1. No generated bytecode

```bash
find . \( -name '__pycache__' -o -name '*.pyc' \) -print
```

Expected output: no tracked files. If any are present:

```bash
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
find . -name '*.pyc' -delete
git rm -r --cached '**/__pycache__' 2>/dev/null || true
git rm --cached '**/*.pyc' 2>/dev/null || true
git add .
git commit -m "Clean generated Python bytecode artifacts"
```

### 2. No large model/checkpoint artifacts

```bash
find . \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -print
```

Expected output: no model checkpoint files in the Git repository.

P0-3 checkpoint outputs should remain in local `outputs/` directories or external artifact storage, not in Git.

### 3. Local Markdown links resolve

```bash
python - <<'PY'
import pathlib, re, urllib.parse
root = pathlib.Path('.')
missing = []
for md in root.rglob('*.md'):
    text = md.read_text(encoding='utf-8', errors='ignore')
    for m in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        link = m.group(1).split('#')[0]
        if not link or '://' in link or link.startswith('mailto:'):
            continue
        p = (md.parent / urllib.parse.unquote(link)).resolve()
        try:
            p.relative_to(root.resolve())
        except ValueError:
            continue
        if not p.exists():
            missing.append((str(md), m.group(1)))
if missing:
    for item in missing:
        print('missing link:', item)
    raise SystemExit(1)
print('all local markdown links ok')
PY
```

### 4. Python syntax check

```bash
python -m py_compile \
  multiscreen_transformers/*.py \
  oracle/*.py \
  scripts/*.py \
  p0_2_three_way_minimal/*.py \
  third_party/multiscreen-pytorch/multiscreen/*.py
```

### 5. P0 quick checks

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

## Version and license consistency

Before tagging, check:

```bash
grep -n "version" pyproject.toml multiscreen_transformers/__init__.py
```

The package version in `pyproject.toml` and `multiscreen_transformers/__init__.py` should match.

Also check root license ownership. The root `LICENSE` should describe the current `Multiscreen-HF` repository. The vendored reference implementation should retain its own license under `third_party/multiscreen-pytorch/`.

## GitHub Actions check

If `.github/workflows/p0-smoke.yml` exists, it should run at least:

```text
- syntax check
- oracle formula/self/smoke tests
- P0-1 quick
- P0-2 quick
```

This is not a substitute for GPU bf16 validation, but it helps prevent accidental breakage in CPU-compatible paths.

## Known acceptable limitations at this audit stage

The following limitations are acceptable for the current P0-qualified baseline if they are documented in `docs/KNOWN_LIMITATIONS.md`:

```text
- no paper-scale pretraining validation
- no long-context efficiency claim
- no Triton/windowed kernel validation
- no PEFT/LoRA/QLoRA/Unsloth integration yet
- no vLLM/SGLang serving validation
- limited generation mode testing
- no production readiness claim
```

## Suggested tag

If all checks pass and the current repository corresponds to the P0-qualified artifact, create:

```bash
git tag p0-qualified-v0
git push origin p0-qualified-v0
```

Suggested release note:

```text
P0-qualified unofficial HF Multiscreen implementation:
- paper oracle equivalence
- three-way reference equivalence
- DynamicCache-compatible generation smoke
- TinyStories Ψ=8/16 bf16 smoke training
```

## Audit conclusion

If the expected files exist, P0 summaries are present, bytecode/checkpoints are absent, and P0 quick checks pass, the repository is ready for:

```text
- public GitHub baseline use
- development handoff
- P0-4 GPT-2 vocab + context 4096 smoke validation
- later P1 ecosystem work
```
