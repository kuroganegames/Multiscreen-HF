# Testing Commands

For project handoff context and recommended next steps, see [HANDOFF.md](HANDOFF.md).

For the long-running local Codex workflow that executes these checks and records P0-4 evidence, see [CODEX_P0_4_HANDOFF.md](CODEX_P0_4_HANDOFF.md).

Run all commands from the repository root unless noted otherwise.

## Setup

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle
```

## P0.5-C1

P0.5-C1 checks the paper architecture, initialization, all-scale state shapes,
normalized tied embeddings, config round trips, and packed-text contract. Its
Psi=8/16/32/48/64 model construction is meta-only and does not allocate real
paper-scale weights.

```bash
python -m py_compile \
  scripts/generate_paper_scale_manifest.py \
  tests/test_paper_architecture_contract.py \
  tests/test_paper_initialization_contract.py \
  tests/test_packed_text_contract.py

python -m unittest discover -s tests -p 'test_paper_architecture_contract.py' -v
python -m unittest discover -s tests -p 'test_paper_initialization_contract.py' -v
python -m unittest discover -s tests -p 'test_packed_text_contract.py' -v

python scripts/generate_paper_scale_manifest.py \
  --check docs/validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json
python -m json.tool \
  docs/validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json \
  >/dev/null
```

See [P0_5_C1_PLAN.md](P0_5_C1_PLAN.md) for the independent count derivation and
[P0_5_C1_SUMMARY.md](validation_results/P0_5_C1_SUMMARY.md) for the focused
result. C1 was reviewed and merged as PR #9.

## P0.5-C2

P0.5-C2 separates the literal paper MiPE rule
(`paper_absolute`) from the historical compatibility rule
(`reference_mod_after_wrap_boundary`). The reference boundary is
inclusive: wrapping begins at
`mipe_reference_wrap_boundary`. Missing fields migrate to reference
mode with the boundary resolved from `max_position_embeddings`;
paper mode is always an explicit choice.

Run the focused C2 position/config/cache suite first:

```bash
export PYTHONPATH=$PWD:$PWD/oracle

python -m py_compile \
  multiscreen_transformers/configuration_multiscreen.py \
  multiscreen_transformers/modeling_multiscreen.py \
  oracle/paper_math_oracle.py \
  tests/test_mipe_position_cache_contract.py

python -m unittest discover \
  -s tests \
  -p 'test_mipe_position_cache_contract.py' \
  -v

python -m unittest discover -s tests -p 'test_paper_architecture_contract.py' -v
```

Then run the stable oracle checks and the strongest required P0-1/P0-2 CPU and
CUDA bf16 comparisons:

```bash
python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py

python oracle/test_against_hf_port.py
python oracle/test_against_hf_port.py --device cuda:0 --dtype bf16

python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle

python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --device cuda:0 \
  --dtype bf16
```

Stable paper/long-boundary correctness uses fp32 auxiliary MiPE and Softmask
math. The low-precision `reference` auxiliary mode is a separate
compatibility lane; bf16 may collapse distinct large positions and fp16 may
become non-finite. A direct test at position 131,071 does not allocate a dense
131K forward and must not be reported as long-context feasibility or
efficiency.

See [P0_5_C2_PLAN.md](P0_5_C2_PLAN.md), the accepted
[MiPE position ADR](adr/ADR-0001-mipe-position-semantics.md), and the compact
[C2 result](validation_results/P0_5_C2_SUMMARY.md). C2 was merged as PR #10;
its separate CUDA-autocast cache-dtype correction was merged as PR #11.

## P1-preflight B

P1-preflight B tests the supported non-reentrant Transformers
gradient-checkpointing API. Run the focused suite under exact isolated 4.57.6
and 5.14.1 lanes; do not substitute a floating resolver result.
Set `TF4576_PYTHON` and `TF5141_PYTHON` to the corresponding exact-lane
Python executables. The commands fail early if either variable is unset or the
executable does not exist.

```bash
set -euo pipefail
: "${TF4576_PYTHON:?set TF4576_PYTHON to the exact 4.57.6 lane executable}"
: "${TF5141_PYTHON:?set TF5141_PYTHON to the exact 5.14.1 lane executable}"
test -x "$TF4576_PYTHON"
test -x "$TF5141_PYTHON"

export PYTHONPATH=$PWD:$PWD/oracle:$PWD/third_party/multiscreen-pytorch

"$TF4576_PYTHON" -m unittest discover \
  -s tests \
  -p 'test_gradient_checkpointing_contract.py' \
  -v

"$TF5141_PYTHON" -m unittest discover \
  -s tests \
  -p 'test_gradient_checkpointing_contract.py' \
  -v
```

Then run the full P0-1/P0-2 CPU fp32 and CUDA bf16 commands below. The
Stage 3 CUDA training-path checks use the recorded 4.57.6 lane.
Before running them, set `HF_CACHE_DIR` to an existing local Hugging Face
cache and `STAGE3_OUTPUT_ROOT` to an absolute writable path outside the
repository. The commands fail early if either variable is unset.

```bash
set -euo pipefail
: "${TF4576_PYTHON:?set TF4576_PYTHON to the exact 4.57.6 lane executable}"
test -x "$TF4576_PYTHON"
: "${HF_CACHE_DIR:?set HF_CACHE_DIR to an existing local cache}"
: "${STAGE3_OUTPUT_ROOT:?set STAGE3_OUTPUT_ROOT outside the repository}"
test -d "$HF_CACHE_DIR"
case "$STAGE3_OUTPUT_ROOT" in
  /*) ;;
  *) echo "STAGE3_OUTPUT_ROOT must be absolute" >&2; exit 2 ;;
esac
case "$STAGE3_OUTPUT_ROOT/" in
  "$PWD/"*) echo "STAGE3_OUTPUT_ROOT must be outside the repository" >&2; exit 2 ;;
esac
mkdir -p "$STAGE3_OUTPUT_ROOT"

"$TF4576_PYTHON" scripts/p0_3_tinystories_stability.py \
  --cache-dir "$HF_CACHE_DIR" \
  --psi 8 16 \
  --steps-per-psi 8:40,16:25 \
  --seq-len 128 \
  --batch-size 4 \
  --max-texts 20000 \
  --max-train-tokens 262144 \
  --device cuda:0 \
  --amp-dtype bf16 \
  --model-compute-dtype fp32 \
  --gradient-checkpointing true \
  --output-dir "$STAGE3_OUTPUT_ROOT/p0-3-checkpointed"

"$TF4576_PYTHON" scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --cache-dir "$HF_CACHE_DIR" \
  --device cuda:0 \
  --amp-dtype bf16 \
  --seq-len 1024 \
  --steps 4 \
  --gradient-accumulation-steps 1 \
  --microbatch-size 1 \
  --gradient-checkpointing true \
  --fused-adamw false \
  --max-texts 20000 \
  --max-train-tokens 32769 \
  --output-dir "$STAGE3_OUTPUT_ROOT/p0-4-checkpointed-diagnostic"
```

The second command must produce `P0-4_DIAGNOSTIC_COMPLETE.md`, not a qualifying
completion marker. See [P1_PREFLIGHT_B_PLAN.md](P1_PREFLIGHT_B_PLAN.md) and the
[Stage 3 result](validation_results/P1_PREFLIGHT_B_SUMMARY.md). Focused PR #12
was reviewed and merged; the commands remain the accepted regression recipe.

## P0.5-C3

P0.5-C3 separates the exact paper recipe, pinned data identity, reduced CUDA
operation, and exact-peak exposure into four evidence lanes. The offline unit
suite mocks Hub access; the data and CUDA commands require the exact pinned
assets and fail closed on a revision, shard hash, library version, fingerprint,
token count, or contract mismatch.

```bash
python -m py_compile \
  scripts/p0_5_c3_paper_training_contract.py \
  tests/test_paper_training_contract.py

python -m unittest discover \
  -s tests \
  -p 'test_paper_training_contract.py' \
  -v

python scripts/p0_5_c3_paper_training_contract.py --mode contract
```

Run the focused suite in separate exact Transformers 4.57.6 and 5.14.1
environments. The `gradient-checkpointing-compat` CI matrix installs and
verifies both exact versions, then runs this same C3 focused suite in each lane.

For the pinned data and CUDA bf16 lanes, set an existing Hugging Face cache and
an absolute output root outside every Git worktree. Every leaf output directory
must be new: the harness refuses to overwrite a previous success or failure.

```bash
set -euo pipefail
: "${HF_CACHE_DIR:?set HF_CACHE_DIR to an existing Hugging Face cache}"
: "${STAGE4_OUTPUT_ROOT:?set STAGE4_OUTPUT_ROOT outside every Git worktree}"
test -d "$HF_CACHE_DIR"
case "$STAGE4_OUTPUT_ROOT" in
  /*) ;;
  *) echo "STAGE4_OUTPUT_ROOT must be absolute" >&2; exit 2 ;;
esac
case "$STAGE4_OUTPUT_ROOT/" in
  "$PWD/"*) echo "STAGE4_OUTPUT_ROOT must be outside the repository" >&2; exit 2 ;;
esac

python scripts/p0_5_c3_paper_training_contract.py \
  --mode data \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/data"

CUDA_VISIBLE_DEVICES=0 python scripts/p0_5_c3_paper_training_contract.py \
  --mode operational \
  --psi 8 \
  --device cuda:0 \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/cuda/psi8/operational"

CUDA_VISIBLE_DEVICES=0 python scripts/p0_5_c3_paper_training_contract.py \
  --mode peak-exposure \
  --psi 8 \
  --device cuda:0 \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/cuda/psi8/peak_exposure"
```

Inspect every Psi=8 metric, completion marker, memory value, and any preserved
`failure.json` before proceeding. Only after Psi=8 passes and memory headroom
is understood, run the corresponding Psi=16 lanes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/p0_5_c3_paper_training_contract.py \
  --mode operational \
  --psi 16 \
  --device cuda:0 \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/cuda/psi16/operational"

CUDA_VISIBLE_DEVICES=0 python scripts/p0_5_c3_paper_training_contract.py \
  --mode peak-exposure \
  --psi 16 \
  --device cuda:0 \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/cuda/psi16/peak_exposure"
```

The operational lane is fixed at context 4096, microbatch 1, accumulation 2,
three optimizer updates, reduced two-update warmup, and diagnostic LR 0.0006.
The separate peak lane performs one bounded context-4096 update at the exact
paper peak LR 0.0625. Both use CUDA bf16, paper-absolute MiPE, fp32 auxiliary
MiPE/Softmask math, AdamW betas `(0.9, 0.95)`, zero weight decay, the explicitly
labeled repository epsilon `1e-8`, non-fused AdamW, supported non-reentrant
checkpointing, and no clipping. Neither lane reproduces the paper global batch,
duration, corpus selection, training precision, quality, or efficiency.

The inspected result is recorded in
[P0_5_C3_SUMMARY.md](validation_results/P0_5_C3_SUMMARY.md), with archive
identity and retention state in
[P0_5_C3_EVIDENCE_ARCHIVE.json](validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json).
Stage 4 remains `REVIEW_REQUIRED` until its focused PR is reviewed and merged.

## P0-1

```bash
python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
python oracle/test_against_hf_port.py --quick
python oracle/test_against_hf_port.py
python oracle/test_against_hf_port.py --device cuda:0 --dtype bf16
python oracle/test_against_hf_port.py --device cuda:0 --dtype fp16 --quick
```

## P0-2

```bash
python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --quick

python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle

python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --device cuda:0 \
  --dtype bf16
```

## P0-3

```bash
python scripts/p0_3_tinystories_stability.py \
  --tokenizer-path tokenizers/tinystories_spm768 \
  --cache-dir /path/to/hf_cache \
  --device cuda:0 \
  --amp-dtype bf16 \
  --seq-len 128 \
  --batch-size 4 \
  --steps-per-psi 8:40,16:25 \
  --output-dir outputs/p0_3_tinystories_stability
```

If tokenizer files are missing:

```bash
python scripts/train_tokenizer_spm.py \
  --dataset_name roneneldan/TinyStories \
  --split train \
  --text_column text \
  --vocab_size 768 \
  --max_samples 200000 \
  --model_max_length 512 \
  --output_dir tokenizers/tinystories_spm768 \
  --cache_dir /path/to/hf_cache
```

## P0-4

P0-4 qualifying CUDA bf16 runs passed for Psi=8 and Psi=16. Reviewed results and raw-artifact hashes are recorded in [P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md) and [P0_4_SUMMARY.json](validation_results/P0_4_SUMMARY.json); the commands below remain the strict reproduction procedure.

Static config preflight; this does not download the tokenizer or dataset:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --validate-config-only

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --validate-config-only
```

Optional non-qualifying 1024-token diagnostic:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --seq-len 1024 \
  --steps 2 \
  --gradient-accumulation-steps 1 \
  --output-dir outputs/p0_4_psi8_ctx1024_diagnostic
```

Qualifying Psi=8 reproduction:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096
```

Run the qualifying Psi=16 reproduction only after the new Psi=8 output is reviewed:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

A qualifying run uses GPT-2 vocab 50,257, sequence length 4,096, CUDA bf16, and at least 50 optimizer steps. Reduced runs write `P0-4_DIAGNOSTIC_COMPLETE.md`, not `P0-4_COMPLETE.md`.

For a future reproduction, retain its ignored raw artifacts under a distinct output directory, compare them with the accepted historical record, and add a new compact sanitized record rather than overwriting the existing evidence. Rerun the P0-1/P0-2 quick suite and both config preflights.
