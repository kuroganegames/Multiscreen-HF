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

See [P0_5_C2_PLAN.md](P0_5_C2_PLAN.md) and the proposed
[MiPE position ADR](adr/ADR-0001-mipe-position-semantics.md), plus the compact
[C2 result](validation_results/P0_5_C2_SUMMARY.md). The local gate has passed,
but C2 remains `REVIEW_REQUIRED` and unaccepted until review and merge.

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
