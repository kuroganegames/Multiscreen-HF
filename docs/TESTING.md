# Testing Commands

For project handoff context and recommended next steps, see [HANDOFF.md](HANDOFF.md).

Run all commands from the repository root unless noted otherwise.

## Setup

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle
```

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

P0-4 is the GPT-2 vocabulary + context-4096 short smoke. See [P0_4_PLAN.md](P0_4_PLAN.md) and [P0_4_RESULTS_TEMPLATE.md](P0_4_RESULTS_TEMPLATE.md) before recording a pass.

Run a cheap diagnostic first:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --psi-values 8 \
  --steps-per-psi 8:3 \
  --seq-len 1024 \
  --microbatch-size 1 \
  --grad-accum-steps 1 \
  --amp-dtype bf16 \
  --synthetic-text \
  --output-dir outputs/p0_4_debug_psi8_ctx1024
```

Then run the actual Ψ=8 gate:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --psi-values 8 \
  --steps-per-psi 8:50 \
  --seq-len 4096 \
  --microbatch-size 1 \
  --grad-accum-steps 8 \
  --amp-dtype bf16 \
  --output-dir outputs/p0_4_gpt2_ctx4096_psi8
```

Only after Ψ=8 passes, run Ψ=16:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --psi-values 16 \
  --steps-per-psi 16:25 \
  --seq-len 4096 \
  --microbatch-size 1 \
  --grad-accum-steps 8 \
  --amp-dtype bf16 \
  --gradient-checkpointing \
  --output-dir outputs/p0_4_gpt2_ctx4096_psi16
```

A P0-4 pass must generate `metrics.jsonl`, `p0_4_results.json`, and `P0-4_COMPLETE.md` under `outputs/`. Do not commit those output files or checkpoints.
