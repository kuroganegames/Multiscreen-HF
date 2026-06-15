# Multiscreen HF P0

Unofficial Hugging Face Transformers-compatible implementation of the **Multiscreen** architecture, with a paper-math oracle and P0 validation tests.

This repository is a research artifact. It is not an official implementation of the Multiscreen paper, and it does not claim paper-scale performance reproduction. The current status is best described as:

> **P0-qualified research implementation:** small-shape formula equivalence, three-way reference equivalence, cache/generation compatibility, and TinyStories Ψ=8/16 smoke training have passed.

For the detailed validation record, see [docs/VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md).

For development restart context, see [docs/HANDOFF.md](docs/HANDOFF.md).

For compact validation run summaries, see [docs/validation_results/VALIDATION_LOG_INDEX.md](docs/validation_results/VALIDATION_LOG_INDEX.md).

For future validation logging rules, see [docs/LOGGING_POLICY.md](docs/LOGGING_POLICY.md).

For repository hygiene and release-readiness checks, see [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md).



## What is included

```text
multiscreen_transformers/       HF Transformers-compatible model/config/data code
scripts/                        tokenizer, training, smoke eval, P0-3 stability scripts
configs/                        TinyStories/debug configs
oracle/                         paper_math_oracle and HF-vs-oracle tests
p0_2_three_way_minimal/          three-way comparison against dieOD/multiscreen-pytorch
third_party/multiscreen-pytorch/ vendored reference implementation used for P0-2
tokenizers/tinystories_spm768/   small 768-vocab TinyStories tokenizer for smoke tests
docs/                           validation status, handoff notes, and reproducibility notes
```

The vendored reference implementation is included under `third_party/` for reproducibility and retains its original Apache-2.0 license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Install

A local editable install is recommended.

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

For a quick non-install workflow, set:

```bash
export PYTHONPATH=$PWD:$PWD/oracle
```

## Minimal usage

```python
import torch
from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM

config = MultiscreenConfig.from_psi(
    8,
    vocab_size=768,
    max_seq_len=128,
    key_dim=16,
    value_dim=64,
)
model = MultiscreenForCausalLM(config).eval()
input_ids = torch.randint(0, config.vocab_size, (1, 16))
with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
print(out.logits.shape)
```

For AutoClass loading in the same process:

```python
from multiscreen_transformers import register_multiscreen_auto_classes
register_multiscreen_auto_classes()
```

## P0 validation commands

### P0-1: paper oracle vs HF implementation

```bash
export PYTHONPATH=$PWD:$PWD/oracle

python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
python oracle/test_against_hf_port.py --quick
python oracle/test_against_hf_port.py
python oracle/test_against_hf_port.py --device cuda:0 --dtype bf16
```

### P0-2: three-way comparison

```bash
export PYTHONPATH=$PWD:$PWD/oracle

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

### P0-3: TinyStories smoke training

If the included tokenizer is present, run:

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

To recreate the tokenizer:

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

## Current validation status

Summary:

- P0-1 complete: `paper_math_oracle` and HF implementation match on small-shape forward/loss/cache/mask sweeps.
- P0-2 complete: vendored unofficial PyTorch reference, HF implementation, and oracle match in CPU fp32 and CUDA bf16 full sweeps.
- P0-3 complete: Ψ=8 and Ψ=16 TinyStories bf16 smoke training passed, including save/load and DynamicCache-compatible generation.

Detailed records are in [docs/VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md), with P0-3 metrics in [docs/validation_results/p0_3_results.json](docs/validation_results/p0_3_results.json).

## Known limitations

Not yet validated:

- paper-scale pretraining or paper-level performance reproduction
- long-context efficiency claims
- custom Triton/windowed kernels
- PEFT/LoRA/QLoRA or Unsloth integration
- vLLM/SGLang serving
- production-scale generation compatibility such as beam search, sampling processors, streamers, or assisted generation
- packed dataset segment-isolation semantics

This implementation should be treated as a validated research baseline, not a production inference stack.

## License

The repository is provided under Apache-2.0. The vendored reference implementation under `third_party/multiscreen-pytorch/` also retains Apache-2.0 licensing.
