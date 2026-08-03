# Multiscreen HF P0

Unofficial Hugging Face Transformers-compatible implementation of the **Multiscreen** architecture, with a paper-math oracle and P0 validation tests.

This repository is a research artifact. It is not an official implementation of the Multiscreen paper, and it does not claim paper-scale performance reproduction.

Current status:

> **P0-qualified research implementation through P0-4.** Reviewed local CUDA bf16 GPT-2-vocabulary, context-4096 qualifying runs passed for both Psi=8 and Psi=16. This remains a correctness/stability smoke result, not a paper-scale or efficiency result.

## Start here

- Development restart: [docs/HANDOFF.md](docs/HANDOFF.md)
- Local Codex `/goal` continuation: [docs/CODEX_P0_4_HANDOFF.md](docs/CODEX_P0_4_HANDOFF.md)
- Repository instructions for Codex: [AGENTS.md](AGENTS.md)
- Detailed validation boundary: [docs/VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md)
- Reproduction commands: [docs/TESTING.md](docs/TESTING.md)
- P0-4 execution plan: [docs/P0_4_PLAN.md](docs/P0_4_PLAN.md)
- Known limitations: [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md)
- Validation log index: [docs/validation_results/VALIDATION_LOG_INDEX.md](docs/validation_results/VALIDATION_LOG_INDEX.md)
- Logging policy: [docs/LOGGING_POLICY.md](docs/LOGGING_POLICY.md)
- Repository audit: [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md)

## What is included

```text
AGENTS.md                      Codex project instructions and validation rules
multiscreen_transformers/     HF-compatible model, config, and data code
scripts/                      tokenizer, training, P0-3, and P0-4 harnesses
configs/                      Tiny/debug/P0-3/P0-4 configs
oracle/                       paper_math_oracle and HF-vs-oracle tests
p0_2_three_way_minimal/        three-way comparison against the reference
third_party/multiscreen-pytorch/
                              vendored dieOD reference used by P0-2
tokenizers/tinystories_spm768/
                              committed tokenizer used by P0-3
docs/                         handoff, validation, testing, and result policy
```

The vendored reference retains its Apache-2.0 license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Install

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
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

## Minimum baseline checks

```bash
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

Full CPU and CUDA comparison commands are in [docs/TESTING.md](docs/TESTING.md).

## Validation status

### P0-1: complete

The dense paper-math oracle and HF implementation match under the recorded small-shape forward, loss, mask, position, and cache sweeps.

### P0-2: complete

The vendored unofficial PyTorch reference, HF implementation, and oracle match under the recorded CPU fp32 and CUDA bf16 three-way comparisons.

### P0-3: complete

Psi=8 and Psi=16 TinyStories bf16 smoke training passed, including:

```text
finite loss
finite gradient norms
probe-loss decrease
save_pretrained / from_pretrained
greedy generate(use_cache=True)
manual cache split vs full-forward suffix
```

Recorded metrics are in [docs/validation_results/p0_3_results.json](docs/validation_results/p0_3_results.json).

### P0-4: complete

Reviewed local runs passed the strict GPT-2-vocabulary, context-4096 CUDA bf16 gate for both intended model orders:

| Metric | Psi=8 | Psi=16 |
|---|---:|---:|
| parameters | 4,134,146 | 27,546,626 |
| optimizer steps | 50 | 50 |
| probe loss | 11.140747 → 4.675382 | 15.799321 → 3.495601 |
| peak CUDA allocated | 3,156,709,888 bytes | 6,622,802,944 bytes |
| reload max abs | 0 | 0 |
| cache max abs | 0 | 0.125, within configured atol/rtol |
| `qualification.qualified` | `true` | `true` |

Both runs recorded finite losses and gradients, configured probe-loss decrease, save/load, tokenizer reload, greedy generation with cache, and manual cache-split agreement within tolerance. The compact reviewed evidence and raw-artifact hashes are in [P0_4_SUMMARY.md](docs/validation_results/P0_4_SUMMARY.md) and [P0_4_SUMMARY.json](docs/validation_results/P0_4_SUMMARY.json).

The qualifying reproduction commands remain:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096
```

Review and accept the new Psi=8 artifacts and memory headroom before running
Psi=16.

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

A CPU, reduced-context, different-dtype, or shorter run remains diagnostic and must not be reported as a P0-4 pass.

## Local Codex reproduction guidance

After cloning, start Codex from the repository root:

```bash
codex
```

Codex reads [AGENTS.md](AGENTS.md) before working. The `/goal` prompt in [docs/CODEX_P0_4_HANDOFF.md](docs/CODEX_P0_4_HANDOFF.md) is retained as a strict P0-4 reproduction/requalification workflow; it is no longer the repository's pending next task.

If `/goal` is unavailable:

```bash
codex features enable goals
codex
```

## Known limitations

Not yet validated:

- paper-scale pretraining or paper-quality reproduction
- long-context retrieval at paper settings
- long-context runtime or memory efficiency
- custom Triton/windowed kernels
- PEFT/LoRA/QLoRA or Unsloth
- torch.compile stability at scale
- broad generation compatibility
- vLLM/SGLang serving
- production readiness

The current HF path is a dense PyTorch correctness baseline. Do not use it to substantiate the paper's speed claims.

## License

The repository is provided under Apache-2.0. The vendored reference under `third_party/multiscreen-pytorch/` retains its original Apache-2.0 licensing.
