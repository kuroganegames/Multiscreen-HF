# Multiscreen HF P0

Unofficial Hugging Face Transformers-compatible implementation of the **Multiscreen** architecture, with a paper-math oracle and P0 validation tests.

This repository is a research artifact. It is not an official implementation of the Multiscreen paper, and it does not claim paper-scale performance reproduction.

Current status:

> **P0-qualified research implementation based on P0-1/P0-2/P0-3.** The P0-4 GPT-2-vocab, context-4096 harness is merged and CPU-diagnosed, while qualifying CUDA bf16 Psi=8/Psi=16 runs remain pending.

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

### P0-4: harness merged; qualifying execution pending

PR #3 added:

```text
scripts/p0_4_gpt2_context4096_smoke.py
configs/p0_4_multiscreen_psi8_gpt2_ctx4096/
configs/p0_4_multiscreen_psi16_gpt2_ctx4096/
docs/P0_4_PLAN.md
docs/P0_4_RESULTS_TEMPLATE.md
CI static preflight and tiny CPU end-to-end diagnostic
```

A qualifying P0-4 run requires:

```text
GPT-2 vocabulary = 50,257
context = 4,096
CUDA
bf16
at least 50 optimizer steps
finite loss and gradients
probe-loss decrease
save/load comparison
generation with cache
manual cache-split comparison
```

A CPU, reduced-context, different-dtype, or shorter run remains diagnostic and must not be reported as a P0-4 pass.

Static preflight:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --validate-config-only

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --validate-config-only
```

Qualifying Psi=8 command:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096
```

Run Psi=16 only after reviewing a qualifying Psi=8 result:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

## Local Codex continuation

After cloning, start Codex from the repository root:

```bash
codex
```

Codex reads [AGENTS.md](AGENTS.md) before working. The ready-to-paste `/goal` prompt in [docs/CODEX_P0_4_HANDOFF.md](docs/CODEX_P0_4_HANDOFF.md) guides environment audit, baseline tests, Psi=8/Psi=16 qualification, evidence sanitization, documentation updates, and PR preparation.

If `/goal` is unavailable:

```bash
codex features enable goals
codex
```

## Known limitations

Not yet validated:

- qualifying P0-4 CUDA bf16 context-4096 execution
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
