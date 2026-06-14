# P0-4 Multiscreen Ψ=16 GPT-2 ctx4096 config

This directory contains a static Hugging Face-style config for the P0-4 GPT-2-vocab, context-4096 smoke gate.

It is a **scaffold**, not a recorded pass result. Use it with `scripts/p0_4_gpt2_context4096_smoke.py` or load it manually after registering the local Multiscreen AutoClasses.

## Suggested smoke command

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

For Ψ=16, run this only after Ψ=8 has passed on the same environment.

## Scope

This config is intended to test construction, GPT-2 vocab sizing, context-4096 forward/backward stability, checkpoint reload, cache split, and greedy cache-enabled generation. It does not imply paper-scale pretraining, long-context efficiency, or serving readiness.
