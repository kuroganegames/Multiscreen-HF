# Repository Audit

This audit records the repository state used to resume development from the P0-qualified baseline and to add the P0-4 scaffold.

## Current repository role

The repository is an unofficial Hugging Face Transformers-compatible Multiscreen implementation. It is a P0-qualified research baseline, not an official implementation, optimized long-context kernel, serving stack, or paper-scale reproduction.

## Key directories

```text
multiscreen_transformers/       HF-compatible config/model/data utilities
oracle/                         dense paper-math oracle and P0-1 tests
p0_2_three_way_minimal/          three-way P0-2 reference/HF/oracle comparison
third_party/multiscreen-pytorch/ vendored unofficial reference used by P0-2
scripts/                        tokenizer, training, smoke, and validation helpers
configs/                        smoke/training config scaffolds
docs/                           handoff, validation, testing, and result records
tokenizers/tinystories_spm768/   TinyStories 768-token smoke tokenizer
```

## P0-qualified baseline files

Treat these as baseline-sensitive files. If any of them change, rerun at least the P0-1 and P0-2 quick checks before merging:

```text
multiscreen_transformers/modeling_multiscreen.py
multiscreen_transformers/configuration_multiscreen.py
oracle/paper_math_oracle.py
oracle/test_against_hf_port.py
p0_2_three_way_minimal/test_three_way_minimal.py
scripts/p0_3_tinystories_stability.py
docs/VALIDATION_STATUS.md
docs/HANDOFF.md
```

## Current P0 status

```text
P0-1: complete
P0-2: complete
P0-3: complete
P0-4: scaffold added; validation not yet recorded here
```

P0-4 should not be marked complete until `scripts/p0_4_gpt2_context4096_smoke.py` passes for the intended Ψ values and the existing P0 quick checks still pass.

## Audit notes

- The current implementation is dense PyTorch and correctness-oriented.
- `paper_math_oracle.py` is a dense equation reference, not a performance reference.
- The `s_r` parameterization conversion must remain `s_r_paper = -s_r_hf`.
- DynamicCache-compatible greedy generation is smoke-validated, but broader generation modes are not.
- `outputs/`, checkpoints, dataset caches, tokenizer caches, and local absolute-path logs should not be committed.
