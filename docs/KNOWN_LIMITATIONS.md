# Known Limitations

This repository is a P0-qualified research implementation, not a full production implementation.

## Not yet validated

- paper-scale pretraining
- 28M/286M/1.3B reproduction
- long-context retrieval benchmarks at paper settings
- long-context runtime efficiency
- fused/windowed/Triton kernels
- training throughput optimization
- PEFT/LoRA/QLoRA
- Unsloth
- torch.compile stability at scale
- vLLM/SGLang serving
- packed dataset segment isolation
- beam search and broad generation compatibility

## P0-4 validated scope

Both intended P0-4 model orders passed the recorded CUDA bf16 context-4096 qualification. This establishes short-run feasibility and stability under one recorded hardware/software environment only; it does not validate paper-scale training, retrieval quality, long-context efficiency, or cross-hardware reproducibility. See [validation_results/P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md).

A reduced-context, CPU, non-bf16, or fewer-than-50-step run is explicitly labeled diagnostic and must not be reported as a P0-4 pass.

## Dense reference path

The current HF path is still a dense PyTorch implementation for screening. It is suitable for correctness and smoke training, but it should not be used to evaluate the paper's speed claims.

At context 4096, dense similarity, mask, activation, gradient, and optimizer-state memory can be substantial. The P0-4 script records memory for diagnosability, not as an efficiency benchmark.

## P0-2 padding masks

The P0-2 three-way comparison does not test padding masks because the vendored unofficial reference implementation has no attention-mask API. Padding behavior is tested in P0-1 against the paper oracle.

## Tokenizer and data access

The included 768-vocab TinyStories tokenizer is provided for smoke-test reproducibility. It is not a claim of optimal tokenization.

P0-4 uses the GPT-2 tokenizer and TinyStories text by default. A first run needs Hub access or a populated local cache; `--tokenizer-name-or-path`, `--text-file`, and cache settings can point to local artifacts.
