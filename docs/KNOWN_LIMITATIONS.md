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

## Dense reference path

The current HF path is still a dense PyTorch implementation for screening. It is suitable for correctness and smoke training, but it should not be used to evaluate the paper's speed claims.

## P0-2 padding masks

The P0-2 three-way comparison does not test padding masks because the vendored unofficial reference implementation has no attention-mask API. Padding behavior is tested in P0-1 against the paper oracle.

## Tokenizer artifact

The included 768-vocab TinyStories tokenizer is provided for smoke-test reproducibility. It is not a claim of optimal tokenization.
