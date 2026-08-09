# Known Limitations

This repository is a P0-qualified research implementation, not a full production implementation.

## Not yet validated

- final Level 1 core requalification and completion of the five-stage program
- qualifying Stage 5 evidence
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

P0.5-C2 uses direct scalar-position and tiny-shape tests for the paper's
long-position MiPE equation, including position 131,071. It does not execute a
dense 131K forward and provides no evidence for 131K memory feasibility,
runtime, retrieval quality, or the paper's optimized window-skipping claims.

## MiPE position and numerical modes

P0.5-C2 uses two accepted explicit serialized position semantics:

```text
paper_absolute
reference_mod_after_wrap_boundary
```

The reference transition is inclusive: a position equal to
`mipe_reference_wrap_boundary` is wrapped per head by the learned,
potentially fractional window. This compatibility rule is not algebraically
equivalent to the paper's unwrapped absolute-position equation.

Configs missing the new fields resolve to the historical reference behavior,
with `mipe_reference_wrap_boundary=max_position_embeddings`. This
preserves pre-C2 checkpoint behavior; paper semantics must be requested
explicitly. The wrap boundary is not a hard supported-position limit and is
separate from the MiPE threshold and learned screening support.

Stable paper/oracle correctness uses fp32 auxiliary MiPE and Softmask math.
The `reference` auxiliary-compute mode follows the incoming low
precision to match the unofficial implementation. At long positions, bf16 can
collapse distinct integer positions and fp16 can become non-finite, so the
low-precision reference lane is compatibility evidence rather than proof of
long-position causal correctness.

These C2 semantics were accepted by merged PR #10. PR #11 separately corrected
CUDA-autocast cache-dtype prediction and did not change the position decision.
This does not mark the full five-stage Level 1 program complete.

## Gradient-checkpointing boundary

P1-preflight B validates the supported non-reentrant full-model training path
under exact Transformers 4.57.6 and 5.14.1; focused PR #12 was reviewed and
merged. Compatibility with later untested Transformers releases is not inferred.

Transformers 5.14.1 installs an input-embedding gradient hook for checkpointed
`input_ids`, while Multiscreen uses normalized `F.embedding` lookup. A focused
test proves finite layer gradients when the first non-reentrant checkpoint
input itself does not require gradients. This does not validate frozen-base
PEFT/LoRA behavior, which remains a separate future gate. Broad Trainer,
distributed, compile, and serving integration are also unvalidated.

## Paper-training-contract boundary

P0.5-C3 is a bounded workstation smoke, not a paper-scale reproduction. Its
checked contract may represent the paper's GPT-2 vocabulary, SlimPajama family,
EOS-concatenated stream, sequence length, AdamW settings, warmup, constant
schedule, peak learning rate, and omitted clipping exactly, while the executed
CUDA bf16 diagnostics use a much smaller local batch and reduced warmup.

The pinned executable data lane uses a specifically identified third-party
SlimPajama reupload test shard because the historically resolved Cerebras Hub
repository could not be re-resolved during the dated audit. It is not claimed
to be byte-identical to the paper corpus or representative of its training
split. Hugging Face dataset fingerprints are recorded as library-state
provenance alongside revision, shard SHA-256, row hashes, and token-stream
hashes; a fingerprint alone is not treated as a content checksum.

Peak-LR exposure checks only finite values and a real bounded update. It does
not establish a loss decrease, convergence, training quality, paper-global
batch behavior, or paper training precision. The focused Stage 4 result was
reviewed and accepted by merged PR #13; acceptance does not broaden this
bounded scope.

The historical C3 evidence descriptor remains partial: exact/private retention
and an explicitly supplied evidence reviewer are still pending. Final Level 1
requalification is the current Stage 5 under
[LEVEL1_CORE_REQUALIFICATION_PLAN.md](LEVEL1_CORE_REQUALIFICATION_PLAN.md), but
it has no accepted qualifying evidence and the five-stage program is not
complete.

## P0-2 padding masks

The P0-2 three-way comparison does not test padding masks because the vendored unofficial reference implementation has no attention-mask API. Padding behavior is tested in P0-1 against the paper oracle.

## Tokenizer and data access

The included 768-vocab TinyStories tokenizer is provided for smoke-test reproducibility. It is not a claim of optimal tokenization.

P0-4 uses the GPT-2 tokenizer and TinyStories text by default. A first run needs Hub access or a populated local cache; `--tokenizer-name-or-path`, `--text-file`, and cache settings can point to local artifacts.
