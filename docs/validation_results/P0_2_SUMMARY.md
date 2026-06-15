# P0-2 Summary: three-way comparison

## Status

```text
passed
```

## Goal

Verify that these three implementations agree on small-shape minimal correctness tests:

```text
dieOD/multiscreen-pytorch reference implementation
HF multiscreen_transformers implementation
paper_math_oracle
```

## Covered areas

```text
prefill logits
external CE loss
KV cache tensors
layer hook outputs
prefix/suffix cache split
cached suffix vs full forward suffix
CPU fp32 and CUDA bf16
reference-compatible MiPE/Softmask low-precision mode
```

## Recorded runs

### CPU fp32 quick

```text
prefill_three_way: 12
cache_split_three_way: 28
```

### CPU fp32 full

```text
prefill_three_way: 45
cache_split_three_way: 237
```

### CUDA bf16 quick

```text
prefill_three_way: 12
cache_split_three_way: 28
```

### CUDA bf16 full

```text
prefill_three_way: 45
cache_split_three_way: 237
```

## Important note: bf16 full

The CUDA bf16 full run originally exposed a cache-K mismatch around the max-position / MiPE modulo branch. This was traced to oracle MiPE/Softmask helper math using stable fp32 while the reference path used activation dtype. The oracle now supports reference-compatible low-precision mode, and the CUDA bf16 full three-way comparison passes.

## Interpretation

P0-2 confirms that the HF implementation is consistent with both the paper oracle and the vendored reference implementation on small-shape prefill/cache tests.

## Not covered

```text
padding masks in the reference implementation, because the reference API does not expose attention_mask
generation matrix beyond cache split behavior
training stability, covered separately by P0-3
```
