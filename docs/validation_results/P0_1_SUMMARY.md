# P0-1 Summary: paper_math_oracle vs HF implementation

## Status

```text
passed
```

## Goal

Verify that `paper_math_oracle` and the current HF Multiscreen implementation agree on small-shape correctness tests.

## Covered areas

```text
formula unit tests
oracle self-checks
oracle smoke tests
HF-vs-oracle shape sweep
loss and shifted-label loss
logits_to_keep
cache split
padding masks
position contract negative tests
zero relevance path
CPU fp32, CUDA bf16, CUDA fp16 quick coverage
```

## Recorded runs

### CPU fp32 quick

```text
cache_split: 10
padding_cache: 24
padding_full: 8
position_contract_negative_cache: 1
position_contract_negative_no_cache: 1
position_ids_zero: 1
shape_forward_loss: 12
shape_logits_to_keep: 24
shape_shifted_loss: 12
zero_relevance: 1
```

### CPU fp32 full

```text
cache_split: 144
padding_cache: 240
padding_full: 88
position_contract_negative_cache: 2
position_contract_negative_no_cache: 2
position_ids_zero: 2
shape_forward_loss: 60
shape_logits_to_keep: 144
shape_shifted_loss: 60
zero_relevance: 2
```

### CUDA bf16 quick

```text
cache_split: 10
padding_cache: 24
padding_full: 8
position_contract_negative_cache: 1
position_contract_negative_no_cache: 1
position_ids_zero: 1
shape_forward_loss: 12
shape_logits_to_keep: 24
shape_shifted_loss: 12
zero_relevance: 1
```

### CUDA bf16 full

```text
cache_split: 144
padding_cache: 240
padding_full: 88
position_contract_negative_cache: 2
position_contract_negative_no_cache: 2
position_ids_zero: 2
shape_forward_loss: 60
shape_logits_to_keep: 144
shape_shifted_loss: 60
zero_relevance: 2
```

### CUDA fp16 quick

```text
cache_split: 10
padding_cache: 24
padding_full: 8
position_contract_negative_cache: 1
position_contract_negative_no_cache: 1
position_ids_zero: 1
shape_forward_loss: 12
shape_logits_to_keep: 24
shape_shifted_loss: 12
zero_relevance: 1
```

## Interpretation

P0-1 confirms small-shape implementation equivalence between the paper-math oracle and the HF port, including masking, cache split, zero relevance, and basic low-precision paths.

## Not covered

```text
paper-scale training
long-context efficiency
Triton/windowed kernels
production generation matrix
```
