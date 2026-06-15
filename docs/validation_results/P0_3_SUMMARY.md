# P0-3 Summary: TinyStories Ψ=8/16 smoke training

## Status

```text
passed
```

## Goal

Verify short-run TinyStories training stability for Ψ=8 and Ψ=16 using the P0-qualified HF implementation.

## Command shape

```text
dataset: roneneldan/TinyStories train[:20000]
tokenizer: tokenizers/tinystories_spm768
packed chunks: 2032
seq_len: 128
device: cuda:0
amp dtype: bf16
batch_size: 4
steps-per-psi: 8:40,16:25
```

## Results

### Ψ=8

```text
params: 966,850
steps: 40
initial_probe_loss: 8.2159
final_probe_loss: 4.3126
absolute_drop: 3.9032
relative_drop: 47.5085%
save/load: passed
generate/use_cache: passed
cache split after training: passed
```

Training trace excerpt:

```text
step=0001 loss=8.1473 grad_norm=10.1225
step=0005 loss=8.0846 grad_norm=9.9547
step=0010 loss=7.6630 grad_norm=9.5738
step=0015 loss=7.4944 grad_norm=9.0501
step=0020 loss=6.1084 grad_norm=7.2572
step=0025 loss=7.0118 grad_norm=8.3237
step=0030 loss=6.8214 grad_norm=7.6891
step=0035 loss=6.5980 grad_norm=7.5221
step=0040 loss=4.3874 grad_norm=3.7362
```

### Ψ=16

```text
params: 14,877,442
steps: 25
initial_probe_loss: 15.8997
final_probe_loss: 5.9280
absolute_drop: 9.9716
relative_drop: 62.7160%
save/load: passed
generate/use_cache: passed
cache split after training: passed
```

Training trace excerpt:

```text
step=0001 loss=15.7802 grad_norm=25.4803
step=0005 loss=15.0056 grad_norm=24.0464
step=0010 loss=13.8717 grad_norm=22.8727
step=0015 loss=13.0933 grad_norm=21.7925
step=0020 loss=8.0660 grad_norm=13.8873
step=0025 loss=9.8976 grad_norm=16.1442
```

## Interpretation

P0-3 confirms that the HF implementation can train briefly on TinyStories in bf16 for Ψ=8 and Ψ=16, with finite loss/gradients, significant probe-loss decrease, checkpoint save/load, and DynamicCache-compatible generation after training.

## Not covered

```text
long training stability
generalization quality
paper-scale pretraining
larger context / GPT-2 vocab
throughput or efficiency
```
