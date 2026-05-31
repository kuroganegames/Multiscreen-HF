# P0-3 TinyStories Smoke Training Result

## Result

Passed.

## Confirmed behavior

- TinyStories text was tokenized and packed.
- Training loss stayed finite.
- Gradients stayed finite.
- Probe-batch loss decreased.
- `save_pretrained` / `from_pretrained` preserved logits.
- `generate()` worked with cache enabled.
- Cached suffix logits matched full forward suffix logits after training.

## Per-Psi metrics

### Psi=8

- params: 966,850
- steps: 40
- seq_len: 128
- batch_size: 4
- amp_dtype: bf16
- initial_probe_loss: 8.215893
- final_probe_loss: 4.312645
- abs_loss_drop: 3.903248
- rel_loss_drop: 47.5085%
- save_load_logits_max_abs: 0
- cache_split_logits_max_abs: 0
- checkpoint_dir: `outputs/p0_3_tinystories_stability_dynamic_cache_patch/psi8`

### Psi=16

- params: 14,877,442
- steps: 25
- seq_len: 128
- batch_size: 4
- amp_dtype: bf16
- initial_probe_loss: 15.899660
- final_probe_loss: 5.928024
- abs_loss_drop: 9.971636
- rel_loss_drop: 62.7160%
- save_load_logits_max_abs: 0
- cache_split_logits_max_abs: 0
- checkpoint_dir: `outputs/p0_3_tinystories_stability_dynamic_cache_patch/psi16`

## Scope

This confirms short-run training stability and basic checkpoint/generation behavior.
It does not confirm paper-scale performance, long-context retrieval, or runtime efficiency.
