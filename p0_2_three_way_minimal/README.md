# P0-2: minimal three-way comparison

This package adds a minimal P0-2 test:

```text
original dieOD/multiscreen-pytorch
  == local HF Transformers port
  == paper_math_oracle
```

The test uses tiny shapes only. It checks:

- prefill logits
- external next-token CE loss computed from logits
- KV cache tensors
- per-layer hidden states via forward hooks
- prefix/suffix cache split
- cached suffix logits vs full-forward suffix logits
- position-modulo boundary case matching the original reference/HF behavior

It intentionally does **not** check padding masks, because the original reference API has no `attention_mask`. Padding/mask checks remain covered by P0-1 HF-vs-oracle tests.

## Requirements

Clone or install the original reference implementation:

```bash
git clone https://github.com/dieOD/multiscreen-pytorch
```

Then run with paths to all three implementations:

```bash
BASE=/path/to/repo
REF=$BASE/third_party/multiscreen-pytorch

python /path/to/p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root $REF \
  --hf-root $BASE \
  --oracle-root $BASE/oracle \
  --quick
```

Full CPU fp32 run:

```bash
python /path/to/p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root $REF \
  --hf-root $BASE \
  --oracle-root $BASE/oracle
```

Optional CUDA bf16 smoke:

```bash
python /path/to/p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root $REF \
  --hf-root $BASE \
  --oracle-root $BASE/oracle \
  --device cuda:0 --dtype bf16 --quick
```

## Expected output

```text
All P0-2 three-way minimal comparisons passed.
  cache_split_three_way: ...
  prefill_three_way: ...
```

## Notes

- The test loads weights from the original reference model into the HF port using `convert_original_state_dict_for_causal_lm()`.
- The oracle then copies the HF weights with `hf_uses_inverse_sr=True`, converting the inverse-width `sr` parameterization to paper-width internally.
- The oracle is set to `position_rule="hf_mod_after_max_position"` for this test, because the original reference and HF port include the modulo branch beyond `max_seq_len` / `max_position_embeddings`.
- Use P0-1 tests, not this script, to test attention masks and padding behavior.
