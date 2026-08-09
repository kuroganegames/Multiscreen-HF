# P0.5-C2 Plan: Long-Position, MiPE, and Cache Semantics

## Status

```text
Gate: P0.5-C2
State: implementation and focused local validation passed
Acceptance: accepted; focused PR #10 merged on 2026-08-08
Accepted P0 boundary: unchanged through P0-4
P1 model/ecosystem capabilities validated by this gate: none
```

P0.5-C2 is the second of five separately reviewed Level 1 Core stages:

```text
P0.5-C1        architecture / initialization / all-scale contract
P0.5-C2        long-position / MiPE / cache semantics
P1-preflight B gradient-checkpointing API modernization
P0.5-C3        paper-training-contract smoke
final Level 1 requalification and evidence
```

The focused C2 implementation was reviewed and merged as PR #10. A separate
post-merge CUDA-autocast cache-dtype prediction correction was reviewed and
merged as PR #11 on 2026-08-09. Neither PR reopens the accepted P0-1 through
P0-4 results, completes the historical P0-4 retention descriptor, or validates
a P1 model/ecosystem capability.

## Objective

Make the paper's absolute-position MiPE equation and the historical
HF/vendored-reference modulo rule explicit, separately serialized behaviors.
Preserve old checkpoints through a tested missing-field migration, separate the
reference wrap boundary from the configured context length, and verify the
scalar contiguous cache contract across the historical 4096 boundary.

The source decision and its alternatives are recorded in
[ADR-0001](adr/ADR-0001-mipe-position-semantics.md). The decision was accepted
with PR #10; PR #11 did not change the position semantics.

## Source audit

All external sources in the C2 audit were accessed on 2026-08-07.

The primary source is
[Multiscreen arXiv:2604.01178v3](https://arxiv.org/abs/2604.01178v3), revised
2026-05-07. The corresponding
[HTML](https://arxiv.org/html/2604.01178v3) and
[TeX source](https://arxiv.org/e-print/2604.01178v3) were inspected. The v3
source archive had SHA-256:

```text
de9ede56a4f845f5dc9abc0b1497018bb3aaebdddde869583735d4bfb5962efd
```

Paper v3 defines:

```text
w = exp(s_w) + 1
r = sigmoid(s_r)

phi(i, w) = pi * i * gamma(w) / w

gamma(w) = 0.5 * (cos(pi * w / w_th) + 1)  when w < w_th
gamma(w) = 0                               when w >= w_th

m_ij(w) = 0.5 * (cos(pi * (j - i) / w) + 1)
          when -w < j - i <= 0, and 0 otherwise
```

The v3 paper and TeX contain no modulo, remainder, or position-wrap rule. They
also distinguish the pretraining context, the learned screening window
`w`, and the MiPE threshold `w_th`. A window at or above
`w_th` makes MiPE the identity; it does not cap Softmask support at
`w_th`.

The current compatibility reference is the explicitly unofficial
[dieOD/multiscreen-pytorch](https://github.com/dieOD/multiscreen-pytorch)
commit
[8abea13c528885e385fe6a853155e20e3827e050](https://github.com/dieOD/multiscreen-pytorch/commit/8abea13c528885e385fe6a853155e20e3827e050).
Its [PR 2](https://github.com/dieOD/multiscreen-pytorch/pull/2) introduced the
modulo branch as a length-extrapolation policy intended to keep rotation angles
in a trained range. The PR does not attribute that rule to the paper or its
author.

No public author/official Multiscreen model implementation was identified in
this bounded audit. The
[Hugging Face Papers API](https://huggingface.co/api/papers/2604.01178)
returned null `githubRepo` and `projectPage` fields, and the
[GitHub public-repository inventory](https://api.github.com/users/ken-nakanishi/repos?per_page=100&sort=full_name)
returned 15 repository names/descriptions with no Multiscreen entry. The author's
[ABCDigits repository](https://github.com/ken-nakanishi/abcdigits) at commit
`777ddef542fa5c1f82ef1819d0b8824a94de5b4c` supplies benchmark
generation, not Multiscreen model code. Exact metadata snapshot hashes and
scope are recorded in the ADR. This name/description and linked-source search
is not a content audit of every public repository, nor proof that private or
unindexed code does not exist; unpublished implementation behavior remains
unknown.

Supplemental unofficial implementations audited at pinned commits generally
use the direct absolute-position equation without the dieOD modulo branch.
They corroborate that the two readings are implementable, but they do not
establish author intent and are not authoritative over paper v3.

## Semantic decision

### Paper-absolute mode

```text
mipe_position_mode = "paper_absolute"
effective_position = absolute_position
```

This is the literal v3 equation for every supported nonnegative position. It
does not introduce a transition at the training context or at
`max_position_embeddings`.

### Reference-compatible mode

```text
mipe_position_mode = "reference_mod_after_wrap_boundary"

effective_position = position               when position < boundary
effective_position = position remainder w   when position >= boundary
```

The separately serialized positive integer boundary is:

```text
mipe_reference_wrap_boundary
```

The transition is inclusive: the position equal to the boundary is the first
wrapped position. The remainder is per head and uses the learned, potentially
fractional `w`. Softmask continues to use absolute query/key
distance.

Applying `position remainder w` is generally not equivalent to the
paper equation. MiPE's rotary position period, when
`gamma(w) > 0`, is `2 * w / gamma(w)`, not
`w`; the reference rule can therefore introduce a phase
discontinuity at the inclusive wrap boundary.

### Separate concepts

The following values must not be used as aliases:

```text
max_position_embeddings       Transformers-facing configured context length
mipe_reference_wrap_boundary  reference-mode transition only
mipe_threshold / w_th         threshold that disables MiPE
w                              learned per-tile screening support
cache length                  number of retained prefix key/value positions
```

`max_position_embeddings` is not enforced by this dense
implementation as a hard supported-position maximum. Screening-window
expansion changes `w`; it is orthogonal to the position mode and is
outside this stage.

## Backward-compatible migration

Pre-C2 serialized configurations contain neither new MiPE field. Loading such a
configuration resolves it as:

```text
mipe_position_mode = "reference_mod_after_wrap_boundary"
mipe_reference_wrap_boundary = max_position_embeddings
```

The canonical resolved fields are emitted by the next serialization. Because
the constructor cannot reliably distinguish an old omitted field from a new
omitted field, the global missing-field default also remains reference
compatible. Literal paper behavior must be requested explicitly.

This is a semantic config migration only; weights are not converted. Existing
P0-3/P0-4 config JSON and historical evidence are not rewritten. For the P0-4
context of 4096, positions `0..4095` remain before the legacy
transition, so the accepted historical run values are unchanged. Outputs and
cached keys at positions at or beyond the boundary can differ between modes.

Config construction, clone, `save_pretrained` /
`from_pretrained`, registered AutoClass loading, and deterministic
serialization must retain the resolved mode and boundary. Invalid modes and
non-positive or non-integral boundaries must fail explicitly.

## Numerical-compute contract

Position semantics and auxiliary compute dtype are independent controls:

```text
stable paper/oracle checks:
  mipe_compute_dtype="fp32"
  softmask_compute_dtype="fp32"

vendored low-precision reference compatibility:
  mipe_compute_dtype="reference"
  softmask_compute_dtype="reference"
```

Stable fp32 auxiliary math is the correctness lane for paper equations and
long-boundary full/cache agreement. Reference auxiliary math follows the input
dtype to reproduce the unofficial implementation. At long positions, bf16 may
map distinct integer positions to the same representable value, and fp16 may
make sufficiently large positions non-finite. Multi-token reference-mode
Softmask can consequently lose causal distinctions.

The low-precision reference lane proves compatibility with the vendored
implementation under its arithmetic, not general long-position causal
correctness. Tests and result records must not merge these two meanings or
silently change dtype rules to obtain agreement.

## Cache and position contract

Both MiPE modes preserve the existing narrow cache API:

```text
- one scalar, batch-shared start position;
- a complete cached prefix beginning at zero;
- start_pos equals the cached prefix length;
- query/key positions form a contiguous nonnegative range;
- a key is cached after the selected MiPE transformation is applied.
```

Arbitrary batch-specific offsets, negative positions, non-contiguous position
IDs, no-cache offset ranges, and cache/start-position conflicts remain
unsupported and must fail loudly. A mode or boundary change alters cached-key
semantics; callers must not mix cache entries created under different config
semantics.

## Focused test matrix

Direct MiPE/oracle tests must cover at least:

```text
positions:
  0, 1, 255, 256, 257,
  4095, 4096, 4097,
  8191, 8192,
  131071

window regimes:
  below 256
  near 256
  equal to 256
  above 256
  fractional learned windows
```

Required comparisons:

```text
- HF paper_absolute vs the paper oracle;
- HF reference_mod_after_wrap_boundary vs the reference-compatible oracle;
- HF reference mode vs the vendored reference;
- stable fp32 auxiliary MiPE and Softmask math;
- reference auxiliary math in low precision where finite and meaningful;
- missing-field migration and explicit config round trips;
- strict invalid-mode, boundary, offset, and cache cases.
```

Cache/full comparisons must include:

```text
prefix 4080 + suffix 32
prefix 4096 + suffix 16
prefix 8192 + suffix 16
one-token decode steps crossing 4096
uneven multiple suffix chunks crossing 4096
```

Use tiny model shapes and direct position tests. The `131071`
matrix entry checks the scalar MiPE equation without constructing a dense 131K
attention matrix.

## Expected focused outputs

```text
docs/adr/ADR-0001-mipe-position-semantics.md
multiscreen_transformers/configuration_multiscreen.py
multiscreen_transformers/modeling_multiscreen.py
oracle/paper_math_oracle.py
focused position/MiPE/cache/config tests
docs/P0_5_C2_PLAN.md
focused CI coverage when justified by runtime
```

The vendored reference and historical P0-4 records are read-only inputs. This
stage must not change the gradient-checkpointing API, training recipe, data
pipeline, or any P1 model/ecosystem feature.

## Validation commands

Run from the repository root in the recorded environment:

```bash
export PYTHONPATH=$PWD:$PWD/oracle

python -m py_compile \
  multiscreen_transformers/configuration_multiscreen.py \
  multiscreen_transformers/modeling_multiscreen.py \
  oracle/paper_math_oracle.py \
  tests/test_mipe_position_cache_contract.py

python -m unittest discover \
  -s tests \
  -p 'test_mipe_position_cache_contract.py' \
  -v

python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py

python oracle/test_against_hf_port.py
python oracle/test_against_hf_port.py --device cuda:0 --dtype bf16

python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle

python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --device cuda:0 \
  --dtype bf16
```

Also rerun the C1 architecture/config round-trip contract and the
P1-preflight A standard-library evidence-tooling suite. Run repository syntax,
JSON, Markdown-link, diff, and artifact-hygiene checks before the focused C2
commit.

These commands are the required stage target, not a statement that they have
already passed. Record exact environment versions, commands, exit status,
counts, tolerances, warnings, and any unavailable CUDA coverage in the C2 pull
request evidence.

## Acceptance boundary

C2 is locally ready for draft-PR review only when:

```text
- the ADR distinguishes proven, inferred, and unknown source claims;
- both canonical modes and the separate inclusive reference boundary are
  serialized and validated;
- missing-field legacy migration preserves pre-C2 behavior;
- paper mode matches the paper oracle at every required matrix position;
- reference mode matches the oracle and vendored reference;
- stable fp32 and reference low-precision evidence remain separately labeled;
- full/cache equality passes for both modes across 4096 and 8192 scenarios;
- unsupported offsets and cache conflicts fail explicitly;
- strongest required P0-1/P0-2 CPU and CUDA comparisons pass;
- configuration, C1, evidence-tooling, syntax, link, diff, and hygiene checks
  pass;
- no dense 131K, retrieval-quality, efficiency, or later-stage claim is made.
```

Merge review was a separate acceptance act and was completed by merged PR #10.
The later PR #11 correction was also separately reviewed and merged before
Stage 3 resumed.

## Explicit exclusions

```text
- dense 131,072-token execution or memory feasibility;
- PG-19, ABCDigits, passkey, lost-in-the-middle, or retrieval validation;
- paper-quality or long-context quality claims;
- Triton, sparse/window-skipping, SWE, or efficiency work;
- gradient-checkpointing API modernization;
- paper optimizer/data/scheduler smoke work;
- new P0-4 training or final Level 1 requalification;
- PEFT/LoRA/QLoRA/Unsloth, compile, serving, or broad generation;
- changes to accepted P0-4 metrics or its historical retention descriptor.
```
