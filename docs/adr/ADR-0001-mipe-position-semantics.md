# ADR-0001: Explicit MiPE position semantics

## Status

```text
Decision: proposed for P0.5-C2
Source audit completed: 2026-08-07
Acceptance: pending focused draft-PR review and merge
```

## Context

Multiscreen paper v3 defines the MiPE angle at position `i` as

```text
phi(i, w) = pi * i * gamma(w) / w
```

and makes MiPE the identity when `w >= w_th`. It does not define a modulo,
remainder, wrap boundary, or transition tied to the training context. The
current Hugging Face implementation and the vendored unofficial PyTorch
reference instead apply a per-head remainder after `max_position_embeddings`:

```text
effective_position = position             when position < boundary
effective_position = position remainder w when position >= boundary
```

The Softmask remains based on absolute query/key distance in both paths. The
two MiPE rules are therefore distinct semantics, not two spellings of the same
equation.

## Source audit

All external sources below were accessed on 2026-08-07.

| Source | Version / commit | Supported claim |
|---|---|---|
| [Multiscreen paper](https://arxiv.org/abs/2604.01178v3) and [HTML](https://arxiv.org/html/2604.01178v3) | arXiv v3, revised 2026-05-07 | Equation (6) uses absolute `i`; Equation (7) disables MiPE at `w >= w_th`; pretraining context and learned window are distinct quantities. |
| [arXiv TeX source](https://arxiv.org/e-print/2604.01178v3) | SHA-256 `de9ede56a4f845f5dc9abc0b1497018bb3aaebdddde869583735d4bfb5962efd` | `main.tex` contains no modulo/remainder/wrap rule for MiPE. |
| `oracle/paper_math_oracle.py` | C2 base `ec805c1ba60c55ea4beb3ad68e0a00c0d718e909` | Already separates literal paper positions from the HF/reference modulo rule. |
| Current HF implementation | C2 base `ec805c1ba60c55ea4beb3ad68e0a00c0d718e909` | Always uses modulo after `max_position_embeddings`; no serialized public mode exists. |
| [dieOD reference commit](https://github.com/dieOD/multiscreen-pytorch/commit/8abea13c528885e385fe6a853155e20e3827e050) and [PR #2](https://github.com/dieOD/multiscreen-pytorch/pull/2) | vendored/upstream HEAD `8abea13c528885e385fe6a853155e20e3827e050` | Adds the modulo branch as an unofficial length-extrapolation policy so angles stay in a trained range; it does not attribute the rule to the paper or author. |
| [Hugging Face Papers API](https://huggingface.co/api/papers/2604.01178) | response observed 2026-08-07; SHA-256 `4350918db69ccb8dce3d80e0c3d9800149a2a626024b1c8a3b98232211ff411c` | Returned paper ID `2604.01178` with null `githubRepo` and `projectPage`; this supports only that the metadata linked no official code/project at access time. |
| [Author GitHub public-repository inventory](https://api.github.com/users/ken-nakanishi/repos?per_page=100&sort=full_name) | 15 rows observed 2026-08-07; response SHA-256 `b311c706b56839732cd3bf4ae3ccf9448ec56c191797a033c06744bc32a10426` | Returned names/descriptions contained no Multiscreen repository; this was an inventory-metadata check, not a content audit of every repository. |
| [Author ABCDigits repository](https://github.com/ken-nakanishi/abcdigits) | `777ddef542fa5c1f82ef1819d0b8824a94de5b4c` | Provides benchmark generation only, not Multiscreen model code. |
| [lucidrains/multiscreen](https://github.com/lucidrains/multiscreen) | `1ee6aa4daae17af4243de2a32146143692d994f2` | Supplemental unofficial implementation uses direct absolute positions and no modulo. |
| [exveria1015/screening-is-enough-pytorch](https://github.com/exveria1015/screening-is-enough-pytorch) | `4501562e9052a8612f9ab83a9d19030bf0aa1762` | Supplemental unofficial implementation uses direct absolute positions and no modulo. |

No public author/official Multiscreen model implementation was identified in
the v3 paper/arXiv links, the two metadata endpoints above, or the inspected
ABCDigits repository during this dated audit. This bounded metadata and
linked-source search is not a content audit of all 15 public repositories and
is not proof that no private or unindexed implementation exists. The
long-position behavior of an unpublished author implementation remains unknown.

### Proven

- Paper v3 uses the unwrapped absolute index `i` in the MiPE equation.
- Paper v3 specifies no position transition or wrap boundary.
- `w_th` controls whether MiPE is active; it is not a context-length limit.
- The paper's optional screening-window expansion changes learned `w` values;
  it is separate from position semantics.
- The vendored reference wraps at and after `max_seq_len`, per head, using
  floating learned `w`; its Softmask still uses absolute distances.
- Cached K is stored after MiPE, so cache values depend on the selected
  position mode, wrap boundary, and auxiliary compute dtype.

### Inferred

- In paper mode, a uniform shift of the position origin cancels in a q-k dot
  product because both vectors use the same linear angle. The repository still
  retains a zero-based scalar contiguous-prefix API for unambiguous caching.
- Applying `position remainder w` only after a boundary is generally not
  equivalent to the paper rule. It can create a phase discontinuity at the
  boundary, especially for fractional learned windows.

### Unknown

- Whether the unpublished author implementation used any undocumented
  long-position transformation.
- Whether an author implementation used zero- or one-based indices. This does
  not change unwrapped pairwise similarities but would matter after wrapping.
- Whether a private or unindexed official model repository exists.

## Decision

### 1. Two canonical serialized modes

The HF config exposes `mipe_position_mode` with exactly these canonical values:

```text
paper_absolute
reference_mod_after_wrap_boundary
```

`paper_absolute` uses the literal paper equation for every nonnegative
position. `reference_mod_after_wrap_boundary` preserves the vendored/HF rule.

### 2. Separate wrap boundary

The HF config exposes `mipe_reference_wrap_boundary` as a positive integer.
It applies only to `reference_mod_after_wrap_boundary`. It is not a maximum
supported position and is not silently treated as the MiPE threshold.

`max_position_embeddings` remains the Transformers-facing configured context
length. The current dense implementation does not enforce it as a hard runtime
position maximum.

### 3. Backward-compatible missing-field migration

Existing serialized configs have neither new field. A missing
`mipe_position_mode` therefore resolves to
`reference_mod_after_wrap_boundary`, and a missing
`mipe_reference_wrap_boundary` resolves to the config's existing
`max_position_embeddings`. The resolved canonical fields are emitted on the
next serialization.

This preserves all pre-C2 logits and cache values. The constructor cannot
reliably distinguish an old missing field from a newly omitted field, so the
global missing-field default also remains reference-compatible. Paper behavior
must be requested explicitly. Silently reinterpreting old configs as paper
mode was rejected.

Existing checked-in P0-3/P0-4 config JSON and historical evidence are not
rewritten. P0-4 positions `0..4095` are unchanged when its legacy boundary
resolves to `4096`.

### 4. Oracle compatibility

The oracle remains paper-oriented by default. It accepts the new canonical
mode names and retains its old `paper` and `hf_mod_after_max_position` names as
compatibility aliases. `from_hf_config` maps the HF mode and explicit reference
wrap boundary instead of requiring an implicit override.

### 5. Orthogonal numerical modes

Position semantics and auxiliary compute dtype remain separate controls:

```text
stable paper/oracle checks: fp32 auxiliary MiPE and Softmask math
vendored low-precision compatibility: reference auxiliary math
```

Reference bf16/fp16 creates position tensors in the incoming low precision,
matching the vendored implementation. At long positions, distinct integers can
round to the same bf16 value; fp16 can become non-finite. Multi-token reference
low-precision Softmask can consequently lose causal distinctions. This is a
documented compatibility limitation, not accepted paper behavior. Long-boundary
cache/full correctness is established with stable fp32 auxiliary math.

### 6. Cache contract remains narrow

Both modes retain one scalar, batch-shared, nonnegative, contiguous position
range. Cache calls require a complete prefix from zero and
`start_pos == past_length`. Arbitrary batch-specific, non-contiguous, negative,
or conflicting offsets remain unsupported and must fail loudly.

## Consequences and required evidence

- Paper-mode HF MiPE must match the paper oracle at positions
  `0,1,255,256,257,4095,4096,4097,8191,8192,131071` across active, threshold,
  inactive, and fractional window regimes.
- Reference-mode HF MiPE must match both the reference-compatible oracle and
  vendored reference.
- Legacy config loading, explicit mode construction, clone, save/load, and
  AutoClass round trips must preserve the resolved mode and boundary.
- Full-context suffix math and cached suffix math must agree for both modes at
  `4080+32`, `4096+16`, and `8192+16`, including one-token and uneven chunks
  crossing 4096, without requiring a dense 131K forward.
- Strong P0-1/P0-2 CPU fp32 and CUDA bf16 regressions remain required. The
  CUDA bf16 reference lane proves compatibility, not long-position causal
  correctness.
- This ADR provides no evidence for 131K dense feasibility, retrieval quality,
  optimized window skipping, or paper efficiency.

## Rejected alternatives

- **Silently switch every missing config to paper mode:** breaks existing
  checkpoints at and after their old transition boundary.
- **Keep a single hidden modulo rule:** cannot support a literal paper claim.
- **Permanently reuse `max_position_embeddings` as the wrap boundary:**
  conflates separate concepts without paper support.
- **Use `w_th` as a wrap or support boundary:** contradicts the paper; MiPE can
  be inactive while Softmask still uses a larger learned `w`.
- **Treat screening-window expansion as a position rule:** it modifies `w` and
  is orthogonal to MiPE indexing.
