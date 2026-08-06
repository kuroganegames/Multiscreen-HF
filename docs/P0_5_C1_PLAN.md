# P0.5-C1 Plan: Architecture, Initialization, and All-Scale Contract

## Status

Implementation and local validation are complete on the focused C1 branch.
Acceptance remains **REVIEW_REQUIRED** until the draft pull request is reviewed
and merged. C2 must not start from this branch.

P0.5-C1 is the first of five separately reviewed Level 1 Core stages:

```text
P0.5-C1       architecture / initialization / all-scale contract
P0.5-C2       long-position / MiPE / cache semantics
P1-preflight B gradient-checkpointing API modernization
P0.5-C3       paper-training-contract smoke
final Level 1 requalification and evidence
```

This gate does not reopen the accepted P0-1 through P0-4 results. It also does
not complete P1-preflight A retention, validate a new training run, or validate
any P1 ecosystem capability.

## Objective

Turn the Multiscreen v3 architecture, parameter-count, and initialization
tables into executable, deterministic contracts. Exercise every paper scale
without allocating real 1.3B or 4B weights, and lock down the existing
configuration, normalized tied embedding, and packed-text behavior needed by
later Level 1 stages.

The current production implementation already satisfies the C1 contract. This
stage therefore adds independent verification and records; it does not change
model, configuration, oracle, cache, position, loss, or dataset behavior.

## Source audit

The primary mathematical source is
[Multiscreen arXiv:2604.01178v3](https://arxiv.org/abs/2604.01178v3), revised
2026-05-07. Tables 1, 2, and 4 and the associated prose specify the C1 scaling,
counts, and initializer values. The v3 TeX source retrieved on 2026-08-06 had
SHA-256:

```text
de9ede56a4f845f5dc9abc0b1497018bb3aaebdddde869583735d4bfb5962efd
```

The paper and its arXiv metadata did not identify a public author model
implementation during the dated audit. This is not proof that private or
unindexed code does not exist. The paper links the author's ABCDigits benchmark,
not Multiscreen model code.

The compatibility reference is the explicitly unofficial
[`dieOD/multiscreen-pytorch`](https://github.com/dieOD/multiscreen-pytorch)
repository. A source audit matched the 20 vendored files to upstream commit
[`8abea13c528885e385fe6a853155e20e3827e050`](https://github.com/dieOD/multiscreen-pytorch/commit/8abea13c528885e385fe6a853155e20e3827e050),
apart from upstream `.gitignore`, which is not vendored. The files entered this
repository in local commit `e4672552ffae1b7a46842ba54b6a23dad16ab581`.

The v3 paper is authoritative for C1. The vendored reference remains relevant
for compatibility, but its approximate parameter helper omits the learned
scalar/vector contribution and must not define the expected table.

## Paper architecture and independent count derivation

For the paper configurations:

```text
N_L = N_H = Psi
d_E = Psi^2
d_K = 16
d_V = 64
w_th = 256
vocabulary = 50,257
```

Using PyTorch storage orientation, the named state shapes are:

```text
embed.weight: [vocabulary, d_E]
s_E, s_F:     scalar

per layer:
  q_proj.weight, k_proj.weight: [N_H * d_K, d_E]
  v_proj.weight, g_proj.weight: [N_H * d_V, d_E]
  o_proj.weight:                [d_E, N_H * d_V]
  s_w, s_r, s_O:                [N_H]
```

The independent accounting is therefore:

```text
non_embedding = N_L * N_H * (d_E * (2*d_K + 3*d_V) + 3) + 2
total         = vocabulary * d_E + non_embedding

paper scaling:
non_embedding = 224 * Psi^4 + 3 * Psi^2 + 2
total         = 224 * Psi^4 + 50,260 * Psi^2 + 2
state keys    = 3 + 8 * Psi
```

The `+3` is the per-tile `s_w`, `s_r`, and `s_O` contribution; `+2` is global
`s_E` and `s_F`. “Non-embedding” excludes only `embed.weight`.

| Psi | Total | Non-embedding | State keys |
|---:|---:|---:|---:|
| 8 | 4,134,146 | 917,698 | 67 |
| 16 | 27,546,626 | 14,680,834 | 131 |
| 32 | 286,347,266 | 234,884,098 | 259 |
| 48 | 1,304,884,226 | 1,189,092,098 | 387 |
| 64 | 3,963,961,346 | 3,758,108,674 | 515 |

`MultiscreenConfig.num_params_estimate` is deliberately not used by the
generator or tests. That compatibility helper is documented as approximate and
is smaller by `3 * N_L * N_H + 2`.

## Initialization contract

```text
W_Q, W_K ~ Normal(0, 0.1 / sqrt(d_K))
W_V      ~ Normal(0, 0.1 / sqrt(d_V))
W_G      ~ Normal(0, 0.1)
W_O      ~ Normal(0, 0.1 / sqrt(d_E))
W_E      ~ Normal(0, 0.1 / sqrt(d_E))
s_w       = headwise linspace(0, log(w_th))
s_r       = 0
s_O       = log(1 / sqrt(N_H * N_L))
s_E       = 0
s_F       = log(sqrt(d_E))
```

The paper prose identifies the Gaussian table values as standard deviations.
The interception test records every `normal_` call by tensor identity and
checks the final requested mean and standard deviation for every random model
parameter. The embedding constructor's initial framework-default call is
expected to occur before the explicit Multiscreen initializer, so the final
call is the architectural contract. A fixed-seed, role-aggregated statistical
check is secondary evidence and is not the sole proof.

The stated `s_w` initializer is preserved literally. Because the implementation
uses `w = exp(s_w) + 1`, its initial realized endpoint windows are 2 and 257;
C1 does not reinterpret the paper initializer to force endpoints 1 and 256.

The manifest also records the current `from_psi` construction default
`max_position_embeddings=256`. That value is not treated as a paper C1
position-semantics contract and is not identified with `w_th`. The distinction
between training context, maximum supported positions, and the reference
modulo-transition boundary remains explicitly deferred to C2.

## Allocation-safe manifest

[`generate_paper_scale_manifest.py`](../scripts/generate_paper_scale_manifest.py)
constructs one model at a time inside `torch.device("meta")`. Before enumerating
state it requires every parameter, buffer, and state tensor to remain on meta,
compares the actual key/shape map with the independently derived named shapes,
and checks the hard-coded paper counts.

The checked artifact is
[`P0_5_C1_ARCHITECTURE_MANIFEST.json`](validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json).
It uses sorted keys, two-space JSON indentation, and one trailing newline. It
contains no timestamp, hostname, private path, runtime metric, weight value, or
environment-dependent field, so regeneration is byte-for-byte deterministic.

This architecture/state manifest is compact checked-in contract data. It is not
a validation evidence archive under `validation_evidence_v1`, and it does not
claim paper-scale runtime, memory, training, quality, or efficiency.

## Focused coverage

[`test_paper_architecture_contract.py`](../tests/test_paper_architecture_contract.py)
covers:

```text
- independent named shapes and hard-coded counts for Psi=8/16/32/48/64;
- allocation-safe meta construction and complete state manifests;
- aliases, conflicts, from_psi, clone, deterministic config serialization;
- direct and registered AutoClass config/model save-load;
- AutoClass metadata;
- identical input/output embedding module identity;
- normalized tied output-weight math;
- no registered lm_head state or Parameter.
```

[`test_paper_initialization_contract.py`](../tests/test_paper_initialization_contract.py)
covers exact scalar/vector values, requested random initializer arguments by
parameter identity, fixed-seed reproducibility, and broad role-aggregated
statistical sanity.

[`test_packed_text_contract.py`](../tests/test_packed_text_contract.py) covers
EOS insertion after every nonempty document, continuous concatenation,
non-overlapping `seq_len + 1` chunks, one-token input/label shifts, exact token
limits, explicit EOS selection, and conventional HF label mode. “No token loss
or duplication” applies to the retained complete-chunk prefix. The existing
documented behavior discards an incomplete final tail, and independent chunks
do not create a training pair across a chunk boundary.

## Validation commands

```bash
export PYTHONPATH=$PWD:$PWD/oracle

python -m py_compile \
  scripts/generate_paper_scale_manifest.py \
  tests/test_paper_architecture_contract.py \
  tests/test_paper_initialization_contract.py \
  tests/test_packed_text_contract.py

python -m unittest discover -s tests -p 'test_paper_architecture_contract.py' -v
python -m unittest discover -s tests -p 'test_paper_initialization_contract.py' -v
python -m unittest discover -s tests -p 'test_packed_text_contract.py' -v

python scripts/generate_paper_scale_manifest.py \
  --check docs/validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json
python -m json.tool \
  docs/validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json \
  >/dev/null
```

The accepted P0 quick baseline and P1-preflight A evidence-tooling suite must
also pass. CI runs the focused C1 contracts in both the regular resolved CPU
lane and a declared-lower-bound lane pinned to Python 3.10, Torch 2.4.0 CPU,
and Transformers 4.57.0. This avoids inferring declared support from either the
local Transformers 4.55 diagnostic environment or only the latest resolver
result.

## Acceptance boundary

C1 is locally ready for review only when:

```text
- all five paper configurations match independent shapes and counts;
- no large-scale real weight allocation occurs;
- state manifests regenerate byte-for-byte;
- exact and random initializer contracts pass;
- config/save-load/AutoClass/tied-head contracts pass;
- packed-text golden tests pass;
- tracked Python and JSON syntax, Markdown links, and hygiene pass;
- the P0 quick baseline and evidence-tooling suite pass;
- no production numerical behavior changed to fit the expected table.
```

Merge review is a separate acceptance act. Until the draft PR is reviewed and
merged, the stage remains `REVIEW_REQUIRED`, C2 remains unstarted, and no Level
1 completion claim is permitted.

## Explicit exclusions

```text
- MiPE absolute/reference position-mode decisions or cache-boundary changes;
- gradient-checkpointing API modernization;
- optimizer, scheduler, clipping, or SlimPajama training-contract work;
- new P0-4 training or final Level 1 requalification;
- PEFT/LoRA/QLoRA/Unsloth, serving, compile, or kernel work;
- changes to historical P0 summaries or the P0-4 retention descriptor.
```
