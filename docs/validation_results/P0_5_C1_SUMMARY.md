# P0.5-C1 Architecture / Initialization / All-Scale Summary

## Verdict

```text
Local gate result: passed
Acceptance state: accepted; focused PR #9 merged 2026-08-07
Later status: P0.5-C2 accepted; PR #10 and correction PR #11 merged
Current staged gate: P1-preflight B local pass; REVIEW_REQUIRED
```

The focused C1 implementation and local validation passed on 2026-08-06. The
focused draft PR #9 was subsequently merged on 2026-08-07, so C1 is accepted.
At that review boundary, C1 acceptance did not validate C2, P1-preflight B,
C3, final Level 1 requalification, or any P1 ecosystem capability. C2 was
later accepted independently by merged PR #10, with the separate CUDA-autocast
correction merged as PR #11. P1-preflight B is the current local-pass,
`REVIEW_REQUIRED` gate.

## Provenance

```text
base commit: a44904fa8f24f81b6f7f67cf575e8e1ac07ddcb7
branch: validation/p0-5-c1-architecture-init-scale
base relation at branch creation: main == origin/main
base worktree: clean
source paper: arXiv:2604.01178v3, revised 2026-05-07
source audit date: 2026-08-06
```

The primary contract comes from paper v3 Tables 1, 2, and 4. No public author
model implementation was identified in the paper/arXiv links during the dated
audit. The 20 vendored unofficial reference files were matched to upstream
`dieOD/multiscreen-pytorch` commit
`8abea13c528885e385fe6a853155e20e3827e050`, apart from the omitted upstream
`.gitignore`.

## Architecture result

The tests derive the expected count from named paper shapes without calling the
implementation's approximate estimate helper:

```text
non_embedding = N_L * N_H * (d_E * (2*d_K + 3*d_V) + 3) + 2
total         = vocabulary * d_E + non_embedding
```

| Psi | Expected/actual total | Expected/actual non-embedding | State keys |
|---:|---:|---:|---:|
| 8 | 4,134,146 | 917,698 | 67 |
| 16 | 27,546,626 | 14,680,834 | 131 |
| 32 | 286,347,266 | 234,884,098 | 259 |
| 48 | 1,304,884,226 | 1,189,092,098 | 387 |
| 64 | 3,963,961,346 | 3,758,108,674 | 515 |

Every model was constructed on the meta device. Every parameter, buffer, and
state tensor remained meta, and no real paper-scale weight storage was
allocated. The full canonical key/shape records are in
[P0_5_C1_ARCHITECTURE_MANIFEST.json](P0_5_C1_ARCHITECTURE_MANIFEST.json).

## Focused behavior result

Passed contracts:

```text
- exact paper C1 architecture fields at all five scales;
- current from_psi max-position default recorded without accepting position
  semantics or equating it with w_th;
- independent total and non-embedding counts;
- deterministic complete state key/shape manifests;
- config aliases, from_psi, clone, conflict rejection, and save/load;
- AutoClass metadata and registered AutoConfig/AutoModelForCausalLM load;
- normalized tied input/output embedding identity;
- no separate trainable lm_head Parameter or state entry;
- exact s_w, s_r, s_O, s_E, and s_F values;
- intercepted requested mean/std for every random model initializer;
- fixed-seed reproducibility and aggregate statistical sanity;
- PackedTextDataset EOS stream, seq_len+1 chunks, one-token shifts,
  max-token boundary, retained-prefix integrity, and HF label mode.
```

No production model, configuration, oracle, cache, position, loss, or dataset
source was changed.

## Local environment

```text
Conda environment: base (unchanged)
Python: 3.12.10
PyTorch: 2.8.0+cu128
Transformers: 4.55.0
NumPy: 2.3.2
safetensors: 0.6.2
tokenizers: 0.21.4
CUDA: 12.8; bf16 available
package installation or upgrade: none
```

The active local Transformers version is below the repository-declared 4.57
minimum. Draft-PR acceptance therefore includes a dedicated Python 3.10,
Torch 2.4.0 CPU, Transformers 4.57.0 lower-bound job as well as the regular
resolved-requirements job. This local run alone replaces neither CI lane.

## Tests recorded locally

Pre-change baseline:

```text
formula unit tests: passed
paper_math_oracle self-checks: passed
paper_math_oracle smoke tests: passed
HF-port CPU fp32 quick: passed
  cache_split 10; padding_cache 24; padding_full 8;
  position negative 2; position_ids_zero 1; shape forward/loss 12;
  logits_to_keep 24; shifted loss 12; zero relevance 1
P0-2 three-way CPU fp32 quick: passed
  prefill_three_way 12; cache_split_three_way 28
P1-preflight A evidence tooling: 58 passed
all tracked Python compile: passed
all tracked JSON parse: passed
```

Focused C1 final pre-PR run:

```text
paper architecture contract: 5 passed
paper initialization contract: 3 passed
packed-text contract: 5 passed
manifest byte-for-byte regeneration: passed
formula units, oracle self-check, and oracle smoke: passed
HF-port CPU fp32 quick: passed
P0-2 three-way CPU fp32 quick: passed
P1-preflight A evidence tooling: 58 passed
all tracked and new Python compile: passed
all tracked and new JSON parse: passed
all local Markdown links: passed
workflow YAML parse: passed
git diff --check: passed
commit-scope artifact/privacy/size/symlink hygiene: passed
```

The final gate repeated the P0 quick baseline after the C1 additions. No
production source changed, so stronger model-core regressions were not invoked.

## Evidence and limitations

The committed JSON is a deterministic architecture/state manifest, not an
evidence archive and not a runtime or memory benchmark. No raw logs,
checkpoints, model weights, hostnames, private paths, or secrets are retained.
The current dense implementation remains quadratic; meta construction and
parameter counts provide no evidence for paper efficiency or scaling quality.

Historical P0-4 evidence retention remains partial/blocked exactly as recorded:
the compact hashes and completion markers were matched and the sanitized
archive verified locally, while external exact/private retention, explicit
acceptance review, and a public asset remain absent. C1 does not rewrite that
descriptor. Final Level 1 evidence packaging and acceptance are deferred to the
separate fifth stage.
