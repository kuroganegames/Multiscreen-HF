# P0.5-C3 Plan: Paper-Training-Contract Smoke

## Status

```text
Gate: P0.5-C3
State: local validation passed; focused draft PR REVIEW_REQUIRED
Acceptance: pending; REVIEW_REQUIRED after the local gate until focused PR merge
Accepted P0 boundary: unchanged through P0-4
P1 model/ecosystem capabilities validated by this gate: none
```

P0.5-C3 is the fourth of five separately reviewed Level 1 Core stages:

```text
P0.5-C1        architecture / initialization / all-scale contract
P0.5-C2        long-position / MiPE / cache semantics
P1-preflight B gradient-checkpointing API modernization
P0.5-C3        paper-training-contract smoke
final Level 1 requalification and evidence
```

C1 was accepted by merged PR #9. C2 was accepted by merged PR #10 and its
separate CUDA-autocast cache-dtype correction by merged PR #11. P1-preflight B
was accepted by merged PR #12. Stage 4 begins from that merged base and remains
independently review-gated.

```text
Stage 4 base / PR #12 merge: a2d43517c45dc39855db81b9286c4abf190a2c14
branch: agent/p0-5-c3-paper-training-contract
base relation at branch creation: HEAD == origin/main
base worktree: clean
base porcelain bytes: 0
base porcelain SHA-256: e3b0c44298fc1c149afbf4f8996fb92427ae41e4649b934ca495991b7852b855
source audit date: 2026-08-09
```

This plan remains the design record. The executed Stage 4 result, CUDA metrics,
dataset fingerprint, and evidence-retention state are recorded in
[P0_5_C3_SUMMARY.md](validation_results/P0_5_C3_SUMMARY.md). The local gate
remains `REVIEW_REQUIRED` until its focused draft PR is reviewed and merged.

## Objective

Encode the paper's tokenizer, data-stream, optimizer, scheduler, and
no-gradient-clipping recipe as executable contracts, then prove that a bounded
version of that path is operational on the available workstation. Keep exact
paper configuration evidence separate from reduced diagnostic execution.

This stage does not attempt the paper token budget, global batch, training
duration, quality, scaling, retrieval, or efficiency experiments. It does not
replace the accepted historical P0-4 result or perform the final Level 1
requalification.

## Primary-source audit

The mathematical and experimental source is
[Multiscreen arXiv:2604.01178v3](https://arxiv.org/abs/2604.01178v3), revised
2026-05-07. Its
[HTML](https://arxiv.org/html/2604.01178v3) and
[TeX source](https://arxiv.org/e-print/2604.01178v3) were inspected on
2026-08-09. The official v3 source archive and top-level TeX had SHA-256:

```text
source archive: de9ede56a4f845f5dc9abc0b1497018bb3aaebdddde869583735d4bfb5962efd
main.tex:       fed987b9591ac8a5f4f10897cffc61d762d8f5a909737deb206e0e8ff7fbe0aa
```

Appendix A states:

```text
- GPT-2 tokenizer with vocabulary 50,257;
- SlimPajama documents concatenated with EOS tokens into an approximately
  628-billion-token continuous stream;
- 2^38 training tokens selected as 2^26 random contiguous sequences of
  length 2^12;
- sequence length 2^12 = 4,096;
- global batch 2^22 tokens;
- AdamW betas (0.9, 0.95);
- 2^12 warmup steps followed by a constant learning rate;
- Multiscreen weight decay and gradient clipping omitted;
- Multiscreen learning rate 2^-4 = 0.0625.
```

The source contains no complete paper training implementation. It does not
specify the scheduler's step-zero convention or optimizer/scheduler call order,
AdamW epsilon or implementation variant, tokenizer revision, EOS ID, dataset
revision/file order, random-sequence seed, microbatch layout, gradient
accumulation, or training precision. The NeurIPS checklist embedded in the TeX
also states that full training/evaluation code and checkpoints are unavailable.
These unknowns must not be presented as recovered author behavior.

The paper's bf16 statement concerns latency measurement, not its pretraining
precision. CUDA bf16 in this gate is therefore a project diagnostic choice.

## Paper contract and derived arithmetic

The checked-in paper-recipe representation must encode these exact values:

```text
tokenizer family: GPT-2
tokenizer vocabulary: 50,257
dataset family: SlimPajama
document handling: append EOS, then concatenate as one continuous stream
model input sequence length: 4,096
paper global batch: 4,194,304 tokens
optimizer: AdamW
betas: (0.9, 0.95)
weight decay: 0
warmup optimizer steps: 4,096
post-warmup schedule: constant
peak learning rate: 0.0625
gradient clipping: disabled
```

The following values are arithmetic consequences of the stated paper values,
not separately reported experimental facts:

```text
paper sequences selected:       2^26 = 67,108,864
paper optimizer steps:          2^38 / 2^22 = 2^16 = 65,536
sequences per global step:      2^22 / 2^12 = 2^10 = 1,024
tokens covered by full warmup:  2^12 * 2^22 = 2^34 = 17,179,869,184
```

The workstation smoke must record its actual microbatch, accumulation, world
size, sequences per update, and tokens per update next to the paper values. A
reduced local batch must never be displayed as if it were `2^22` tokens.

## Scheduler operationalization

Paper v3 gives only “4096 warmup steps, then constant.” This repository defines
the missing indexing convention explicitly:

```text
step = zero-based optimizer-update index, evaluated before that update
W = warmup_steps
peak = peak_learning_rate

lr(step) = peak * min(step + 1, W) / W, for step >= 0
```

For the exact paper contract (`W=4096`, `peak=0.0625`):

| Step | Exact value | Decimal |
|---:|---:|---:|
| 0 | `1 / 65536` | `0.0000152587890625` |
| 1 | `1 / 32768` | `0.000030517578125` |
| 4095 | `1 / 16` | `0.0625` |
| 4096 | `1 / 16` | `0.0625` |
| 4097 | `1 / 16` | `0.0625` |

This `(step + 1)` convention is a repository operationalization, not an
equation quoted from the paper. The vendored unofficial trainer provides a
local precedent for the warmup ramp, but its post-warmup cosine decay is not
the paper contract and must not be reused here. Negative steps, non-positive
warmup lengths, and non-finite or non-positive peaks must fail explicitly.

A reduced diagnostic warmup may use the same formula with a smaller `W` to
exercise the transition. Its metrics must carry both the paper warmup and the
executed diagnostic warmup, with an explicit non-paper-scale label.

## Tokenizer identity

The paper names GPT-2 and its vocabulary size but does not pin tokenizer
artifacts. Stage 4 uses an explicit Hub snapshot already available to the
validation environment:

```text
repo ID: gpt2
revision: 607a30d783dfa663caf39e06633721c8d4cfcd7e
vocabulary size: 50,257
EOS token: <|endoftext|>
EOS token ID: 50,256
```

Pinned asset hashes:

```text
merges.txt             1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
tokenizer.json         8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6
tokenizer_config.json  5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f
vocab.json             196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
```

The tokenizer manifest SHA-256 is computed from UTF-8 bytes of canonical JSON
with lexicographically sorted keys, compact separators, and one trailing
newline. The payload contains `repo_id`, `revision`, and the four-name `files`
hash mapping above. Its expected SHA-256 is:

```text
07c45937a89b33f30016aef5b3982f13f25bf2c6ba940c535d1b5daa90459a71
```

The contract must verify actual bytes, vocabulary, and EOS identity. A matching
repository name or mutable `main` reference alone is insufficient.

## SlimPajama provenance and fingerprint contract

The paper cites the Cerebras SlimPajama release. The official dataset card and
Cerebras preprocessing documentation identify the family as a cleaned and
deduplicated RedPajama-derived corpus with `text` and source metadata fields,
distributed across train, validation, and test data.

The historical official Hub repository identity and last locally resolved full
revision in the dated audit were:

```text
repo ID: cerebras/SlimPajama-627B
revision: 2d0accdd58c5d5511943ca1f5ff0e3eb5e293543
```

Anonymous live page/API/Git resolution for that repository returned
unauthorized/not-found responses during the 2026-08-09 audit. The commit is an
immutable historical pin, but this audit does not claim it was re-resolved as
the current `main`, and the available cache did not contain raw rows.

An executable data contract must therefore record, before a run:

```text
- source repo ID and full 40-character revision;
- whether the source is the Cerebras release or a labeled SlimPajama-family
  derivative/reupload;
- config and split;
- an ordered explicit raw-shard list;
- each raw shard's repository path, size, and SHA-256/LFS identity;
- ordered row coordinates and SHA-256 of each UTF-8 text payload used;
- observed datasets-library version and `_fingerprint` when available;
- a canonical selected-source manifest SHA-256;
- tokenizer identity and asset-manifest SHA-256;
- EOS-concatenated token-stream SHA-256 and exact accounting.
```

Hugging Face `_fingerprint` is useful supplementary cache/state provenance,
but it changes with data transforms and is not a substitute for raw source and
content hashes. A third-party reupload must be labeled as such and must not be
claimed byte-identical to the Cerebras repository without a cryptographic
comparison. No dataset fingerprint value may be invented before the exact
selected data is available and hashed.

Because the dataset card refers users to the licenses of component sources,
raw text excerpts remain outside Git. Compact manifests should retain hashes,
counts, source-relative identifiers, and license/source labels without raw
document bodies or private local paths.

## Continuous-stream and token-accounting contract

For every selected nonempty document, tokenize with no implicit special tokens,
append exactly one explicit EOS token, and concatenate documents in the pinned
order. The Stage 4 implementation reuses the accepted C1 packed-text rule:

```text
- source chunks contain `sequence_length + 1` consecutive tokens;
- `input_ids = chunk[:-1]`;
- `labels = chunk[1:]` for explicit shifted mode;
- model input length is exactly 4,096;
- complete source chunks are non-overlapping;
- the final incomplete tail is reported and discarded;
- no source token inside the retained complete-chunk prefix is lost or
  duplicated;
- independent chunks do not synthesize a prediction pair across a chunk
  boundary.
```

`sequence_length + 1` chunking is this repository's causal-shift
operationalization. The paper states sequence length 4096 but does not specify
its loader's label-shift accounting. Metrics must distinguish source tokens,
EOS tokens, retained tokens, discarded tail tokens, model input tokens, target
tokens, and optimizer-accounted tokens.

## Four separate evidence lanes

### A. Exact unit/config contract

The checked config or manifest and focused tests must prove:

```text
- GPT-2 vocabulary and pinned tokenizer assets;
- paper dataset-family and data-stream fields;
- sequence length 4,096 and paper global batch 2^22 tokens;
- AdamW betas (0.9, 0.95) and weight decay zero in every parameter group;
- full 4,096-step warmup values at 0, 1, 4095, 4096, and 4097;
- constant post-warmup learning rate;
- peak LR exactly 0.0625;
- no gradient-clipping call or hidden finite max-norm substitute;
- deterministic serialization and rejection of malformed contracts.
```

### B. Deterministic data contract

Use synthetic golden documents for exhaustive EOS/packing edge cases and a
pinned SlimPajama-family selection for source provenance. Two independent loads
of the same pinned selection must produce identical ordered row hashes, token
stream, chunks, labels, counts, and fingerprints. Network-dependent full-corpus
loading is not a CI requirement.

### C. Short CUDA bf16 operational smoke

Run Psi=8 first with context 4096, CUDA bf16, actual local accumulation, an
explicit reduced warmup, weight decay zero, and clipping disabled. Require:

```text
- finite forward loss, gradients, gradient norm observation, and parameters;
- at least one real optimizer update and a changed trainable parameter;
- the expected reduced-warmup transition and constant phase;
- exact actual tokens-per-update accounting;
- truthful peak memory and environment records;
- no paper-global-batch or training-quality claim.
```

Inspect Psi=8 artifacts and memory before running Psi=16 under the same
contract. Preserve every failed attempt separately. Do not lower the defined
paper peak or add clipping merely to obtain a pass.

### D. Bounded peak-LR exposure diagnostic

Exercise one or more explicitly bounded optimizer updates at the exact
`0.0625` learning rate for Psi=8 and then Psi=16. Require finite values, a valid
optimizer update, and honest before/after parameter evidence. Do not require a
loss decrease, convergence, smooth training trajectory, or paper-quality
result from this diagnostic.

## Execution and review order

```text
1. run static/unit/config/scheduler/data contracts;
2. run required CPU baselines and repository hygiene;
3. run Psi=8 reduced-warmup CUDA bf16 operational smoke;
4. inspect Psi=8 raw events, completion state, and memory headroom;
5. run Psi=8 bounded peak-LR exposure;
6. only then run corresponding Psi=16 diagnostics;
7. preserve and explain failures before any correction and rerun;
8. sanitize compact results without rewriting raw events;
9. open one focused Stage 4 draft PR and stop at REVIEW_REQUIRED.
```

## Expected focused outputs

Implementation filenames may be refined with the focused diff, but Stage 4 is
expected to contain only narrowly scoped recipe work such as:

```text
checked paper-training-contract config/manifest
focused optimizer/scheduler/no-clipping/data tests
bounded diagnostic harness
focused offline CI coverage
docs/P0_5_C3_PLAN.md
compact Stage 4 result records created only after execution and review
```

Model, configuration, oracle, MiPE, cache, generation, and state-dict semantics
are read-only for this gate unless an independently demonstrated core defect
forces a separately reviewed scope decision.

## Validation target

The implementation must document the final exact filenames and commands in
[`TESTING.md`](TESTING.md). At minimum run:

```text
- Stage 4 Python/config syntax and deterministic manifest checks;
- focused paper-recipe, scheduler, optimizer, no-clipping, tokenizer, data,
  packing, and accounting tests;
- C1 packed-text and architecture regressions;
- P1-preflight B focused checkpointing regression;
- P1-preflight A standard-library evidence-tooling suite;
- formula units, oracle self-check, and oracle smoke;
- P0-1 and P0-2 quick baseline;
- stronger CPU/CUDA P0 comparisons if any core model/config/oracle/cache source
  changes;
- Psi=8 then Psi=16 CUDA bf16 operational and peak-exposure diagnostics;
- JSON, workflow YAML, Markdown-link, diff, privacy, artifact, symlink, and size
  hygiene.
```

These are targets, not statements that they have passed. Record exact commands,
environment versions, warnings, counts, tolerances, exit status, and unavailable
coverage in the later Stage 4 result record.

## Evidence handling

Use the P1-preflight A provenance, packaging, sanitization, and offline verifier
contracts. Raw metrics, outputs, failed-attempt artifacts, checkpoints, model
weights, tokenizer copies, and source documents remain ignored and outside Git.
Only compact sanitized summaries, deterministic manifests, and evidence
descriptors may be committed after inspection.

Stage 4 acceptance does not complete the historical P0-4 retention descriptor.
Do not infer an acceptance reviewer from ambient GitHub or Hugging Face
identity. Final Level 1 acceptance still requires the explicitly configured
reviewer and durable external archive inputs defined by the program.

## Acceptance boundary

P0.5-C3 is locally ready for draft-PR review only when:

```text
- the exact paper recipe and all repository operationalizations are separately
  labeled and deterministically serialized;
- scheduler points 0, 1, 4095, 4096, and 4097 match the defined contract;
- optimizer betas, zero weight decay, and absence of clipping are executable
  assertions;
- tokenizer revision/assets and selected dataset provenance/fingerprints are
  verified rather than inferred;
- EOS continuous-stream packing and all token counts are exact;
- Psi=8 operational and peak-exposure diagnostics pass and are reviewed before
  Psi=16 begins;
- Psi=16 operational and peak-exposure diagnostics pass;
- every diagnostic uses CUDA bf16 context 4096 and records its reduced local
  batch/warmup without paper-scale claims;
- required P0, C1, checkpointing, evidence-tooling, syntax, link, diff, privacy,
  and artifact-hygiene checks pass;
- no accepted historical evidence is rewritten;
- one focused draft PR is opened and the stage stops at REVIEW_REQUIRED.
```

Merge review remains a separate acceptance act. Until the focused Stage 4 PR
is reviewed and merged, final Level 1 requalification must not begin.

## Explicit exclusions

```text
- paper token budget, paper global-batch execution, or paper-scale training;
- a quality, convergence, superiority, scaling-law, or benchmark claim;
- dense long-context feasibility, retrieval evaluation, or efficiency claims;
- a new qualifying P0-4 run or final Level 1 requalification;
- model/config/oracle/MiPE/cache/state-dict/generation semantic changes;
- PEFT/LoRA/QLoRA/Unsloth, frozen-base adapters, compile, serving, distributed
  training, or Triton/window-skipping kernels;
- completion of historical P0-4 exact/private retention or acceptance review;
- raw dataset text, logs, outputs, checkpoints, weights, archives, secrets, or
  private absolute paths in Git.
```
