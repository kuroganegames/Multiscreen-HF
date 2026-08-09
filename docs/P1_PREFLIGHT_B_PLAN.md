# P1-preflight B Plan: Gradient-Checkpointing API Modernization

## Status

```text
Gate: P1-preflight B
State: implementation and focused local validation passed
Acceptance: REVIEW_REQUIRED; pending focused draft-PR review and merge
Accepted P0 boundary: unchanged through P0-4
P1 model/ecosystem capabilities validated by this gate: none
```

This is the third of five separately reviewed Level 1 Core stages. P0.5-C1
was accepted by merged PR #9. P0.5-C2 was accepted by merged PR #10, and its
separate CUDA-autocast cache-dtype correction was merged as PR #11. This gate
does not combine C2, P0.5-C3, final requalification, or PEFT work.

## Objective

Replace Multiscreen's legacy Transformers checkpointing hook with the supported
runtime contract while preserving the previously validated non-reentrant
training behavior. The forward path must invoke the checkpoint function
installed by Transformers, not a directly imported function that ignores
runtime kwargs.

## Provenance

```text
Stage 3 base / PR #11 merge: 0c83be6b4b043f4b965df4528534f24e9d5ab4f1
branch: agent/p1-preflight-b-gradient-checkpointing
base relation at branch creation: HEAD == origin/main
base worktree: clean
base porcelain bytes: 0
base porcelain SHA-256: e3b0c44298fc1c149afbf4f8996fb92427ae41e4649b934ca495991b7852b855
source/API audit date: 2026-08-09
```

An earlier dirty Stage 3 worktree based on the PR #10 merge was preserved as a
read-only transfer source. Its tracked diff and untracked focused test were
hashed before and after transfer. The current branch was created from the
merged PR #11 `main`, and only the Stage 3 hunks were applied; the PR #11
model/oracle/cache-test correction remains in the base.

## Transformers API audit

The supported contract was inspected in the installed release sources and the
official release records for these exact lanes:

```text
recorded P0-4 lane: Transformers 4.57.6
current supported lane: Transformers 5.14.1
```

Primary release references:

- [Transformers v4.57.6 modeling_utils.py](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)
- [Transformers v5.14.1 modeling_utils.py](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/modeling_utils.py)
- [Transformers 5.14.1 release](https://github.com/huggingface/transformers/releases/tag/v5.14.1)
- [Transformers release history on PyPI](https://pypi.org/project/transformers/)

Both releases expose the new `_set_gradient_checkpointing(enable=...,
gradient_checkpointing_func=...)` implementation. The old Multiscreen override
contained a `value` parameter, so 4.57.6 classified it as the deprecated old
format and ignored supplied kwargs. Release 4.57.6 defaults checkpointing to
`use_reentrant=True`; release 5.14.1 defaults to `False`. Multiscreen therefore
sets a cross-version default of `False` without mutating or overriding an
explicit caller value.

PyPI marks 4.57.0 as yanked because of setup/installation problems. The active
declared lower bound and exact lower-bound CI lane are raised to the already
recorded, non-yanked 4.57.6 patch release. The current lane is pinned exactly to
5.14.1 rather than floating during validation.

## Implementation contract

```text
- remove the model's legacy `_set_gradient_checkpointing` override;
- inherit the supported Transformers propagation implementation;
- retain `supports_gradient_checkpointing = True`;
- default `gradient_checkpointing_enable()` to `use_reentrant=False`;
- copy the caller kwargs mapping before adding that default;
- keep caller-supplied checkpoint kwargs intact;
- initialize the runtime boolean false until Transformers installs a function;
- invoke `self._gradient_checkpointing_func` in each checkpointed layer;
- keep checkpointing disabled in eval and when the runtime flag is false;
- do not serialize transient function objects or checkpoint kwargs.
```

The P0-3 and P0-4 harnesses enable checkpointing at runtime with explicit
`use_reentrant=False` and record that setting in their diagnostic metrics.
P0-3 also narrowly loads the committed Transformers-5 tokenizer metadata under
4.57.6 through `PreTrainedTokenizerFast` when and only when AutoTokenizer
reports the known `TokenizersBackend` class-resolution error.

## Focused executable contract

The focused suite must run under both exact Transformers lanes and prove:

```text
- supported enable/disable propagation;
- inherited hook has no legacy `value` parameter;
- non-reentrant default and arbitrary kwargs are installed;
- caller kwargs are not mutated;
- disable prevents the installed function from being invoked;
- config opt-in installs a callable before the runtime flag is true;
- custom checkpoint-function injection is invoked once per layer;
- the first checkpoint input may have requires_grad=False without the
  reentrant missing-input-gradient warning or lost layer gradients;
- deterministic logits, loss, and every parameter gradient match the plain path;
- forward, loss, gradients, and one optimizer step remain finite;
- transient functions/kwargs are absent from state_dict and config JSON;
- save/reload logits and greedy generated token IDs match exactly;
- the committed P0-3 tokenizer loads with identical vocabulary and special IDs
  in both compatibility lanes.
```

The small deterministic comparison tolerances are `rtol=1e-5, atol=1e-6` for
logits and gradients and `rtol=1e-6, atol=1e-7` for loss. Save/reload logits
and generated token IDs are required to match exactly.

## Required regression and smoke matrix

Because the model training path changes, final local validation includes:

```text
- focused contract under Transformers 4.57.6 and 5.14.1;
- all repository unit tests in the recorded lane;
- formula units, oracle self-check, and oracle smoke;
- P0-1 full CPU fp32 and CUDA bf16;
- P0-2 full CPU fp32 and CUDA bf16;
- C1 focused contracts and manifest check;
- C2 focused position/cache contract, including CUDA bf16 regression;
- P1-preflight A standard-library-only evidence-tooling suite;
- P0-3 checkpointed CUDA bf16 Psi=8/Psi=16 short smoke;
- reduced P0-4 checkpointed CUDA bf16 diagnostic;
- save/load, cache split, and greedy generation postchecks in both smokes;
- Python syntax, JSON, workflow YAML, Markdown links, diff, and artifact hygiene.
```

The P0-3 and reduced P0-4 runs are Stage 3 diagnostics. They do not replace or
rewrite the accepted historical P0-3/P0-4 evidence. Full P0-4 requalification
is reserved for Stage 5.

## Evidence handling

Compact truthful results belong in
`docs/validation_results/P1_PREFLIGHT_B_SUMMARY.md`. Raw logs and smoke outputs
remain outside Git. No checkpoint, tokenizer copy, generated weight, raw log,
archive, environment, cache, or private absolute path may be committed.

Historical P0-4 evidence retention remains partial/blocked exactly as already
recorded. This gate neither supplies an acceptance reviewer for that historical
handoff nor creates durable private retention.

## Acceptance boundary

P1-preflight B is locally ready for draft-PR review only when the exact API
matrix and all required regressions pass, both checkpointed smokes complete
their postchecks, and final scope/privacy/hygiene inspection is clean. Merge
review is a separate acceptance act. Until the focused draft PR is reviewed and
merged, this stage remains `REVIEW_REQUIRED` and Stage 4 must not begin.

## Explicit exclusions

```text
- PEFT/LoRA/QLoRA/Unsloth or frozen-base adapter validation;
- paper optimizer/data/scheduler changes from P0.5-C3;
- a new qualifying P0-4 run or final Level 1 requalification;
- cache, MiPE, long-position, or oracle semantic changes;
- broad generation, beam search, serving, compile, distributed, or Triton work;
- efficiency, paper-scale, retrieval-quality, or production claims.
```

Transformers 5.14.1 installs an input-embedding gradient hook when checkpointing
is enabled for `input_ids`. Multiscreen currently performs normalized embedding
lookup with `F.embedding`, so that module hook is not the basis of this gate's
full-parameter training correctness. The required non-reentrant no-grad-input
test proves the current full-model path. Frozen-base adapter training remains a
future PEFT-stage contract and is not inferred here.
