# Testing Commands

For project handoff context and recommended next steps, see [HANDOFF.md](HANDOFF.md).

For the long-running local Codex workflow that executes these checks and records P0-4 evidence, see [CODEX_P0_4_HANDOFF.md](CODEX_P0_4_HANDOFF.md).

Run all commands from the repository root unless noted otherwise.

## Setup

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/oracle
```

## P0.5-C1

P0.5-C1 checks the paper architecture, initialization, all-scale state shapes,
normalized tied embeddings, config round trips, and packed-text contract. Its
Psi=8/16/32/48/64 model construction is meta-only and does not allocate real
paper-scale weights.

```bash
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

See [P0_5_C1_PLAN.md](P0_5_C1_PLAN.md) for the independent count derivation and
[P0_5_C1_SUMMARY.md](validation_results/P0_5_C1_SUMMARY.md) for the focused
result. C1 was reviewed and merged as PR #9.

## P0.5-C2

P0.5-C2 separates the literal paper MiPE rule
(`paper_absolute`) from the historical compatibility rule
(`reference_mod_after_wrap_boundary`). The reference boundary is
inclusive: wrapping begins at
`mipe_reference_wrap_boundary`. Missing fields migrate to reference
mode with the boundary resolved from `max_position_embeddings`;
paper mode is always an explicit choice.

Run the focused C2 position/config/cache suite first:

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

python -m unittest discover -s tests -p 'test_paper_architecture_contract.py' -v
```

Then run the stable oracle checks and the strongest required P0-1/P0-2 CPU and
CUDA bf16 comparisons:

```bash
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

Stable paper/long-boundary correctness uses fp32 auxiliary MiPE and Softmask
math. The low-precision `reference` auxiliary mode is a separate
compatibility lane; bf16 may collapse distinct large positions and fp16 may
become non-finite. A direct test at position 131,071 does not allocate a dense
131K forward and must not be reported as long-context feasibility or
efficiency.

See [P0_5_C2_PLAN.md](P0_5_C2_PLAN.md), the accepted
[MiPE position ADR](adr/ADR-0001-mipe-position-semantics.md), and the compact
[C2 result](validation_results/P0_5_C2_SUMMARY.md). C2 was merged as PR #10;
its separate CUDA-autocast cache-dtype correction was merged as PR #11.

## P1-preflight B

P1-preflight B tests the supported non-reentrant Transformers
gradient-checkpointing API. Run the focused suite under exact isolated 4.57.6
and 5.14.1 lanes; do not substitute a floating resolver result.
Set `TF4576_PYTHON` and `TF5141_PYTHON` to the corresponding exact-lane
Python executables. The commands fail early if either variable is unset or the
executable does not exist.

```bash
set -euo pipefail
: "${TF4576_PYTHON:?set TF4576_PYTHON to the exact 4.57.6 lane executable}"
: "${TF5141_PYTHON:?set TF5141_PYTHON to the exact 5.14.1 lane executable}"
test -x "$TF4576_PYTHON"
test -x "$TF5141_PYTHON"

export PYTHONPATH=$PWD:$PWD/oracle:$PWD/third_party/multiscreen-pytorch

"$TF4576_PYTHON" -m unittest discover \
  -s tests \
  -p 'test_gradient_checkpointing_contract.py' \
  -v

"$TF5141_PYTHON" -m unittest discover \
  -s tests \
  -p 'test_gradient_checkpointing_contract.py' \
  -v
```

Then run the full P0-1/P0-2 CPU fp32 and CUDA bf16 commands below. The
Stage 3 CUDA training-path checks use the recorded 4.57.6 lane.
Before running them, set `HF_CACHE_DIR` to an existing local Hugging Face
cache and `STAGE3_OUTPUT_ROOT` to an absolute writable path outside the
repository. The commands fail early if either variable is unset.

```bash
set -euo pipefail
: "${TF4576_PYTHON:?set TF4576_PYTHON to the exact 4.57.6 lane executable}"
test -x "$TF4576_PYTHON"
: "${HF_CACHE_DIR:?set HF_CACHE_DIR to an existing local cache}"
: "${STAGE3_OUTPUT_ROOT:?set STAGE3_OUTPUT_ROOT outside the repository}"
test -d "$HF_CACHE_DIR"
case "$STAGE3_OUTPUT_ROOT" in
  /*) ;;
  *) echo "STAGE3_OUTPUT_ROOT must be absolute" >&2; exit 2 ;;
esac
case "$STAGE3_OUTPUT_ROOT/" in
  "$PWD/"*) echo "STAGE3_OUTPUT_ROOT must be outside the repository" >&2; exit 2 ;;
esac
mkdir -p "$STAGE3_OUTPUT_ROOT"

"$TF4576_PYTHON" scripts/p0_3_tinystories_stability.py \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --cache-dir "$HF_CACHE_DIR" \
  --psi 8 16 \
  --steps-per-psi 8:40,16:25 \
  --seq-len 128 \
  --batch-size 4 \
  --max-texts 20000 \
  --max-train-tokens 262144 \
  --device cuda:0 \
  --amp-dtype bf16 \
  --model-compute-dtype fp32 \
  --gradient-checkpointing true \
  --output-dir "$STAGE3_OUTPUT_ROOT/p0-3-checkpointed"

"$TF4576_PYTHON" scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --cache-dir "$HF_CACHE_DIR" \
  --device cuda:0 \
  --amp-dtype bf16 \
  --seq-len 1024 \
  --steps 4 \
  --gradient-accumulation-steps 1 \
  --microbatch-size 1 \
  --gradient-checkpointing true \
  --fused-adamw false \
  --max-texts 20000 \
  --max-train-tokens 32769 \
  --output-dir "$STAGE3_OUTPUT_ROOT/p0-4-checkpointed-diagnostic"
```

The second command must produce `P0-4_DIAGNOSTIC_COMPLETE.md`, not a qualifying
completion marker. See [P1_PREFLIGHT_B_PLAN.md](P1_PREFLIGHT_B_PLAN.md) and the
[Stage 3 result](validation_results/P1_PREFLIGHT_B_SUMMARY.md). Focused PR #12
was reviewed and merged; the commands remain the accepted regression recipe.

## P0.5-C3

P0.5-C3 separates the exact paper recipe, pinned data identity, reduced CUDA
operation, and exact-peak exposure into four evidence lanes. The offline unit
suite mocks Hub access; the data and CUDA commands require the exact pinned
assets and fail closed on a revision, shard hash, library version, fingerprint,
token count, or contract mismatch.

When the production loader sees both `HF_DATASETS_OFFLINE=1` and
`HF_HUB_OFFLINE=1`, it uses a deliberately narrow datasets 5.0.1 prepared-cache
rehydration route for the fixed C3 SlimPajama source. That route requires the
canonical absolute cache root and exact non-symlink prepared-cache layout, and
binds the `Cache` implementation to the installed `datasets` distribution. It
still verifies the pinned raw parquet revision, size, SHA-256, full and selected
fingerprints, and all selected-row records. This is not a general offline
fallback; online production and injected test loaders keep their existing
behavior, and the C3 output schema and training semantics are unchanged.

```bash
python -m py_compile \
  scripts/p0_5_c3_paper_training_contract.py \
  tests/test_paper_training_contract.py

python -m unittest discover \
  -s tests \
  -p 'test_paper_training_contract.py' \
  -v

python scripts/p0_5_c3_paper_training_contract.py --mode contract
```

Run the focused suite in separate exact Transformers 4.57.6 and 5.14.1
environments. The `gradient-checkpointing-compat` CI matrix installs and
verifies both exact versions, then runs this same C3 focused suite in each lane.

For the pinned data and CUDA bf16 lanes, set an existing Hugging Face cache and
an absolute output root outside every Git worktree. Every leaf output directory
must be new: the harness refuses to overwrite a previous success or failure.

```bash
set -euo pipefail
: "${HF_CACHE_DIR:?set HF_CACHE_DIR to an existing Hugging Face cache}"
: "${STAGE4_OUTPUT_ROOT:?set STAGE4_OUTPUT_ROOT outside every Git worktree}"
test -d "$HF_CACHE_DIR"
case "$STAGE4_OUTPUT_ROOT" in
  /*) ;;
  *) echo "STAGE4_OUTPUT_ROOT must be absolute" >&2; exit 2 ;;
esac
case "$STAGE4_OUTPUT_ROOT/" in
  "$PWD/"*) echo "STAGE4_OUTPUT_ROOT must be outside the repository" >&2; exit 2 ;;
esac

python scripts/p0_5_c3_paper_training_contract.py \
  --mode data \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/data"

CUDA_VISIBLE_DEVICES=0 python scripts/p0_5_c3_paper_training_contract.py \
  --mode operational \
  --psi 8 \
  --device cuda:0 \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/cuda/psi8/operational"

CUDA_VISIBLE_DEVICES=0 python scripts/p0_5_c3_paper_training_contract.py \
  --mode peak-exposure \
  --psi 8 \
  --device cuda:0 \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/cuda/psi8/peak_exposure"
```

Inspect every Psi=8 metric, completion marker, memory value, and any preserved
`failure.json` before proceeding. Only after Psi=8 passes and memory headroom
is understood, run the corresponding Psi=16 lanes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/p0_5_c3_paper_training_contract.py \
  --mode operational \
  --psi 16 \
  --device cuda:0 \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/cuda/psi16/operational"

CUDA_VISIBLE_DEVICES=0 python scripts/p0_5_c3_paper_training_contract.py \
  --mode peak-exposure \
  --psi 16 \
  --device cuda:0 \
  --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$STAGE4_OUTPUT_ROOT/cuda/psi16/peak_exposure"
```

The operational lane is fixed at context 4096, microbatch 1, accumulation 2,
three optimizer updates, reduced two-update warmup, and diagnostic LR 0.0006.
The separate peak lane performs one bounded context-4096 update at the exact
paper peak LR 0.0625. Both use CUDA bf16, paper-absolute MiPE, fp32 auxiliary
MiPE/Softmask math, AdamW betas `(0.9, 0.95)`, zero weight decay, the explicitly
labeled repository epsilon `1e-8`, non-fused AdamW, supported non-reentrant
checkpointing, and no clipping. Both summaries therefore record
`diagnostic_reduced_from_paper=true`: the operational lane reduces warmup and
LR, while peak exposure reduces warmup even though its exposed peak LR is exact.
Neither lane reproduces the paper global batch,
duration, corpus selection, training precision, quality, or efficiency.

The inspected result is recorded in
[P0_5_C3_SUMMARY.md](validation_results/P0_5_C3_SUMMARY.md), with its historical
packaging record in
[P0_5_C3_EVIDENCE_ARCHIVE.json](validation_results/P0_5_C3_EVIDENCE_ARCHIVE.json).
Stage 4 was reviewed and accepted by merged PR #13. The historical descriptor
remains an accurate packaging-time partial snapshot. A later
[external-retention closure](validation_results/P0_5_C3_EVIDENCE_CLOSURE.json)
records exact/private external retention and verification plus sanitized archive
reverification. Codex reviewed all 26 source artifacts and all 8 optimizer-step
raw events; acceptance review is recorded and overall evidence status is
complete.
Neither archive is published, and the exact archive must remain private.

After restoring both archive files to explicit local paths, reverify them fully offline:

```bash
export MULTISCREEN_C3_EXACT_ARCHIVE=/absolute/private/path/validation-evidence-exact-p0-5-c3-8fa5dbf1-v2.tar.gz
export MULTISCREEN_C3_SANITIZED_ARCHIVE=/absolute/staging/path/validation-evidence-sanitized-p0-5-c3-8fa5dbf1-v2.tar.gz

C3_CLOSURE=docs/validation_results/P0_5_C3_EVIDENCE_CLOSURE.json
C3_SCHEMA=schemas/validation_evidence_v1.schema.json

python -S scripts/verify_validation_evidence.py \
  --archive "$MULTISCREEN_C3_EXACT_ARCHIVE" \
  --expected-sha256 db882b8eb5d871b4ca8696a324d4a67aa6bd36389dd173db4ea857587d57319e \
  --evidence-document "$C3_CLOSURE" \
  --schema "$C3_SCHEMA" \
  --json

python -S scripts/verify_validation_evidence.py \
  --archive "$MULTISCREEN_C3_SANITIZED_ARCHIVE" \
  --expected-sha256 274e489f4b4872f8f8c797b56b9d49aebc3a8c0e005fe2c65694f136616a9573 \
  --evidence-document "$C3_CLOSURE" \
  --schema "$C3_SCHEMA" \
  --json
```

The separate final Level 1 requalification subsequently passed locally under
[LEVEL1_CORE_REQUALIFICATION_PLAN.md](LEVEL1_CORE_REQUALIFICATION_PLAN.md). Its
reviewed [summary](validation_results/LEVEL1_CORE_SUMMARY.md) and complete
[evidence descriptor](validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json)
record the Stage 5 result, which was reviewed and accepted as merged PR #14.
This remains an unofficial correctness-first result; the dense quadratic path
is not efficiency evidence, and it does not validate paper-scale reproduction,
retrieval benchmarks, optimized long-context efficiency, distributed training,
or any P1 model/ecosystem capability.

## P0-1

```bash
python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
python oracle/test_against_hf_port.py --quick
python oracle/test_against_hf_port.py
python oracle/test_against_hf_port.py --device cuda:0 --dtype bf16
python oracle/test_against_hf_port.py --device cuda:0 --dtype fp16 --quick
```

## P0-2

```bash
python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --quick

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

## P0-3

```bash
python scripts/p0_3_tinystories_stability.py \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --tokenizer-path tokenizers/tinystories_spm768 \
  --cache-dir /path/to/hf_cache \
  --device cuda:0 \
  --amp-dtype bf16 \
  --seq-len 128 \
  --batch-size 4 \
  --steps-per-psi 8:40,16:25 \
  --output-dir outputs/p0_3_tinystories_stability
```

If tokenizer files are missing:

```bash
python scripts/train_tokenizer_spm.py \
  --dataset_name roneneldan/TinyStories \
  --split train \
  --text_column text \
  --vocab_size 768 \
  --max_samples 200000 \
  --model_max_length 512 \
  --output_dir tokenizers/tinystories_spm768 \
  --cache_dir /path/to/hf_cache
```

## P0-4

P0-4 qualifying CUDA bf16 runs passed for Psi=8 and Psi=16. Reviewed results and raw-artifact hashes are recorded in [P0_4_SUMMARY.md](validation_results/P0_4_SUMMARY.md) and [P0_4_SUMMARY.json](validation_results/P0_4_SUMMARY.json); the commands below remain the strict reproduction procedure.

Static config preflight; this does not download the tokenizer or dataset:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --validate-config-only

python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --validate-config-only
```

Optional non-qualifying 1024-token diagnostic:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --seq-len 1024 \
  --steps 2 \
  --gradient-accumulation-steps 1 \
  --output-dir outputs/p0_4_psi8_ctx1024_diagnostic
```

Qualifying Psi=8 reproduction:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096
```

Run the qualifying Psi=16 reproduction only after the new Psi=8 output is reviewed:

```bash
python scripts/p0_4_gpt2_context4096_smoke.py \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096
```

A qualifying run uses GPT-2 vocab 50,257, sequence length 4,096, CUDA bf16,
microbatch 1, at least 50 actually completed optimizer steps, runtime gradient
checkpointing enabled, and the supported non-reentrant checkpointing path
(`use_reentrant=False`). Gradient accumulation is not a qualification
condition. A run that misses any qualification condition writes
`P0-4_DIAGNOSTIC_COMPLETE.md`, not `P0-4_COMPLETE.md`.

Each fresh run writes a canonical `data_contract.json` before training. It
binds the resolved default TinyStories source and fingerprint, the ordered
selected-text manifest, the exact packed `uint32` token stream and packing
parameters, and the normalized GPT-2 tokenizer identity projection. Its digest
must occur exactly once in stdout and match the run-start, preflight, summary,
and run-complete references. The final Level 1 focused and full reviewers also
require the Psi=8 and Psi=16 contracts to be byte-identical and cross-bind the
tokenizer projection to the single-cache offline preflight and checkpoint
reload reports.

For a future reproduction, retain its ignored raw artifacts under a distinct output directory, compare them with the accepted historical record, and add a new compact sanitized record rather than overwriting the existing evidence. Rerun the P0-1/P0-2 quick suite and both config preflights.

## Final Level 1 requalification

The exact Stage 5 execution matrix passed locally on tested source
`b224ca1a127ee18fc5fd4b00a5df639401d60679`. The reviewed result is recorded in
[LEVEL1_CORE_SUMMARY.md](validation_results/LEVEL1_CORE_SUMMARY.md); complete
private retention, sanitization, and verification are recorded in
[LEVEL1_CORE_EVIDENCE_ARCHIVE.json](validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json).
Neither archive is published. The focused Stage 5 result was reviewed and
accepted as merged PR #14.
This remains an unofficial correctness-first result; the dense quadratic path
is not efficiency evidence, and it does not validate paper-scale reproduction,
retrieval benchmarks, optimized long-context efficiency, distributed training,
or any P1 model/ecosystem capability.

The commands below remain the exact future reproduction and closure procedure.
They apply only after P0.5-C3 was accepted by merged PR #13 and after the
evidence-support changes in this section are committed. They do not reuse
historical P0-4 or C3 metrics as evidence for a new tested commit.

A future final execution requires an explicitly named reviewer and durable private
storage. The run root must be a new absolute path outside every Git worktree;
the archive directory and Hugging Face cache must already exist. The setup uses
an owner-only default umask for every external private artifact it creates. Do
not infer the reviewer from an authenticated GitHub account.

```bash
set -euo pipefail
umask 077

: "${TF4576_PYTHON:?exact Transformers 4.57.6 Python is required}"
: "${TF5141_PYTHON:?exact Transformers 5.14.1 Python is required}"
: "${HF_CACHE_DIR:?existing Hugging Face cache is required}"
: "${LEVEL1_RUN_ROOT:?new absolute private run root is required}"
: "${MULTISCREEN_EVIDENCE_REVIEWERS:?explicit reviewer list is required}"
: "${MULTISCREEN_EVIDENCE_ARCHIVE_DIR:?durable private archive directory is required}"

test -x "$TF4576_PYTHON"
test -x "$TF5141_PYTHON"
test -d "$HF_CACHE_DIR"
test -d "$MULTISCREEN_EVIDENCE_ARCHIVE_DIR"
test -d "$(dirname "$LEVEL1_RUN_ROOT")"
test ! -e "$LEVEL1_RUN_ROOT"

case "$TF4576_PYTHON" in /*) ;; *) exit 2 ;; esac
case "$TF5141_PYTHON" in /*) ;; *) exit 2 ;; esac
case "$LEVEL1_RUN_ROOT" in /*) ;; *) exit 2 ;; esac
case "$HF_CACHE_DIR" in /*) ;; *) exit 2 ;; esac
case "$MULTISCREEN_EVIDENCE_ARCHIVE_DIR" in /*) ;; *) exit 2 ;; esac
test "$MULTISCREEN_EVIDENCE_ARCHIVE_DIR" != /
test -w "$MULTISCREEN_EVIDENCE_ARCHIVE_DIR"
test "$(realpath -e "$MULTISCREEN_EVIDENCE_ARCHIVE_DIR")" = \
  "$MULTISCREEN_EVIDENCE_ARCHIVE_DIR"

LEVEL1_REPO=$(git rev-parse --show-toplevel)
cd -P "$LEVEL1_REPO"
test "$PWD" = "$LEVEL1_REPO"
test "$(git symbolic-ref --quiet --short HEAD)" = \
  validation/level1-core-requalification
while IFS= read -r LEVEL1_WORKTREE; do
  case "$MULTISCREEN_EVIDENCE_ARCHIVE_DIR/" in
    "$LEVEL1_WORKTREE/"*) exit 2 ;;
  esac
done < <(git worktree list --porcelain | sed -n 's/^worktree //p')
unset LEVEL1_WORKTREE
LEVEL1_TESTED_COMMIT=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)"

LEVEL1_COMMON_ENV=(
  /usr/bin/env -i
  HOME="$LEVEL1_RUN_ROOT"
  PATH=/usr/bin:/bin
  LANG=C.UTF-8
  LC_ALL=C.UTF-8
  TZ=UTC
  HF_DATASETS_DISABLE_PROGRESS_BARS=1
  HF_DATASETS_OFFLINE=1
  HF_HUB_DISABLE_TELEMETRY=1
  HF_HUB_OFFLINE=1
  PYTHONDONTWRITEBYTECODE=1
  PYTHONHASHSEED=0
  PYTHONNOUSERSITE=1
  PYTHONOPTIMIZE=0
  PYTHONUNBUFFERED=1
  PYTHONUTF8=1
  TOKENIZERS_PARALLELISM=false
  TRANSFORMERS_OFFLINE=1
)
LEVEL1_CPU_ENV=("${LEVEL1_COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES=)
LEVEL1_CUDA_ENV=(
  "${LEVEL1_COMMON_ENV[@]}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=0
)
readonly -a LEVEL1_COMMON_ENV LEVEL1_CPU_ENV LEVEL1_CUDA_ENV

record4576() {
  "${LEVEL1_CPU_ENV[@]}" \
    "$TF4576_PYTHON" -P -S -B scripts/run_level1_requalification_command.py \
    --repo-root "$LEVEL1_REPO" \
    --run-root "$LEVEL1_RUN_ROOT" \
    "$@"
}

record5141() {
  "${LEVEL1_CPU_ENV[@]}" \
    "$TF5141_PYTHON" -P -S -B scripts/run_level1_requalification_command.py \
    --repo-root "$LEVEL1_REPO" \
    --run-root "$LEVEL1_RUN_ROOT" \
    "$@"
}
```

The recorder owns the private run root at mode `0700`, reserves every command
name once, streams complete merged output, and records log size/hash and exit
status. `--require-absent` records and checks fresh run-root-relative output
paths immediately before the child starts. Never rerun a failed name or reuse
a prior run root.

### Environment and repository preflight

Create the run root with two dry runtime records, then record the exact package
and CUDA identities. These commands contain no cache path, token, username, or
other ambient environment value in their reports.

```bash
record4576 --name runtime-tf4576 --environment-record
record5141 --name runtime-tf5141 --environment-record

record4576 --name environment-tf4576 -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" scripts/report_level1_environment.py --lane tf4576

record5141 --name environment-tf5141 -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF5141_PYTHON" scripts/report_level1_environment.py --lane tf5141

record4576 --name environment-cuda0 -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  "$TF4576_PYTHON" scripts/report_level1_environment.py --lane cuda0

record4576 --name offline-cache-preflight -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -B scripts/check_level1_offline_cache.py \
  --repo-root "$LEVEL1_REPO" --cache-dir "$HF_CACHE_DIR"

record4576 --name repository-hygiene -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -S -B scripts/check_level1_repository.py \
  --repo-root "$LEVEL1_REPO" --check hygiene
```

`offline-cache-preflight` must record exactly one canonical, path-free JSON
object proving that the one explicit cache satisfies all fixed P0-3, P0-4, and
C3 public input identities while every loader is offline. If it fails, stop
immediately and preserve the private run root for diagnosis; do not continue to
repository hygiene, tests, or CUDA work in that run.

For C3, this preflight also exercises the fixed datasets 5.0.1 prepared-cache
rehydration route. The cache must contain both the pinned raw parquet object and
the exact prepared output; the harness independently rechecks their bound
identity, the full and selected dataset fingerprints, and the 64 selected row
records before any C3 CUDA lane may start.

The first hygiene report is the clean tested-source observation. It must name
`LEVEL1_TESTED_COMMIT`, record the SHA-256 of empty porcelain bytes, and finish
before `syntax-level1` starts.

### Evidence support and repository-wide static checks

```bash
record4576 --name syntax-level1 \
  --require-absent pycache/syntax-level1 -- \
  "${LEVEL1_CPU_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  PYTHONPYCACHEPREFIX="$LEVEL1_RUN_ROOT/pycache/syntax-level1" \
  "$TF4576_PYTHON" -m py_compile \
  $(git ls-files '*.py')

record4576 --name level1-evidence-support-tests -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S -m unittest discover \
  -s tests -p 'test_level1_*.py' -v

record4576 --name tokenizer-reload-tests-tf4576 -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -m unittest discover \
  -s tests -p 'test_tokenizer_reload_check.py' -v

record5141 --name tokenizer-reload-tests-tf5141 -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF5141_PYTHON" -m unittest discover \
  -s tests -p 'test_tokenizer_reload_check.py' -v

record4576 --name validation-evidence-tests -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S -m unittest discover \
  -s tests -p 'test_validation_evidence*.py' -v

record4576 --name json-validation -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -S -B scripts/check_level1_repository.py \
  --repo-root "$LEVEL1_REPO" --check json

record4576 --name workflow-yaml -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -S -B scripts/check_level1_repository.py \
  --repo-root "$LEVEL1_REPO" --check workflow-yaml

record4576 --name markdown-links -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -S -B scripts/check_level1_repository.py \
  --repo-root "$LEVEL1_REPO" --check markdown-links
```

### C1, C2, checkpointing, and full P0 regressions

Run the architecture and initialization contracts in the recorded 4.57.6
lane. C2 runs with CUDA visible so its CUDA-autocast cache-dtype regression is
not skipped. Gradient checkpointing is checked in both exact lanes.

```bash
record4576 --name c1-architecture -- \
  "${LEVEL1_CPU_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" -m unittest discover \
  -s tests -p 'test_paper_architecture_contract.py' -v

record4576 --name c1-initialization -- \
  "${LEVEL1_CPU_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" -m unittest discover \
  -s tests -p 'test_paper_initialization_contract.py' -v

record4576 --name c1-packed-data -- \
  "${LEVEL1_CPU_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" -m unittest discover \
  -s tests -p 'test_packed_text_contract.py' -v

record4576 --name c1-manifest -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" scripts/generate_paper_scale_manifest.py \
  --check docs/validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json

record4576 --name c2-position-cache -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" -m unittest discover \
  -s tests -p 'test_mipe_position_cache_contract.py' -v

record4576 --name gradient-checkpointing-tf4576 -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" -m unittest discover \
  -s tests -p 'test_gradient_checkpointing_contract.py' -v

record5141 --name gradient-checkpointing-tf5141 -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF5141_PYTHON" -m unittest discover \
  -s tests -p 'test_gradient_checkpointing_contract.py' -v

record4576 --name formula-units -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" oracle/test_formula_units.py

record4576 --name oracle-selfcheck -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" oracle/test_paper_math_oracle_selfcheck.py

record4576 --name oracle-smoke -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" oracle/test_paper_math_oracle_smoke.py

record4576 --name p0-1-cpu-fp32 -- \
  "${LEVEL1_CPU_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" oracle/test_against_hf_port.py \
  --device cpu --dtype fp32 --seed 1234 --rtol 1e-5 --atol 1e-5

record4576 --name p0-1-cuda-bf16 -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" oracle/test_against_hf_port.py \
  --device cuda:0 --dtype bf16 --seed 1234 --rtol 0.03 --atol 0.03

record4576 --name p0-2-cpu-fp32 -- \
  "${LEVEL1_CPU_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch --hf-root . \
  --oracle-root oracle --device cpu --dtype fp32 --seed 4321 \
  --rtol 1e-5 --atol 1e-5

record4576 --name p0-2-cuda-bf16 -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch --hf-root . \
  --oracle-root oracle --device cuda:0 --dtype bf16 --seed 4321 \
  --rtol 0.03 --atol 0.03
```

Do not add `--quick` or `--no-layer-hooks`. The full P0-1 and P0-2 scripts are
the assertion oracles for their complete case counts.

### C3 contract, pinned data, and CUDA lanes

Run the C3 unit suite in both exact Transformers lanes and the CLI contract in
4.57.6. Then run five fresh outputs in the documented order.

```bash
record4576 --name c3-contracts-tf4576 -- \
  "${LEVEL1_CUDA_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" -m unittest discover \
  -s tests -p 'test_paper_training_contract.py' -v

record5141 --name c3-contracts-tf5141 -- \
  "${LEVEL1_CUDA_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF5141_PYTHON" -m unittest discover \
  -s tests -p 'test_paper_training_contract.py' -v

record4576 --name c3-contract-cli -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" scripts/p0_5_c3_paper_training_contract.py \
  --manifest configs/p0_5_c3_paper_training_contract.json --mode contract

record4576 --name c3-data --require-absent artifacts/c3/data -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" scripts/p0_5_c3_paper_training_contract.py \
  --manifest configs/p0_5_c3_paper_training_contract.json \
  --mode data --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$LEVEL1_RUN_ROOT/artifacts/c3/data"

record4576 --name c3-psi8-operational \
  --require-absent artifacts/c3/cuda/psi8/operational -- \
  "${LEVEL1_CUDA_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" scripts/p0_5_c3_paper_training_contract.py \
  --manifest configs/p0_5_c3_paper_training_contract.json \
  --mode operational --psi 8 --device cuda:0 --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$LEVEL1_RUN_ROOT/artifacts/c3/cuda/psi8/operational"

record4576 --name c3-psi8-peak-exposure \
  --require-absent artifacts/c3/cuda/psi8/peak-exposure -- \
  "${LEVEL1_CUDA_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" scripts/p0_5_c3_paper_training_contract.py \
  --manifest configs/p0_5_c3_paper_training_contract.json \
  --mode peak-exposure --psi 8 --device cuda:0 --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$LEVEL1_RUN_ROOT/artifacts/c3/cuda/psi8/peak-exposure"
```

Stop here. Inspect both Psi=8 metric streams and completion markers, confirm
that neither output contains `failure.json`, and confirm sufficient GPU memory
headroom. Preserve a failure and start no Psi=16 lane if this inspection does
not pass. These runs are bounded dense diagnostics, not paper-scale or
efficiency evidence.

Only after the Psi=8 inspection passes:

```bash

record4576 --name c3-psi16-operational \
  --require-absent artifacts/c3/cuda/psi16/operational -- \
  "${LEVEL1_CUDA_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" scripts/p0_5_c3_paper_training_contract.py \
  --manifest configs/p0_5_c3_paper_training_contract.json \
  --mode operational --psi 16 --device cuda:0 --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$LEVEL1_RUN_ROOT/artifacts/c3/cuda/psi16/operational"

record4576 --name c3-psi16-peak-exposure \
  --require-absent artifacts/c3/cuda/psi16/peak-exposure -- \
  "${LEVEL1_CUDA_ENV[@]}" PYTHONPATH=.:oracle \
  "$TF4576_PYTHON" scripts/p0_5_c3_paper_training_contract.py \
  --manifest configs/p0_5_c3_paper_training_contract.json \
  --mode peak-exposure --psi 16 --device cuda:0 --cache-dir "$HF_CACHE_DIR" \
  --output-dir "$LEVEL1_RUN_ROOT/artifacts/c3/cuda/psi16/peak-exposure"
```

Inspect the two Psi=16 outputs with the same criteria before continuing.

### P0-3 checkpointed CUDA bf16 smoke

The single P0-3 command must log every one of its 65 steps. Its two saved
tokenizers are then reloaded independently from the actual checkpoint paths.

```bash
record4576 --name p0-3-checkpointed \
  --require-absent artifacts/p0-3 -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" scripts/p0_3_tinystories_stability.py \
  --repo-root "$LEVEL1_REPO" \
  --tokenizer-path tokenizers/tinystories_spm768 \
  --cache-dir "$HF_CACHE_DIR" --dataset-name roneneldan/TinyStories \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --train-split 'train[:20000]' --text-column text --max-texts 20000 \
  --max-train-tokens 262144 --seq-len 128 --psi 8 16 \
  --steps-per-psi 8:40,16:25 --batch-size 4 --num-workers 0 \
  --device cuda:0 --amp-dtype bf16 --model-compute-dtype fp32 \
  --key-dim 16 --value-dim 64 --gradient-checkpointing true \
  --mipe-threshold 256 --initializer-range 0.1 --learning-rate 0.0006 \
  --weight-decay 0 --max-grad-norm 1 --seed 42 --log-every 1 \
  --train-probe-every 4 --min-loss-drop 0.01 --min-rel-loss-drop 0.001 \
  --reload-atol 1e-5 --reload-rtol 1e-5 \
  --cache-atol 0.03 --cache-rtol 0.03 \
  --prompt 'Once upon a time' --max-new-tokens 12 \
  --output-dir "$LEVEL1_RUN_ROOT/artifacts/p0-3"

record4576 --name p0-3-tokenizer-psi8 \
  --require-absent artifacts/p0-3/tokenizer-reload-psi8.json -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" scripts/check_tokenizer_reload.py \
  --source-tokenizer tokenizers/tinystories_spm768 \
  --checkpoint "$LEVEL1_RUN_ROOT/artifacts/p0-3/psi8" \
  --logical-name p0_3_psi8 --source-id tinystories-spm768 \
  --checkpoint-id p0-3-psi8-checkpoint \
  --output "$LEVEL1_RUN_ROOT/artifacts/p0-3/tokenizer-reload-psi8.json"

record4576 --name p0-3-tokenizer-psi16 \
  --require-absent artifacts/p0-3/tokenizer-reload-psi16.json -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" scripts/check_tokenizer_reload.py \
  --source-tokenizer tokenizers/tinystories_spm768 \
  --checkpoint "$LEVEL1_RUN_ROOT/artifacts/p0-3/psi16" \
  --logical-name p0_3_psi16 --source-id tinystories-spm768 \
  --checkpoint-id p0-3-psi16-checkpoint \
  --output "$LEVEL1_RUN_ROOT/artifacts/p0-3/tokenizer-reload-psi16.json"
```

Do not pass P0-4 source-normalization flags to the P0-3 tokenizer checks.
P0-3 uses its committed tokenizer contract without runtime normalization.
The fresh `artifacts/p0-3/data_contract.json` must be canonical and record the
pinned dataset revision, selected-text manifest, exact packed-token stream,
and tokenizer identity. Its file SHA-256 must match the reference in the
aggregate result and both per-Psi metrics; the full reviewer enforces that
binding.

### P0-4 strict Psi=8 qualification and gate review

Run both config preflights before allocating the qualifying models. The actual
runs repeat every exposed qualification setting on the command line; betas,
epsilon, probe replay, check tolerances, and generation length remain fixed by
the checked `run.json` files and are independently reviewed from raw events.

```bash
record4576 --name p0-4-psi8-preflight -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" scripts/p0_4_gpt2_context4096_smoke.py \
  --repo-root "$LEVEL1_REPO" \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --validate-config-only

record4576 --name p0-4-psi8 \
  --require-absent artifacts/p0-4/psi8 -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" scripts/p0_4_gpt2_context4096_smoke.py \
  --repo-root "$LEVEL1_REPO" \
  --config-dir configs/p0_4_multiscreen_psi8_gpt2_ctx4096 \
  --output-dir "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8" \
  --tokenizer-name-or-path gpt2 --cache-dir "$HF_CACHE_DIR" \
  --dataset-name roneneldan/TinyStories --train-split 'train[:20000]' \
  --text-column text --streaming false --max-texts 20000 \
  --max-train-tokens 524416 --seq-len 4096 --steps 50 \
  --microbatch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 0.0006 --weight-decay 0 --max-grad-norm 1 \
  --amp-dtype bf16 --gradient-checkpointing true --fused-adamw true \
  --device cuda:0 --allow-cpu false --num-workers 0 --seed 42 --log-every 1

record4576 --name p0-4-tokenizer-psi8 \
  --require-absent artifacts/p0-4/psi8/tokenizer-reload.json -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" scripts/check_tokenizer_reload.py \
  --source-tokenizer gpt2 --cache-dir "$HF_CACHE_DIR" \
  --checkpoint "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8/checkpoint" \
  --logical-name p0_4_psi8 --source-id gpt2 \
  --checkpoint-id p0-4-psi8-checkpoint \
  --source-pad-token-from-eos --source-padding-side right \
  --source-model-max-length 4096 \
  --output "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8/tokenizer-reload.json"
```

At this point, stop. Run the focused `p0-4-lane` raw reviewer through the
recorder and preserve its passing report as command `p0-4-review-psi8`. The
focused report must bind the Psi=8 record, log, 57-event stream, completion
marker, tokenizer report, tested commit, and fresh-output observation. It does
not substitute for the explicitly named human evidence reviewer.

Only after that focused report passes may the Psi=16 command begin. The final
full-review command and its exact arguments follow the reviewer CLI shown
below.

```bash
record4576 --name p0-4-review-psi8 \
  --require-absent artifacts/p0-4/psi8/raw-review.json -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -S -B scripts/review_level1_requalification.py \
  --mode p0-4-lane --tested-commit "$LEVEL1_TESTED_COMMIT" \
  --command-ledger "$LEVEL1_RUN_ROOT/commands.jsonl" \
  --p0-4-root "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8" \
  --tokenizer-reload-report \
  "p0_4_psi8=$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8/tokenizer-reload.json" \
  --output "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8/raw-review.json"
```

Stop again and inspect the focused report. Only a passing report permits the
following Psi=16 preflight and fresh qualification:

```bash
record4576 --name p0-4-psi16-preflight -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" scripts/p0_4_gpt2_context4096_smoke.py \
  --repo-root "$LEVEL1_REPO" \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --validate-config-only

record4576 --name p0-4-psi16 \
  --require-absent artifacts/p0-4/psi16 -- \
  "${LEVEL1_CUDA_ENV[@]}" \
  PYTHONPATH=.:oracle:third_party/multiscreen-pytorch \
  "$TF4576_PYTHON" scripts/p0_4_gpt2_context4096_smoke.py \
  --repo-root "$LEVEL1_REPO" \
  --config-dir configs/p0_4_multiscreen_psi16_gpt2_ctx4096 \
  --output-dir "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi16" \
  --tokenizer-name-or-path gpt2 --cache-dir "$HF_CACHE_DIR" \
  --dataset-name roneneldan/TinyStories --train-split 'train[:20000]' \
  --text-column text --streaming false --max-texts 20000 \
  --max-train-tokens 524416 --seq-len 4096 --steps 50 \
  --microbatch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 0.0006 --weight-decay 0 --max-grad-norm 1 \
  --amp-dtype bf16 --gradient-checkpointing true --fused-adamw true \
  --device cuda:0 --allow-cpu false --num-workers 0 --seed 42 --log-every 1

record4576 --name p0-4-tokenizer-psi16 \
  --require-absent artifacts/p0-4/psi16/tokenizer-reload.json -- \
  "${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" scripts/check_tokenizer_reload.py \
  --source-tokenizer gpt2 --cache-dir "$HF_CACHE_DIR" \
  --checkpoint "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi16/checkpoint" \
  --logical-name p0_4_psi16 --source-id gpt2 \
  --checkpoint-id p0-4-psi16-checkpoint \
  --source-pad-token-from-eos --source-padding-side right \
  --source-model-max-length 4096 \
  --output "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi16/tokenizer-reload.json"

record4576 --name repository-hygiene-final -- \
  "${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -S -B scripts/check_level1_repository.py \
  --repo-root "$LEVEL1_REPO" --check hygiene
```

`repository-hygiene-final` is the last recorder command. The ledger must now
contain exactly the fixed command matrix and the two runtime records. Do not
record the full reviewer because doing so would create an extra ledger entry
and a self-reference.

```bash
test ! -e "$LEVEL1_RUN_ROOT/review"
mkdir -m 0700 "$LEVEL1_RUN_ROOT/review"

"${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -S -B scripts/review_level1_requalification.py \
  --mode full --tested-commit "$LEVEL1_TESTED_COMMIT" \
  --command-ledger "$LEVEL1_RUN_ROOT/commands.jsonl" \
  --p0-3-root "$LEVEL1_RUN_ROOT/artifacts/p0-3" \
  --p0-3-stdout "$LEVEL1_RUN_ROOT/logs/p0-3-checkpointed.log" \
  --p0-4-psi8-root "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8" \
  --p0-4-psi16-root "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi16" \
  --p0-4-psi8-review \
  "$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8/raw-review.json" \
  --c3-data-root "$LEVEL1_RUN_ROOT/artifacts/c3/data" \
  --c3-psi8-operational-root \
  "$LEVEL1_RUN_ROOT/artifacts/c3/cuda/psi8/operational" \
  --c3-psi8-peak-exposure-root \
  "$LEVEL1_RUN_ROOT/artifacts/c3/cuda/psi8/peak-exposure" \
  --c3-psi16-operational-root \
  "$LEVEL1_RUN_ROOT/artifacts/c3/cuda/psi16/operational" \
  --c3-psi16-peak-exposure-root \
  "$LEVEL1_RUN_ROOT/artifacts/c3/cuda/psi16/peak-exposure" \
  --tokenizer-reload-report \
  "p0_3_psi8=$LEVEL1_RUN_ROOT/artifacts/p0-3/tokenizer-reload-psi8.json" \
  --tokenizer-reload-report \
  "p0_3_psi16=$LEVEL1_RUN_ROOT/artifacts/p0-3/tokenizer-reload-psi16.json" \
  --tokenizer-reload-report \
  "p0_4_psi8=$LEVEL1_RUN_ROOT/artifacts/p0-4/psi8/tokenizer-reload.json" \
  --tokenizer-reload-report \
  "p0_4_psi16=$LEVEL1_RUN_ROOT/artifacts/p0-4/psi16/tokenizer-reload.json" \
  --output "$LEVEL1_RUN_ROOT/review/level1-core.json"
```

The full machine reviewer is necessary but is not the human acceptance review.
An explicitly named reviewer must next inspect every raw event and lossless log,
the focused and full reports, and the compact artifacts. Preserve any rejected
run. Only after that review actually occurs may provenance record a non-empty
method, the full tested commit, and `raw-events-reviewed=true`.

### Acceptance provenance, archive sealing, and descriptor closure

Stop here until the explicitly named reviewer has genuinely inspected every
raw event and lossless log. The reviewer must supply a concise, path-free
method description. Setting the variables below is an assertion that this
review has actually happened; it is not a substitute for the review.

The commands below keep the exact archive at its final durable private path,
put the sanitized archive in a distinct owner-only staging directory, verify
both archives offline, and use the two-commit closure described in
[LEVEL1_CORE_REQUALIFICATION_PLAN.md](LEVEL1_CORE_REQUALIFICATION_PLAN.md).
Neither archive is published by these commands.

```bash
: "${LEVEL1_REVIEW_METHOD:?actual raw-event review method is required}"

LEVEL1_RESULTS_ROOT="$LEVEL1_REPO/docs/validation_results"
LEVEL1_SCHEMA="$LEVEL1_REPO/schemas/validation_evidence_v1.schema.json"
LEVEL1_IMPLEMENTATION_BASE=3282eae7cb97ecfe01753460f6bce63d03e3cf88
LEVEL1_PACKAGE_REPORT="$LEVEL1_RUN_ROOT/review/package-report.json"
LEVEL1_EXACT_PRIMARY_REPORT="$LEVEL1_RUN_ROOT/review/exact-primary-verification.json"
LEVEL1_SANITIZED_PRIMARY_REPORT="$LEVEL1_RUN_ROOT/review/sanitized-primary-verification.json"
LEVEL1_EXACT_ARCHIVE="$MULTISCREEN_EVIDENCE_ARCHIVE_DIR/level1-core-${LEVEL1_TESTED_COMMIT}.exact.tar.gz"
LEVEL1_SANITIZED_STAGING_DIR="$MULTISCREEN_EVIDENCE_ARCHIVE_DIR/level1-core-${LEVEL1_TESTED_COMMIT}.sanitized-staging"
LEVEL1_SANITIZED_ARCHIVE="$LEVEL1_SANITIZED_STAGING_DIR/level1-core-${LEVEL1_TESTED_COMMIT}.sanitized.tar.gz"
LEVEL1_EXACT_LOCATOR="private-external:level1-core-${LEVEL1_TESTED_COMMIT}"
LEVEL1_SANITIZED_LOCATOR="sanitized-staging:level1-core-${LEVEL1_TESTED_COMMIT}"

test ! -e "$LEVEL1_RUN_ROOT/review/acceptance-provenance.json"
test ! -e "$LEVEL1_PACKAGE_REPORT"
test ! -e "$LEVEL1_EXACT_PRIMARY_REPORT"
test ! -e "$LEVEL1_SANITIZED_PRIMARY_REPORT"
test ! -e "$LEVEL1_EXACT_ARCHIVE"
test ! -e "$LEVEL1_SANITIZED_STAGING_DIR"
mkdir -m 0700 "$LEVEL1_SANITIZED_STAGING_DIR"
test "$(realpath -e "$LEVEL1_SANITIZED_STAGING_DIR")" = \
  "$LEVEL1_SANITIZED_STAGING_DIR"

"${LEVEL1_CPU_ENV[@]}" \
  MULTISCREEN_EVIDENCE_REVIEWERS="$MULTISCREEN_EVIDENCE_REVIEWERS" \
  PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/collect_validation_provenance.py \
  --repo "$LEVEL1_REPO" \
  --review-method "$LEVEL1_REVIEW_METHOD" \
  --review-commit "$LEVEL1_TESTED_COMMIT" \
  --raw-events-reviewed true \
  --output "$LEVEL1_RUN_ROOT/review/acceptance-provenance.json" \
  --json

"${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/build_level1_evidence.py prepare \
  --run-root "$LEVEL1_RUN_ROOT" \
  --results-root "$LEVEL1_RESULTS_ROOT"

"${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/package_validation_evidence.py \
  --input "$LEVEL1_RUN_ROOT/review/level1-package-input.json" \
  --root "run=$LEVEL1_RUN_ROOT" \
  --root "results=$LEVEL1_RESULTS_ROOT" \
  --mode both \
  --exact-output "$LEVEL1_EXACT_ARCHIVE" \
  --sanitized-output "$LEVEL1_SANITIZED_ARCHIVE" \
  --repository-root "$LEVEL1_REPO" \
  --sensitive-value "$LEVEL1_REPO" \
  --sensitive-value "$LEVEL1_RUN_ROOT" \
  --sensitive-value "$HF_CACHE_DIR" \
  --sensitive-value "$TF4576_PYTHON" \
  --sensitive-value "$TF5141_PYTHON" \
  --output "$LEVEL1_PACKAGE_REPORT" \
  --json

package_archive_sha() {
  "${LEVEL1_CPU_ENV[@]}" "$TF4576_PYTHON" -S -c '
import json
import re
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    report = json.load(handle)
matches = [
    item["sha256"]
    for item in report["archives"]
    if item.get("archive_kind") == sys.argv[2]
]
assert len(matches) == 1 and re.fullmatch(r"[0-9a-f]{64}", matches[0])
print(matches[0])
' "$LEVEL1_PACKAGE_REPORT" "$1"
}

LEVEL1_EXACT_SHA=$(package_archive_sha exact_private)
LEVEL1_SANITIZED_SHA=$(package_archive_sha sanitized_shareable)
LEVEL1_VERIFY_AT=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
readonly LEVEL1_EXACT_SHA LEVEL1_SANITIZED_SHA LEVEL1_VERIFY_AT

"${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/verify_validation_evidence.py \
  --archive "$LEVEL1_EXACT_ARCHIVE" \
  --expected-sha256 "$LEVEL1_EXACT_SHA" \
  --output "$LEVEL1_EXACT_PRIMARY_REPORT" \
  --timestamp-utc "$LEVEL1_VERIFY_AT" \
  --json

"${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/verify_validation_evidence.py \
  --archive "$LEVEL1_SANITIZED_ARCHIVE" \
  --expected-sha256 "$LEVEL1_SANITIZED_SHA" \
  --output "$LEVEL1_SANITIZED_PRIMARY_REPORT" \
  --timestamp-utc "$LEVEL1_VERIFY_AT" \
  --json

"${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/build_level1_evidence.py seal \
  --run-root "$LEVEL1_RUN_ROOT" \
  --results-root "$LEVEL1_RESULTS_ROOT" \
  --schema "$LEVEL1_SCHEMA" \
  --package-report "$LEVEL1_PACKAGE_REPORT" \
  --exact-archive "$LEVEL1_EXACT_ARCHIVE" \
  --sanitized-archive "$LEVEL1_SANITIZED_ARCHIVE" \
  --sanitized-staging-dir "$LEVEL1_SANITIZED_STAGING_DIR" \
  --exact-primary-report "$LEVEL1_EXACT_PRIMARY_REPORT" \
  --sanitized-primary-report "$LEVEL1_SANITIZED_PRIMARY_REPORT" \
  --implementation-base-commit "$LEVEL1_IMPLEMENTATION_BASE" \
  --exact-storage-locator "$LEVEL1_EXACT_LOCATOR" \
  --sanitized-storage-locator "$LEVEL1_SANITIZED_LOCATOR" \
  --verification-timestamp-utc "$LEVEL1_VERIFY_AT"
```

Inspect the generated summaries, partial descriptor, both committed
descriptor-aware verification reports, package report, and both final archive
hashes. Update no canonical completion language yet. Commit A contains exactly
the two summaries, partial descriptor, and two descriptor-aware verification
reports:

```bash
git diff --check
git add -- \
  docs/validation_results/LEVEL1_CORE_SUMMARY.json \
  docs/validation_results/LEVEL1_CORE_SUMMARY.md \
  docs/validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json \
  docs/validation_results/LEVEL1_CORE_EXACT_VERIFICATION.json \
  docs/validation_results/LEVEL1_CORE_SANITIZED_VERIFICATION.json
git diff --cached --check
git diff --cached --name-status
git commit -m "docs: record Level 1 core requalification evidence"
LEVEL1_COMMIT_A=$(git rev-parse HEAD)
test -z "$(git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)"
```

Collect a fresh clean observation of commit A, then close the canonical
descriptor. The builder independently confirms the live HEAD, named branch,
clean porcelain/diffs/submodules, and the exact five evidence blobs committed
in commit A before it replaces the partial descriptor.

```bash
test ! -e "$LEVEL1_RUN_ROOT/review/commit-a-provenance.json"
"${LEVEL1_CPU_ENV[@]}" \
  MULTISCREEN_EVIDENCE_REVIEWERS="$MULTISCREEN_EVIDENCE_REVIEWERS" \
  PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/collect_validation_provenance.py \
  --repo "$LEVEL1_REPO" \
  --review-method "$LEVEL1_REVIEW_METHOD" \
  --review-commit "$LEVEL1_TESTED_COMMIT" \
  --raw-events-reviewed true \
  --output "$LEVEL1_RUN_ROOT/review/commit-a-provenance.json" \
  --json

"${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/build_level1_evidence.py close \
  --run-root "$LEVEL1_RUN_ROOT" \
  --results-root "$LEVEL1_RESULTS_ROOT" \
  --schema "$LEVEL1_SCHEMA" \
  --package-report "$LEVEL1_PACKAGE_REPORT" \
  --commit-provenance "$LEVEL1_RUN_ROOT/review/commit-a-provenance.json" \
  --commit-a "$LEVEL1_COMMIT_A" \
  --exact-archive "$LEVEL1_EXACT_ARCHIVE" \
  --sanitized-archive "$LEVEL1_SANITIZED_ARCHIVE" \
  --sanitized-staging-dir "$LEVEL1_SANITIZED_STAGING_DIR" \
  --implementation-base-commit "$LEVEL1_IMPLEMENTATION_BASE" \
  --exact-storage-locator "$LEVEL1_EXACT_LOCATOR" \
  --sanitized-storage-locator "$LEVEL1_SANITIZED_LOCATOR" \
  --verification-timestamp-utc "$LEVEL1_VERIFY_AT"
```

Only now update `README.md`, `AGENTS.md`, the Stage 5 plan, handoff, validation
status, testing guide, known limitations, release checklist, and validation-log
index from the reviewed evidence. The accepted statement must immediately keep
the documented exclusions. Commit B contains the completed descriptor and
those canonical status updates; it must not change either summary or either
verification report.

```bash
git diff --check
git add -- \
  AGENTS.md \
  README.md \
  docs/HANDOFF.md \
  docs/KNOWN_LIMITATIONS.md \
  docs/LEVEL1_CORE_REQUALIFICATION_PLAN.md \
  docs/RELEASE_CHECKLIST.md \
  docs/TESTING.md \
  docs/VALIDATION_STATUS.md \
  docs/validation_results/LEVEL1_CORE_EVIDENCE_ARCHIVE.json \
  docs/validation_results/VALIDATION_LOG_INDEX.md
git diff --cached --check
git diff --cached --name-status
git commit -m "docs: close Level 1 core evidence descriptor"
```

Finally reverify the complete descriptor against both unchanged archives,
compare the byte-stable reports, and require a clean branch tip. These final
reports stay in the private run root and are not added to Git.

```bash
test ! -e "$LEVEL1_RUN_ROOT/review/exact-final-verification.json"
test ! -e "$LEVEL1_RUN_ROOT/review/sanitized-final-verification.json"

"${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/verify_validation_evidence.py \
  --archive "$LEVEL1_EXACT_ARCHIVE" \
  --expected-sha256 "$LEVEL1_EXACT_SHA" \
  --evidence-document \
  "$LEVEL1_RESULTS_ROOT/LEVEL1_CORE_EVIDENCE_ARCHIVE.json" \
  --schema "$LEVEL1_SCHEMA" \
  --output "$LEVEL1_RUN_ROOT/review/exact-final-verification.json" \
  --timestamp-utc "$LEVEL1_VERIFY_AT" \
  --json

"${LEVEL1_CPU_ENV[@]}" PYTHONPATH=. \
  "$TF4576_PYTHON" -S scripts/verify_validation_evidence.py \
  --archive "$LEVEL1_SANITIZED_ARCHIVE" \
  --expected-sha256 "$LEVEL1_SANITIZED_SHA" \
  --evidence-document \
  "$LEVEL1_RESULTS_ROOT/LEVEL1_CORE_EVIDENCE_ARCHIVE.json" \
  --schema "$LEVEL1_SCHEMA" \
  --output "$LEVEL1_RUN_ROOT/review/sanitized-final-verification.json" \
  --timestamp-utc "$LEVEL1_VERIFY_AT" \
  --json

cmp \
  "$LEVEL1_RUN_ROOT/review/exact-final-verification.json" \
  "$LEVEL1_RESULTS_ROOT/LEVEL1_CORE_EXACT_VERIFICATION.json"
cmp \
  "$LEVEL1_RUN_ROOT/review/sanitized-final-verification.json" \
  "$LEVEL1_RESULTS_ROOT/LEVEL1_CORE_SANITIZED_VERIFICATION.json"

"${LEVEL1_CPU_ENV[@]}" \
  "$TF4576_PYTHON" -P -S -B scripts/check_level1_repository.py \
  --repo-root "$LEVEL1_REPO" --check hygiene
test -z "$(git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)"
git diff --check
git diff --cached --check
```

Push and open the focused Stage 5 draft PR only after all commands above pass.
Do not publish either archive, merge the PR, or create an immutable tag as part
of this procedure.

## HF contract hardening Stage E

Stage E requalifies the integrated post-Level-1 hardening baseline. Its
authoritative scope, 53-command matrix, two environment records, raw-review
contract, retention boundary, and two-commit closure are fixed in
[HF_CONTRACT_HARDENING_PLAN.md](HF_CONTRACT_HARDENING_PLAN.md).

The Stage E support path is separate from the accepted Level 1 reviewer and
builder. Run both the new and legacy fixture suites before selecting a clean
tested-source commit:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.

python -S -m py_compile \
  scripts/build_hf_contract_hardening_evidence.py \
  scripts/check_hf_contract_hardening_offline_cache.py \
  scripts/review_hf_contract_hardening.py \
  tests/test_hf_contract_hardening_evidence_builder.py \
  tests/test_hf_contract_hardening_evidence_review.py \
  tests/test_hf_contract_hardening_offline_cache.py

python -S -m unittest discover \
  -s tests \
  -p 'test_hf_contract_hardening_*.py' \
  -v

python -S -m unittest discover \
  -s tests \
  -p 'test_level1_*.py' \
  -v

python -S -m unittest discover \
  -s tests \
  -p 'test_validation_evidence*.py' \
  -v
```

The new offline preflight checks only the fixed P0-3 and P0-4 public inputs;
it deliberately does not require the historical C3 prepared cache. A real
preflight must run offline with one explicit canonical cache:

```bash
: "${HF_CACHE_DIR:?set HF_CACHE_DIR to the existing offline cache}"
test -d "$HF_CACHE_DIR"

HF_DATASETS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -P -B scripts/check_hf_contract_hardening_offline_cache.py \
  --repo-root "$PWD" \
  --cache-dir "$HF_CACHE_DIR"
```

Do not begin the recorded matrix until the exact 4.57.6 and 5.14.1 Python
lanes, CUDA bf16, a new external private run root, an explicit reviewer, and a
writable external exact-archive directory are all available. Psi=16 P0-4 must
not start until the recorded Psi=8 v2 focused review passes and is inspected.
Historical Level 1, P0-4, and P0.5-C3 evidence files remain immutable and are
not members of the Stage E archive.
