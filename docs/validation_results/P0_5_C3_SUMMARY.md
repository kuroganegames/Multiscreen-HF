# P0.5-C3 Summary: paper-training-contract smoke

## Recorded pre-merge verdict

```text
Local gate result: passed
Acceptance state: REVIEW_REQUIRED
Final Level 1 core requalification: not started
```

The executable paper-training contract, pinned data lane, two-version focused
test matrix, full P0 regressions, and four bounded CUDA bf16 diagnostics passed
against the tested source commit below. This is a local Stage 4 result awaiting
review; it is not accepted until the focused draft PR is reviewed and merged.

## Provenance

```text
Stage 4 base / PR #12 merge: a2d43517c45dc39855db81b9286c4abf190a2c14
tested source commit: 8fa5dbf13530c942b2c9e5f03a572bd0cd5ca74f
source branch: agent/p0-5-c3-paper-training-contract
run date (UTC): 2026-08-09
run-end worktree observation: clean
run-end observation time: 2026-08-09T08:12:09Z
run-end porcelain SHA-256: e3b0c44298fc1c149afbf4f8996fb92427ae41e4649b934ca495991b7852b855
acceptance reviewer: pending; none supplied
```

A tested-source identity record was captured immediately before the CUDA
diagnostics, after the contract and pinned-data preflights. It does not retain a
timestamped porcelain observation, so the evidence descriptor records run-start
provenance as `not_recorded_in_original_run`. Only the timestamped run-end
observation records an auditable clean-worktree hash.

## Checked contract

The harness keeps three kinds of facts distinct: paper statements, arithmetic
derived from those statements, and repository operational choices where the
paper is silent.

| Field | Checked value | Source class |
|---|---|---|
| tokenizer repository | GPT-2 | paper contract |
| tokenizer revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | reproducibility pin |
| vocabulary | 50,257 | paper contract |
| EOS token ID | 50,256 | repository operationalization; paper unspecified |
| document stream | append one EOS and concatenate continuously | paper contract |
| document tokenization | no truncation or implicit special tokens | repository operationalization |
| prediction context | 4,096 tokens from stored chunks of 4,097 | paper contract plus existing shifted-label API |
| global batch | 4,194,304 tokens | paper contract |
| optimizer | AdamW, betas `(0.9, 0.95)`, weight decay 0 | paper contract |
| epsilon | `1e-8` | repository operationalization; paper unspecified |
| gradient clipping | disabled | paper contract |
| schedule | 4,096-step linear warmup, then constant | paper contract |
| peak learning rate | `0.0625` | paper contract |
| update indexing | zero-based update uses `(step + 1) / 4096` during warmup | explicit repository operationalization |

The exact scheduler checkpoints passed:

| Zero-based update | Learning rate |
|---:|---:|
| 0 | 0.0000152587890625 |
| 1 | 0.000030517578125 |
| 4095 | 0.0625 |
| 4096 | 0.0625 |
| 4097 | 0.0625 |

The implementation uses `AdamW(fused=False)` so the checked optimizer path is
explicit and does not silently vary with hardware.

## Pinned data lane

The data contract passed with the canonical Hub loader and no local-text
fallback:

| Field | Recorded value |
|---|---|
| family | SlimPajama |
| pinned source | `gmongaras/SlimPajama-627B_Reupload` |
| revision | `c34c22dbb10ae6b264a2f357a909d1a537141b36` |
| shard | `data/test-00000-of-00030.parquet` |
| shard bytes | 43,263,929 |
| shard SHA-256 | `d9a83d59b72f4c303f0c0e46d0e73a8446eabb56b9aa5fd992347c358ab65743` |
| Datasets version | 5.0.1 |
| full fingerprint | `507a47fcec5cbfdc` |
| selected rows | contiguous rows 0 through 63 |
| selection fingerprint | `f1e6c1c09434a7e4` |
| row-manifest SHA-256 | `942f9b3397ff7073342973082efa4cddf3ace16bc7e3d180c827df3203243831` |
| tokenizer asset-manifest SHA-256 | `07c45937a89b33f30016aef5b3982f13f25bf2c6ba940c535d1b5daa90459a71` |
| uint32-LE token-stream SHA-256 | `3232bc3996272d563b6cc4e63a8d7a7d3769c7ec33e74d3d008d97cd290d7496` |

Accounting was reproduced exactly:

| Item | Count |
|---|---:|
| selected / nonempty documents | 64 / 64 |
| text tokens | 58,645 |
| EOS tokens | 64 |
| concatenated tokens | 58,709 |
| complete 4,097-token stored chunks | 14 |
| usable stored tokens | 57,358 |
| discarded incomplete tail | 1,351 |

The compact output retains per-row and per-chunk hashes but no raw source text.
This source is a pinned third-party SlimPajama-family reupload test shard. It is
not claimed byte-identical to the paper corpus or representative of its train
split.

## Environment

Primary validation environment:

```text
Python: 3.12.11
PyTorch: 2.7.1+cu128
Transformers: 4.57.6
Datasets: 5.0.1
Tokenizers: 0.22.0
Safetensors: 0.5.3
Accelerate: 1.6.0
CUDA runtime: 12.8
device: cuda:0
GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
GPU total memory: 101,973,491,712 bytes
compute capability: 12.0
NVIDIA driver: 595.71.05
bf16 supported: true
```

An unretained session check reported Python 3.12.10, PyTorch 2.8.0+cu128, and
Transformers 5.14.1 for the latest-compatibility focused lane. The retained
18-test log has no version header, so exact version identity is session metadata
rather than self-contained archive evidence. No install or upgrade was
performed, but no before/after package-state snapshot was retained.

## Test results

| Check | Result |
|---|---:|
| C3 focused suite, Transformers 4.57.6 | 18 passed |
| C3 focused suite, Transformers 5.14.1 | 18 passed |
| actual pinned data lane | passed |
| all repository tests, Transformers 4.57.6 | 121 passed |
| standard-library evidence tooling | 58 passed |
| formula units / oracle self-check / oracle smoke | passed |
| P0-1 full CPU fp32 | 744 checks passed |
| P0-1 full CUDA bf16 | 744 checks passed |
| P0-2 full CPU fp32 | 282 checks passed |
| P0-2 full CUDA bf16 | 282 checks passed |

The strong P0-1 and P0-2 reruns were required because the Stage 4 branch follows
the merged gradient-checkpointing model-core change from PR #12. CPU fp32 and
CUDA bf16 both remained green.

The retained 18-test logs apply to tested commit `8fa5dbf…`. A later
branch-head-only hardening makes checked manifest comparison strict across JSON
boolean and number types. Both focused lanes and the full primary suite were
rerun; no model, optimizer, data, or CUDA training path changed.

## CUDA bf16 diagnostics

All four runs used context 4,096, GPT-2 vocabulary 50,257,
`mipe_position_mode="paper_absolute"`, fp32 MiPE/softmask auxiliaries,
non-reentrant gradient checkpointing, tied embeddings, AdamW without clipping,
and the pinned data contract above. Psi=8 ran before Psi=16.

Operational mode deliberately reduces warmup to 2 updates and peak learning
rate to 0.0006. It checks the workstation training path, accumulation,
scheduler order, finite values, and nonzero updates. Peak-exposure mode performs
one bounded update at the exact paper peak learning rate of 0.0625; it checks
finite values and an actual update, not loss decrease.

| Psi | Mode | Steps / accumulation | Observed LR | Post loss | Peak bytes, allocated / reserved |
|---:|---|---:|---|---:|---:|
| 8 | operational | 3 / 2 | 0.0003, 0.0006, 0.0006 | 11.23622 | 3,568,145,920 / 5,330,960,384 |
| 8 | peak exposure | 1 / 1 | 0.0625 | 11.31011 | 3,517,948,416 / 5,320,474,624 |
| 16 | operational | 3 / 2 | 0.0003, 0.0006, 0.0006 | 15.72905 | 7,034,909,696 / 9,124,708,352 |
| 16 | peak exposure | 1 / 1 | 0.0625 | 15.22273 | 6,705,063,424 / 8,050,966,528 |

Operational event vectors:

```text
Psi=8  mean losses: 11.35676, 10.96895, 11.01792
Psi=8  gradient L2: 3.36630, 2.62152, 2.98998
Psi=16 mean losses: 15.96629, 15.22286, 15.33546
Psi=16 gradient L2: 23.15300, 21.83766, 22.18719
```

Peak-exposure mean loss / gradient L2 was 11.38174 / 4.22524 for Psi=8
and 16.04001 / 23.78993 for Psi=16.

Every loss, gradient norm, parameter, and update was finite. No step applied
gradient clipping. The tracked parameter changed on every update; the exact
peak-exposure delta was `-0.0625` for both Psi values. All four completion
markers were present and no `failure.json` existed.

## Evidence retention

The final sanitized package contains 26 source artifacts and 3 control members.
Offline verification passed the whole-archive digest, canonical single-member
gzip and normalized USTAR structure, member paths and types, manifest, every
member hash, `SHA256SUMS`, and an independent scan of all 29 members.

```text
sanitized archive:
  validation-evidence-sanitized-p0-5-c3-8fa5dbf1-v2.tar.gz
archive SHA-256:
  274e489f4b4872f8f8c797b56b9d49aebc3a8c0e005fe2c65694f136616a9573
archive size:
  16,810 bytes
manifest SHA-256:
  75fd240b2da86b2ea46258e354a5f5321552d1be19d6ea5f01697e32433d6a72
historical packaging verification report SHA-256:
  8579a1724a693faeec9934416ba8e11bf55c0d91487ac0de2b0f8cd2a55da81b
sanitization report SHA-256:
  04f02677674a722716157c8ad919d74d275c88c8e7130fbabcc8b27c691be910
files scanned:
  26
replacements / unresolved findings:
  0 / 0
public asset:
  none
```

At the original Stage 4 packaging time, retention was partial:
`MULTISCREEN_EVIDENCE_ARCHIVE_DIR` was not configured, so no exact/private
archive was created. The verified sanitized archive remained unpublished, and
no explicit evidence reviewer was supplied. Those dated facts remain unchanged
in [the historical descriptor](P0_5_C3_EVIDENCE_ARCHIVE.json).

### Post-acceptance external-retention closure

On 2026-08-14, the same 26 original source artifacts were reread from the
retained raw root. Every size and SHA-256 matched the historical descriptor.
An allowlist-only exact/private archive was then created directly in the
existing external private-retention class and verified offline:

```text
exact/private archive:
  validation-evidence-exact-p0-5-c3-8fa5dbf1-v2.tar.gz
archive SHA-256:
  db882b8eb5d871b4ca8696a324d4a67aa6bd36389dd173db4ea857587d57319e
archive size:
  15,932 bytes
manifest SHA-256:
  94ad9e97a9cc2681a6cb0b48bca2de4578828195d0b4905905112b5a6956654b
archive members:
  28 (26 source artifacts plus MANIFEST.json and SHA256SUMS)
public:
  false
```

The unchanged 29-member sanitized archive was also reverified against the new
closure descriptor. Both descriptor-aware verification reports are committed
with the [closure descriptor](P0_5_C3_EVIDENCE_CLOSURE.json). The exact archive
remains private and the sanitized archive remains unpublished.

No explicit evidence reviewer was supplied for this later closure, so
`acceptance_review` and overall `evidence_status` remain `pending`/`partial`.
The retention state itself is now `verified`. This follow-up does not rewrite
original-run provenance, accepted metrics, PR #13 acceptance, or any model
capability claim.

## Interpretation and limits

P0.5-C3 locally validates that the repository can encode the selected paper
training recipe as executable, reproducible contracts and can exercise that
contract through bounded Psi=8/Psi=16 CUDA bf16 updates. It does not validate
the paper corpus, global batch execution, training duration, convergence,
quality, retrieval performance, runtime or memory efficiency, optimized
kernels, cross-hardware reproducibility, PEFT/LoRA, broad generation, serving,
or production readiness.

The `REVIEW_REQUIRED` field above records the pre-merge Stage 4 handoff state.
The focused result was subsequently reviewed and accepted by merged PR #13, and
final Level 1 core requalification was later accepted by merged PR #14. The
post-acceptance retention closure recorded here changes neither acceptance
decision nor capability scope.
