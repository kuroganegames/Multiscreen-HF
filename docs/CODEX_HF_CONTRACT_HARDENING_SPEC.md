# Codex Specification — Post-Level-1 Hugging Face Contract Hardening

## 0. この文書の位置づけ

この文書を、Codex Goal

```text
Post-Level-1 Hugging Face contract hardening
```

の **authoritative specification** とする。

短い `/goal` prompt は、この文書を全文読んで従うことだけを要求する。実装対象、変更方針、テスト、合格基準、PR分割、環境保護、停止条件はすべて本書を優先する。

このGateは、既に受理済みの

```text
Level 1 — Core mathematical Hugging Face implementation
```

を取り消すものではない。Level 1で検証された数学・初期化・MiPE/cache・gradient checkpointing・bounded paper training contract・P0-1〜P0-4を保持したまま、HF API境界とedge-caseを堅牢化し、新しいbaselineを作る。

このGateは以下を検証しない。

```text
PEFT/LoRA
QLoRA / bitsandbytes
Unsloth
torch.compile
vLLM / SGLang
distributed training
Triton / windowed kernel
paper-scale pretraining
retrieval benchmark
broad generation compatibility
production serving
```

---

# 1. 対象問題

前回レビューで列挙された内容を漏れなく対象にする。

## 必須修正 5件

### A. Hugging Face output-head contract

現在の実装では、実際のlogitsはparameter-freeなnormalized tied `lm_head` proxyで計算される一方、`get_output_embeddings()`は入力側`nn.Embedding`を返す。

必要な修正:

```text
get_input_embeddings():
  token ids -> hidden lookup

get_output_embeddings():
  hidden states -> vocabulary logits
```

をHF APIとして明示的に成立させる。

### B. normalized tied headのdeepcopy lifecycle

`_NormalizedTiedLMHead`がweak referenceでownerを保持しているため、`copy.deepcopy(model)`後にcopy側proxyが元モデルを参照し続けないことを保証する。

### C. P0-4 qualification predicate

accepted P0-4 contractには少なくとも

```text
GPT-2 vocab = 50,257
context = 4096
CUDA
bf16
microbatch = 1
optimizer steps >= 50
gradient checkpointing = true
```

が含まれる。

現行のqualification predicateがこれらをすべて実行時に強制するよう修正する。

### D. gradient-checkpointed training + past_key_values

training + gradient checkpointing + supplied cacheの組み合わせで、cacheを受け取ったにもかかわらずlayerへ`past_kv=None`を渡すsilent miscomputeを禁止する。

このGoalでは、cached checkpointed training自体を新規実装せず、明示的fail-fastを推奨する。

### E. zero-valid-target loss

次の場合にNaNやdetached zeroを返さない。

```text
labelsがすべて-100
attention_maskがすべて0
causal shift後にvalid targetが0
sequence length 1
```

graph-connected finite zeroを返し、`backward()`可能にする。

---

## 堅牢化 2件

### F. cached generation suffix ambiguity

cacheありの`prepare_inputs_for_generation()`が、

```text
full input including prefix
already-sliced suffix
```

を長さだけで曖昧に推定し、already-sliced suffixのtokenを誤って削除しないようにする。

### G. PackedTextDataset EOS fail-fast

`eos_token_id`が

```text
明示引数にもない
tokenizer.eos_token_idにもない
```

場合、token ID 0を暗黙fallbackにせず明示的に失敗させる。

---

# 2. 明示的スコープ外

以下は、今回の新しい failing test が必要性を示さない限り変更しない。

```text
state-dict conversion key collision detection
MiPE数式
Softmask数式
Trim数式
TanhNorm
gating
projection initialization
paper oracle math
accepted MiPE position semantics
cache position semanticsそのもの
P1 ecosystem support
```

---

# 3. 開発環境契約

現在の開発環境を壊してはいけない。

Python環境はConda管理である。
`uv`は導入済みだがinstall補助としてのみ使用する。

## 許可

```text
既存Conda環境の利用
隔離Conda環境の新規作成
隔離venvの新規作成
明示した対象環境へのuv/pip install
```

## 禁止

```text
Conda baseの削除・破壊
global install
conda update --all
無制約 pip install -U
broad uv sync
unrelated lock file変更
unrelated package upgrade
```

packageを変更した場合はbefore/after versionを記録する。

環境ディレクトリ、cache、checkpoint、outputsはcommitしない。

---

# 4. Git / PR運用

すべて最新reviewed `main`から開始する。

作業開始時:

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
git remote -v
git fetch --prune
```

working treeがdirtyなら、user workを破棄しない。

以下を禁止する。

```text
mainへの直接commit
git reset --hardによるuser work破棄
stacked implementation PR
自動merge
自動tag
unrelated changeの混入
```

---

# 5. Stage構成

## Stage A
output-head + deepcopy lifecycle

推奨branch:

```text
hardening/hf-output-head-lifecycle
```

## Stage B
training edge correctness

推奨branch:

```text
hardening/training-edge-contracts
```

## Stage C
P0-4 qualification hardening

推奨branch:

```text
hardening/p0-4-qualification-contract
```

## Stage D
generation + dataset hardening

推奨branch:

```text
hardening/generation-data-contracts
```

## Stage E
final integrated requalification + evidence

推奨branch:

```text
validation/hf-contract-hardening-requalification
```

Stages A-Dは各Stageごとに:

```text
focused implementation
focused tests
regression
commit
draft PR
REVIEW_REQUIRED
停止
```

userがreview/merge後:

```bash
git switch main
git fetch --prune
git pull --ff-only
```

その後 `/goal resume` で次Stageへ進む。

---

# 6. 最初に行うbaseline reproduction

production codeを変更する前に、7項目それぞれについて以下のいずれかに分類する。

```text
REPRODUCED
NOT REPRODUCED WITH EVIDENCE
ALREADY FIXED ON CURRENT MAIN
SPECIFICATION DECISION REQUIRED
```

## A. output head

確認:

```python
model._compute_logits(hidden_states)
model.lm_head(hidden_states)
model.get_output_embeddings()
```

現在のreturned moduleがhidden->vocab projectionとしてcallableかを確認する。

## B. deepcopy

```python
copied = copy.deepcopy(model)
```

確認:

```text
owner identity
元/copyのlogits
元だけweight変更
copyだけweight変更
元model削除後のcopy forward
copy loss.backward()のgradient destination
```

## C. P0-4 predicate

long GPU jobは不要。

synthetic settingsで:

```text
microbatch=2
gradient_checkpointing=false
```

でも現行predicateがqualifiedになるか確認する。

## D. GC + past cache

spy layerまたはmonkeypatchで、cacheがforwardへ渡されてもcheckpoint branchが`past_kv=None`を使うか確認する。

## E. zero-valid loss

以下を実行する。

```text
all labels = -100
all attention mask = 0
sequence length = 1
shifted labels
unshifted labels
```

loss finite / backward可否を記録する。

## F. generation ambiguity

例:

```text
past_length = 2
already-sliced suffix length = 5
cache_positionなし
```

suffix先頭が削除されるか確認する。

## G. missing EOS

```text
tokenizer.eos_token_id = None
explicit eos_token_idなし
```

token ID 0が挿入されるか確認する。

---

# 7. Stage A — output-head + deepcopy

## 7.1 get_output_embeddings

最終契約:

```python
output_module = model.get_output_embeddings()
logits = output_module(hidden_states)
```

shape:

```text
hidden_states: [B,T,dE]
logits:        [B,T,V]
```

必須:

```text
output_module(hidden) == model._compute_logits(hidden)
output_module(hidden) == model.lm_head(hidden)
```

preferred:

```python
model.get_output_embeddings() is model.lm_head
```

ただしexact Transformers sourceを確認し、安全な別設計が必要なら根拠を記録する。

## 7.2 parameter contract

修正後も:

```text
新規trainable output parameterなし
lm_head.weight Parameterなし
lm_head.bias Parameterなし
state_dictにlm_head.*なし
paper parameter count不変
s_F維持
normalized input embeddingによるlogit計算維持
```

## 7.3 set_output_embeddings

任意のuntied headをsilentに受理してはいけない。

推奨:

```text
現在のnormalized tied proxy -> no-op
任意のreplacement -> ValueErrorまたはNotImplementedError
```

partial mutation禁止。

## 7.4 resize_token_embeddings

Transformers 4.57.6と5.14.1の実装を実際にauditする。

次のどちらかにする。

### fully supported

```text
expand
shrink
config.vocab_size更新
model.vocab_size更新
input embedding更新
output proxyが新embeddingを反映
parameter-free維持
save/load成功
```

### explicit fail-fast

```text
明示例外
failure前後でmodel/config不変
```

可能ならfull supportを優先する。

## 7.5 deepcopy

以下をすべて満たす。

```text
copied.lm_head owner == copied
initial logits完全一致
original mutationでcopied logits不変
copy mutationでoriginal logits不変
original削除後もcopy forward成功
copy backwardのgradはcopy parameterのみ
module cycleなし
state_dict不変
save/load成功
```

親modelをproxyのregistered childとして保持する設計は禁止。

## Stage A focused tests

新規:

```text
tests/test_hf_output_head_contract.py
```

最低限:

```text
output head callability
shape
exact logits
no output parameters
state_dict identity
paper parameter counts
save/load
AutoModelForCausalLM
deepcopy isolation
garbage collection
gradient isolation
resize contract
set_output_embeddings contract
```

4.57.6 / 5.14.1双方で実行する。

さらに:

```bash
python -m unittest discover -s tests -p 'test_paper_architecture_contract.py' -v
python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
python oracle/test_against_hf_port.py --quick
python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle \
  --quick
```

合格後draft PRを作り`REVIEW_REQUIRED`で停止。

---

# 8. Stage B — training edge correctness

## 8.1 GC + past cache

次の場合:

```text
training == true
gradient checkpointing == true
non-empty past_key_values
```

silent ignore禁止。

このGoalではfail-fastを推奨。

例外はlayer equation実行前に発生させる。

対象:

```text
legacy tuple/list
supported DynamicCache
kv_caches alias
```

エラーメッセージには:

```text
gradient-checkpointed training with past_key_values is unsupported
```

相当の明示性を持たせる。

正常な:

```text
checkpointed training without cache
eval cached decode
checkpointing disabled path
```

は壊さない。

## 8.2 zero-valid-target loss

valid targetが0の場合:

```python
loss = logits.sum() * 0.0
```

相当のgraph-connected zeroにする。

対象:

```text
labels_are_shifted=True, all -100
labels_are_shifted=False, all -100
attention_mask all zero
seq_len=1
mixed valid/ignored
left padding
right padding
```

要件:

```text
loss == 0
finite
requires grad経路あり
backward成功
parameter grad finite
zero-valid caseのgradはzero
mixed valid caseの通常CE値は変更しない
```

valid targetがあるのにNaNが出た場合を隠してはいけない。

新規test候補:

```text
tests/test_training_edge_contract.py
```

Stage Bでは:

```bash
python -m unittest discover -s tests -p 'test_training_edge_contract.py' -v
python -m unittest discover -s tests -p 'test_gradient_checkpointing_contract.py' -v
python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
python oracle/test_against_hf_port.py
python p0_2_three_way_minimal/test_three_way_minimal.py \
  --reference-root third_party/multiscreen-pytorch \
  --hf-root . \
  --oracle-root oracle
```

CUDA availableならP0-1/P0-2 CUDA bf16 fullも実行。

checkpointed reduced smokeも実行。

合格後draft PR、`REVIEW_REQUIRED`で停止。

---

# 9. Stage C — P0-4 qualification

authoritative qualification:

```text
GPT-2 vocab == 50257
seq_len == 4096
CUDA
bf16
microbatch == 1
optimizer steps >= 50
gradient checkpointing == true
supported runtimeのnon-reentrant checkpointing
```

gradient accumulation 8は、既存accepted runの設定ではあるが、authoritative qualificationで必須と明記されていない限り、新たに必須条件へ追加しない。

## 新predicate例

```text
gpt2_vocab_50257
context_4096
cuda_device
bf16_amp
microbatch_size_1
optimizer_steps_at_least_50
gradient_checkpointing_enabled
gradient_checkpointing_non_reentrant
```

## static preflight

少なくとも:

```text
run microbatch = 1
run gradient_checkpointing = true
```

を確認。

`use_reentrant=False`がruntime code側のcontractなら専用unit testで確認。

## negative qualification tests

以下は必ずdiagnostic:

```text
microbatch=2
gradient_checkpointing=false
use_reentrant=true（override可能なら）
CPU
non-bf16
seq_len<4096
steps<50
```

必須:

```text
qualification.qualified=false
P0-4_DIAGNOSTIC_COMPLETE.md
P0-4_COMPLETE.mdなし
```

## historical compatibility

旧accepted P0-4 evidenceはqualification conditionsが旧schemaでも、reviewerが別途microbatch/checkpointingを確認している。

したがって:

```text
historical evidenceを書き換えない
historical schemaはhistorical commitとして読める
new runで旧incomplete predicateをqualified扱いしない
```

必要ならschema version / tested commit boundaryを導入する。

新規:

```text
tests/test_p0_4_qualification_contract.py
```

Stage Cでlong GPU runは不要。

合格後draft PR、`REVIEW_REQUIRED`で停止。

---

# 10. Stage D — generation / data hardening

## 10.1 cached suffix ambiguity

まずTransformers 4.57.6 / 5.14.1の`GenerationMixin`をaudit。

確認:

```text
cached iterationのinput_ids形状
full inputかsuffixか
cache_positionの有無
one-token suffix
multi-token suffix
DynamicCache
```

normal `model.generate()`はuserに追加flagを要求せず動作させる。

direct APIで曖昧な場合、silent token drop禁止。

必要なら明示kwarg:

```text
input_ids_include_prefix
```

等を導入してもよいが:

```text
strict bool
non-serialized
normal generate()には不要
test必須
```

### supported

```text
greedy generate(use_cache=True)
full sequence including prefix
explicit one-token suffix
explicit multi-token suffix
consistent cache_position
legacy cache
DynamicCache
```

### rejected

```text
full/suffix解釈が曖昧でmetadata不足
cache_position不整合
empty suffix
explicit metadata矛盾
```

両MiPE modeで:

```text
full forward
one-shot cached suffix
chunked cached suffix
greedy generation
```

の一致を確認。

## 10.2 missing EOS

以下の優先順位:

```text
explicit eos_token_id
tokenizer.eos_token_id
```

両方NoneならValueError。

禁止:

```text
padをEOS代用
bosをEOS代用
unkをEOS代用
token ID 0への暗黙fallback
```

ただし明示`eos_token_id=0`やtokenizerがEOS=0を持つ場合は有効。

`from_hf_dataset()`も同じ契約。

テスト:

```text
explicit EOS
tokenizer EOS
EOS=0
missing EOS
from_hf_dataset path
```

新規候補:

```text
tests/test_generation_input_contract.py
tests/test_packed_text_contract.py更新
```

さらに:

```bash
python -m unittest discover -s tests -p 'test_mipe_position_cache_contract.py' -v
```

合格後draft PR、`REVIEW_REQUIRED`で停止。

---

# 11. Stage E — final requalification

Stages A-D merge後のみ開始。

新しいbaselineについてfresh evidenceを作成する。

旧Level 1証拠は書き換えない。

推奨追加:

```text
docs/HF_CONTRACT_HARDENING_PLAN.md
docs/validation_results/HF_CONTRACT_HARDENING_SUMMARY.md
docs/validation_results/HF_CONTRACT_HARDENING_SUMMARY.json
docs/validation_results/HF_CONTRACT_HARDENING_EVIDENCE_ARCHIVE.json
```

---

# 12. Final test matrix

## focused

```text
HF output head
deepcopy lifecycle
training edge
gradient checkpointing
P0-4 qualification
generation input
packed text
paper architecture
paper initialization
MiPE/cache
paper training contract
evidence tooling
Level 1 reviewer/evidence support
```

## exact Transformers lanes

```text
4.57.6
5.14.1
```

## P0-1 full

```bash
python oracle/test_against_hf_port.py
python oracle/test_against_hf_port.py --device cuda:0 --dtype bf16
```

## P0-2 full

```bash
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

## formula/oracle

```bash
python oracle/test_formula_units.py
python oracle/test_paper_math_oracle_selfcheck.py
python oracle/test_paper_math_oracle_smoke.py
```

## fresh training

```text
P0-3 checkpointed CUDA bf16 Psi=8/16
P0-4 strict CUDA bf16 Psi=8
review
P0-4 strict CUDA bf16 Psi=16
```

P0-4は新しいhardened qualification predicateをすべて満たすこと。

確認:

```text
finite losses
finite grad norms
loss decrease
save/load
tokenizer reload
cache split
generation
data contract
complete marker
failure artifactなし
diagnostic markerなし
```

historical P0-4 metricsを新commitの証拠に流用しない。

---

# 13. Evidence

P1-preflight A toolingを使う。

Stage E前に:

```bash
export MULTISCREEN_EVIDENCE_REVIEWERS=kuroganegames
export MULTISCREEN_EVIDENCE_ARCHIVE_DIR=/absolute/path/outside/repository
```

必要:

```text
clean worktree at start
HEAD SHA
branch
exact environment
explicit reviewer
raw events reviewed
exact/private external archive
sanitized archive
offline verification
archive hashes
no secrets/private paths in Git
```

reviewer identityをGitHub authから推測しない。

区別して記録:

```text
historical Level 1 tested commit
new hardening tested commit
evidence commit
review commit
closure tip
```

---

# 14. Documentation

Stage Eで必要に応じて更新:

```text
README.md
AGENTS.md
docs/HANDOFF.md
docs/VALIDATION_STATUS.md
docs/TESTING.md
docs/KNOWN_LIMITATIONS.md
docs/RELEASE_CHECKLIST.md
docs/validation_results/VALIDATION_LOG_INDEX.md
```

明記:

```text
output headはhidden->vocab callable
normalized tied storageはparameter-free
deepcopy isolation
GC+past trainingは明示unsupported
zero-valid lossはgraph-connected zero
P0-4 predicateはmicrobatch/checkpointingまで強制
ambiguous cached suffixはsilent dropしない
missing EOSはfail-fast
```

維持するlimitations:

```text
unofficial implementation
dense quadratic screening
no paper-scale reproduction
no retrieval validation
no optimized efficiency evidence
no distributed qualification
no PEFT qualification
no broad serving claim
```

---

# 15. Hygiene

各PR前:

```bash
git diff --check
git status --short
git diff
git diff --cached
```

確認:

```text
syntax
JSON
YAML
Markdown links
no __pycache__
no *.pyc
no env
no outputs
no weights/checkpoints
no raw evidence in Git
no private absolute path
no secret
no unexpected large file
no nested .git
```

package versionを変更する場合:

```text
pyproject.toml
multiscreen_transformers/__init__.py
```

を一致させる。

---

# 16. 停止条件

通常のscope内作業では毎回permissionを求めない。

以下は停止:

```text
model math変更が必要
oracle semantics変更が必要
accepted MiPE/cache semantics変更が必要
accepted tolerance弱化が必要
環境を破壊する操作が必要
Transformers lane間で解決不能な仕様衝突
required CUDA unavailable
evidence retention不能
previous accepted regressionが再現可能に失敗
```

この場合:

```text
PARTIAL/BLOCKED WITH EVIDENCE
```

で終了し、テストを弱化しない。

---

# 17. Progress report

各checkpoint:

```text
Checkpoint:
Baseline:
Findings:
Files changed:
Tests:
Artifacts:
Risks:
Next action:
```

A-D終了:

```text
REVIEW_REQUIRED
```

E完了:

```text
COMPLETE
```

---

# 18. 最終完了条件

`COMPLETE`は以下すべて必要。

```text
全7項目fix/resolution
Stages A-D review/merge済み
final integrated matrix passed
fresh P0-3 passed
fresh P0-4 Psi=8 passed
fresh P0-4 Psi=16 passed
4.57.6 / 5.14.1 focused tests passed
output-head contract passed
deepcopy isolation passed
negative qualification tests passed
exact/private evidence retained
sanitized archive verified
explicit reviewer recorded
docs current
final draft PR作成
自動mergeなし
自動tagなし
```

---

# 19. Final response format

```text
Terminal state:
Stage:

Baseline:
  branch:
  tested commit:
  environment:

Findings:
  A output head:
  B deepcopy:
  C P0-4 qualification:
  D GC + past:
  E zero-valid loss:
  F generation ambiguity:
  G missing EOS:

変更ファイル:
  - ...

追加ファイル:
  - ...

実行テスト:
  - command:
    result:
    count:

CUDA:
  P0-3 Psi=8:
  P0-3 Psi=16:
  P0-4 Psi=8:
  P0-4 Psi=16:

Evidence:
  reviewer:
  worktree:
  exact archive:
  sanitized archive:
  verification:

Commit / PR:
  - ...

未確認:
  - ...

次:
  - ...
```

---

# 20. 最終原則

目的は単にtestをpassさせることではない。

既に受理済みのMultiscreen数学・Level 1証拠を保持しながら、明示的で監査可能なHugging Face API contractを成立させる。

禁止:

```text
untied output head導入
supplied cacheのsilent ignore
suffix tokenのsilent drop
token 0をEOSとしてsilent使用
incomplete P0-4 runのqualification
zero-valid-target NaNの隠蔽
accepted P0 toleranceの弱化
historical evidenceの書換え
```
