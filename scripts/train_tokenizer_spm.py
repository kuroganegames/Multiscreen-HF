#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers

from cache_utils import apply_hf_cache_env, make_cache_paths

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"


def choose_text_column(dataset, requested: str) -> str:
    columns = list(getattr(dataset, "column_names", []) or [])
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"text_column={requested!r} not found. Available columns: {columns}")
        return requested
    for name in ("text", "story", "content", "completion", "document"):
        if name in columns:
            return name
    if len(columns) == 1:
        return columns[0]
    raise ValueError(f"Could not infer text column. Available columns: {columns}")


def iter_texts(dataset, text_column: str, max_samples: int | None) -> Iterable[str]:
    n = 0
    for row in dataset:
        if max_samples is not None and n >= max_samples:
            break
        value = row.get(text_column)
        if value is None:
            continue
        text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        # Keep one story per line. ByteLevel BPE can handle newlines too, but this
        # keeps tokenizer training comparable to the previous SentencePiece script.
        yield text.replace("\n", " ")
        n += 1


def write_corpus(dataset, text_column: str, output_file: Path, max_samples: int | None) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_file.open("w", encoding="utf-8") as f:
        for text in iter_texts(dataset, text_column, max_samples):
            f.write(text)
            f.write("\n")
            count += 1
    return count


def build_hf_native_bpe_tokenizer(corpus: Path, *, vocab_size: int, model_max_length: int) -> PreTrainedTokenizerFast:
    """Train an HF-native ByteLevel BPE tokenizer and wrap it as PreTrainedTokenizerFast.

    Previous attempts trained a SentencePiece .model correctly, but the current
    Transformers environment failed to reload it through LlamaTokenizer/T5Tokenizer,
    producing 4- or 5-token tokenizers. Training directly with the `tokenizers`
    backend writes a tokenizer.json that AutoTokenizer can reload without requiring
    any model-specific slow tokenizer wrapper.

    This is not a T5/GPT-2 model tokenizer semantically; it is just a small BPE
    tokenizer whose token ids can be fed to Multiscreen or any CausalLM.
    """
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN, fuse_unk=False))

    # NFKC helps keep TinyStories punctuation/unicode variants stable. ByteLevel
    # then gives full byte coverage, avoiding the need for SentencePiece byte_fallback.
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train(files=[str(corpus)], trainer=trainer)

    # Do not auto-add BOS/EOS here. SFTTrainer can append EOS via eos_token, and
    # the training script uses plain text LM with SFTTrainer.
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        model_max_length=int(model_max_length),
        padding_side="right",
        truncation_side="right",
    )
    return hf_tokenizer


def validate_saved_tokenizer(output_dir: Path, *, expected_vocab_size: int) -> None:
    for use_fast in (True, False):
        tok = AutoTokenizer.from_pretrained(str(output_dir), use_fast=use_fast)
        actual_len = len(tok)
        print(f"[check] AutoTokenizer(use_fast={use_fast}) class: {tok.__class__.__name__}")
        print(f"[check] AutoTokenizer(use_fast={use_fast}) len: {actual_len}")
        print(f"[check] AutoTokenizer(use_fast={use_fast}) vocab_size: {getattr(tok, 'vocab_size', None)}")
        print(
            f"[check] AutoTokenizer(use_fast={use_fast}) special ids:",
            {"unk": tok.unk_token_id, "bos": tok.bos_token_id, "eos": tok.eos_token_id, "pad": tok.pad_token_id},
        )
        if actual_len != expected_vocab_size:
            raise RuntimeError(
                f"Tokenizer save/load failed for AutoTokenizer(use_fast={use_fast}): "
                f"expected len={expected_vocab_size}, actual len={actual_len}. Refusing to continue."
            )
        expected_ids = {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
        actual_ids = {"unk": tok.unk_token_id, "bos": tok.bos_token_id, "eos": tok.eos_token_id, "pad": tok.pad_token_id}
        if actual_ids != expected_ids:
            raise RuntimeError(f"Special token ids changed: expected={expected_ids}, actual={actual_ids}")

        sample = "Once upon a time, Timmy went to the park."
        ids = tok.encode(sample, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids[:30])
        print(f"[check] sample ids: {ids[:60]}")
        print(f"[check] sample tokens: {pieces}")
        if not ids or max(ids) <= 3 or len(set(ids)) < 4:
            raise RuntimeError("Tokenizer maps normal text almost only to special/unk IDs. Refusing to continue.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny HF-native ByteLevel BPE tokenizer. The filename is kept "
            "as train_tokenizer_spm.py for compatibility with previous commands, "
            "but the implementation intentionally avoids SentencePiece wrappers."
        )
    )
    parser.add_argument("--dataset_name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--vocab_size", type=int, default=768)
    parser.add_argument("--max_samples", type=int, default=200_000)
    parser.add_argument("--output_dir", default="tokenizers/tinystories_spm768")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--hf_home", default=None)
    parser.add_argument("--hub_cache_dir", default=None)
    parser.add_argument("--datasets_cache_dir", default=None)
    parser.add_argument("--model_cache_dir", default=None)
    parser.add_argument("--tokenizer_cache_dir", default=None)
    parser.add_argument("--modules_cache_dir", default=None)
    parser.add_argument("--assets_cache_dir", default=None)
    parser.add_argument("--overwrite", action="store_true", default=True)
    # Kept for CLI compatibility with the SentencePiece version; ignored by this implementation.
    parser.add_argument("--input_sentence_size", type=int, default=0)
    parser.add_argument("--max_sentence_length", type=int, default=8192)
    parser.add_argument("--hard_vocab_limit", action="store_true", default=False)
    args = parser.parse_args()

    if args.vocab_size < 300:
        raise ValueError("ByteLevel BPE needs at least 260 tokens for bytes + special tokens. Use vocab_size>=300.")

    cache_paths = make_cache_paths(
        None,
        cache_dir=args.cache_dir,
        hf_home=args.hf_home,
        hub_cache_dir=args.hub_cache_dir,
        datasets_cache_dir=args.datasets_cache_dir,
        model_cache_dir=args.model_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        modules_cache_dir=args.modules_cache_dir,
        assets_cache_dir=args.assets_cache_dir,
    )
    apply_hf_cache_env(cache_paths)

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        cache_dir=str(cache_paths.datasets_cache_dir) if cache_paths.datasets_cache_dir else None,
    )
    text_column = choose_text_column(dataset, args.text_column)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        n = write_corpus(dataset, text_column, corpus, args.max_samples)
        if n == 0:
            raise RuntimeError("No text was written for tokenizer training.")
        print(f"[info] Wrote {n:,} rows to {corpus}")

        hf_tokenizer = build_hf_native_bpe_tokenizer(
            corpus,
            vocab_size=args.vocab_size,
            model_max_length=args.model_max_length,
        )

    hf_tokenizer.save_pretrained(str(output_dir))
    (output_dir / "TOKENIZER_NOTE.txt").write_text(
        "HF-native ByteLevel BPE tokenizer trained with tokenizers. "
        "The model architecture can be Multiscreen or another CausalLM; it only consumes token ids.\n",
        encoding="utf-8",
    )
    validate_saved_tokenizer(output_dir, expected_vocab_size=args.vocab_size)
    print(f"Saved tokenizer to {output_dir}")
    print(f"Text column used: {text_column}")
    print("[info] tokenizer implementation: HF-native ByteLevel BPE tokenizer.json")


if __name__ == "__main__":
    main()
