"""Dataset utilities for Multiscreen causal LM training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTextDataset(Dataset):
    """In-memory packed dataset for autoregressive language-model training.

    Texts are tokenized, separated by EOS, concatenated, and chunked into fixed
    length sequences.

    By default this dataset follows the original ``dieOD/multiscreen-pytorch``
    trainer: each stored chunk has ``seq_len + 1`` tokens, ``input_ids`` are
    ``chunk[:-1]``, and ``labels`` are ``chunk[1:]``.  The item also includes a
    scalar ``labels_are_shifted=True`` flag so a standard Transformers data
    collator/Trainer can forward it to ``MultiscreenForCausalLM`` and avoid a
    second internal next-token shift.

    Set ``legacy_shifted_labels=False`` for conventional Hugging Face causal-LM
    batches where ``labels == input_ids`` and the model performs the standard
    internal shift. In that mode the dataset emits ``labels_are_shifted=False``.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer,
        seq_len: int = 256,
        eos_token_id: Optional[int] = None,
        max_tokens: Optional[int] = None,
        legacy_shifted_labels: bool = True,
        return_labels_are_shifted: bool = True,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = int(seq_len)
        self.legacy_shifted_labels = bool(legacy_shifted_labels)
        self.return_labels_are_shifted = bool(return_labels_are_shifted)

        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = 0
        self.eos_token_id = int(eos_token_id)

        all_ids: list[int] = []
        for text in texts:
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(int(i) for i in ids)
            all_ids.append(self.eos_token_id)
            if max_tokens is not None and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

        chunk_size = self.seq_len + 1 if self.legacy_shifted_labels else self.seq_len
        usable = (len(all_ids) // chunk_size) * chunk_size
        if usable == 0:
            raise ValueError(f"Not enough tokens for one chunk (need {chunk_size}, got {len(all_ids)})")

        self.tokens = np.array(all_ids[:usable], dtype=np.int64).reshape(-1, chunk_size)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.tokens[idx]
        if self.legacy_shifted_labels:
            input_ids = torch.from_numpy(chunk[:-1].copy())
            labels = torch.from_numpy(chunk[1:].copy())
        else:
            input_ids = torch.from_numpy(chunk.copy())
            labels = input_ids.clone()

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        }
        if self.return_labels_are_shifted:
            item["labels_are_shifted"] = torch.tensor(self.legacy_shifted_labels, dtype=torch.bool)
        return item

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name: str,
        tokenizer,
        seq_len: int = 256,
        split: str = "train",
        text_column: str = "text",
        config_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        legacy_shifted_labels: bool = True,
        return_labels_are_shifted: bool = True,
        cache_dir: Optional[str] = None,
        data_files: Optional[str | list[str] | dict[str, str | list[str]]] = None,
        data_dir: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> "PackedTextDataset":
        """Load and pack a Hugging Face dataset.

        ``cache_dir`` is forwarded to :func:`datasets.load_dataset`, which is
        useful when training from TinyStories or other Hub datasets on machines
        with a dedicated dataset cache volume.  ``data_files`` / ``data_dir`` /
        ``revision`` are kept as narrow passthroughs for local or pinned data
        sources while preserving the original in-memory packing behavior.
        """

        from datasets import load_dataset

        load_kwargs = {
            "split": split,
            "cache_dir": cache_dir,
            "data_files": data_files,
            "data_dir": data_dir,
            "revision": revision,
        }
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
        ds = load_dataset(dataset_name, config_name, **load_kwargs)
        return cls(
            texts=(row[text_column] for row in ds),
            tokenizer=tokenizer,
            seq_len=seq_len,
            max_tokens=max_tokens,
            legacy_shifted_labels=legacy_shifted_labels,
            return_labels_are_shifted=return_labels_are_shifted,
        )
