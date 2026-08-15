"""Deterministic contracts for PackedTextDataset."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch

from multiscreen_transformers import PackedTextDataset


class DeterministicTokenizer:
    eos_token_id = 99

    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings = encodings
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return list(self.encodings[text])


class MissingEosTokenizer:
    def __init__(self, *, eos_is_none: bool) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.unk_token_id = 2
        if eos_is_none:
            self.eos_token_id = None
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        raise AssertionError("encode must not be called before EOS validation")


class PackedTextContractTests(unittest.TestCase):
    def test_eos_concatenation_exact_chunks_shift_and_no_loss_or_duplication(self) -> None:
        tokenizer = DeterministicTokenizer(
            {
                "alpha": [1, 2],
                "beta": [3],
                "gamma": [4, 5, 6],
            }
        )
        dataset = PackedTextDataset(
            ["alpha", "", "beta", "gamma"],
            tokenizer,
            seq_len=2,
        )
        expected_stream = [1, 2, 99, 3, 99, 4, 5, 6, 99]

        self.assertEqual(tokenizer.calls, [("alpha", False), ("beta", False), ("gamma", False)])
        self.assertEqual(dataset.tokens.shape, (3, 3))
        self.assertEqual(dataset.tokens.reshape(-1).tolist(), expected_stream)
        self.assertEqual(len(dataset), 3)

        expected_chunks = [expected_stream[index : index + 3] for index in range(0, 9, 3)]
        for index, chunk in enumerate(expected_chunks):
            with self.subTest(index=index):
                item = dataset[index]
                self.assertEqual(item["input_ids"].tolist(), chunk[:-1])
                self.assertEqual(item["labels"].tolist(), chunk[1:])
                self.assertTrue(torch.equal(item["attention_mask"], torch.ones(2, dtype=torch.long)))
                self.assertEqual(item["labels_are_shifted"].dtype, torch.bool)
                self.assertTrue(bool(item["labels_are_shifted"]))

    def test_max_tokens_truncates_exactly_and_stops_before_later_documents(self) -> None:
        tokenizer = DeterministicTokenizer(
            {
                "alpha": [10, 11],
                "beta": [20, 21, 22],
                "gamma": [30, 31],
            }
        )
        dataset = PackedTextDataset(
            ["alpha", "beta", "gamma"],
            tokenizer,
            seq_len=4,
            max_tokens=5,
        )

        self.assertEqual(tokenizer.calls, [("alpha", False), ("beta", False)])
        self.assertEqual(dataset.tokens.tolist(), [[10, 11, 99, 20, 21]])
        self.assertEqual(dataset[0]["input_ids"].tolist(), [10, 11, 99, 20])
        self.assertEqual(dataset[0]["labels"].tolist(), [11, 99, 20, 21])

    def test_only_the_incomplete_tail_is_discarded(self) -> None:
        tokenizer = DeterministicTokenizer(
            {
                "one": [1, 2, 3],
                "two": [4, 5, 6],
                "three": [7],
            }
        )
        dataset = PackedTextDataset(
            ["one", "two", "three"],
            tokenizer,
            seq_len=3,
        )
        complete_stream = [1, 2, 3, 99, 4, 5, 6, 99, 7, 99]

        self.assertEqual(dataset.tokens.shape, (2, 4))
        self.assertEqual(dataset.tokens.reshape(-1).tolist(), complete_stream[:8])
        self.assertEqual(complete_stream[8:], [7, 99])

    def test_hugging_face_label_mode_uses_seq_len_chunks_without_a_second_shift(self) -> None:
        tokenizer = DeterministicTokenizer(
            {
                "left": [1, 2],
                "right": [3, 4],
            }
        )
        dataset = PackedTextDataset(
            ["left", "right"],
            tokenizer,
            seq_len=3,
            legacy_shifted_labels=False,
        )

        self.assertEqual(dataset.tokens.tolist(), [[1, 2, 99], [3, 4, 99]])
        for index in range(len(dataset)):
            item = dataset[index]
            self.assertEqual(item["input_ids"].tolist(), dataset.tokens[index].tolist())
            self.assertTrue(torch.equal(item["labels"], item["input_ids"]))
            self.assertFalse(bool(item["labels_are_shifted"]))

    def test_explicit_eos_and_short_stream_errors_are_deterministic(self) -> None:
        tokenizer = DeterministicTokenizer({"tiny": [7]})
        with self.assertRaisesRegex(ValueError, r"need 4, got 2"):
            PackedTextDataset(["tiny"], tokenizer, seq_len=3, eos_token_id=8)

        dataset = PackedTextDataset(
            ["tiny", "tiny"],
            tokenizer,
            seq_len=3,
            eos_token_id=8,
            return_labels_are_shifted=False,
        )
        self.assertEqual(dataset.tokens.tolist(), [[7, 8, 7, 8]])
        self.assertNotIn("labels_are_shifted", dataset[0])

    def test_explicit_eos_takes_priority_over_tokenizer_eos(self) -> None:
        tokenizer = DeterministicTokenizer({"tiny": [7]})
        dataset = PackedTextDataset(
            ["tiny", "tiny"],
            tokenizer,
            seq_len=3,
            eos_token_id=8,
        )

        self.assertEqual(dataset.eos_token_id, 8)
        self.assertEqual(dataset.tokens.tolist(), [[7, 8, 7, 8]])

    def test_zero_is_a_valid_explicit_or_tokenizer_eos(self) -> None:
        explicit_tokenizer = DeterministicTokenizer({"tiny": [7]})
        explicit_dataset = PackedTextDataset(
            ["tiny", "tiny"],
            explicit_tokenizer,
            seq_len=3,
            eos_token_id=0,
        )
        self.assertEqual(explicit_dataset.eos_token_id, 0)
        self.assertEqual(explicit_dataset.tokens.tolist(), [[7, 0, 7, 0]])

        tokenizer_eos = DeterministicTokenizer({"tiny": [7]})
        tokenizer_eos.eos_token_id = 0
        tokenizer_dataset = PackedTextDataset(
            ["tiny", "tiny"],
            tokenizer_eos,
            seq_len=3,
        )
        self.assertEqual(tokenizer_dataset.eos_token_id, 0)
        self.assertEqual(tokenizer_dataset.tokens.tolist(), [[7, 0, 7, 0]])

    def test_missing_eos_attribute_or_none_fails_before_encode(self) -> None:
        for eos_is_none in (False, True):
            with self.subTest(eos_is_none=eos_is_none):
                tokenizer = MissingEosTokenizer(eos_is_none=eos_is_none)
                with self.assertRaisesRegex(
                    ValueError,
                    r"eos_token_id must be provided explicitly or set on tokenizer",
                ):
                    PackedTextDataset(["must-not-encode"], tokenizer, seq_len=1)
                self.assertEqual(tokenizer.calls, [])

    def test_from_hf_dataset_forwards_explicit_eos(self) -> None:
        tokenizer = DeterministicTokenizer({"alpha": [1, 2]})
        load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def load_dataset(*args, **kwargs):
            load_calls.append((args, kwargs))
            return [{"text": "alpha"}, {"text": "alpha"}]

        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = load_dataset
        with mock.patch.dict(sys.modules, {"datasets": datasets_module}):
            dataset = PackedTextDataset.from_hf_dataset(
                "fixture",
                tokenizer,
                seq_len=2,
                split="validation",
                eos_token_id=8,
            )

        self.assertEqual(load_calls, [(("fixture", None), {"split": "validation"})])
        self.assertEqual(dataset.eos_token_id, 8)
        self.assertEqual(dataset.tokens.tolist(), [[1, 2, 8], [1, 2, 8]])

    def test_from_hf_dataset_uses_tokenizer_eos_including_zero(self) -> None:
        tokenizer = DeterministicTokenizer({"alpha": [1, 2]})
        tokenizer.eos_token_id = 0

        def load_dataset(*args, **kwargs):
            self.assertEqual(args, ("fixture", None))
            self.assertEqual(kwargs, {"split": "train"})
            return [{"text": "alpha"}, {"text": "alpha"}]

        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = load_dataset
        with mock.patch.dict(sys.modules, {"datasets": datasets_module}):
            dataset = PackedTextDataset.from_hf_dataset(
                "fixture",
                tokenizer,
                seq_len=2,
            )

        self.assertEqual(dataset.eos_token_id, 0)
        self.assertEqual(dataset.tokens.tolist(), [[1, 2, 0], [1, 2, 0]])

    def test_from_hf_dataset_missing_eos_fails_before_load(self) -> None:
        tokenizer = MissingEosTokenizer(eos_is_none=True)
        load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def load_dataset(*args, **kwargs):
            load_calls.append((args, kwargs))
            raise AssertionError("load_dataset must not be called before EOS validation")

        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = load_dataset
        with mock.patch.dict(sys.modules, {"datasets": datasets_module}):
            with self.assertRaisesRegex(
                ValueError,
                r"eos_token_id must be provided explicitly or set on tokenizer",
            ):
                PackedTextDataset.from_hf_dataset("fixture", tokenizer, seq_len=1)

        self.assertEqual(load_calls, [])
        self.assertEqual(tokenizer.calls, [])


if __name__ == "__main__":
    unittest.main()
