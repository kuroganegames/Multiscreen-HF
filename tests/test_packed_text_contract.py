"""Deterministic P0.5-C1 golden tests for PackedTextDataset."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
