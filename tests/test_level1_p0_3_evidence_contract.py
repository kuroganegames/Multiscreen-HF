"""Focused stdlib-only fixtures for the P0-3 data evidence contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import p0_3_evidence_contract as contract


class FakePackedTokens:
    def __init__(
        self,
        values: tuple[int, ...],
        *,
        shape: tuple[int, int] = (2, 3),
    ) -> None:
        self.shape = shape
        self.flat = values


class FakeTokenizer:
    is_fast = True

    def __init__(
        self,
        *,
        vocabulary_variant: bool = False,
        special_variant: bool = False,
        probe_variant: bool = False,
    ) -> None:
        self.vocabulary_variant = vocabulary_variant
        self.special_variant = special_variant
        self.probe_variant = probe_variant


class FakeVerifier:
    @staticmethod
    def vocabulary_manifest(tokenizer: FakeTokenizer) -> dict[str, object]:
        vocabulary = {"<eos>": 0, "story": 1}
        if tokenizer.vocabulary_variant:
            vocabulary["changed"] = 2
        return {
            "added_vocabulary_mapping": {"<eos>": 0},
            "full_vocabulary_mapping": vocabulary,
            "tokenizer_length": len(vocabulary),
            "vocab_size": len(vocabulary),
        }

    @staticmethod
    def special_tokens_manifest(tokenizer: FakeTokenizer) -> dict[str, object]:
        tokens = ["<eos>"]
        if tokenizer.special_variant:
            tokens.append("<pad>")
        return {
            "all_special_ids": list(range(len(tokens))),
            "all_special_tokens": tokens,
            "special_tokens_map": {"eos_token": "<eos>"},
        }

    @staticmethod
    def probe_manifest(tokenizer: FakeTokenizer) -> dict[str, object]:
        probe_ids = [1, 0] if tokenizer.probe_variant else [1]
        return {
            "probes": [{"input_ids": probe_ids, "probe_id": "story"}],
            "special_token_boundary_probes": [
                {"input_ids": [0], "probe_id": "exact"}
            ],
        }

    @staticmethod
    def operationalization_manifest(_tokenizer: FakeTokenizer) -> dict[str, object]:
        return {
            "model_input_names": ["input_ids", "attention_mask"],
            "model_max_length": 512,
            "padding_side": "right",
            "truncation_side": "right",
        }

    @staticmethod
    def _sha256_manifest(value: object) -> str:
        data = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
            )
            + "\n"
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


def build_projection(tokenizer: FakeTokenizer | None = None) -> dict[str, object]:
    return contract.build_tokenizer_projection(
        tokenizer or FakeTokenizer(), verifier=FakeVerifier
    )


def build(
    *,
    texts: tuple[str, ...] = ("first story", "second story"),
    tokens: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    shape: tuple[int, int] = (2, 3),
    dataset_name: str = "roneneldan/TinyStories",
    train_split: str = "train[:20000]",
    revision: str = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    dataset_fingerprint: str = "0123456789abcdef",
    data_files: object = None,
    data_dir: str | None = None,
    text_file: str | None = None,
    tokenizer_projection: dict[str, object] | None = None,
) -> dict[str, object]:
    return contract.build_data_contract(
        source_kind="huggingface_dataset",
        dataset_name=dataset_name,
        dataset_config=None,
        train_split=train_split,
        revision=revision,
        text_column="text",
        dataset_fingerprint=dataset_fingerprint,
        data_files=data_files,
        data_dir=data_dir,
        text_file=text_file,
        max_texts=20000,
        max_train_tokens=262144,
        texts=texts,
        packed_tokens=FakePackedTokens(tokens, shape=shape),
        seq_len=2,
        eos_token_id=2,
        tokenizer=tokenizer_projection or build_projection(),
    )


def build_local(*, text_file: str) -> dict[str, object]:
    return contract.build_data_contract(
        source_kind="local_text_file",
        dataset_name="/private/dataset/name",
        dataset_config="/private/dataset/config",
        train_split="/private/split",
        revision=None,
        text_column="/private/column",
        dataset_fingerprint="/private/fingerprint",
        data_files=None,
        data_dir=None,
        text_file=text_file,
        max_texts=2,
        max_train_tokens=6,
        texts=("first story", "second story"),
        packed_tokens=FakePackedTokens((1, 2, 3, 4, 5, 6)),
        seq_len=2,
        eos_token_id=2,
        tokenizer=build_projection(),
    )


class P0ThreeEvidenceContractTests(unittest.TestCase):
    def test_contract_is_canonical_path_free_and_deterministic(self) -> None:
        report = build()
        raw = contract.canonical_json_bytes(report)
        self.assertEqual(raw, contract.canonical_json_bytes(json.loads(raw)))
        self.assertEqual(report["schema_version"], contract.SCHEMA_VERSION)
        self.assertEqual(report["source"]["selected_text_count"], 2)
        self.assertEqual(report["packing"]["chunk_count"], 2)
        self.assertEqual(report["packing"]["chunk_size"], 3)
        self.assertEqual(report["packing"]["usable_token_count"], 6)
        self.assertEqual(report["tokenizer"]["class"], "FakeTokenizer")
        self.assertTrue(report["tokenizer"]["is_fast"])
        self.assertNotIn("/home/", raw.decode("utf-8"))
        self.assertEqual(build(), report)

    def test_text_and_token_mutations_change_their_respective_hashes(self) -> None:
        original = build()
        text_changed = build(texts=("first story", "changed story"))
        token_changed = build(tokens=(1, 2, 3, 4, 5, 7))
        self.assertNotEqual(
            original["source"]["selected_text_manifest_sha256"],
            text_changed["source"]["selected_text_manifest_sha256"],
        )
        self.assertEqual(
            original["packing"]["packed_token_stream_sha256"],
            text_changed["packing"]["packed_token_stream_sha256"],
        )
        self.assertNotEqual(
            original["packing"]["packed_token_stream_sha256"],
            token_changed["packing"]["packed_token_stream_sha256"],
        )

    def test_tokenizer_projection_uses_exact_reload_verifier_manifests(self) -> None:
        tokenizer = FakeTokenizer()
        projection = build_projection(tokenizer)
        expected = {
            "probe_manifest_sha256": FakeVerifier._sha256_manifest(
                FakeVerifier.probe_manifest(tokenizer)
            ),
            "special_tokens_manifest_sha256": FakeVerifier._sha256_manifest(
                FakeVerifier.special_tokens_manifest(tokenizer)
            ),
            "vocabulary_manifest_sha256": FakeVerifier._sha256_manifest(
                FakeVerifier.vocabulary_manifest(tokenizer)
            ),
        }
        self.assertEqual(projection["hashes"], expected)
        self.assertEqual(
            projection["operationalization"],
            FakeVerifier.operationalization_manifest(tokenizer),
        )

    def test_tokenizer_manifest_mutations_change_bound_hashes(self) -> None:
        original = build_projection()
        variants = (
            ("vocabulary_manifest_sha256", FakeTokenizer(vocabulary_variant=True)),
            ("special_tokens_manifest_sha256", FakeTokenizer(special_variant=True)),
            ("probe_manifest_sha256", FakeTokenizer(probe_variant=True)),
        )
        for field, tokenizer in variants:
            with self.subTest(field=field):
                changed = build_projection(tokenizer)
                self.assertNotEqual(
                    original["hashes"][field], changed["hashes"][field]
                )

        malformed = copy.deepcopy(original)
        malformed["hashes"]["vocabulary_manifest_sha256"] = "f" * 63
        with self.assertRaisesRegex(
            contract.P0ThreeEvidenceContractError, "lowercase SHA-256"
        ):
            build(tokenizer_projection=malformed)

    def test_hub_revision_and_packed_shape_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            contract.P0ThreeEvidenceContractError, "full lowercase commit"
        ):
            build(revision="main")
        with self.assertRaisesRegex(contract.P0ThreeEvidenceContractError, "shape"):
            build(tokens=(1, 2, 3, 4, 5), shape=(1, 5))

    def test_hub_identifiers_reject_private_paths(self) -> None:
        invalid = (
            {"dataset_name": "C:/private/dataset"},
            {"dataset_name": "FILE:/private/dataset"},
            {"train_split": "/home/private/split"},
            {"train_split": "train\x7fprivate"},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                contract.P0ThreeEvidenceContractError,
                "printable text|path-free public identifier",
            ):
                build(**override)

    def test_dataset_fingerprint_rejects_paths_and_non_hashes(self) -> None:
        invalid = (
            "/home/private/dataset",
            "synthetic-fingerprint",
            "ABCDEF0123456789",
            "f" * 15,
            "f" * 65,
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(
                contract.P0ThreeEvidenceContractError,
                "16-64 lowercase hexadecimal",
            ):
                build(dataset_fingerprint=value)

    def test_hub_overrides_are_forbidden_and_recorded_null(self) -> None:
        source = build()["source"]
        self.assertIsNone(source["data_files"])
        self.assertIsNone(source["data_dir"])
        self.assertIsNone(source["text_file"])
        overrides = (
            {"data_files": {"train": "/home/private/data.json"}},
            {"data_dir": "/home/private/data"},
            {"text_file": "/home/private/data.txt"},
        )
        for override in overrides:
            with self.subTest(override=override), self.assertRaisesRegex(
                contract.P0ThreeEvidenceContractError, "forbids"
            ):
                build(**override)

    def test_local_source_records_only_a_path_free_marker(self) -> None:
        private = "/home/private/stories.txt"
        report = build_local(text_file=private)
        raw = contract.canonical_json_bytes(report).decode("utf-8")
        self.assertNotIn(private, raw)
        self.assertEqual(
            report["source"]["text_file"], contract.LOCAL_TEXT_FILE_MARKER
        )
        for field in (
            "data_dir",
            "data_files",
            "dataset_config",
            "dataset_fingerprint",
            "dataset_name",
            "revision",
            "text_column",
            "train_split",
        ):
            self.assertIsNone(report["source"][field])

    def test_report_write_is_exclusive_canonical_and_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            output = parent / "data_contract.json"
            report = build()
            expected = contract.canonical_json_bytes(report)
            digest = contract.write_new_report(output, report)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            self.assertEqual(list(parent.iterdir()), [output])
            with self.assertRaisesRegex(
                contract.P0ThreeEvidenceContractError, "already exists"
            ):
                contract.write_new_report(output, report)

    def test_report_write_rejects_symlink_and_noncanonical_parents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(
                contract.P0ThreeEvidenceContractError, "symlink"
            ):
                contract.write_new_report(alias / "data_contract.json", build())
            with self.assertRaisesRegex(
                contract.P0ThreeEvidenceContractError, "canonical"
            ):
                contract.write_new_report(
                    real / ".." / "real" / "data_contract.json", build()
                )
            self.assertEqual(list(real.iterdir()), [])

    def test_report_write_failure_leaves_no_partial_final_or_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            output = parent / "data_contract.json"
            with mock.patch.object(
                contract, "_write_all", side_effect=OSError("injected")
            ), self.assertRaisesRegex(
                contract.P0ThreeEvidenceContractError, "written safely"
            ):
                contract.write_new_report(output, build())
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_report_publish_occurs_after_complete_durable_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            output = parent / "data_contract.json"
            report = build()
            expected = contract.canonical_json_bytes(report)
            original_link = os.link
            observed = {"called": False}

            def inspect_then_link(
                source: str,
                destination: str,
                **kwargs: object,
            ) -> None:
                self.assertFalse(output.exists())
                temporary_path = parent / source
                self.assertEqual(temporary_path.read_bytes(), expected)
                self.assertEqual(os.stat(temporary_path).st_mode & 0o777, 0o600)
                observed["called"] = True
                original_link(source, destination, **kwargs)

            with mock.patch.object(contract.os, "link", side_effect=inspect_then_link):
                contract.write_new_report(output, report)
            self.assertTrue(observed["called"])
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(list(parent.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
