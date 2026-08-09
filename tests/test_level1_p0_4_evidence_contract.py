#!/usr/bin/env python3
"""Focused fixtures for the P0-4 selected-data evidence contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from scripts import p0_4_evidence_contract as contract


class FakePackedTokens:
    def __init__(
        self,
        values: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        *,
        shape: tuple[int, int] = (2, 3),
    ) -> None:
        self.shape = shape
        self.flat = iter(values)


class FakeTokenizer:
    pass


class FakeVerifier:
    @staticmethod
    def vocabulary_manifest(_tokenizer: object) -> dict[str, object]:
        return {
            "added_vocabulary_mapping": {"<eos>": 2},
            "full_vocabulary_mapping": {"a": 0, "b": 1, "<eos>": 2},
            "tokenizer_length": 3,
            "vocab_size": 3,
        }

    @staticmethod
    def special_tokens_manifest(_tokenizer: object) -> dict[str, object]:
        return {"all_special_tokens": ["<eos>"], "eos_token_id": 2}

    @staticmethod
    def probe_manifest(_tokenizer: object) -> dict[str, object]:
        return {
            "probes": [{"name": "story", "ids": [0, 1]}],
            "special_token_boundary_probes": [{"name": "exact", "ids": [2]}],
        }

    @staticmethod
    def operationalization_manifest(_tokenizer: object) -> dict[str, object]:
        return {
            "model_input_names": ["input_ids", "attention_mask"],
            "model_max_length": 4096,
            "padding_side": "right",
            "truncation_side": "right",
        }

    @staticmethod
    def _sha256_manifest(value: object) -> str:
        raw = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
            )
            + "\n"
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def tokenizer_projection() -> dict[str, object]:
    return contract.build_tokenizer_projection(
        FakeTokenizer(), verifier=FakeVerifier
    )


def build(
    *,
    texts: tuple[str, ...] = ("first story", "second story"),
    tokens: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    shape: tuple[int, int] = (2, 3),
    revision: str | None = None,
    dataset_fingerprint: str = "0123456789abcdef",
    data_files: object = None,
    data_dir: str | None = None,
) -> dict[str, object]:
    return contract.build_data_contract(
        source_kind="huggingface_dataset",
        dataset_name="roneneldan/TinyStories",
        dataset_config=None,
        train_split="train[:20000]",
        revision=revision,
        text_column="text",
        dataset_fingerprint=dataset_fingerprint,
        data_files=data_files,
        data_dir=data_dir,
        text_file=None,
        streaming=False,
        max_texts=20000,
        max_train_tokens=524416,
        texts=texts,
        packed_tokens=FakePackedTokens(tokens, shape=shape),
        seq_len=2,
        eos_token_id=2,
        tokenizer=tokenizer_projection(),
    )


def load_smoke_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: (lambda function: function)
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_data = types.ModuleType("torch.utils.data")
    fake_torch_data.DataLoader = object
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *_args, **_kwargs: None
    module_name = "level1_p0_4_smoke_fixture"
    path = Path(__file__).resolve().parents[1] / "scripts" / "p0_4_gpt2_context4096_smoke.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct P0-4 smoke module fixture")
    module = importlib.util.module_from_spec(spec)
    modules = {
        module_name: module,
        "torch": fake_torch,
        "torch.utils": fake_torch_utils,
        "torch.utils.data": fake_torch_data,
        "datasets": fake_datasets,
    }
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class P0FourEvidenceContractTests(unittest.TestCase):
    def test_default_ref_contract_is_canonical_path_free_and_complete(self) -> None:
        report = build()
        raw = contract.canonical_json_bytes(report)
        self.assertEqual(raw, contract.canonical_json_bytes(json.loads(raw)))
        self.assertEqual(report["schema_version"], contract.SCHEMA_VERSION)
        self.assertEqual(report["status"], "recorded")
        self.assertEqual(report["source"]["dataset_name"], "roneneldan/TinyStories")
        self.assertEqual(report["source"]["dataset_fingerprint"], "0123456789abcdef")
        self.assertIsNone(report["source"]["revision"])
        self.assertEqual(report["source"]["revision_resolution"], "default_ref")
        self.assertEqual(report["source"]["text_column"], "text")
        self.assertEqual(report["source"]["selected_text_count"], 2)
        self.assertEqual(report["packing"]["chunk_count"], 2)
        self.assertEqual(report["packing"]["chunk_size"], 3)
        self.assertEqual(report["packing"]["usable_token_count"], 6)
        self.assertEqual(report["packing"]["seq_len"], 2)
        self.assertEqual(report["packing"]["eos_token_id"], 2)
        self.assertTrue(report["packing"]["legacy_shifted_labels"])
        self.assertTrue(report["packing"]["return_labels_are_shifted"])
        self.assertEqual(report["tokenizer"]["class"], "FakeTokenizer")
        self.assertNotIn("/home/", raw.decode("utf-8"))
        self.assertEqual(build(), report)

    def test_text_and_exact_uint32_token_mutations_change_bound_hashes(self) -> None:
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

    def test_tokenizer_projection_uses_reload_verifier_manifests(self) -> None:
        projection = tokenizer_projection()
        expected = {
            "probe_manifest_sha256": FakeVerifier._sha256_manifest(
                FakeVerifier.probe_manifest(None)
            ),
            "special_tokens_manifest_sha256": FakeVerifier._sha256_manifest(
                FakeVerifier.special_tokens_manifest(None)
            ),
            "vocabulary_manifest_sha256": FakeVerifier._sha256_manifest(
                FakeVerifier.vocabulary_manifest(None)
            ),
        }
        self.assertEqual(projection["hashes"], expected)
        self.assertEqual(
            projection["operationalization"],
            FakeVerifier.operationalization_manifest(None),
        )

    def test_revision_fingerprint_shape_and_token_range_fail_closed(self) -> None:
        explicit = build(revision="f" * 40)
        self.assertEqual(explicit["source"]["revision_resolution"], "explicit_commit")
        for revision in ("main", "/private/revision", "F" * 40):
            with self.subTest(revision=revision), self.assertRaisesRegex(
                contract.P0FourEvidenceContractError,
                "default ref or a full lowercase commit",
            ):
                build(revision=revision)
        with self.assertRaisesRegex(
            contract.P0FourEvidenceContractError, "16-64 lowercase hexadecimal"
        ):
            build(dataset_fingerprint="not-a-fingerprint")
        with self.assertRaisesRegex(
            contract.P0FourEvidenceContractError, "shape"
        ):
            build(tokens=(1, 2, 3, 4, 5), shape=(1, 5))
        with self.assertRaisesRegex(
            contract.P0FourEvidenceContractError, "outside uint32"
        ):
            build(tokens=(1, 2, 3, 4, 5, 2**32))

    def test_overrides_and_local_path_are_recorded_only_as_markers(self) -> None:
        overridden = build(
            data_files={"train": "/private/data.json"},
            data_dir="/private/data",
        )
        raw = contract.canonical_json_bytes(overridden).decode("utf-8")
        self.assertEqual(
            overridden["source"]["data_files"],
            contract.CONFIGURED_OVERRIDE_MARKER,
        )
        self.assertEqual(
            overridden["source"]["data_dir"],
            contract.CONFIGURED_OVERRIDE_MARKER,
        )
        self.assertNotIn("/private/", raw)

        local = contract.build_data_contract(
            source_kind="local_text_file",
            dataset_name="ignored",
            dataset_config=None,
            train_split="ignored",
            revision=None,
            text_column=None,
            dataset_fingerprint=None,
            data_files=None,
            data_dir=None,
            text_file="/private/stories.txt",
            streaming=False,
            max_texts=2,
            max_train_tokens=6,
            texts=("first story", "second story"),
            packed_tokens=FakePackedTokens(),
            seq_len=2,
            eos_token_id=2,
            tokenizer=tokenizer_projection(),
        )
        local_raw = contract.canonical_json_bytes(local).decode("utf-8")
        self.assertEqual(
            local["source"]["text_file"], contract.LOCAL_TEXT_FILE_MARKER
        )
        self.assertNotIn("/private/", local_raw)

    def test_report_write_is_exclusive_canonical_and_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "data_contract.json"
            report = build()
            expected = contract.canonical_json_bytes(report)
            digest = contract.write_new_report(output, report)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(
                contract.P0FourEvidenceContractError, "already exists"
            ):
                contract.write_new_report(output, report)

    def test_hub_loader_returns_actual_fingerprint_and_selected_column(self) -> None:
        smoke = load_smoke_module()

        class FakeDataset:
            column_names = ["id", "story"]
            _fingerprint = "89abcdef01234567"

            def __iter__(self):
                return iter(
                    (
                        {"id": 1, "story": "first"},
                        {"id": 2, "story": "   "},
                        {"id": 3, "story": "second"},
                        {"id": 4, "story": "third"},
                    )
                )

        seen: dict[str, object] = {}

        def fake_load(name, config, **kwargs):
            seen.update({"name": name, "config": config, **kwargs})
            return FakeDataset()

        settings = Namespace(
            text_file=None,
            dataset_name="roneneldan/TinyStories",
            dataset_config=None,
            train_split="train[:20000]",
            text_column="auto",
            cache_dir="cache-not-recorded",
            data_files=None,
            data_dir=None,
            revision=None,
            streaming=False,
            max_texts=2,
        )
        with mock.patch.object(smoke, "load_dataset", side_effect=fake_load):
            loaded = smoke.load_texts(settings)
        self.assertEqual(loaded.texts, ["first", "second"])
        self.assertEqual(loaded.dataset_fingerprint, FakeDataset._fingerprint)
        self.assertEqual(loaded.text_column, "story")
        self.assertEqual(loaded.source_kind, "huggingface_dataset")
        self.assertEqual(seen["name"], "roneneldan/TinyStories")
        self.assertIsNone(seen["revision"] if "revision" in seen else None)

    def test_contract_reference_and_existing_output_fail_closed(self) -> None:
        smoke = load_smoke_module()
        digest = "a" * 64
        self.assertEqual(
            smoke.build_data_contract_reference(
                digest, schema_version=contract.SCHEMA_VERSION
            ),
            {
                "file": "data_contract.json",
                "schema_version": contract.SCHEMA_VERSION,
                "sha256": digest,
            },
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            smoke.build_data_contract_reference(
                "A" * 64, schema_version=contract.SCHEMA_VERSION
            )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary).resolve()
            target = smoke.require_new_data_contract_path(output_dir)
            self.assertEqual(target, output_dir / "data_contract.json")
            target.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
                smoke.require_new_data_contract_path(output_dir)


if __name__ == "__main__":
    unittest.main()
