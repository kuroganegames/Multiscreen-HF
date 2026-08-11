"""Adversarial tests for the single-cache, offline Stage 5 preflight."""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import check_level1_offline_cache as check


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ENVIRONMENT = dict(check.REQUIRED_OFFLINE_ENVIRONMENT)


class FakeTokenizer:
    def __init__(
        self,
        *,
        vocab_size: int = 50_257,
        eos_token_id: int = 50_256,
        is_fast: bool = True,
        pad_token: str | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.eos_token = "<|endoftext|>"
        self.is_fast = is_fast
        self._pad_token: str | None = None
        self.pad_token_id: int | None = None
        self.pad_token = pad_token
        self.padding_side = "left"
        self.truncation_side = "right"
        self.model_max_length = 1_000_000
        self.model_input_names = ["input_ids", "attention_mask"]

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._pad_token = value
        self.pad_token_id = self.eos_token_id if value == self.eos_token else None

    def __len__(self) -> int:
        return self.vocab_size


class FakeDataset:
    def __init__(
        self,
        *,
        row_count: int = check.TINY_STORIES_ROWS,
        fingerprint: str = "0123456789abcdef",
        columns: tuple[str, ...] = ("text",),
        nonstring_at: int | None = None,
    ) -> None:
        self._fingerprint = fingerprint
        self.column_names = list(columns)
        self._texts: list[object] = ["private source text"] * row_count
        if nonstring_at is not None and 0 <= nonstring_at < row_count:
            self._texts[nonstring_at] = None

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, key: str) -> list[object]:
        if key != "text" or key not in self.column_names:
            raise KeyError(key)
        return self._texts


class FakeC3:
    def __init__(self, *, mutation: str | None = None, fail_validate: bool = False) -> None:
        self.mutation = mutation
        self.fail_validate = fail_validate
        self.cache_arguments: list[str] = []
        self.validated = False

    def validate_manifest(self, manifest: dict[str, object]) -> dict[str, str]:
        self.validated = True
        if self.fail_validate:
            raise ValueError("private validation diagnostic")
        return {"status": "passed"}

    def load_pinned_tokenizer(
        self,
        manifest: dict[str, object],
        *,
        cache_dir: str,
    ) -> tuple[FakeTokenizer, dict[str, object]]:
        self.cache_arguments.append(cache_dir)
        config = manifest["tokenizer"]
        assert isinstance(config, dict)
        provenance = {
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "revision": config["revision"],
        }
        tokenizer = FakeTokenizer(
            vocab_size=int(config["expected_vocab_size"]),
            eos_token_id=int(config["expected_eos_token_id"]),
        )
        if self.mutation == "tokenizer_revision":
            provenance["revision"] = "0" * 40
        if self.mutation == "asset_manifest":
            provenance["asset_manifest_sha256"] = "0" * 64
        return tokenizer, provenance

    def load_pinned_rows(
        self,
        manifest: dict[str, object],
        *,
        cache_dir: str,
    ) -> SimpleNamespace:
        self.cache_arguments.append(cache_dir)
        config = manifest["dataset"]
        assert isinstance(config, dict)
        selection = config["selection"]
        assert isinstance(selection, dict)
        data_contract = config["expected_data_contract"]
        assert isinstance(data_contract, dict)
        row_count = int(selection["stop"]) - int(selection["start"])
        provenance = {
            "data_files": copy.deepcopy(config["data_files"]),
            "file": {
                "sha256": config["expected_file_sha256"],
                "size_bytes": config["expected_file_size_bytes"],
            },
            "full_fingerprint": config["expected_full_fingerprint"],
            "revision": config["revision"],
            "row_manifest_sha256": data_contract["row_manifest_sha256"],
            "selection": {"fingerprint": selection["expected_fingerprint"]},
        }
        if self.mutation == "full_fingerprint":
            provenance["full_fingerprint"] = "drifted"
        if self.mutation == "selected_fingerprint":
            provenance["selection"]["fingerprint"] = "drifted"
        if self.mutation == "shard_hash":
            provenance["file"]["sha256"] = "0" * 64
        if self.mutation == "row_manifest":
            provenance["row_manifest_sha256"] = "0" * 64
        if self.mutation == "row_count":
            row_count -= 1
        return SimpleNamespace(
            texts=tuple("private slim pajama text" for _ in range(row_count)),
            provenance=provenance,
        )


def fake_tokenizer_loader(**kwargs: object) -> FakeTokenizer:
    del kwargs
    return FakeTokenizer()


def fake_dataset_loader(repository: str, **kwargs: object) -> FakeDataset:
    del repository, kwargs
    return FakeDataset()


def fake_tokenizer_projector(tokenizer: FakeTokenizer) -> dict[str, object]:
    return {
        "class": type(tokenizer).__name__,
        "counts": {
            "added_vocabulary": 1,
            "all_special_tokens": 1,
            "probes": 5,
            "special_token_boundary_probes": 7,
            "tokenizer_length": len(tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "vocabulary": tokenizer.vocab_size,
        },
        "hashes": {
            "probe_manifest_sha256": "1" * 64,
            "special_tokens_manifest_sha256": "2" * 64,
            "vocabulary_manifest_sha256": "3" * 64,
        },
        "is_fast": tokenizer.is_fast,
        "operationalization": {
            "model_input_names": list(tokenizer.model_input_names),
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
        },
    }


def normalized_fake_projection() -> dict[str, object]:
    tokenizer = FakeTokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = check.GPT2_CONTEXT_LENGTH
    return fake_tokenizer_projector(tokenizer)


class Level1OfflineCacheTests(unittest.TestCase):
    def test_complete_check_uses_one_cache_and_public_identities_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            tokenizer_loader = mock.Mock(return_value=FakeTokenizer())
            dataset_loader = mock.Mock(
                side_effect=[
                    FakeDataset(fingerprint=str(cache / "default-private-fingerprint")),
                    FakeDataset(fingerprint=str(cache / "pinned-private-fingerprint")),
                ]
            )
            c3 = FakeC3()
            report = check.check_offline_cache(
                REPOSITORY_ROOT,
                cache,
                tokenizer_loader=tokenizer_loader,
                tokenizer_projector=fake_tokenizer_projector,
                dataset_loader=dataset_loader,
                c3_module=c3,
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["schema_version"], check.SCHEMA_VERSION)
            self.assertEqual(
                report["offline_environment"],
                dict(sorted(check.REQUIRED_OFFLINE_ENVIRONMENT.items())),
            )
            self.assertEqual(
                report["cache"],
                {"explicit": True, "path_recorded": False, "single_cache": True},
            )
            tokenizer_loader.assert_called_once_with(
                pretrained_model_name_or_path="gpt2",
                cache_dir=str(cache),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
            self.assertEqual(len(dataset_loader.call_args_list), 2)
            default_call, pinned_call = dataset_loader.call_args_list
            self.assertEqual(default_call.args, (check.TINY_STORIES_REPOSITORY,))
            self.assertNotIn("revision", default_call.kwargs)
            self.assertEqual(
                pinned_call.kwargs["revision"], check.TINY_STORIES_PINNED_REVISION
            )
            for call in dataset_loader.call_args_list:
                self.assertEqual(call.kwargs["cache_dir"], str(cache))
                self.assertEqual(call.kwargs["split"], "train[:20000]")
                self.assertFalse(call.kwargs["streaming"])
            self.assertTrue(c3.validated)
            self.assertEqual(c3.cache_arguments, [str(cache), str(cache)])
            serialized = check.canonical_json_bytes(report).decode("utf-8")
            self.assertNotIn(str(cache), serialized)
            self.assertNotIn(str(REPOSITORY_ROOT), serialized)
            self.assertNotIn("private source text", serialized)
            self.assertNotIn("private slim pajama text", serialized)
            self.assertEqual(serialized.count("\n"), 1)
            self.assertEqual(
                report["checks"]["p0_4_gpt2_tokenizer"]["identity_projection"],
                fake_tokenizer_projector(tokenizer_loader.return_value),
            )
            self.assertEqual(
                report["checks"]["p0_5_c3"]["dataset"]["row_manifest_sha256"],
                "942f9b3397ff7073342973082efa4cddf3ace16bc7e3d180c827df3203243831",
            )
            self.assertEqual(tokenizer_loader.return_value.pad_token_id, 50_256)
            self.assertEqual(tokenizer_loader.return_value.padding_side, "right")
            self.assertEqual(
                tokenizer_loader.return_value.model_max_length,
                check.GPT2_CONTEXT_LENGTH,
            )

    def test_cli_emits_exactly_one_canonical_path_free_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            output = io.BytesIO()
            expected = {
                "cache": {"explicit": True, "path_recorded": False, "single_cache": True},
                "checks": {},
                "offline_environment": dict(sorted(OFFLINE_ENVIRONMENT.items())),
                "schema_version": check.SCHEMA_VERSION,
                "status": "passed",
            }
            with mock.patch.object(check, "check_offline_cache", return_value=expected):
                result = check.main(
                    ["--repo-root", str(REPOSITORY_ROOT), "--cache-dir", str(cache)],
                    environment=OFFLINE_ENVIRONMENT,
                    stdout=output,
                )
            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue(), check.canonical_json_bytes(expected))
            self.assertEqual(output.getvalue().count(b"\n"), 1)
            self.assertNotIn(str(cache).encode(), output.getvalue())
            self.assertNotIn(str(REPOSITORY_ROOT).encode(), output.getvalue())

    def test_loader_stdout_stderr_are_not_part_of_the_cli_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            output = io.BytesIO()

            def noisy_tokenizer_loader(**kwargs: object) -> FakeTokenizer:
                del kwargs
                print("private tokenizer diagnostic")
                print("private tokenizer error", file=sys.stderr)
                return FakeTokenizer()

            datasets = iter((FakeDataset(), FakeDataset()))

            def noisy_dataset_loader(repository: str, **kwargs: object) -> FakeDataset:
                del repository, kwargs
                print("private dataset diagnostic")
                print("private dataset error", file=sys.stderr)
                return next(datasets)

            def noisy_tokenizer_projector(
                tokenizer: FakeTokenizer,
            ) -> dict[str, object]:
                print("private projector diagnostic")
                print("private projector error", file=sys.stderr)
                return fake_tokenizer_projector(tokenizer)

            with (
                mock.patch.object(
                    check,
                    "_default_tokenizer_loader",
                    return_value=noisy_tokenizer_loader,
                ),
                mock.patch.object(
                    check,
                    "_default_dataset_loader",
                    return_value=noisy_dataset_loader,
                ),
                mock.patch.object(
                    check,
                    "_default_tokenizer_projector",
                    return_value=noisy_tokenizer_projector,
                ),
                mock.patch.object(check, "_default_c3_module", return_value=FakeC3()),
            ):
                result = check.main(
                    ["--repo-root", str(REPOSITORY_ROOT), "--cache-dir", str(cache)],
                    environment=OFFLINE_ENVIRONMENT,
                    stdout=output,
                )

            self.assertEqual(result, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(output.getvalue(), check.canonical_json_bytes(report))
            self.assertEqual(output.getvalue().count(b"\n"), 1)
            self.assertNotIn(b"private", output.getvalue())

    def test_all_offline_flags_are_required_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            arguments = [
                "--repo-root",
                str(REPOSITORY_ROOT),
                "--cache-dir",
                str(cache),
            ]
            cases = [
                {},
                {"HF_HUB_OFFLINE": "1"},
                {**OFFLINE_ENVIRONMENT, "HF_HUB_OFFLINE": "true"},
                {**OFFLINE_ENVIRONMENT, "TRANSFORMERS_OFFLINE": "0"},
                {**OFFLINE_ENVIRONMENT, "HF_DATASETS_OFFLINE": ""},
            ]
            for environment in cases:
                with self.subTest(environment=environment):
                    output = io.BytesIO()
                    result = check.main(
                        arguments,
                        environment=environment,
                        stdout=output,
                    )
                    self.assertEqual(result, 1)
                    self.assertEqual(
                        json.loads(output.getvalue()),
                        {
                            "failure": "offline_environment_required",
                            "schema_version": check.SCHEMA_VERSION,
                            "status": "failed",
                        },
                    )
                    self.assertEqual(output.getvalue().count(b"\n"), 1)

    def test_path_validation_rejects_relative_noncanonical_missing_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            file_path = root / "file"
            file_path.write_text("not a cache\n", encoding="utf-8")
            values = (
                "relative/cache",
                f"{root}/.",
                str(root / "missing"),
                str(file_path),
            )
            for value in values:
                with self.subTest(value=value):
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check.validate_cache_directory(value)
                    self.assertEqual(raised.exception.code, "invalid_cache_dir")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_path_validation_rejects_leaf_and_parent_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            real.mkdir()
            leaf = root / "leaf"
            leaf.symlink_to(real, target_is_directory=True)
            parent = root / "parent"
            parent.symlink_to(root, target_is_directory=True)
            nested = parent / "real"
            for value in (leaf, nested):
                with self.subTest(value=value.name):
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check.validate_cache_directory(str(value))
                    self.assertEqual(raised.exception.code, "invalid_cache_dir")

    def test_repo_root_must_be_exact_git_top_level(self) -> None:
        nested = REPOSITORY_ROOT / "tests"
        with self.assertRaises(check.OfflineCacheError) as raised:
            check.validate_repository_root(str(nested.resolve()))
        self.assertEqual(raised.exception.code, "invalid_repo_root")

    def test_default_gpt2_identity_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            cases = (
                FakeTokenizer(vocab_size=50_256),
                FakeTokenizer(eos_token_id=0),
                FakeTokenizer(is_fast=False),
            )
            for tokenizer in cases:
                with self.subTest(tokenizer=tokenizer.__dict__):
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check._check_default_gpt2(
                            cache,
                            mock.Mock(return_value=tokenizer),
                            fake_tokenizer_projector,
                        )
                    self.assertEqual(
                        raised.exception.code, "p0_4_tokenizer_identity_mismatch"
                    )

    def test_default_gpt2_projection_is_built_after_source_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            tokenizer = FakeTokenizer()
            projector = mock.Mock(side_effect=fake_tokenizer_projector)

            report = check._check_default_gpt2(
                cache,
                mock.Mock(return_value=tokenizer),
                projector,
            )

            projector.assert_called_once_with(tokenizer)
            self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
            self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
            self.assertEqual(tokenizer.padding_side, "right")
            self.assertEqual(tokenizer.model_max_length, 4_096)
            self.assertEqual(
                report["identity_projection"]["operationalization"],
                {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "model_max_length": 4_096,
                    "padding_side": "right",
                    "truncation_side": "right",
                },
            )

    def test_default_gpt2_projection_errors_fail_closed_without_private_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            private = str(cache / "private" / "tokenizer.json")
            cases: tuple[object, ...] = (
                RuntimeError(private),
                check.OfflineCacheError(private),
                {"class": private},
                {
                    **normalized_fake_projection(),
                    "class": private,
                },
                {
                    **normalized_fake_projection(),
                    "hashes": {
                        **normalized_fake_projection()["hashes"],
                        "probe_manifest_sha256": "A" * 64,
                    },
                },
                {
                    **normalized_fake_projection(),
                    "counts": {
                        **normalized_fake_projection()["counts"],
                        "tokenizer_length": True,
                    },
                },
                {
                    **normalized_fake_projection(),
                    "operationalization": {
                        **normalized_fake_projection()["operationalization"],
                        "model_input_names": [private],
                    },
                },
            )
            for value in cases:
                with self.subTest(value=type(value).__name__):
                    if isinstance(value, Exception):
                        projector = mock.Mock(side_effect=value)
                    else:
                        projector = mock.Mock(return_value=value)
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check._check_default_gpt2(
                            cache,
                            mock.Mock(return_value=FakeTokenizer()),
                            projector,
                        )
                    self.assertEqual(
                        raised.exception.code,
                        "p0_4_tokenizer_projection_failed",
                    )
                    serialized = check.canonical_json_bytes(
                        check._failure_report(raised.exception.code)
                    )
                    self.assertNotIn(private.encode(), serialized)

    def test_default_gpt2_projection_must_cross_bind_observed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            projection = normalized_fake_projection()
            cases = (
                ("tokenizer_length", 50_256),
                ("vocab_size", 50_256),
                ("vocabulary", 50_256),
            )
            for field, value in cases:
                with self.subTest(field=field):
                    malformed = copy.deepcopy(projection)
                    malformed["counts"][field] = value
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check._check_default_gpt2(
                            cache,
                            mock.Mock(return_value=FakeTokenizer()),
                            mock.Mock(return_value=malformed),
                        )
                    self.assertEqual(
                        raised.exception.code,
                        "p0_4_tokenizer_projection_failed",
                    )

    def test_default_gpt2_projection_rejects_wrong_operational_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            cases = (
                ("model_input_names", ["attention_mask", "input_ids"]),
                ("truncation_side", "left"),
            )
            for field, value in cases:
                with self.subTest(field=field):
                    malformed = normalized_fake_projection()
                    malformed["operationalization"][field] = value
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check._check_default_gpt2(
                            cache,
                            mock.Mock(return_value=FakeTokenizer()),
                            mock.Mock(return_value=malformed),
                        )
                    self.assertEqual(
                        raised.exception.code,
                        "p0_4_tokenizer_projection_failed",
                    )

    def test_default_gpt2_rejects_wrong_tokenizer_operational_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            cases = (
                ("model_input_names", ["input_ids"]),
                ("truncation_side", "left"),
            )
            for field, value in cases:
                with self.subTest(field=field):
                    tokenizer = FakeTokenizer()
                    setattr(tokenizer, field, value)
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check._check_default_gpt2(
                            cache,
                            mock.Mock(return_value=tokenizer),
                            mock.Mock(return_value=normalized_fake_projection()),
                        )
                    self.assertEqual(
                        raised.exception.code,
                        "p0_4_tokenizer_identity_mismatch",
                    )

    def test_production_projector_is_the_p0_4_evidence_contract_builder(self) -> None:
        from scripts import p0_4_evidence_contract

        self.assertIs(
            check._default_tokenizer_projector(REPOSITORY_ROOT),
            p0_4_evidence_contract.build_tokenizer_projection,
        )

    def test_exact_script_invocation_bootstraps_without_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            scripts = root / "scripts"
            configs = root / "configs"
            cache = root / "cache"
            home = root / "home"
            environment_root = root / "environment"
            for directory in (scripts, configs, cache, home):
                directory.mkdir()
            venv.EnvBuilder(with_pip=False).create(environment_root)
            environment_python = environment_root / "bin" / "python"
            environment_site_packages = (
                environment_root
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            environment_site_packages.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                REPOSITORY_ROOT / "scripts/check_level1_offline_cache.py",
                scripts / "check_level1_offline_cache.py",
            )
            (environment_site_packages / "transformers.py").write_text(
                textwrap.dedent(
                    """
                    class Tokenizer:
                        def __init__(self):
                            self.eos_token_id = 50256
                            self.eos_token = "<|endoftext|>"
                            self.is_fast = True
                            self._pad_token = None
                            self.pad_token_id = None
                            self.padding_side = "left"
                            self.truncation_side = "right"
                            self.model_max_length = 1000000
                            self.model_input_names = ["input_ids", "attention_mask"]

                        @property
                        def pad_token(self):
                            return self._pad_token

                        @pad_token.setter
                        def pad_token(self, value):
                            self._pad_token = value
                            self.pad_token_id = 50256 if value == self.eos_token else None

                        def __len__(self):
                            return 50257

                    class AutoTokenizer:
                        @staticmethod
                        def from_pretrained(**kwargs):
                            assert kwargs["pretrained_model_name_or_path"] == "gpt2"
                            assert kwargs["local_files_only"] is True
                            return Tokenizer()
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (environment_site_packages / "datasets.py").write_text(
                textwrap.dedent(
                    """
                    class Dataset:
                        _fingerprint = "0123456789abcdef"
                        column_names = ["text"]

                        def __len__(self):
                            return 20000

                        def __getitem__(self, key):
                            assert key == "text"
                            return ("public text",) * 20000

                    def load_dataset(repository, **kwargs):
                        assert repository == "roneneldan/TinyStories"
                        assert kwargs["streaming"] is False
                        return Dataset()
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (scripts / "p0_4_evidence_contract.py").write_text(
                textwrap.dedent(
                    """
                    def build_tokenizer_projection(tokenizer):
                        assert tokenizer.pad_token_id == 50256
                        assert tokenizer.padding_side == "right"
                        assert tokenizer.model_max_length == 4096
                        return {
                            "class": type(tokenizer).__name__,
                            "counts": {
                                "added_vocabulary": 1,
                                "all_special_tokens": 1,
                                "probes": 5,
                                "special_token_boundary_probes": 7,
                                "tokenizer_length": 50257,
                                "vocab_size": 50257,
                                "vocabulary": 50257,
                            },
                            "hashes": {
                                "probe_manifest_sha256": "1" * 64,
                                "special_tokens_manifest_sha256": "2" * 64,
                                "vocabulary_manifest_sha256": "3" * 64,
                            },
                            "is_fast": True,
                            "operationalization": {
                                "model_input_names": ["input_ids", "attention_mask"],
                                "model_max_length": 4096,
                                "padding_side": "right",
                                "truncation_side": "right",
                            },
                        }
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            manifest = {
                "dataset": {
                    "data_files": {"train": "public.jsonl"},
                    "expected_file_sha256": "4" * 64,
                    "expected_file_size_bytes": 1,
                    "expected_full_fingerprint": "0123456789abcdef",
                    "repository": "public/dataset",
                    "revision": "5" * 40,
                    "selection": {
                        "expected_fingerprint": "fedcba9876543210",
                        "start": 0,
                        "stop": 2,
                    },
                    "expected_data_contract": {
                        "row_manifest_sha256": "8" * 64,
                    },
                    "split": "train",
                    "text_column": "text",
                },
                "tokenizer": {
                    "asset_manifest_sha256": "6" * 64,
                    "expected_eos_token_id": 50_256,
                    "expected_vocab_size": 50_257,
                    "repository": "gpt2",
                    "revision": "7" * 40,
                },
            }
            (configs / check.C3_MANIFEST_RELATIVE.name).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            (scripts / "p0_5_c3_paper_training_contract.py").write_text(
                textwrap.dedent(
                    """
                    from types import SimpleNamespace
                    from transformers import Tokenizer

                    def validate_manifest(manifest):
                        return {"status": "passed"}

                    def load_pinned_tokenizer(manifest, *, cache_dir):
                        del cache_dir
                        config = manifest["tokenizer"]
                        return Tokenizer(), {
                            "asset_manifest_sha256": config["asset_manifest_sha256"],
                            "revision": config["revision"],
                        }

                    def load_pinned_rows(manifest, *, cache_dir):
                        del cache_dir
                        config = manifest["dataset"]
                        selection = config["selection"]
                        return SimpleNamespace(
                            texts=("public",) * (selection["stop"] - selection["start"]),
                            provenance={
                                "data_files": dict(config["data_files"]),
                                "file": {
                                    "sha256": config["expected_file_sha256"],
                                    "size_bytes": config["expected_file_size_bytes"],
                                },
                                "full_fingerprint": config["expected_full_fingerprint"],
                                "revision": config["revision"],
                                "row_manifest_sha256": config[
                                    "expected_data_contract"
                                ]["row_manifest_sha256"],
                                "selection": {
                                    "fingerprint": selection["expected_fingerprint"]
                                },
                            },
                        )
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "init", "-q", str(root)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            environment = {
                "HOME": str(home),
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "TZ": "UTC",
                "CUDA_VISIBLE_DEVICES": "",
                "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONOPTIMIZE": "0",
                "PYTHONUNBUFFERED": "1",
                "PYTHONUTF8": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            }
            self.assertNotIn("PYTHONPATH", environment)
            result = subprocess.run(
                [
                    str(environment_python),
                    "-P",
                    "-B",
                    "scripts/check_level1_offline_cache.py",
                    "--repo-root",
                    str(root),
                    "--cache-dir",
                    str(cache),
                ],
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(result.stdout, check.canonical_json_bytes(report))
            self.assertEqual(result.stderr, b"")
            self.assertFalse(any(root.rglob("*.pyc")))
            self.assertFalse(any(root.rglob("*.pyo")))
            self.assertFalse(any(root.rglob("__pycache__")))

    def test_tinystories_identity_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            cases = (
                FakeDataset(row_count=19_999),
                FakeDataset(fingerprint=""),
                FakeDataset(columns=("content",)),
                FakeDataset(nonstring_at=10),
            )
            for dataset in cases:
                with self.subTest(dataset=dataset.__dict__):
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check._load_tiny_stories(
                            cache,
                            mock.Mock(return_value=dataset),
                            revision=check.TINY_STORIES_PINNED_REVISION,
                            failure_prefix="p0_3_dataset",
                        )
                    self.assertEqual(
                        raised.exception.code, "p0_3_dataset_identity_mismatch"
                    )

    def test_loader_failure_diagnostic_and_private_path_are_not_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            private = str(cache / "secret" / "source.txt")
            loader = mock.Mock(side_effect=RuntimeError(private))
            with self.assertRaises(check.OfflineCacheError) as raised:
                check._load_tiny_stories(
                    cache,
                    loader,
                    revision=None,
                    failure_prefix="p0_4_dataset",
                )
            report = check._failure_report(raised.exception.code)
            serialized = check.canonical_json_bytes(report)
            self.assertEqual(raised.exception.code, "p0_4_dataset_load_failed")
            self.assertNotIn(private.encode(), serialized)
            self.assertEqual(serialized.count(b"\n"), 1)

    def test_c3_rejects_manifest_validation_and_every_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            cases = (
                FakeC3(fail_validate=True),
                FakeC3(mutation="tokenizer_revision"),
                FakeC3(mutation="asset_manifest"),
                FakeC3(mutation="full_fingerprint"),
                FakeC3(mutation="selected_fingerprint"),
                FakeC3(mutation="shard_hash"),
                FakeC3(mutation="row_manifest"),
                FakeC3(mutation="row_count"),
            )
            for c3 in cases:
                with self.subTest(c3=c3.__dict__):
                    with self.assertRaises(check.OfflineCacheError) as raised:
                        check._check_c3(REPOSITORY_ROOT, cache, c3)
                    self.assertEqual(raised.exception.code, "c3_contract_check_failed")

    def test_invalid_arguments_also_emit_one_canonical_object(self) -> None:
        output = io.BytesIO()
        result = check.main([], environment=OFFLINE_ENVIRONMENT, stdout=output)
        expected = check._failure_report("invalid_arguments")
        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), check.canonical_json_bytes(expected))
        self.assertEqual(output.getvalue().count(b"\n"), 1)

    def test_c3_manifest_symlink_is_rejected_before_loading(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            configs = root / "configs"
            configs.mkdir()
            target = root / "manifest.json"
            source = REPOSITORY_ROOT / check.C3_MANIFEST_RELATIVE
            shutil.copyfile(source, target)
            (configs / check.C3_MANIFEST_RELATIVE.name).symlink_to(target)
            with self.assertRaises(check.OfflineCacheError) as raised:
                check._check_c3(root, root, FakeC3())
            self.assertEqual(raised.exception.code, "invalid_c3_manifest")

    def test_git_top_level_output_must_be_single_canonical_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cases = (
                SimpleNamespace(returncode=1, stdout=b"", stderr=b"private"),
                SimpleNamespace(returncode=0, stdout=str(root).encode(), stderr=b""),
                SimpleNamespace(
                    returncode=0,
                    stdout=(str(root) + "\nextra\n").encode(),
                    stderr=b"",
                ),
            )
            for result in cases:
                with self.subTest(stdout=result.stdout):
                    with mock.patch.object(check.subprocess, "run", return_value=result):
                        with self.assertRaises(check.OfflineCacheError) as raised:
                            check._git_top_level(root)
                    self.assertEqual(raised.exception.code, "invalid_repo_root")


if __name__ == "__main__":
    unittest.main()
