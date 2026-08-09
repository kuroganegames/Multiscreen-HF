"""Focused P0.5-C3 paper-training-contract tests.

All tests are CPU-only and offline. Actual pinned SlimPajama loading and CUDA bf16
training diagnostics are separate local evidence runs.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from scripts import p0_5_c3_paper_training_contract as c3


class FakeMapDataset:
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        fingerprint: str,
        selection_fingerprint: str,
    ) -> None:
        self.rows = rows
        self._fingerprint = fingerprint
        self.selection_fingerprint = selection_fingerprint
        self.selected_indices: list[int] | None = None

    def select(self, indices) -> "FakeMapDataset":
        selected_indices = list(indices)
        self.selected_indices = selected_indices
        return FakeMapDataset(
            [self.rows[index] for index in selected_indices],
            fingerprint=self.selection_fingerprint,
            selection_fingerprint=self.selection_fingerprint,
        )

    def __iter__(self):
        return iter(self.rows)


class DeterministicTokenizer:
    eos_token = "<eos>"

    def __init__(
        self,
        encodings: dict[str, list[int]] | None = None,
        *,
        vocab_size: int = 50_257,
        eos_token_id: int = 50_256,
    ) -> None:
        self.encodings = encodings or {}
        self.vocab_size_for_len = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = None
        self._pad_token = None
        self.model_max_length = 1024
        self.calls: list[dict[str, object]] = []

    def __len__(self) -> int:
        return self.vocab_size_for_len

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value) -> None:
        self._pad_token = value
        self.pad_token_id = self.eos_token_id

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
    ) -> list[int]:
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
            }
        )
        return list(self.encodings[text])


class ManifestContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = c3.read_json(c3.DEFAULT_MANIFEST)

    def test_checked_manifest_and_exact_scheduler_checkpoints(self) -> None:
        summary = c3.validate_manifest(self.manifest)

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(
            summary["scheduler_checkpoints"],
            {
                "0": 1 / 65_536,
                "1": 1 / 32_768,
                "4095": 1 / 16,
                "4096": 1 / 16,
                "4097": 1 / 16,
            },
        )
        self.assertFalse(summary["gradient_clipping_enabled"])
        self.assertEqual(summary["paper_global_batch_tokens"], 2**22)

    def test_asset_aggregate_uses_the_audited_canonical_payload(self) -> None:
        tokenizer = self.manifest["tokenizer"]
        payload = {
            "repo_id": tokenizer["repository"],
            "revision": tokenizer["revision"],
            "files": tokenizer["assets"],
        }
        actual = hashlib.sha256(c3.canonical_json_bytes(payload)).hexdigest()

        self.assertEqual(
            actual,
            "07c45937a89b33f30016aef5b3982f13f25bf2c6ba940c535d1b5daa90459a71",
        )
        self.assertEqual(actual, tokenizer["asset_manifest_sha256"])

    def test_contract_mutations_fail_closed(self) -> None:
        mutations = (
            ("paper beta", lambda value: value["paper_recipe"]["optimizer"]["betas"].__setitem__(1, 0.999)),
            ("weight decay", lambda value: value["paper_recipe"]["optimizer"].__setitem__("weight_decay", 0.1)),
            (
                "weight decay JSON type",
                lambda value: value["paper_recipe"]["optimizer"].__setitem__(
                    "weight_decay", False
                ),
            ),
            (
                "gradient clipping",
                lambda value: value["paper_recipe"]["gradient_clipping"].__setitem__(
                    "enabled", True
                ),
            ),
            ("paper peak", lambda value: value["paper_recipe"]["scheduler"].__setitem__("peak_learning_rate", 0.01)),
            (
                "gradient clipping JSON type",
                lambda value: value["paper_recipe"]["gradient_clipping"].__setitem__(
                    "enabled", 0
                ),
            ),
            ("warmup", lambda value: value["paper_recipe"]["scheduler"].__setitem__("warmup_steps", 4095)),
            ("sequence", lambda value: value["model"].__setitem__("sequence_length", 2048)),
            ("dataset revision", lambda value: value["dataset"].__setitem__("revision", "0" * 40)),
            ("fingerprint", lambda value: value["dataset"].__setitem__("expected_full_fingerprint", "bad")),
            ("extra key", lambda value: value["paper_recipe"].__setitem__("clip_grad_norm", 1.0)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(self.manifest)
                mutate(changed)
                with self.assertRaises(ValueError):
                    c3.validate_manifest(changed)

    def test_diagnostics_are_explicitly_non_reproducing(self) -> None:
        operational = self.manifest["diagnostics"]["operational"]
        exposure = self.manifest["diagnostics"]["peak_exposure"]

        self.assertEqual(operational["sequence_length"], 4096)
        self.assertEqual(
            operational["microbatch_size"]
            * operational["gradient_accumulation_steps"]
            * operational["sequence_length"],
            8192,
        )
        self.assertEqual(8192 / self.manifest["paper_recipe"]["paper_global_batch_tokens"], 1 / 512)
        self.assertNotEqual(
            operational["peak_learning_rate"],
            self.manifest["paper_recipe"]["scheduler"]["peak_learning_rate"],
        )
        self.assertEqual(exposure["peak_learning_rate"], 0.0625)
        self.assertFalse(exposure["require_loss_decrease"])
        self.assertFalse(operational["gradient_clipping"])
        self.assertFalse(exposure["gradient_clipping"])


class SchedulerAndOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = c3.read_json(c3.DEFAULT_MANIFEST)

    def test_reference_aligned_zero_based_warmup(self) -> None:
        expected = {
            0: 1 / 65_536,
            1: 1 / 32_768,
            4095: 1 / 16,
            4096: 1 / 16,
            4097: 1 / 16,
        }
        for step, learning_rate in expected.items():
            with self.subTest(step=step):
                self.assertEqual(
                    c3.paper_learning_rate(
                        step,
                        peak_learning_rate=0.0625,
                        warmup_steps=4096,
                    ),
                    learning_rate,
                )

    def test_scheduler_rejects_invalid_inputs(self) -> None:
        invalid = (
            (-1, 0.0625, 4096),
            (True, 0.0625, 4096),
            (0, 0.0, 4096),
            (0, float("inf"), 4096),
            (0, 0.0625, 0),
            (0, 0.0625, True),
        )
        for step, peak, warmup in invalid:
            with self.subTest(step=step, peak=peak, warmup=warmup):
                with self.assertRaises(ValueError):
                    c3.paper_learning_rate(
                        step,
                        peak_learning_rate=peak,
                        warmup_steps=warmup,
                    )

    def test_adamw_and_no_clipping_update_contract(self) -> None:
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 2, bias=True)
        named_parameters = list(model.named_parameters())
        optimizer = c3.make_optimizer(
            (parameter for _, parameter in named_parameters),
            manifest=self.manifest,
            learning_rate=0.0625,
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0)
        self.assertEqual(optimizer.defaults["eps"], 1e-8)
        self.assertFalse(optimizer.defaults["fused"])
        self.assertEqual(
            self.manifest["paper_recipe"]["optimizer"]["eps_source"],
            "repository_operationalization_paper_unspecified",
        )
        for group in optimizer.param_groups:
            self.assertEqual(group["betas"], (0.9, 0.95))
            self.assertEqual(group["weight_decay"], 0.0)

        inputs = torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.0, -1.0]])
        targets = torch.tensor([[0.5, -0.25], [-0.75, 0.125]])
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()

        with mock.patch.object(
            torch.nn.utils,
            "clip_grad_norm_",
            side_effect=AssertionError("gradient clipping must not be called"),
        ) as clipping:
            result = c3.optimizer_step_without_clipping(
                optimizer,
                named_parameters,
            )

        clipping.assert_not_called()
        self.assertGreater(result["gradient_l2_norm"], 0.0)
        self.assertNotEqual(result["tracked_parameter_delta"], 0.0)
        self.assertFalse(result["gradient_clipping_applied"])

    def test_learning_rate_is_set_on_every_parameter_group(self) -> None:
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = torch.optim.AdamW(
            [{"params": [first], "lr": 0.1}, {"params": [second], "lr": 0.2}],
            lr=0.3,
        )
        c3.set_optimizer_learning_rate(optimizer, 1 / 16)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1 / 16, 1 / 16],
        )


class PinnedSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = c3.read_json(c3.DEFAULT_MANIFEST)

    def _verified_file_patch(self):
        return mock.patch.object(
            c3,
            "_verify_hub_file_identity",
            return_value={
                "filename": "data/test-00000-of-00030.parquet",
                "size_bytes": 43_263_929,
                "sha256": self.manifest["dataset"]["expected_file_sha256"],
                "revision": self.manifest["dataset"]["revision"],
            },
        )

    def _canonical_cache_root(self, parent: Path) -> tuple[Path, Path]:
        root = parent / "canonical-cache"
        output = root / c3.OFFLINE_CACHE_RELATIVE_DIRECTORY
        output.mkdir(parents=True)
        (output / "dataset_info.json").write_text("{}\n", encoding="utf-8")
        (output / "slim_pajama-627_b_reupload-test.arrow").write_bytes(
            b"fixture-arrow"
        )
        return root.resolve(), output.resolve()

    @staticmethod
    def _canonical_rows(
        *,
        full_fingerprint: str = "507a47fcec5cbfdc",
        selection_fingerprint: str = "f1e6c1c09434a7e4",
    ) -> FakeMapDataset:
        return FakeMapDataset(
            [{"text": f"row-{index}"} for index in range(64)],
            fingerprint=full_fingerprint,
            selection_fingerprint=selection_fingerprint,
        )

    @staticmethod
    def _fake_cache_class(
        expected_output: Path,
        dataset: FakeMapDataset,
        calls: list[tuple[str, object]],
    ) -> type:
        class FakeCache:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))
                self.cache_dir = str(expected_output)

            def as_dataset(self, *, split):
                calls.append(("as_dataset", {"split": split}))
                return dataset

        return FakeCache

    def test_hub_file_identity_checks_bytes_not_only_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.bin"
            path.write_bytes(b"checked bytes")
            expected = hashlib.sha256(b"checked bytes").hexdigest()
            calls: list[dict[str, object]] = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                return str(path)

            result = c3._verify_hub_file_identity(
                repository="owner/repo",
                revision="1" * 40,
                filename="asset.bin",
                repo_type="dataset",
                expected_size=len(b"checked bytes"),
                expected_sha256=expected,
                cache_dir=None,
                hub_download_fn=fake_download,
            )

        self.assertEqual(result["sha256"], expected)
        self.assertEqual(calls[0]["repo_type"], "dataset")
        self.assertEqual(calls[0]["revision"], "1" * 40)

    def test_hub_file_identity_rejects_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.bin"
            path.write_bytes(b"wrong")
            with self.assertRaises(AssertionError):
                c3._verify_hub_file_identity(
                    repository="owner/repo",
                    revision="1" * 40,
                    filename="asset.bin",
                    repo_type="model",
                    expected_size=5,
                    expected_sha256="0" * 64,
                    cache_dir=None,
                    hub_download_fn=lambda **_: str(path),
                )

    def test_canonical_hub_map_loader_and_fixed_selection(self) -> None:
        rows = [{"text": f"row-{index}"} for index in range(64)]
        dataset = FakeMapDataset(
            rows,
            fingerprint="507a47fcec5cbfdc",
            selection_fingerprint="f1e6c1c09434a7e4",
        )
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_load(name, **kwargs):
            calls.append((name, kwargs))
            return dataset

        with mock.patch.object(
            c3,
            "_verify_hub_file_identity",
            return_value={
                "filename": "data/test-00000-of-00030.parquet",
                "size_bytes": 43_263_929,
                "sha256": self.manifest["dataset"]["expected_file_sha256"],
                "revision": self.manifest["dataset"]["revision"],
            },
        ):
            selected = c3.load_pinned_rows(
                self.manifest,
                cache_dir=None,
                load_dataset_fn=fake_load,
                datasets_version="5.0.1",
            )

        self.assertEqual(calls[0][0], "gmongaras/SlimPajama-627B_Reupload")
        self.assertEqual(
            calls[0][1]["data_files"],
            {"test": "data/test-00000-of-00030.parquet"},
        )
        self.assertEqual(calls[0][1]["split"], "test")
        self.assertEqual(
            calls[0][1]["revision"],
            "c34c22dbb10ae6b264a2f357a909d1a537141b36",
        )
        self.assertFalse(calls[0][1]["streaming"])
        self.assertEqual(calls[0][1]["verification_mode"], "no_checks")
        self.assertEqual(len(selected.texts), 64)
        self.assertEqual(selected.texts[0], "row-0")
        self.assertEqual(selected.texts[-1], "row-63")
        self.assertEqual(
            selected.provenance["full_fingerprint"],
            "507a47fcec5cbfdc",
        )
        self.assertEqual(
            selected.provenance["selection"]["fingerprint"],
            "f1e6c1c09434a7e4",
        )
        self.assertTrue(selected.provenance["canonical_hub_loader"])
        self.assertTrue(
            selected.provenance["local_parquet_fingerprint_not_accepted"]
        )
        self.assertEqual(len(selected.provenance["row_records"]), 64)
        self.assertEqual(
            selected.provenance["row_records"][0]["row_index"],
            0,
        )
        self.assertNotIn("text", selected.provenance["row_records"][0])

    def test_offline_canonical_cache_uses_exact_factory_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, expected_output = self._canonical_cache_root(Path(temporary))
            dataset = self._canonical_rows()
            calls: list[tuple[str, object]] = []
            cache_class = self._fake_cache_class(
                expected_output,
                dataset,
                calls,
            )
            online_loader = mock.Mock(
                side_effect=AssertionError("online loader must not run")
            )
            datasets_module = types.SimpleNamespace(
                __version__="5.0.1",
                load_dataset=online_loader,
            )
            with (
                self._verified_file_patch(),
                mock.patch.object(
                    c3,
                    "_import_datasets_module",
                    return_value=datasets_module,
                ),
                mock.patch.object(
                    c3,
                    "_bound_datasets_cache_class",
                    return_value=cache_class,
                ) as bound,
            ):
                selected = c3.load_pinned_rows(
                    self.manifest,
                    cache_dir=str(root),
                    environment={
                        "HF_DATASETS_OFFLINE": "1",
                        "HF_HUB_OFFLINE": "1",
                    },
                )

            bound.assert_called_once_with(datasets_module)
            online_loader.assert_not_called()
            self.assertEqual(
                calls,
                [
                    (
                        "init",
                        {
                            "cache_dir": str(root),
                            "repo_id": "gmongaras/SlimPajama-627B_Reupload",
                            "dataset_name": "slim_pajama-627_b_reupload",
                            "config_name": "default-8884724778247ab6",
                            "version": "0.0.0",
                            "hash": "c34c22dbb10ae6b264a2f357a909d1a537141b36",
                        },
                    ),
                    ("as_dataset", {"split": "test"}),
                ],
            )
            self.assertEqual(len(selected.texts), 64)
            self.assertEqual(
                selected.provenance["full_fingerprint"],
                "507a47fcec5cbfdc",
            )
            self.assertEqual(
                selected.provenance["selection"]["fingerprint"],
                "f1e6c1c09434a7e4",
            )

    def test_offline_canonical_cache_rejects_partial_or_invalid_environment(self) -> None:
        cases = (
            {"HF_DATASETS_OFFLINE": "1"},
            {"HF_HUB_OFFLINE": "1"},
            {"HF_DATASETS_OFFLINE": "true", "HF_HUB_OFFLINE": "1"},
            {"HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "false"},
        )
        with self._verified_file_patch():
            for environment in cases:
                with self.subTest(environment=environment):
                    with self.assertRaises(RuntimeError):
                        c3.load_pinned_rows(
                            self.manifest,
                            cache_dir=None,
                            environment=environment,
                        )

    def test_offline_canonical_cache_rejects_wrong_version_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _ = self._canonical_cache_root(Path(temporary))
            wrong_version = types.SimpleNamespace(
                __version__="5.0.0",
                load_dataset=mock.Mock(),
            )
            with (
                self._verified_file_patch(),
                mock.patch.object(
                    c3,
                    "_import_datasets_module",
                    return_value=wrong_version,
                ),
                mock.patch.object(c3, "_bound_datasets_cache_class") as bound,
            ):
                with self.assertRaises(RuntimeError):
                    c3.load_pinned_rows(
                        self.manifest,
                        cache_dir=str(root),
                        environment={
                            "HF_DATASETS_OFFLINE": "1",
                            "HF_HUB_OFFLINE": "1",
                        },
                    )
            bound.assert_not_called()

            changed = copy.deepcopy(self.manifest)
            changed["dataset"]["repository"] = "other/repository"
            with self.assertRaises(RuntimeError):
                c3._load_offline_canonical_dataset(
                    changed["dataset"],
                    cache_dir=str(root),
                    datasets_module=types.SimpleNamespace(__version__="5.0.1"),
                )

    def test_offline_canonical_cache_rejects_noncanonical_and_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            file_path = parent / "file"
            file_path.write_text("not a directory\n", encoding="utf-8")
            missing = parent / "missing"
            cases = (
                None,
                "relative-cache",
                "/",
                f"{parent}/.",
                str(file_path),
                str(missing),
                str(parent),
            )
            for cache_dir in cases:
                with self.subTest(cache_dir=cache_dir):
                    with self.assertRaises(ValueError):
                        c3._canonical_offline_cache_paths(cache_dir)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_offline_canonical_cache_rejects_root_and_output_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root, _ = self._canonical_cache_root(parent)
            alias = parent / "alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                c3._canonical_offline_cache_paths(str(alias))

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = parent / "cache"
            config_parent = (
                root / "gmongaras___slim_pajama-627_b_reupload"
            )
            config_parent.mkdir(parents=True)
            real_config = parent / "real-config"
            (real_config / c3.OFFLINE_CACHE_VERSION / c3.OFFLINE_CACHE_HASH).mkdir(
                parents=True
            )
            (config_parent / c3.OFFLINE_CACHE_CONFIG_NAME).symlink_to(
                real_config,
                target_is_directory=True,
            )
            with self.assertRaises(ValueError):
                c3._canonical_offline_cache_paths(str(root.resolve()))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_offline_canonical_cache_requires_exact_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, output = self._canonical_cache_root(Path(temporary))
            arrow = output / "slim_pajama-627_b_reupload-test.arrow"
            arrow.unlink()
            with self.assertRaises(ValueError):
                c3._validate_offline_cache_files(output)

            arrow.write_bytes(b"fixture-arrow")
            extra = output / "unexpected"
            extra.write_bytes(b"unexpected")
            with self.assertRaises(ValueError):
                c3._validate_offline_cache_files(output)

            extra.unlink()
            target = output.parent / "outside.arrow"
            target.write_bytes(b"fixture-arrow")
            arrow.unlink()
            arrow.symlink_to(target)
            with self.assertRaises(ValueError):
                c3._validate_offline_cache_files(output)

    def test_offline_canonical_cache_rejects_builder_output_and_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, expected_output = self._canonical_cache_root(Path(temporary))
            datasets_module = types.SimpleNamespace(
                __version__="5.0.1",
                load_dataset=mock.Mock(),
            )

            class WrongOutputCache:
                def __init__(self, **_kwargs):
                    self.cache_dir = str(root)

            with mock.patch.object(
                c3,
                "_bound_datasets_cache_class",
                return_value=WrongOutputCache,
            ):
                with self.assertRaises(RuntimeError):
                    c3._load_offline_canonical_dataset(
                        self.manifest["dataset"],
                        cache_dir=str(root),
                        datasets_module=datasets_module,
                    )

            for full, selected_fingerprint in (
                ("wrong", "f1e6c1c09434a7e4"),
                ("507a47fcec5cbfdc", "wrong"),
            ):
                calls: list[tuple[str, object]] = []
                cache_class = self._fake_cache_class(
                    expected_output,
                    self._canonical_rows(
                        full_fingerprint=full,
                        selection_fingerprint=selected_fingerprint,
                    ),
                    calls,
                )
                with (
                    self._verified_file_patch(),
                    mock.patch.object(
                        c3,
                        "_import_datasets_module",
                        return_value=datasets_module,
                    ),
                    mock.patch.object(
                        c3,
                        "_bound_datasets_cache_class",
                        return_value=cache_class,
                    ),
                ):
                    with self.subTest(
                        full=full,
                        selected=selected_fingerprint,
                    ):
                        with self.assertRaises(AssertionError):
                            c3.load_pinned_rows(
                                self.manifest,
                                cache_dir=str(root),
                                environment={
                                    "HF_DATASETS_OFFLINE": "1",
                                    "HF_HUB_OFFLINE": "1",
                                },
                            )

    def test_online_and_injected_loaders_bypass_offline_cache_factory(self) -> None:
        dataset = self._canonical_rows()
        online_loader = mock.Mock(return_value=dataset)
        datasets_module = types.SimpleNamespace(
            __version__="5.0.1",
            load_dataset=online_loader,
        )
        with (
            self._verified_file_patch(),
            mock.patch.object(
                c3,
                "_import_datasets_module",
                return_value=datasets_module,
            ),
            mock.patch.object(c3, "_bound_datasets_cache_class") as bound,
        ):
            c3.load_pinned_rows(
                self.manifest,
                cache_dir="/public/cache",
                environment={},
            )
        bound.assert_not_called()
        online_loader.assert_called_once_with(
            "gmongaras/SlimPajama-627B_Reupload",
            data_files={"test": "data/test-00000-of-00030.parquet"},
            split="test",
            revision="c34c22dbb10ae6b264a2f357a909d1a537141b36",
            streaming=False,
            verification_mode="no_checks",
            cache_dir="/public/cache",
        )

        injected_loader = mock.Mock(return_value=self._canonical_rows())
        with (
            self._verified_file_patch(),
            mock.patch.object(
                c3,
                "_import_datasets_module",
                side_effect=AssertionError("production import must not run"),
            ),
        ):
            c3.load_pinned_rows(
                self.manifest,
                cache_dir="relative-value-preserved-for-injected-loader",
                load_dataset_fn=injected_loader,
                datasets_version="5.0.1",
                environment={
                    "HF_DATASETS_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                },
            )
        self.assertEqual(
            injected_loader.call_args.kwargs["cache_dir"],
            "relative-value-preserved-for-injected-loader",
        )

    def test_cache_class_is_bound_to_installed_package_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary).resolve()
            package = site / "datasets"
            module_path = package / "packaged_modules" / "cache" / "cache.py"
            module_path.parent.mkdir(parents=True)
            package_init = package / "__init__.py"
            package_init.write_text("# fixture\n", encoding="utf-8")
            module_path.write_text("# fixture\n", encoding="utf-8")

            cache_class = type("Cache", (), {})
            cache_class.__module__ = "datasets.packaged_modules.cache.cache"
            cache_module = types.SimpleNamespace(
                __file__=str(module_path),
                __name__="datasets.packaged_modules.cache.cache",
                Cache=cache_class,
            )
            datasets_module = types.SimpleNamespace(__file__=str(package_init))
            distribution = types.SimpleNamespace(
                locate_file=lambda name: package if name == "datasets" else site
            )
            with (
                mock.patch.object(
                    c3.importlib,
                    "import_module",
                    return_value=cache_module,
                ),
                mock.patch.object(
                    c3.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
            ):
                self.assertIs(
                    c3._bound_datasets_cache_class(datasets_module),
                    cache_class,
                )

                cache_module.__file__ = str(package / "wrong.py")
                (package / "wrong.py").write_text("# wrong\n", encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    c3._bound_datasets_cache_class(datasets_module)

    def test_dataset_version_and_fingerprint_mismatches_fail(self) -> None:
        rows = [{"text": f"row-{index}"} for index in range(64)]

        def load_with_fingerprint(fingerprint: str):
            return lambda *_args, **_kwargs: FakeMapDataset(
                rows,
                fingerprint=fingerprint,
                selection_fingerprint="f1e6c1c09434a7e4",
            )

        verification = mock.patch.object(
            c3,
            "_verify_hub_file_identity",
            return_value={
                "filename": "data/test-00000-of-00030.parquet",
                "size_bytes": 43_263_929,
                "sha256": self.manifest["dataset"]["expected_file_sha256"],
                "revision": self.manifest["dataset"]["revision"],
            },
        )
        with verification:
            with self.assertRaises(RuntimeError):
                c3.load_pinned_rows(
                    self.manifest,
                    cache_dir=None,
                    load_dataset_fn=load_with_fingerprint("507a47fcec5cbfdc"),
                    datasets_version="4.0.0",
                )
            with self.assertRaises(AssertionError):
                c3.load_pinned_rows(
                    self.manifest,
                    cache_dir=None,
                    load_dataset_fn=load_with_fingerprint("wrong"),
                    datasets_version="5.0.1",
                )

    def test_tokenizer_loader_is_pinned_and_disables_document_truncation(self) -> None:
        tokenizer = DeterministicTokenizer()
        calls: list[dict[str, object]] = []

        def fake_loader(**kwargs):
            calls.append(kwargs)
            return tokenizer

        with mock.patch.object(
            c3,
            "validate_tokenizer_assets",
            return_value={
                "repository": "gpt2",
                "revision": self.manifest["tokenizer"]["revision"],
                "assets": {},
                "asset_manifest_sha256": self.manifest["tokenizer"]["asset_manifest_sha256"],
            },
        ):
            loaded, summary = c3.load_pinned_tokenizer(
                self.manifest,
                cache_dir=None,
                tokenizer_loader=fake_loader,
            )

        self.assertIs(loaded, tokenizer)
        self.assertEqual(
            calls[0]["pretrained_model_name_or_path"],
            "gpt2",
        )
        self.assertEqual(
            calls[0]["revision"],
            "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        )
        self.assertTrue(calls[0]["use_fast"])
        self.assertEqual(tokenizer.model_max_length, 1_000_000_000)
        self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
        self.assertFalse(summary["document_tokenization_truncation"])


class PackingAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = c3.read_json(c3.DEFAULT_MANIFEST)

    def test_exact_eos_stream_accounting_and_no_raw_text_output(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["packing"]["sequence_length"] = 2
        manifest["packing"]["stored_chunk_tokens"] = 3
        tokenizer = DeterministicTokenizer(
            {
                "alpha": [1, 2],
                "beta": [3, 4],
            }
        )
        selected = c3.SelectedRows(
            texts=("alpha", "", "beta"),
            provenance={
                "row_manifest_sha256": "a" * 64,
                "source_scope": "synthetic unit fixture",
            },
        )
        packed = c3.pack_selected_rows(
            manifest,
            selected_rows=selected,
            tokenizer=tokenizer,
            tokenizer_provenance={
                "tokenizer_class": "DeterministicTokenizer",
                "vocab_size": 50_257,
                "eos_token_id": 50_256,
            },
            enforce_checked_accounting=False,
        )

        expected_stream = [1, 2, 50_256, 3, 4, 50_256]
        self.assertEqual(packed.dataset.tokens.tolist(), [[1, 2, 50_256], [3, 4, 50_256]])
        self.assertEqual(
            packed.summary["accounting"],
            {
                "selected_rows": 3,
                "nonempty_documents": 2,
                "text_tokens": 4,
                "eos_tokens": 2,
                "concatenated_tokens": 6,
                "packed_chunks": 2,
                "usable_tokens": 6,
                "discarded_tail_tokens": 0,
                "sequence_length": 2,
                "stored_chunk_tokens": 3,
                "prediction_tokens_per_chunk": 2,
            },
        )
        self.assertEqual(
            packed.summary["hashes"]["token_stream_sha256"],
            c3.uint32_little_endian_sha256(expected_stream),
        )
        self.assertEqual(
            packed.summary["hashes"]["packed_chunk_sha256"],
            [
                c3.uint32_little_endian_sha256(expected_stream[:3]),
                c3.uint32_little_endian_sha256(expected_stream[3:]),
            ],
        )
        serialized = json.dumps(packed.summary, sort_keys=True)
        self.assertNotIn("alpha", serialized)
        self.assertNotIn("beta", serialized)
        self.assertEqual(
            [call["truncation"] for call in tokenizer.calls],
            [False, False],
        )

    def test_checked_accounting_mismatch_fails_closed(self) -> None:
        tokenizer = DeterministicTokenizer({"short": list(range(4096))})
        selected = c3.SelectedRows(
            texts=("short",),
            provenance={
                "row_manifest_sha256": "b" * 64,
                "source_scope": "synthetic unit fixture",
            },
        )
        with self.assertRaises(AssertionError):
            c3.pack_selected_rows(
                self.manifest,
                selected_rows=selected,
                tokenizer=tokenizer,
                tokenizer_provenance={
                    "tokenizer_class": "DeterministicTokenizer",
                    "vocab_size": 50_257,
                    "eos_token_id": 50_256,
                },
                enforce_checked_accounting=True,
            )


class ModelAndOutputBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = c3.read_json(c3.DEFAULT_MANIFEST)

    def test_stage3_checkpointing_and_paper_position_mode_are_used(self) -> None:
        model = c3.build_model(
            self.manifest,
            psi=8,
            tokenizer_summary={"eos_token_id": 50_256},
        )
        self.assertEqual(model.config.hidden_size, 64)
        self.assertEqual(model.config.num_hidden_layers, 8)
        self.assertEqual(model.config.num_attention_heads, 8)
        self.assertEqual(model.config.max_position_embeddings, 4096)
        self.assertEqual(model.config.mipe_position_mode, "paper_absolute")
        self.assertEqual(model.config.mipe_compute_dtype, "fp32")
        self.assertEqual(model.config.softmask_compute_dtype, "fp32")
        self.assertTrue(model.multiscreen.gradient_checkpointing)
        self.assertFalse(model.config.use_cache)
        with self.assertRaises(ValueError):
            c3.build_model(
                self.manifest,
                psi=32,
                tokenizer_summary={"eos_token_id": 50_256},
            )

    def test_output_must_be_new_absolute_and_outside_repository(self) -> None:
        repository = c3.DEFAULT_MANIFEST.parents[1]
        with self.assertRaises(ValueError):
            c3.prepare_output_directory(
                "relative-output",
                repository_root=repository,
            )
        with self.assertRaises(ValueError):
            c3.prepare_output_directory(
                str(repository / "outputs" / "forbidden-c3"),
                repository_root=repository,
            )

        with tempfile.TemporaryDirectory() as temporary:
            other_worktree = Path(temporary) / "other-worktree"
            other_worktree.mkdir()
            (other_worktree / ".git").write_text(
                "gitdir: /tmp/synthetic-git-dir\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                c3.prepare_output_directory(
                    str(other_worktree / "raw-output"),
                    repository_root=repository,
                )

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new-output"
            result = c3.prepare_output_directory(
                str(target),
                repository_root=repository,
            )
            self.assertEqual(result, target.resolve())
            self.assertTrue(result.is_dir())
            with self.assertRaises(FileExistsError):
                c3.prepare_output_directory(
                    str(target),
                    repository_root=repository,
                )

    def test_contract_cli_is_cpu_only_and_writes_no_artifacts(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = c3.main(["--mode", "contract"])
        result = json.loads(stdout.getvalue())

        self.assertEqual(status, 0)
        self.assertEqual(result["stage"], "P0.5-C3")
        self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
