"""Adversarial tests for the Stage E P0-3/P0-4 cache preflight."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_hf_contract_hardening_offline_cache as check


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ENVIRONMENT = dict(check.REQUIRED_OFFLINE_ENVIRONMENT)


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab_size = 50_257
        self.eos_token_id = 50_256
        self.eos_token = "<|endoftext|>"
        self.is_fast = True
        self._pad_token: str | None = None
        self.pad_token_id: int | None = None
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
    def __init__(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint
        self.column_names = ["text"]
        self._texts = ["private source text"] * check.TINY_STORIES_ROWS

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, key: str) -> list[str]:
        if key != "text":
            raise KeyError(key)
        return self._texts


def fake_projection(tokenizer: FakeTokenizer) -> dict[str, object]:
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


class HardeningOfflineCacheTests(unittest.TestCase):
    def test_complete_check_uses_one_cache_and_excludes_c3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            tokenizer_loader = mock.Mock(return_value=FakeTokenizer())
            dataset_loader = mock.Mock(
                side_effect=[
                    FakeDataset(str(cache / "default-private-fingerprint")),
                    FakeDataset(str(cache / "pinned-private-fingerprint")),
                ]
            )
            with mock.patch.object(
                check._level1,
                "_default_c3_module",
                side_effect=AssertionError("Stage E must not load C3"),
            ):
                report = check.check_offline_cache(
                    REPOSITORY_ROOT,
                    cache,
                    tokenizer_loader=tokenizer_loader,
                    tokenizer_projector=fake_projection,
                    dataset_loader=dataset_loader,
                )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                report["scope"],
                {
                    "fresh_p0_3": True,
                    "fresh_p0_4": True,
                    "fresh_p0_5_c3": False,
                },
            )
            self.assertNotIn("p0_5_c3", report["checks"])
            self.assertEqual(tokenizer_loader.call_count, 1)
            self.assertEqual(dataset_loader.call_count, 2)
            tokenizer_kwargs = tokenizer_loader.call_args.kwargs
            self.assertEqual(tokenizer_kwargs["cache_dir"], os.fspath(cache))
            self.assertTrue(tokenizer_kwargs["local_files_only"])
            self.assertFalse(tokenizer_kwargs["trust_remote_code"])
            self.assertTrue(tokenizer_kwargs["use_fast"])
            dataset_calls = dataset_loader.call_args_list
            self.assertEqual(
                [call.kwargs["cache_dir"] for call in dataset_calls],
                [os.fspath(cache), os.fspath(cache)],
            )
            self.assertNotIn("revision", dataset_calls[0].kwargs)
            self.assertEqual(
                dataset_calls[1].kwargs["revision"],
                check.TINY_STORIES_PINNED_REVISION,
            )

            serialized = check.canonical_json_bytes(report)
            self.assertEqual(serialized.count(b"\n"), 1)
            self.assertNotIn(os.fspath(cache).encode(), serialized)
            self.assertEqual(json.loads(serialized), report)

    def test_success_report_has_exact_top_level_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            report = check.check_offline_cache(
                REPOSITORY_ROOT,
                cache,
                tokenizer_loader=mock.Mock(return_value=FakeTokenizer()),
                tokenizer_projector=fake_projection,
                dataset_loader=mock.Mock(
                    side_effect=[FakeDataset("default"), FakeDataset("pinned")]
                ),
            )
        self.assertEqual(
            set(report),
            {
                "cache",
                "checks",
                "offline_environment",
                "schema_version",
                "scope",
                "status",
            },
        )
        self.assertEqual(report["schema_version"], check.SCHEMA_VERSION)
        self.assertEqual(
            set(report["checks"]),
            {
                "p0_3_tinystories",
                "p0_4_gpt2_tokenizer",
                "p0_4_tinystories",
            },
        )

    def test_missing_offline_environment_fails_before_path_validation(self) -> None:
        output = io.BytesIO()
        result = check.main(
            ["--repo-root", "/private/repository", "--cache-dir", "/private/cache"],
            environment={},
            stdout=output,
        )
        self.assertEqual(result, 1)
        self.assertEqual(
            output.getvalue(),
            check.canonical_json_bytes(
                check._failure_report("offline_environment_required")
            ),
        )
        self.assertNotIn(b"/private/", output.getvalue())

    def test_invalid_arguments_emit_one_path_free_canonical_object(self) -> None:
        output = io.BytesIO()
        result = check.main([], environment=OFFLINE_ENVIRONMENT, stdout=output)
        expected = check._failure_report("invalid_arguments")
        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), check.canonical_json_bytes(expected))
        self.assertEqual(output.getvalue().count(b"\n"), 1)

    def test_safe_path_cli_loads_fixed_sibling_without_pythonpath(self) -> None:
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        self.assertNotIn("PYTHONPATH", environment)
        result = subprocess.run(
            [
                sys.executable,
                "-P",
                "-S",
                "-B",
                "scripts/check_hf_contract_hardening_offline_cache.py",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            result.stdout,
            check.canonical_json_bytes(check._failure_report("invalid_arguments")),
        )
        self.assertEqual(result.stderr, b"")

    def test_loader_diagnostic_and_private_path_are_not_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            private = os.fspath(cache / "secret" / "source.txt")
            with self.assertRaises(check.OfflineCacheError) as raised:
                check.check_offline_cache(
                    REPOSITORY_ROOT,
                    cache,
                    tokenizer_loader=mock.Mock(side_effect=RuntimeError(private)),
                    tokenizer_projector=fake_projection,
                    dataset_loader=mock.Mock(),
                )
            report = check._failure_report(raised.exception.code)
            serialized = check.canonical_json_bytes(report)
            self.assertEqual(raised.exception.code, "p0_4_tokenizer_load_failed")
            self.assertNotIn(private.encode(), serialized)

    def test_cache_directory_rejects_relative_and_symlink_paths(self) -> None:
        with self.assertRaises(check.OfflineCacheError) as relative:
            check.validate_cache_directory("relative/cache")
        self.assertEqual(relative.exception.code, "invalid_cache_dir")

        if not hasattr(os, "symlink"):
            return
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "target"
            target.mkdir()
            alias = root / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(check.OfflineCacheError) as symlink:
                check.validate_cache_directory(os.fspath(alias))
            self.assertEqual(symlink.exception.code, "invalid_cache_dir")


if __name__ == "__main__":
    unittest.main()
