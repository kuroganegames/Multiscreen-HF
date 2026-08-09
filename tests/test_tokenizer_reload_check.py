"""Focused tests for the standalone tokenizer reload verifier."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_tokenizer_reload as check


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    SPECIAL_TOKENS_ATTRIBUTES = (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    )

    def __init__(
        self,
        name_or_path: str,
        *,
        vocabulary: dict[str, int] | None = None,
        decode_suffix: str = "",
    ) -> None:
        self.name_or_path = name_or_path
        self._vocabulary = vocabulary or {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
            "story": 4,
            "private_vocab_token": 5,
        }
        self._decode_suffix = decode_suffix
        self.vocab_size = len(self._vocabulary)
        self.is_fast = True
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.sep_token = None
        self.pad_token = "<pad>"
        self.cls_token = None
        self.mask_token = None
        self.additional_special_tokens = []
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 0
        self.sep_token_id = None
        self.pad_token_id = 3
        self.cls_token_id = None
        self.mask_token_id = None
        self.additional_special_tokens_ids = []
        self.all_special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
        self.all_special_ids = [1, 2, 0, 3]
        self.special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        self.special_tokens_map_extended = dict(self.special_tokens_map)
        self.added_tokens_decoder = {
            token_id: FakeAddedToken(token)
            for token, token_id in zip(self.all_special_tokens, self.all_special_ids)
        }
        self.model_input_names = ["input_ids", "attention_mask"]
        self.model_max_length = 512
        self.padding_side = "right"
        self.truncation_side = "right"

    def __len__(self) -> int:
        return len(self._vocabulary)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocabulary)

    def get_added_vocab(self) -> dict[str, int]:
        return {token: self._vocabulary[token] for token in self.all_special_tokens}

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        del padding, truncation
        middle = 4 + sum(text.encode("utf-8")) % 2
        input_ids = [middle]
        if add_special_tokens:
            input_ids = [self.bos_token_id, *input_ids, self.eos_token_id]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del clean_up_tokenization_spaces
        values = token_ids
        if skip_special_tokens:
            values = [token_id for token_id in values if token_id not in self.all_special_ids]
        return ",".join(str(value) for value in values) + self._decode_suffix


class FakeAddedToken:
    def __init__(
        self,
        content: str,
        *,
        lstrip: bool = False,
        normalized: bool = True,
        rstrip: bool = False,
        single_word: bool = False,
        special: bool = True,
    ) -> None:
        self.content = content
        self.lstrip = lstrip
        self.normalized = normalized
        self.rstrip = rstrip
        self.single_word = single_word
        self.special = special


class FakeP0FourTokenizer(FakeTokenizer):
    def __init__(self, name_or_path: str, *, checkpoint_form: bool) -> None:
        super().__init__(name_or_path)
        self.model_max_length = 4096 if checkpoint_form else 1_000_000_000_000
        self.pad_token = self.eos_token if checkpoint_form else None
        if checkpoint_form:
            self.special_tokens_map_extended = {
                key: FakeAddedToken(token)
                for key, token in self.special_tokens_map.items()
            }

    @property
    def pad_token(self) -> str | None:
        return getattr(self, "_pad_token", None)

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._pad_token = value
        self.pad_token_id = None if value is None else self._vocabulary[value]
        if hasattr(self, "special_tokens_map"):
            self._refresh_special_metadata()

    def _refresh_special_metadata(self) -> None:
        pairs = (
            ("bos_token", self.bos_token, self.bos_token_id),
            ("eos_token", self.eos_token, self.eos_token_id),
            ("unk_token", self.unk_token, self.unk_token_id),
            ("pad_token", self.pad_token, self.pad_token_id),
        )
        self.special_tokens_map = {
            name: token for name, token, _ in pairs if token is not None
        }
        self.special_tokens_map_extended = dict(self.special_tokens_map)
        self.all_special_tokens = []
        self.all_special_ids = []
        for _, token, token_id in pairs:
            if token is not None and token not in self.all_special_tokens:
                self.all_special_tokens.append(token)
                self.all_special_ids.append(token_id)


def make_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"FakeTokenizer"}\n', encoding="utf-8"
    )
    (checkpoint / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return checkpoint


class TokenizerReloadCheckTests(unittest.TestCase):
    def test_pass_report_is_canonical_compact_and_local_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            output = root / "report.json"
            source = FakeTokenizer("public-source")
            reloaded = FakeTokenizer(str(checkpoint.resolve()))
            with mock.patch.object(
                check.AutoTokenizer,
                "from_pretrained",
                side_effect=[source, reloaded],
            ) as loader:
                result = check.main(
                    [
                        "--source-tokenizer",
                        "org/public-tokenizer",
                        "--checkpoint",
                        str(checkpoint),
                        "--logical-name",
                        "p0_3_psi8",
                        "--source-id",
                        "org/public-tokenizer",
                        "--checkpoint-id",
                        "p0-3-psi8-checkpoint",
                        "--cache-dir",
                        str(root / "cache"),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(output.read_bytes(), check.canonical_json_bytes(report))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["logical_name"], "p0_3_psi8")
            self.assertEqual(report["checked_fields"], list(check.CHECKED_FIELDS))
            self.assertTrue(report["checkpoint"]["reloaded_from_checkpoint"])
            self.assertEqual(report["counts"]["probes"], 5)
            self.assertEqual(
                report["counts"]["special_token_boundary_probes"],
                len(source.all_special_tokens) * len(check.SPECIAL_TOKEN_BOUNDARIES),
            )
            self.assertEqual(
                report["source_normalization"],
                {
                    "model_max_length": None,
                    "pad_token_from_eos": False,
                    "padding_side": None,
                },
            )
            self.assertEqual(report["operationalization"]["model_max_length"], 512)
            self.assertEqual(report["operationalization"]["truncation_side"], "right")
            for value in report["hashes"].values():
                self.assertRegex(value, r"^[0-9a-f]{64}$")
            serialized = output.read_text(encoding="utf-8")
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("private_vocab_token", serialized)
            self.assertEqual(len(loader.call_args_list), 2)
            self.assertTrue(loader.call_args_list[0].kwargs["local_files_only"])
            self.assertTrue(loader.call_args_list[1].kwargs["local_files_only"])
            self.assertFalse(loader.call_args_list[0].kwargs["trust_remote_code"])
            self.assertEqual(stat_mode(output), 0o600)

    def test_allow_nonlocal_applies_only_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            with mock.patch.object(
                check.AutoTokenizer,
                "from_pretrained",
                side_effect=[
                    FakeTokenizer("public-source"),
                    FakeTokenizer(str(checkpoint.resolve())),
                ],
            ) as loader:
                result = check.main(
                    [
                        "--source-tokenizer",
                        "org/public-tokenizer",
                        "--checkpoint",
                        str(checkpoint),
                        "--logical-name",
                        "p0_4_psi16",
                        "--allow-nonlocal",
                        "--output",
                        str(root / "report.json"),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertFalse(loader.call_args_list[0].kwargs["local_files_only"])
            self.assertTrue(loader.call_args_list[1].kwargs["local_files_only"])

    def test_p0_4_source_operationalization_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            unadjusted_output = root / "unadjusted.json"
            with mock.patch.object(
                check.AutoTokenizer,
                "from_pretrained",
                side_effect=[
                    FakeP0FourTokenizer("gpt2", checkpoint_form=False),
                    FakeP0FourTokenizer(
                        str(checkpoint.resolve()), checkpoint_form=True
                    ),
                ],
            ):
                unadjusted = check.main(
                    [
                        "--source-tokenizer",
                        "gpt2",
                        "--checkpoint",
                        str(checkpoint),
                        "--logical-name",
                        "p0_4_psi8",
                        "--output",
                        str(unadjusted_output),
                    ]
                )
            self.assertEqual(unadjusted, 1)
            self.assertFalse(unadjusted_output.exists())

            adjusted_output = root / "adjusted.json"
            with mock.patch.object(
                check.AutoTokenizer,
                "from_pretrained",
                side_effect=[
                    FakeP0FourTokenizer("gpt2", checkpoint_form=False),
                    FakeP0FourTokenizer(
                        str(checkpoint.resolve()), checkpoint_form=True
                    ),
                ],
            ):
                adjusted = check.main(
                    [
                        "--source-tokenizer",
                        "gpt2",
                        "--checkpoint",
                        str(checkpoint),
                        "--logical-name",
                        "p0_4_psi8",
                        "--source-pad-token-from-eos",
                        "--source-padding-side",
                        "right",
                        "--source-model-max-length",
                        "4096",
                        "--output",
                        str(adjusted_output),
                    ]
                )
            self.assertEqual(adjusted, 0)
            report = json.loads(adjusted_output.read_text(encoding="utf-8"))
            self.assertEqual(
                report["source_normalization"],
                {
                    "model_max_length": 4096,
                    "pad_token_from_eos": True,
                    "padding_side": "right",
                },
            )
            self.assertEqual(
                report["operationalization"],
                {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "model_max_length": 4096,
                    "padding_side": "right",
                    "truncation_side": "right",
                },
            )

    def test_vocabulary_and_probe_mismatches_fail_without_output(self) -> None:
        truncation_changed = FakeTokenizer("placeholder")
        truncation_changed.truncation_side = "left"
        added_flag_changed = FakeTokenizer("placeholder")
        added_flag_changed.added_tokens_decoder[1] = FakeAddedToken("<s>", lstrip=True)
        mutations = (
            FakeTokenizer(
                "placeholder",
                vocabulary={
                    "<unk>": 0,
                    "<s>": 1,
                    "</s>": 2,
                    "<pad>": 3,
                    "story": 4,
                    "different": 5,
                },
            ),
            FakeTokenizer("placeholder", decode_suffix="changed"),
            truncation_changed,
            added_flag_changed,
        )
        for index, changed in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkpoint = make_checkpoint(root)
                changed.name_or_path = str(checkpoint.resolve())
                output = root / "report.json"
                with mock.patch.object(
                    check.AutoTokenizer,
                    "from_pretrained",
                    side_effect=[FakeTokenizer("public-source"), changed],
                ):
                    result = check.main(
                        [
                            "--source-tokenizer",
                            "public-source",
                            "--checkpoint",
                            str(checkpoint),
                            "--logical-name",
                            "p0_3_psi16",
                            "--output",
                            str(output),
                        ]
                    )
                self.assertEqual(result, 1)
                self.assertFalse(output.exists())

    def test_checkpoint_must_exist_and_must_not_be_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing"
            output = root / "report.json"
            args = [
                "--source-tokenizer",
                "public-source",
                "--checkpoint",
                str(missing),
                "--logical-name",
                "p0_3_psi8",
                "--output",
                str(output),
            ]
            self.assertEqual(check.main(args), 1)
            real = make_checkpoint(root)
            linked = root / "linked-checkpoint"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            args[3] = str(linked)
            self.assertEqual(check.main(args), 1)
            self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            output = root / "report.json"
            output.write_bytes(b"existing\n")
            with mock.patch.object(
                check.AutoTokenizer,
                "from_pretrained",
                side_effect=[
                    FakeTokenizer("public-source"),
                    FakeTokenizer(str(checkpoint.resolve())),
                ],
            ):
                result = check.main(
                    [
                        "--source-tokenizer",
                        "public-source",
                        "--checkpoint",
                        str(checkpoint),
                        "--logical-name",
                        "p0_4_psi8",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(result, 1)
            self.assertEqual(output.read_bytes(), b"existing\n")

    def test_output_path_must_be_absolute(self) -> None:
        with self.assertRaisesRegex(check.TokenizerReloadError, "must be absolute"):
            check.safe_write_new(Path("relative-report.json"), b"{}\n")


    def test_private_identifiers_and_wrong_reload_origin_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            base_args = [
                "--source-tokenizer",
                "public-source",
                "--checkpoint",
                str(checkpoint),
                "--logical-name",
                "p0_4_psi8",
                "--output",
                str(root / "report.json"),
            ]
            self.assertEqual(check.main([*base_args, "--source-id", "/private/source"]), 1)
            with mock.patch.object(
                check.AutoTokenizer,
                "from_pretrained",
                side_effect=[FakeTokenizer("public-source"), FakeTokenizer("wrong-origin")],
            ):
                self.assertEqual(check.main(base_args), 1)
            self.assertFalse((root / "report.json").exists())

    def test_committed_tinystories_tokenizer_compatibility_reload(self) -> None:
        source = REPOSITORY_ROOT / "tokenizers" / "tinystories_spm768"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            shutil.copytree(source, checkpoint)
            output = root / "report.json"
            result = check.main(
                [
                    "--source-tokenizer",
                    str(source),
                    "--checkpoint",
                    str(checkpoint),
                    "--logical-name",
                    "p0_3_psi8",
                    "--source-id",
                    "tinystories_spm768",
                    "--checkpoint-id",
                    "copied-checkpoint",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["counts"]["vocabulary"], 768)
            self.assertEqual(report["counts"]["vocab_size"], 768)
            self.assertEqual(report["counts"]["tokenizer_length"], 768)
            self.assertEqual(report["counts"]["probes"], 5)


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
