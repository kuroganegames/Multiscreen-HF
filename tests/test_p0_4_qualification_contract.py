"""Executable contracts for prospective P0-4 qualification decisions."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from multiscreen_transformers import MultiscreenConfig
from scripts import p0_4_gpt2_context4096_smoke as smoke


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONDITIONS = {
    "gpt2_vocab_50257",
    "context_4096",
    "cuda_device",
    "bf16_amp",
    "optimizer_steps_at_least_50",
}


def settings(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "seq_len": 4096,
        "amp_dtype": "bf16",
        "batch_size": 1,
        "grad_accum": 8,
    }
    values.update(overrides)
    return Namespace(**values)


def checkpointing_runtime(
    *, enabled: bool = True, non_reentrant: bool = True
) -> dict[str, object]:
    return {
        "enabled": enabled,
        "non_reentrant": non_reentrant,
        "kwargs": {"use_reentrant": False} if non_reentrant else {"use_reentrant": True},
    }


def qualify(
    *,
    run_settings: Namespace | None = None,
    vocab: int = smoke.GPT2_VOCAB_SIZE,
    device: str = "cuda:0",
    optimizer_steps: int = 50,
    runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    return smoke.qualification(
        run_settings or settings(),
        vocab,
        torch.device(device),
        optimizer_steps=optimizer_steps,
        checkpointing_runtime=(
            checkpointing_runtime() if runtime is None else runtime
        ),
    )


def note_summary(qualification: dict[str, object]) -> dict[str, object]:
    return {
        "qualification": qualification,
        "model": {"psi": 8, "parameter_count": 1234},
        "training": {
            "optimizer_steps": 50,
            "initial_probe_loss": 2.0,
            "final_probe_loss": 1.0,
            "abs_loss_drop": 1.0,
            "rel_loss_drop": 0.5,
            "peak_allocated_bytes": 0,
        },
        "checks": {
            "save_reload": {"loaded_logits_max_abs": 0.0},
            "cache": {"cache_split_logits_max_abs": 0.0},
        },
    }


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 43


class P0FourQualificationContractTests(unittest.TestCase):
    def test_authoritative_positive_predicate(self) -> None:
        for amp_dtype in ("bf16", "bfloat16"):
            with self.subTest(amp_dtype=amp_dtype):
                result = qualify(run_settings=settings(amp_dtype=amp_dtype))
                self.assertEqual(
                    result["schema_version"], smoke.QUALIFICATION_SCHEMA_VERSION
                )
                self.assertEqual(
                    tuple(result["conditions"]), smoke.QUALIFICATION_CONDITION_NAMES
                )
                self.assertTrue(result["qualified"])
                self.assertTrue(
                    all(value is True for value in result["conditions"].values())
                )
                smoke.validate_qualification(result)

    def test_each_incomplete_run_is_diagnostic(self) -> None:
        cases = {
            "wrong_vocab": {
                "vocab": smoke.GPT2_VOCAB_SIZE - 1,
                "false": {"gpt2_vocab_50257"},
            },
            "short_context": {
                "run_settings": settings(seq_len=4095),
                "false": {"context_4096"},
            },
            "cpu": {"device": "cpu", "false": {"cuda_device"}},
            "non_bf16": {
                "run_settings": settings(amp_dtype="fp16"),
                "false": {"bf16_amp"},
            },
            "microbatch_two": {
                "run_settings": settings(batch_size=2),
                "false": {"microbatch_size_1"},
            },
            "steps_49": {
                "optimizer_steps": 49,
                "false": {"optimizer_steps_at_least_50"},
            },
            "checkpointing_disabled": {
                "runtime": checkpointing_runtime(
                    enabled=False, non_reentrant=False
                ),
                "false": {
                    "gradient_checkpointing_enabled",
                    "gradient_checkpointing_non_reentrant",
                },
            },
            "reentrant": {
                "runtime": checkpointing_runtime(
                    enabled=True, non_reentrant=False
                ),
                "false": {"gradient_checkpointing_non_reentrant"},
            },
            "runtime_witness_missing": {
                "runtime": {},
                "false": {
                    "gradient_checkpointing_enabled",
                    "gradient_checkpointing_non_reentrant",
                },
            },
            "runtime_witness_inconsistent": {
                "runtime": {
                    "enabled": True,
                    "non_reentrant": True,
                    "kwargs": {"use_reentrant": True},
                },
                "false": {"gradient_checkpointing_non_reentrant"},
            },
        }
        for name, raw_case in cases.items():
            case = dict(raw_case)
            expected_false = case.pop("false")
            with self.subTest(name=name):
                result = qualify(**case)
                false_conditions = {
                    condition
                    for condition, passed in result["conditions"].items()
                    if passed is False
                }
                self.assertEqual(false_conditions, expected_false)
                self.assertFalse(result["qualified"])
                smoke.validate_qualification(result)

    def test_actual_optimizer_steps_are_authoritative(self) -> None:
        requested_fewer = settings()
        requested_fewer.steps = 1
        self.assertTrue(
            qualify(run_settings=requested_fewer, optimizer_steps=50)["qualified"]
        )
        requested_more = settings()
        requested_more.steps = 50
        result = qualify(run_settings=requested_more, optimizer_steps=49)
        self.assertFalse(result["qualified"])
        self.assertFalse(result["conditions"]["optimizer_steps_at_least_50"])

    def test_gradient_accumulation_is_not_a_qualification_condition(self) -> None:
        results = [qualify(run_settings=settings(grad_accum=value)) for value in (1, 8, 13)]
        self.assertTrue(all(result["qualified"] for result in results))
        self.assertTrue(
            all(
                "gradient_accumulation" not in result["conditions"]
                for result in results
            )
        )
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_complete_and_diagnostic_markers_are_exclusive(self) -> None:
        cases = {
            "positive": qualify(),
            "microbatch_two": qualify(run_settings=settings(batch_size=2)),
            "checkpointing_disabled": qualify(
                runtime=checkpointing_runtime(enabled=False, non_reentrant=False)
            ),
            "reentrant": qualify(
                runtime=checkpointing_runtime(enabled=True, non_reentrant=False)
            ),
            "cpu": qualify(device="cpu"),
            "non_bf16": qualify(run_settings=settings(amp_dtype="none")),
            "short_context": qualify(run_settings=settings(seq_len=1024)),
            "short_run": qualify(optimizer_steps=49),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, result in cases.items():
                with self.subTest(name=name):
                    output = root / name
                    output.mkdir()
                    complete = output / smoke.QUALIFIED_MARKER
                    diagnostic = output / smoke.DIAGNOSTIC_MARKER
                    complete.write_text("stale\n", encoding="utf-8")
                    diagnostic.write_text("stale\n", encoding="utf-8")
                    smoke.write_note(note_summary(result), output)
                    self.assertEqual(complete.exists(), bool(result["qualified"]))
                    self.assertEqual(
                        diagnostic.exists(), not bool(result["qualified"])
                    )
                    self.assertFalse((output / "P0-4_FAILED.md").exists())
                    if not result["qualified"]:
                        note = diagnostic.read_text(encoding="utf-8")
                        for condition, passed in result["conditions"].items():
                            if passed is False:
                                self.assertIn(f"`{condition}`", note)

    def test_incomplete_or_inconsistent_schema_cannot_write_complete_marker(self) -> None:
        legacy = {
            "qualified": True,
            "conditions": {name: True for name in LEGACY_CONDITIONS},
        }
        all_true = qualify()
        inconsistent = dict(all_true)
        inconsistent["qualified"] = False
        unknown = dict(all_true)
        unknown["schema_version"] = "unknown"

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name, value in {
                "legacy": legacy,
                "inconsistent": inconsistent,
                "unknown": unknown,
            }.items():
                with self.subTest(name=name), self.assertRaises(ValueError):
                    smoke.write_note(note_summary(value), output)
                self.assertFalse((output / smoke.QUALIFIED_MARKER).exists())

    def test_committed_static_preflights_include_stage_c_conditions(self) -> None:
        for psi in (8, 16):
            config_dir = (
                REPO_ROOT / f"configs/p0_4_multiscreen_psi{psi}_gpt2_ctx4096"
            )
            result = smoke.validate_config_files(Namespace(config_dir=config_dir))
            self.assertTrue(result["checks"]["run_microbatch_1"])
            self.assertTrue(result["checks"]["run_gradient_checkpointing_true"])

    def test_static_preflight_rejects_noncanonical_stage_c_settings(self) -> None:
        source = REPO_ROOT / "configs/p0_4_multiscreen_psi8_gpt2_ctx4096"
        model_json = (source / "config.json").read_text(encoding="utf-8")
        original_run = json.loads((source / "run.json").read_text(encoding="utf-8"))
        mutations = {
            "microbatch_two": ("microbatch_size", 2, "run_microbatch_1"),
            "microbatch_bool": ("microbatch_size", True, "run_microbatch_1"),
            "checkpointing_false": (
                "gradient_checkpointing",
                False,
                "run_gradient_checkpointing_true",
            ),
            "checkpointing_string": (
                "gradient_checkpointing",
                "true",
                "run_gradient_checkpointing_true",
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, (key, value, expected) in mutations.items():
                with self.subTest(name=name):
                    config_dir = root / name
                    config_dir.mkdir()
                    (config_dir / "config.json").write_text(
                        model_json, encoding="utf-8"
                    )
                    run = json.loads(json.dumps(original_run))
                    run["training"][key] = value
                    (config_dir / "run.json").write_text(
                        json.dumps(run), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(AssertionError, expected):
                        smoke.validate_config_files(
                            Namespace(config_dir=config_dir)
                        )

    def test_effective_cli_overrides_remain_diagnostic_not_static_failures(self) -> None:
        config_dir = REPO_ROOT / "configs/p0_4_multiscreen_psi8_gpt2_ctx4096"
        args = smoke.parse_args(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--config-dir",
                str(config_dir),
                "--microbatch-size",
                "2",
                "--gradient-checkpointing",
                "false",
            ]
        )
        resolved = smoke.load_settings(args)
        smoke.validate_config_files(resolved)
        self.assertEqual(resolved.batch_size, 2)
        self.assertFalse(resolved.gradient_checkpointing)
        result = qualify(
            run_settings=resolved,
            runtime=checkpointing_runtime(enabled=False, non_reentrant=False),
        )
        self.assertFalse(result["qualified"])

    def test_harness_installs_observable_non_reentrant_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_dir = Path(temporary)
            config = MultiscreenConfig(
                vocab_size=43,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                key_dim=4,
                value_dim=8,
                max_position_embeddings=16,
                use_cache=False,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
            config.save_pretrained(config_dir)
            run_settings = Namespace(
                config_dir=config_dir,
                seq_len=16,
                gradient_checkpointing=True,
            )
            model = smoke.build_model(run_settings, FakeTokenizer())
            runtime = smoke.inspect_gradient_checkpointing_runtime(model)
            self.assertTrue(runtime["enabled"])
            self.assertTrue(runtime["non_reentrant"])
            self.assertEqual(runtime["kwargs"], {"use_reentrant": False})

            model.gradient_checkpointing_disable()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
            reentrant = smoke.inspect_gradient_checkpointing_runtime(model)
            self.assertTrue(reentrant["enabled"])
            self.assertFalse(reentrant["non_reentrant"])
            self.assertEqual(reentrant["kwargs"], {"use_reentrant": True})

    def test_historical_accepted_summary_keeps_legacy_shape(self) -> None:
        path = REPO_ROOT / "docs/validation_results/P0_4_SUMMARY.json"
        historical = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            historical["tested_source_commit"],
            "3d734d74e04ce6a320fb31cf3d8241f823ff43fa",
        )
        self.assertEqual(historical["qualification_contract"]["microbatch_size"], 1)
        self.assertTrue(historical["qualification_contract"]["gradient_checkpointing"])
        for run in historical["runs"]:
            self.assertEqual(set(run["qualification_conditions"]), LEGACY_CONDITIONS)
            self.assertTrue(run["qualification_qualified"])
            self.assertEqual(run["training"]["microbatch_size"], 1)
            self.assertTrue(run["model"]["gradient_checkpointing"])


if __name__ == "__main__":
    unittest.main()
