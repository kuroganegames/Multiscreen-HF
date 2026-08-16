"""Adversarial fixtures for the Stage E raw-evidence reviewer."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_SPEC = importlib.util.spec_from_file_location(
    "review_hf_contract_hardening",
    REPO_ROOT / "scripts" / "review_hf_contract_hardening.py",
)
assert REVIEW_SPEC is not None and REVIEW_SPEC.loader is not None
REVIEW = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(REVIEW)

LEGACY_TEST_SPEC = importlib.util.spec_from_file_location(
    "test_level1_evidence_review_for_stage_e",
    REPO_ROOT / "tests" / "test_level1_evidence_review.py",
)
assert LEGACY_TEST_SPEC is not None and LEGACY_TEST_SPEC.loader is not None
LEGACY_TEST = importlib.util.module_from_spec(LEGACY_TEST_SPEC)
LEGACY_TEST_SPEC.loader.exec_module(LEGACY_TEST)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(REVIEW._legacy._runner_canonical_bytes(value) for value in values)
    )


class EvidenceFixture(LEGACY_TEST.EvidenceFixture):
    """Build the fixed Stage E matrix, including its 48-command midpoint."""

    def _make_p0_4(self, root: Path, *, psi: int) -> None:
        super()._make_p0_4(root, psi=psi)
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        qualification = {
            "schema_version": REVIEW.P0_4_QUALIFICATION_SCHEMA_VERSION,
            "qualified": True,
            "conditions": {name: True for name in REVIEW.P0_4_CONDITIONS},
        }
        summary["qualification"] = qualification
        write_json(summary_path, summary)
        metrics_path = root / "metrics.jsonl"
        events = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        events[-1]["qualification"] = copy.deepcopy(qualification)
        write_jsonl(metrics_path, events)

    def _offline_cache_log(self) -> dict[str, object]:
        report = super()._offline_cache_log()
        report["checks"].pop("p0_5_c3")
        report["schema_version"] = REVIEW.OFFLINE_CACHE_SCHEMA_VERSION
        report["scope"] = {
            "fresh_p0_3": True,
            "fresh_p0_4": True,
            "fresh_p0_5_c3": False,
        }
        return report

    def _preflight_log(self, psi: int) -> dict[str, object]:
        checks = {
            "model_type_multiscreen",
            "vocab_size_50257",
            "max_position_embeddings_4096",
            "hidden_size_is_psi_squared",
            "psi_is_8_or_16",
            "layers_equal_psi",
            "heads_equal_psi",
            "tie_word_embeddings",
            "run_expected_vocab_50257",
            "run_seq_len_4096",
            "run_amp_bf16",
            "run_microbatch_1",
            "run_steps_at_least_50",
            "run_gradient_checkpointing_true",
        }
        return {
            "checks": {name: True for name in checks},
            "config_dir": os.fspath(
                self.repository
                / f"configs/p0_4_multiscreen_psi{psi}_gpt2_ctx4096"
            ),
            "psi": psi,
        }

    def _focused_log(self, name: str) -> bytes:
        expected = REVIEW._FOCUSED_TEST_BY_COMMAND[name][2]
        return (
            f"test_fixture ({name}) ... ok\n\n"
            "----------------------------------------------------------------------\n"
            f"Ran {expected} tests in 0.001s\n\n"
            "OK\n"
        ).encode("utf-8")

    def _make_ledger(self) -> None:
        encode = REVIEW._legacy._runner_canonical_bytes
        marker = {
            "created_at_utc": self._time(0),
            "format_version": "level1-requalification-run-v1",
            "repository": {
                "head_commit": self.tested_commit,
                "worktree_path_sha256": hashlib.sha256(
                    os.fsencode(self.repository)
                ).hexdigest(),
            },
            "tool_version": "1.0.0",
        }
        (self.root / ".level1-requalification-run.json").write_bytes(encode(marker))

        environment_records: list[dict[str, object]] = []
        for index, name in enumerate(REVIEW.REQUIRED_ENVIRONMENT_NAMES):
            started = 1 + index * 2
            record = {
                "cwd": {"base": "repository_root", "path": "."},
                "duration_ns": 1_000_000_000,
                "duration_seconds": 1.0,
                "ended_at_utc": self._time(started + 1),
                "format_version": "level1-requalification-command-record-v1",
                "name": name,
                "record_type": "environment",
                "repository": {"head_commit": self.tested_commit},
                "runtime": self._runtime(
                    "3.12.10" if name == "runtime-tf5141" else "3.12.11"
                ),
                "started_at_utc": self._time(started),
            }
            raw = encode(record)
            (self.root / "records" / f"{name}.json").write_bytes(raw)
            environment_records.append(record)
        (self.root / "environment.jsonl").write_bytes(
            b"".join(encode(item) for item in environment_records)
        )

        tails = REVIEW._expected_command_tails(
            repository=self.repository,
            run_root=self.root,
            cache=self.cache,
            tf4576_python=self.tf4576_python,
            tf5141_python=self.tf5141_python,
            tested_commit=self.tested_commit,
        )
        semantic_names = {
            "environment-tf4576",
            "environment-tf5141",
            "environment-cuda0",
        }
        repository_names = {
            "json-validation",
            "workflow-yaml",
            "markdown-links",
            "repository-hygiene",
            "repository-hygiene-final",
        }
        command_records: list[dict[str, object]] = []
        for index, name in enumerate(REVIEW.REQUIRED_COMMAND_NAMES):
            started = 10 + index * 3
            argv = [
                "/usr/bin/env",
                "-i",
                *REVIEW._expected_environment(name, self.root),
                *tails[name],
            ]
            log_path = self.root / "logs" / f"{name}.log"
            if name == "p0-3-checkpointed":
                log_raw = log_path.read_bytes()
            elif name == "offline-cache-preflight":
                log_raw = encode(self._offline_cache_log())
                log_path.write_bytes(log_raw)
            elif name in semantic_names:
                log_raw = encode(self._environment_log(name))
                log_path.write_bytes(log_raw)
            elif name in repository_names:
                log_raw = encode(self._repository_log(name))
                log_path.write_bytes(log_raw)
            elif name in REVIEW._FOCUSED_TEST_BY_COMMAND:
                log_raw = self._focused_log(name)
                log_path.write_bytes(log_raw)
            elif name in {"p0-4-psi8-preflight", "p0-4-psi16-preflight"}:
                psi = 8 if "psi8" in name else 16
                log_raw = encode(self._preflight_log(psi))
                log_path.write_bytes(log_raw)
            elif name in {"p0-4-psi8", "p0-4-psi16"}:
                psi = 8 if name.endswith("psi8") else 16
                log_raw = (
                    f"[P0-4] data_contract sha256="
                    f"{self.p0_4_contract_sha256[psi]}\nfixture passed\n"
                ).encode("utf-8")
                log_path.write_bytes(log_raw)
            else:
                log_raw = b"fixture passed\n"
                log_path.write_bytes(log_raw)
            record = {
                "argv": argv,
                "cwd": {"base": "repository_root", "path": "."},
                "duration_ns": 1_000_000_000,
                "duration_seconds": 1.0,
                "ended_at_utc": self._time(started + 1),
                "exit_code": 0,
                "format_version": "level1-requalification-command-record-v1",
                "log": {
                    "path": f"logs/{name}.log",
                    "sha256": hashlib.sha256(log_raw).hexdigest(),
                    "size_bytes": len(log_raw),
                },
                "name": name,
                "preconditions": {
                    "absent_paths": REVIEW._expected_absent_paths(name, self.root)
                },
                "record_type": "command",
                "returncode": 0,
                "runtime": self._runtime(
                    "3.12.10" if name in REVIEW.TF5141_COMMAND_NAMES else "3.12.11"
                ),
                "started_at_utc": self._time(started),
                "termination_signal": None,
            }
            raw = encode(record)
            (self.root / "records" / f"{name}.json").write_bytes(raw)
            command_records.append(record)
        self._all_command_records = command_records
        prefix_count = len(REVIEW.P0_4_PSI8_REVIEW_COMMAND_NAMES)
        self.ledger.write_bytes(
            b"".join(encode(item) for item in command_records[:prefix_count])
        )

    def _make_focused_review(self) -> None:
        self.p0_4_psi8_review = self.p0_4_psi8 / "raw-review.json"
        report = REVIEW.review_p0_4_lane_inputs(
            p0_4_root=self.p0_4_psi8,
            tokenizer_reports={
                "p0_4_psi8": self.tokenizer_reports["p0_4_psi8"]
            },
            command_ledger=self.ledger,
            tested_commit=self.tested_commit,
        )
        self.p0_4_psi8_review.write_bytes(REVIEW._pretty_canonical_bytes(report))
        self.ledger.write_bytes(
            b"".join(
                REVIEW._legacy._runner_canonical_bytes(item)
                for item in self._all_command_records
            )
        )

    def kwargs(self) -> dict[str, object]:
        return {
            "p0_3_root": self.p0_3,
            "p0_3_stdout": self.p0_3_stdout,
            "p0_4_psi8_root": self.p0_4_psi8,
            "p0_4_psi16_root": self.p0_4_psi16,
            "p0_4_psi8_review": self.p0_4_psi8_review,
            "tokenizer_reports": self.tokenizer_reports,
            "command_ledger": self.ledger,
            "tested_commit": self.tested_commit,
        }


class HfContractHardeningEvidenceReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = EvidenceFixture(Path(self.temporary.name) / "fixture")

    def _history(self) -> dict[str, object]:
        return {
            "status": "passed",
            "implementation_baseline": REVIEW.IMPLEMENTATION_BASELINE,
            "tested_commit": self.fixture.tested_commit,
            "artifact_count": len(REVIEW.HISTORICAL_PATHS),
            "tree_material_sha256": "a" * 64,
        }

    def review(self) -> dict[str, object]:
        with mock.patch.object(
            REVIEW,
            "_review_historical_evidence_immutability",
            return_value=self._history(),
        ):
            return REVIEW.review_inputs(**self.fixture.kwargs())

    def lane(self, psi: int = 8) -> dict[str, object]:
        root = self.fixture.p0_4_psi8 if psi == 8 else self.fixture.p0_4_psi16
        return REVIEW._review_p0_4_lane_v2(
            root, psi=psi, expected_cache=self.fixture.cache, hashes={}
        )

    def assert_rejected(self, pattern: str = "") -> None:
        with self.assertRaisesRegex(REVIEW.ReviewError, pattern or ".+"):
            self.review()

    def assert_lane_rejected(self, pattern: str = "") -> None:
        with self.assertRaisesRegex(REVIEW.ReviewError, pattern or ".+"):
            self.lane()

    def _rewrite_records(self, records: list[dict[str, object]]) -> None:
        encode = REVIEW._legacy._runner_canonical_bytes
        for record in records:
            (self.fixture.root / "records" / f"{record['name']}.json").write_bytes(
                encode(record)
            )
        self.fixture.ledger.write_bytes(b"".join(encode(item) for item in records))

    def _ledger_only(self) -> dict[str, object]:
        hashes: dict[str, str] = {}
        result, _cache, _focused = REVIEW._review_command_ledger(
            self.fixture.ledger,
            tested_commit=self.fixture.tested_commit,
            required_names=REVIEW.REQUIRED_COMMAND_NAMES,
            bind_ledgers=True,
            hashes=hashes,
        )
        return result

    def test_complete_fixture_binds_fixed_matrix_and_179_raw_events(self) -> None:
        first = self.review()
        second = self.review()
        self.assertEqual(first, second)
        ledger = first["command_ledger"]
        self.assertEqual(ledger["observed_command_count"], 53)
        self.assertEqual(ledger["required_command_count"], 53)
        self.assertEqual(
            ledger["required_commands"], sorted(REVIEW.REQUIRED_COMMAND_NAMES)
        )
        self.assertEqual(ledger["observed_environment_record_count"], 2)
        self.assertEqual(
            ledger["ordering_checks"],
            [
                "runtime-environments-before-commands",
                "all-command-records-nonoverlapping-in-fixed-order",
                "p0-4-psi8-focused-review-before-psi16",
            ],
        )
        for lane in ("tf4576", "tf5141"):
            observed = first["focused_tests"]["lanes"][lane]
            self.assertEqual(observed["command_count"], 10)
            self.assertEqual(observed["test_count"], 117)
        self.assertEqual(
            first["aggregate"]["raw_event_counts"],
            {
                "p0_3_stdout_step_events": 65,
                "p0_4_jsonl_events": 114,
                "total": 179,
            },
        )

    def test_command_and_environment_inventory_rejects_order_or_count_changes(self) -> None:
        records = self.fixture.command_records()
        records[0], records[1] = records[1], records[0]
        self.fixture.ledger.write_bytes(
            b"".join(REVIEW._legacy._runner_canonical_bytes(item) for item in records)
        )
        self.assert_rejected("fixed order")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "missing-command")
        records = self.fixture.command_records()[:-1]
        self.fixture.ledger.write_bytes(
            b"".join(REVIEW._legacy._runner_canonical_bytes(item) for item in records)
        )
        self.assert_rejected("missing, duplicated, extra")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "environment-order")
        path = self.fixture.root / "environment.jsonl"
        environments = [json.loads(line) for line in path.read_text().splitlines()]
        environments.reverse()
        path.write_bytes(
            b"".join(
                REVIEW._legacy._runner_canonical_bytes(item)
                for item in environments
            )
        )
        self.assert_rejected("exactly two ordered records")

    def test_focused_report_binds_exact_48_command_prefix_and_inner_state(self) -> None:
        stored = json.loads(self.fixture.p0_4_psi8_review.read_text(encoding="utf-8"))
        self.assertEqual(stored["command_ledger"]["required_command_count"], 48)
        self.assertEqual(
            stored["command_ledger"]["required_commands"],
            sorted(REVIEW.P0_4_PSI8_REVIEW_COMMAND_NAMES),
        )
        self.assertEqual(
            stored["command_ledger"]["ordering_checks"],
            [
                "runtime-environments-before-commands",
                "all-command-records-nonoverlapping-in-fixed-order",
                "p0-4-psi8-inputs-complete-for-focused-review",
            ],
        )
        self.assertEqual(stored["focused_tests"]["lanes"]["tf4576"]["test_count"], 117)
        self.assertEqual(stored["focused_tests"]["lanes"]["tf5141"]["test_count"], 117)

        stored["command_ledger"]["status"] = "failed"
        write_json(self.fixture.p0_4_psi8_review, stored)
        self.assert_rejected("focused report ledger|focused ledger")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "focused-tests")
        stored = json.loads(self.fixture.p0_4_psi8_review.read_text(encoding="utf-8"))
        stored["focused_tests"]["lanes"]["tf5141"]["test_count"] = 116
        write_json(self.fixture.p0_4_psi8_review, stored)
        self.assert_rejected("focused tests|117")

    def test_focused_lane_logs_require_exact_per_file_and_per_lane_counts(self) -> None:
        name = "hf-output-head-tf4576"
        self.fixture.replace_log(
            name,
            b"test_fixture ... ok\n\nRan 3 tests in 0.001s\n\nOK\n",
        )
        self.assert_rejected("exactly 4 tests")

    def test_p0_4_v2_accepts_canonical_sorted_exact_eight_conditions(self) -> None:
        summary = json.loads(
            (self.fixture.p0_4_psi8 / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            list(summary["qualification"]["conditions"]),
            sorted(REVIEW.P0_4_CONDITIONS),
        )
        reviewed = self.lane()
        self.assertEqual(
            reviewed["qualification_schema_version"],
            REVIEW.P0_4_QUALIFICATION_SCHEMA_VERSION,
        )
        self.assertEqual(
            reviewed["qualification_conditions"], list(REVIEW.P0_4_CONDITIONS)
        )

    def test_p0_4_v2_rejects_missing_extra_false_or_wrong_schema(self) -> None:
        cases = ("missing", "extra", "false", "schema")
        for case in cases:
            with self.subTest(case=case):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"qualification-{case}"
                )
                path = self.fixture.p0_4_psi8 / "summary.json"
                summary = json.loads(path.read_text(encoding="utf-8"))
                qualification = summary["qualification"]
                if case == "missing":
                    qualification["conditions"].pop("context_4096")
                elif case == "extra":
                    qualification["conditions"]["unreviewed"] = True
                elif case == "false":
                    qualification["conditions"]["cuda_device"] = False
                else:
                    qualification["schema_version"] = "unexpected"
                write_json(path, summary)
                with self.assertRaises(REVIEW.ReviewError):
                    self.lane()

    def test_p0_4_summary_event_and_path_cross_bindings_reject_tampering(self) -> None:
        cases = (
            "run-complete",
            "run-start-settings",
            "output-dir",
            "summary-checkpoint",
            "event-checkpoint",
        )
        for case in cases:
            with self.subTest(case=case):
                self.fixture = EvidenceFixture(Path(self.temporary.name) / case)
                summary_path = self.fixture.p0_4_psi8 / "summary.json"
                metrics_path = self.fixture.p0_4_psi8 / "metrics.jsonl"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                events = [
                    json.loads(line)
                    for line in metrics_path.read_text(encoding="utf-8").splitlines()
                ]
                if case == "run-complete":
                    events[-1]["qualification"]["conditions"]["cuda_device"] = False
                elif case == "run-start-settings":
                    events[0]["settings"]["batch_size"] = 2
                elif case == "output-dir":
                    summary["settings"]["output_dir"] = os.fspath(self.fixture.root)
                    events[0]["settings"] = copy.deepcopy(summary["settings"])
                elif case == "summary-checkpoint":
                    summary["checks"]["save_reload"]["checkpoint_dir"] = os.fspath(
                        self.fixture.root
                    )
                else:
                    events[53]["checkpoint_dir"] = os.fspath(self.fixture.root)
                write_json(summary_path, summary)
                write_jsonl(metrics_path, events)
                with self.assertRaises(REVIEW.ReviewError):
                    self.lane()

    def test_p0_4_rejects_microbatch_checkpointing_and_reentrant_witness(self) -> None:
        cases = (
            ("microbatch", lambda value: value["training"].__setitem__("microbatch_size", 2)),
            ("settings-gc", lambda value: value["settings"].__setitem__("gradient_checkpointing", False)),
            ("model-gc", lambda value: value["model"].__setitem__("gradient_checkpointing", False)),
            ("reentrant", lambda value: value["model"].__setitem__("gradient_checkpointing_kwargs", {"use_reentrant": True})),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                self.fixture = EvidenceFixture(Path(self.temporary.name) / name)
                path = self.fixture.p0_4_psi8 / "summary.json"
                summary = json.loads(path.read_text(encoding="utf-8"))
                mutate(summary)
                write_json(path, summary)
                with self.assertRaises(REVIEW.ReviewError):
                    self.lane()

    def test_artifact_hash_and_raw_event_tampering_are_rejected(self) -> None:
        stored = json.loads(self.fixture.p0_4_psi8_review.read_text(encoding="utf-8"))
        label = next(iter(stored["aggregate"]["artifact_hashes"]))
        stored["aggregate"]["artifact_hashes"][label] = "0" * 64
        write_json(self.fixture.p0_4_psi8_review, stored)
        self.assert_rejected("focused artifact changed|invalid digest")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "artifact")
        summary_path = self.fixture.p0_4_psi8 / "summary.json"
        metrics_path = self.fixture.p0_4_psi8 / "metrics.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        summary["checks"]["generation"]["sample_text"] = "valid but changed"
        events[55]["sample_text"] = "valid but changed"
        write_json(summary_path, summary)
        write_jsonl(metrics_path, events)
        self.assert_rejected("focused artifact changed|focused report projections differ")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "event")
        metrics_path = self.fixture.p0_4_psi8 / "metrics.jsonl"
        events = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        events[2], events[3] = events[3], events[2]
        write_jsonl(metrics_path, events)
        self.assert_lane_rejected("57-event sequence|optimizer_step")

    def test_offline_scope_requires_real_json_booleans(self) -> None:
        report = self.fixture._offline_cache_log()
        report["scope"]["fresh_p0_3"] = 1
        self.fixture.replace_log(
            "offline-cache-preflight",
            REVIEW._legacy._runner_canonical_bytes(report),
        )
        self.assert_rejected("boolean|scope")

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "offline-bool")
        report = self.fixture._offline_cache_log()
        report["cache"]["explicit"] = 1
        self.fixture.replace_log(
            "offline-cache-preflight",
            REVIEW._legacy._runner_canonical_bytes(report),
        )
        self.assert_rejected("boolean|cache")

    def test_initial_and_final_hygiene_semantics_must_match(self) -> None:
        report = self.fixture._repository_log("repository-hygiene-final")
        report["result"]["privacy"]["artifact_manifest_sha256"] = "c" * 64
        self.fixture.replace_log(
            "repository-hygiene-final",
            REVIEW._legacy._runner_canonical_bytes(report),
        )
        self.assert_rejected("initial.*final|hygiene.*identity")

    def test_stage_e_adds_hub_progress_suppression_without_changing_legacy(self) -> None:
        legacy_environment = (
            "PATH=/usr/bin:/bin",
            "LANG=C.UTF-8",
            "LC_ALL=C.UTF-8",
            "TZ=UTC",
            "HF_DATASETS_DISABLE_PROGRESS_BARS=1",
            "HF_DATASETS_OFFLINE=1",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "HF_HUB_OFFLINE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONHASHSEED=0",
            "PYTHONNOUSERSITE=1",
            "PYTHONOPTIMIZE=0",
            "PYTHONUNBUFFERED=1",
            "PYTHONUTF8=1",
            "TOKENIZERS_PARALLELISM=false",
            "TRANSFORMERS_OFFLINE=1",
        )
        stage_e_environment = (
            "PATH=/usr/bin:/bin",
            "LANG=C.UTF-8",
            "LC_ALL=C.UTF-8",
            "TZ=UTC",
            "HF_DATASETS_DISABLE_PROGRESS_BARS=1",
            "HF_DATASETS_OFFLINE=1",
            "HF_HUB_DISABLE_PROGRESS_BARS=1",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "HF_HUB_OFFLINE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONHASHSEED=0",
            "PYTHONNOUSERSITE=1",
            "PYTHONOPTIMIZE=0",
            "PYTHONUNBUFFERED=1",
            "PYTHONUTF8=1",
            "TOKENIZERS_PARALLELISM=false",
            "TRANSFORMERS_OFFLINE=1",
        )
        self.assertEqual(REVIEW._legacy.HERMETIC_FIXED_ENVIRONMENT, legacy_environment)
        self.assertEqual(
            REVIEW.STAGE_E_HERMETIC_FIXED_ENVIRONMENT,
            stage_e_environment,
        )

        progress_assignment = "HF_HUB_DISABLE_PROGRESS_BARS=1"
        for name in REVIEW.REQUIRED_COMMAND_NAMES:
            with self.subTest(name=name):
                environment = REVIEW._expected_environment(name, self.fixture.root)
                self.assertEqual(environment.count(progress_assignment), 1)
                self.assertEqual(
                    environment[1 : 1 + len(stage_e_environment)],
                    stage_e_environment,
                )

    def test_ledger_rejects_missing_or_duplicate_hub_progress_suppression(self) -> None:
        progress_assignment = "HF_HUB_DISABLE_PROGRESS_BARS=1"
        for label in ("missing", "duplicate"):
            with self.subTest(label=label):
                self.fixture = EvidenceFixture(
                    Path(self.temporary.name) / f"hub-progress-{label}"
                )
                records = self.fixture.command_records()
                record = next(
                    item
                    for item in records
                    if item["name"] == "hf-output-head-tf5141"
                )
                argv = record["argv"]
                index = argv.index(progress_assignment)
                if label == "missing":
                    argv.pop(index)
                else:
                    argv.insert(index + 1, progress_assignment)
                self._rewrite_records(records)
                with self.assertRaisesRegex(
                    REVIEW.ReviewError,
                    "non-hermetic or incorrectly classified environment",
                ):
                    self._ledger_only()

    def test_interpreters_must_be_absolute_and_distinct(self) -> None:
        records = self.fixture.command_records()
        for record in records:
            argv = record["argv"]
            index = REVIEW._executable_index(
                argv, name=record["name"], run_root=self.fixture.root
            )
            if argv[index] == self.fixture.tf4576_python:
                argv[index] = "python"
        self._rewrite_records(records)
        with self.assertRaisesRegex(REVIEW.ReviewError, "absolute"):
            self._ledger_only()

        self.fixture = EvidenceFixture(Path(self.temporary.name) / "same-python")
        records = self.fixture.command_records()
        for record in records:
            argv = record["argv"]
            index = REVIEW._executable_index(
                argv, name=record["name"], run_root=self.fixture.root
            )
            if argv[index] == self.fixture.tf5141_python:
                argv[index] = self.fixture.tf4576_python
        self._rewrite_records(records)
        with self.assertRaisesRegex(REVIEW.ReviewError, "distinct"):
            self._ledger_only()

    def _git(self, repository: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    def _history_repository(self, name: str) -> tuple[Path, str, str]:
        repository = Path(self.temporary.name) / name
        repository.mkdir()
        self._git(repository, "init", "-q")
        self._git(repository, "config", "user.name", "Stage E Fixture")
        self._git(repository, "config", "user.email", "stage-e@example.invalid")
        for relative in REVIEW.HISTORICAL_PATHS:
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{relative}\n", encoding="utf-8")
        self._git(repository, "add", "--all")
        self._git(repository, "commit", "-q", "-m", "baseline")
        baseline = self._git(repository, "rev-parse", "HEAD")
        (repository / "unrelated.txt").write_text("unchanged history\n", encoding="utf-8")
        self._git(repository, "add", "unrelated.txt")
        self._git(repository, "commit", "-q", "-m", "descendant")
        tested = self._git(repository, "rev-parse", "HEAD")
        return repository, baseline, tested

    def test_historical_fourteen_path_mode_blob_and_ancestry_contract(self) -> None:
        repository, baseline, tested = self._history_repository("history-pass")
        report = REVIEW._review_historical_evidence_immutability(
            repository=repository,
            tested_commit=tested,
            implementation_baseline=baseline,
        )
        self.assertEqual(report["artifact_count"], 14)
        with self.assertRaisesRegex(REVIEW.ReviewError, "does not descend"):
            REVIEW._review_historical_evidence_immutability(
                repository=repository,
                tested_commit=baseline,
                implementation_baseline=tested,
            )

        repository, baseline, _tested = self._history_repository("history-blob")
        changed = repository / REVIEW.HISTORICAL_PATHS[0]
        changed.write_text("changed blob\n", encoding="utf-8")
        self._git(repository, "add", os.fspath(changed.relative_to(repository)))
        self._git(repository, "commit", "-q", "-m", "change historical blob")
        tested = self._git(repository, "rev-parse", "HEAD")
        with self.assertRaisesRegex(REVIEW.ReviewError, "historical evidence"):
            REVIEW._review_historical_evidence_immutability(
                repository=repository,
                tested_commit=tested,
                implementation_baseline=baseline,
            )

        repository, baseline, _tested = self._history_repository("history-mode")
        relative = REVIEW.HISTORICAL_PATHS[0]
        self._git(repository, "update-index", "--chmod=+x", relative)
        self._git(repository, "commit", "-q", "-m", "change historical mode")
        tested = self._git(repository, "rev-parse", "HEAD")
        with self.assertRaisesRegex(REVIEW.ReviewError, "unsupported object|historical evidence"):
            REVIEW._review_historical_evidence_immutability(
                repository=repository,
                tested_commit=tested,
                implementation_baseline=baseline,
            )

        repository, baseline, _tested = self._history_repository("history-extra")
        unexpected = (
            repository
            / "docs/validation_results/LEVEL1_CORE_UNEXPECTED.json"
        )
        unexpected.write_text("{}\n", encoding="utf-8")
        self._git(repository, "add", os.fspath(unexpected.relative_to(repository)))
        self._git(repository, "commit", "-q", "-m", "add unexpected history")
        tested = self._git(repository, "rev-parse", "HEAD")
        with self.assertRaisesRegex(
            REVIEW.ReviewError,
            "escaped its fixed pathspec|incomplete or ambiguous",
        ):
            REVIEW._review_historical_evidence_immutability(
                repository=repository,
                tested_commit=tested,
                implementation_baseline=baseline,
            )

    def test_live_reviewer_pins_stage_e_and_legacy_reviewer_blobs(self) -> None:
        repository = Path(REVIEW.__file__).resolve().parents[1]
        tested_commit = "d" * 40
        expected_branch = "validation/hf-contract-hardening-requalification"
        calls: list[tuple[str, ...]] = []

        def capture(
            _repository: Path,
            arguments: tuple[str, ...],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[bytes]:
            calls.append(arguments)
            if arguments == ("rev-parse", "--show-toplevel"):
                raw = os.fsencode(repository) + b"\n"
            elif arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
                raw = tested_commit.encode("ascii") + b"\n"
            elif arguments == ("symbolic-ref", "--quiet", "--short", "HEAD"):
                raw = expected_branch.encode("ascii") + b"\n"
            elif arguments[0] == "status":
                raw = b""
            elif arguments[:2] == ("ls-files", "--error-unmatch"):
                raw = arguments[-1].encode("ascii") + b"\n"
            elif arguments[:2] == ("cat-file", "blob"):
                relative = arguments[2].split(":", 1)[1]
                raw = (repository / relative).read_bytes()
            elif (
                arguments[0] == "rev-parse"
                and len(arguments) == 2
                and arguments[1].endswith(
                    ":scripts/review_level1_requalification.py"
                )
            ):
                raw = REVIEW.LEGACY_REVIEWER_GIT_BLOB.encode("ascii") + b"\n"
            else:  # pragma: no cover - new Git operation must be explicit
                raise AssertionError(arguments)
            return subprocess.CompletedProcess(arguments, 0, stdout=raw, stderr=b"")

        with mock.patch.object(REVIEW._legacy, "_git_capture", side_effect=capture):
            REVIEW._verify_live_reviewer_checkout(tested_commit)
        self.assertIn(
            ("symbolic-ref", "--quiet", "--short", "HEAD"),
            calls,
        )
        pinned = {args[2] for args in calls if args[:2] == ("cat-file", "blob")}
        self.assertEqual(
            pinned,
            {
                f"{tested_commit}:scripts/review_hf_contract_hardening.py",
            },
        )
        legacy_pins = {
            args[1]
            for args in calls
            if args[0] == "rev-parse"
            and len(args) == 2
            and args[1].endswith(":scripts/review_level1_requalification.py")
        }
        self.assertEqual(
            legacy_pins,
            {
                f"{REVIEW.IMPLEMENTATION_BASELINE}:scripts/review_level1_requalification.py",
                f"{tested_commit}:scripts/review_level1_requalification.py",
            },
        )

        def stale_capture(
            _repository: Path,
            arguments: tuple[str, ...],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[bytes]:
            result = capture(_repository, arguments, **kwargs)
            if arguments == (
                "rev-parse",
                f"{tested_commit}:scripts/review_level1_requalification.py",
            ):
                result.stdout = b"0" * 40 + b"\n"
            return result

        with mock.patch.object(REVIEW._legacy, "_git_capture", side_effect=stale_capture):
            with self.assertRaisesRegex(REVIEW.ReviewError, "legacy|Level 1|source"):
                REVIEW._verify_live_reviewer_checkout(tested_commit)

        def wrong_branch_capture(
            _repository: Path,
            arguments: tuple[str, ...],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[bytes]:
            result = capture(_repository, arguments, **kwargs)
            if arguments == ("symbolic-ref", "--quiet", "--short", "HEAD"):
                result.stdout = b"validation/wrong-stage-e-branch\n"
            return result

        with mock.patch.object(
            REVIEW._legacy,
            "_git_capture",
            side_effect=wrong_branch_capture,
        ):
            with self.assertRaisesRegex(REVIEW.ReviewError, "branch"):
                REVIEW._verify_live_reviewer_checkout(tested_commit)

        def detached_capture(
            _repository: Path,
            arguments: tuple[str, ...],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[bytes]:
            result = capture(_repository, arguments, **kwargs)
            if arguments == ("symbolic-ref", "--quiet", "--short", "HEAD"):
                result.returncode = 1
                result.stdout = b""
            return result

        with mock.patch.object(
            REVIEW._legacy,
            "_git_capture",
            side_effect=detached_capture,
        ):
            with self.assertRaisesRegex(REVIEW.ReviewError, "branch|detached"):
                REVIEW._verify_live_reviewer_checkout(tested_commit)

    def test_safe_path_cli_and_relative_inputs_fail_closed(self) -> None:
        kwargs = self.fixture.kwargs()
        kwargs["p0_3_root"] = "artifacts/p0-3"
        with self.assertRaisesRegex(REVIEW.ReviewError, "absolute"):
            with mock.patch.object(
                REVIEW,
                "_review_historical_evidence_immutability",
                return_value=self._history(),
            ):
                REVIEW.review_inputs(**kwargs)

        environment = {
            "HOME": os.fspath(Path(self.temporary.name) / "cli-home"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        help_result = subprocess.run(
            [
                sys.executable,
                "-P",
                "-S",
                "-B",
                os.fspath(REPO_ROOT / "scripts/review_hf_contract_hardening.py"),
                "--help",
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--mode", help_result.stdout)

        relative = subprocess.run(
            [
                sys.executable,
                "-P",
                "-S",
                "-B",
                os.fspath(REPO_ROOT / "scripts/review_hf_contract_hardening.py"),
                "--tested-commit",
                "d" * 40,
                "--command-ledger",
                "commands.jsonl",
                "--output",
                "review.json",
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(relative.returncode, 1)
        self.assertIn("absolute", relative.stderr)


if __name__ == "__main__":
    unittest.main()
