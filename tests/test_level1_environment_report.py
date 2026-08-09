from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "report_level1_environment.py"
SPEC = importlib.util.spec_from_file_location("report_level1_environment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def exact_versions(lane: str) -> dict[str, str | None]:
    values = {name: None for name in MODULE.PACKAGE_NAMES}
    values.update(MODULE.EXPECTED_PACKAGES[lane])
    return values


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def is_bf16_supported(self) -> bool:
        return True

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(name="Synthetic GPU", total_memory=96 * 1024**3)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (12, 0)

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (95 * 1024**3, 96 * 1024**3)

    def memory_allocated(self, index: int) -> int:
        assert index == 0
        return 0

    memory_reserved = memory_allocated


def nvidia_smi_fixture() -> dict[str, object]:
    return {
        "compute_capability": "12.0",
        "device_name": "Synthetic GPU",
        "driver_version": "595.71.05",
        "memory_free_mib": 95_000,
        "memory_total_mib": 97_887,
        "other_compute_process_count": 1,
        "other_compute_used_memory_mib": 64,
        "physical_index": 0,
        "reporter_compute_process_present": True,
        "reporter_used_memory_mib": 550,
    }


class Level1EnvironmentReportTests(unittest.TestCase):
    def test_exact_lane_report_is_canonical_and_path_free(self) -> None:
        versions = exact_versions("tf4576")
        torch = SimpleNamespace(__version__=versions["torch"])
        transformers = SimpleNamespace(__version__=versions["transformers"])
        with (
            mock.patch.object(MODULE, "installed_versions", return_value=versions),
            mock.patch.object(MODULE, "_runtime_versions", return_value=(torch, transformers)),
            mock.patch.object(MODULE, "_nvidia_smi_snapshot", return_value=nvidia_smi_fixture()),
            mock.patch.object(MODULE.platform, "python_implementation", return_value="CPython"),
            mock.patch.object(MODULE.platform, "python_version", return_value="3.12.11"),
        ):
            report = MODULE.compatibility_report("tf4576")
        raw = MODULE.canonical_json(report)
        self.assertEqual(raw, MODULE.canonical_json(json.loads(raw)))
        self.assertNotIn("/home/", raw)
        self.assertNotIn("/tmp/", raw)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["python"]["optimization_level"], 0)
        self.assertIs(report["python"]["assertions_enabled"], True)

    def test_python_optimization_fails_closed(self) -> None:
        with (
            mock.patch.object(MODULE.sys, "flags", SimpleNamespace(optimize=1)),
            self.assertRaisesRegex(MODULE.EnvironmentReportError, "optimization level 0"),
        ):
            MODULE.python_identity()

    def test_version_drift_fails_closed(self) -> None:
        versions = exact_versions("tf5141")
        versions["transformers"] = "5.14.2"
        with self.assertRaisesRegex(MODULE.EnvironmentReportError, "exact lane"):
            MODULE.assert_expected_versions("tf5141", versions)

    def test_cuda_report_requires_exact_selection_and_capabilities(self) -> None:
        versions = exact_versions("tf4576")
        cuda = FakeCuda()
        torch = SimpleNamespace(
            __version__=versions["torch"],
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90701)),
            cuda=cuda,
            version=SimpleNamespace(cuda="12.8"),
        )
        transformers = SimpleNamespace(__version__="4.57.6")
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
            mock.patch.object(MODULE, "installed_versions", return_value=versions),
            mock.patch.object(MODULE, "_runtime_versions", return_value=(torch, transformers)),
            mock.patch.object(MODULE, "_nvidia_smi_snapshot", return_value=nvidia_smi_fixture()),
        ):
            report = MODULE.cuda_report()
        self.assertEqual(report["selection"]["logical_device"], "cuda:0")
        self.assertTrue(report["cuda"]["bf16_supported"])
        self.assertEqual(report["cuda"]["total_memory_bytes"], 96 * 1024**3)
        self.assertEqual(report["runtime"]["torch"], "2.7.1+cu128")
        self.assertEqual(report["packages"], versions)
        self.assertEqual(report["python"]["optimization_level"], 0)
        self.assertIs(report["python"]["assertions_enabled"], True)
        self.assertEqual(report["nvidia_smi"]["driver_version"], "595.71.05")

        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
            self.assertRaisesRegex(MODULE.EnvironmentReportError, "requires CUDA_VISIBLE_DEVICES"),
        ):
            MODULE.cuda_report()

    def test_cuda_report_rejects_missing_bf16(self) -> None:
        versions = exact_versions("tf4576")
        cuda = FakeCuda()
        cuda.is_bf16_supported = lambda: False
        torch = SimpleNamespace(
            __version__=versions["torch"],
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90701)),
            cuda=cuda,
            version=SimpleNamespace(cuda="12.8"),
        )
        transformers = SimpleNamespace(__version__="4.57.6")
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
            mock.patch.object(MODULE, "installed_versions", return_value=versions),
            mock.patch.object(MODULE, "_runtime_versions", return_value=(torch, transformers)),
            mock.patch.object(MODULE, "_nvidia_smi_snapshot", return_value=nvidia_smi_fixture()),
            self.assertRaisesRegex(MODULE.EnvironmentReportError, "does not support bf16"),
        ):
            MODULE.cuda_report()

    def test_nvidia_smi_snapshot_is_path_free_and_selected_gpu_only(self) -> None:
        rows = [
            [["GPU-selected", "0", "Synthetic GPU", "595.71.05", "97887", "95000", "12.0"]],
            [
                ["GPU-other", "9", "/private/other-python", "2048"],
                ["GPU-selected", str(os.getpid()), "/private/bin/worker", "64"],
            ],
        ]
        with mock.patch.object(MODULE, "_run_nvidia_smi", side_effect=rows):
            report = MODULE._nvidia_smi_snapshot()
        self.assertEqual(report["physical_index"], 0)
        self.assertTrue(report["reporter_compute_process_present"])
        self.assertEqual(report["reporter_used_memory_mib"], 64)
        self.assertEqual(report["other_compute_process_count"], 0)
        self.assertEqual(report["other_compute_used_memory_mib"], 0)
        self.assertNotIn("/private/", MODULE.canonical_json(report))

    def test_tf5141_rejects_unexpected_relevant_package_install(self) -> None:
        versions = exact_versions("tf5141")
        versions["datasets"] = "5.0.1"
        with self.assertRaisesRegex(MODULE.EnvironmentReportError, "exact lane"):
            MODULE.assert_expected_versions("tf5141", versions)

    def test_main_reports_failure_without_json(self) -> None:
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            mock.patch.object(
                MODULE,
                "compatibility_report",
                side_effect=MODULE.EnvironmentReportError("synthetic drift"),
            ),
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(stdout),
        ):
            code = MODULE.main(["--lane", "tf4576"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("synthetic drift", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
