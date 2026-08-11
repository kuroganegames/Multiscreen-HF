#!/usr/bin/env python3
"""Emit a path-free, fail-closed Level 1 compatibility environment report."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "multiscreen-level1-environment-v1"
REPORTER_VERSION = "1.0.0"
PACKAGE_NAMES = (
    "PyYAML",
    "accelerate",
    "datasets",
    "huggingface-hub",
    "numpy",
    "pyarrow",
    "safetensors",
    "sentencepiece",
    "tokenizers",
    "torch",
    "transformers",
    "trl",
)
EXPECTED_PACKAGES: Mapping[str, Mapping[str, str | None]] = {
    "tf4576": {
        "PyYAML": "6.0.1",
        "accelerate": "1.6.0",
        "datasets": "5.0.1",
        "huggingface-hub": "0.34.3",
        "numpy": "1.26.4",
        "pyarrow": "25.0.0",
        "safetensors": "0.5.3",
        "sentencepiece": "0.2.0",
        "tokenizers": "0.22.0",
        "torch": "2.7.1+cu128",
        "transformers": "4.57.6",
        "trl": "1.9.2",
    },
    "tf5141": {
        "PyYAML": "6.0.2",
        "accelerate": None,
        "datasets": None,
        "huggingface-hub": "1.27.0",
        "numpy": "2.3.2",
        "pyarrow": None,
        "safetensors": "0.8.0",
        "sentencepiece": "0.2.0",
        "tokenizers": "0.22.2",
        "torch": "2.8.0",
        "transformers": "5.14.1",
        "trl": None,
    },
}
EXPECTED_TORCH_RUNTIME = {
    "tf4576": "2.7.1+cu128",
    "tf5141": "2.8.0+cu128",
}


class EnvironmentReportError(RuntimeError):
    """The selected compatibility lane is unavailable or has drifted."""


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def assert_expected_versions(
    lane: str,
    versions: Mapping[str, str | None],
) -> None:
    expected = EXPECTED_PACKAGES[lane]
    differences = {
        name: {"expected": wanted, "actual": versions.get(name)}
        for name, wanted in expected.items()
        if versions.get(name) != wanted
    }
    if differences:
        raise EnvironmentReportError(
            f"{lane} package versions do not match the exact lane: {differences}"
        )


def _runtime_versions() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except (ImportError, OSError) as exc:
        raise EnvironmentReportError("cannot import torch and transformers") from exc
    return torch, transformers


def python_identity() -> dict[str, Any]:
    optimization_level = int(sys.flags.optimize)
    if optimization_level != 0 or not __debug__:
        raise EnvironmentReportError(
            "Level 1 evidence requires Python assertions with optimization level 0"
        )
    return {
        "assertions_enabled": True,
        "implementation": platform.python_implementation(),
        "optimization_level": 0,
        "version": platform.python_version(),
    }


def compatibility_report(lane: str) -> dict[str, Any]:
    versions = installed_versions()
    assert_expected_versions(lane, versions)
    torch, transformers = _runtime_versions()
    if str(torch.__version__) != EXPECTED_TORCH_RUNTIME[lane]:
        raise EnvironmentReportError("torch runtime version does not match the exact lane")
    if str(transformers.__version__) != versions["transformers"]:
        raise EnvironmentReportError(
            "transformers runtime and distribution versions differ"
        )
    return {
        "lane": lane,
        "packages": versions,
        "python": python_identity(),
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "tool_version": REPORTER_VERSION,
    }


def _run_nvidia_smi(arguments: Sequence[str]) -> list[list[str]]:
    binary = "/usr/bin/nvidia-smi"
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        raise EnvironmentReportError("nvidia-smi is unavailable at the recorded system path")
    try:
        completed = subprocess.run(
            [binary, *arguments, "--format=csv,noheader,nounits"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise EnvironmentReportError("nvidia-smi query failed") from exc
    if completed.returncode != 0:
        raise EnvironmentReportError("nvidia-smi query returned a non-zero status")
    return [
        [field.strip() for field in row]
        for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True)
        if row
    ]


def _positive_int(value: str, *, field: str, allow_zero: bool = False) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise EnvironmentReportError(f"nvidia-smi {field} is not an integer") from exc
    if result < 0 or (result == 0 and not allow_zero):
        raise EnvironmentReportError(f"nvidia-smi {field} is outside its valid range")
    return result


def _nvidia_smi_snapshot() -> dict[str, Any]:
    gpu_rows = _run_nvidia_smi(
        [
            "--id=0",
            "--query-gpu=uuid,index,name,driver_version,memory.total,memory.free,compute_cap",
        ]
    )
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != 7:
        raise EnvironmentReportError("nvidia-smi selected-GPU query was ambiguous")
    gpu_uuid, index, name, driver, total, free, capability = gpu_rows[0]
    if index != "0" or not gpu_uuid or not name or not driver or not capability:
        raise EnvironmentReportError("nvidia-smi selected-GPU identity is incomplete")
    process_rows = _run_nvidia_smi(
        ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory"]
    )
    current_pid = os.getpid()
    other_process_count = 0
    other_used_memory_mib = 0
    reporter_process_present = False
    reporter_used_memory_mib = 0
    for row in process_rows:
        if len(row) != 4:
            raise EnvironmentReportError("nvidia-smi compute-process query was malformed")
        process_uuid, pid, process_name, used_memory = row
        if process_uuid != gpu_uuid:
            continue
        if not process_name:
            raise EnvironmentReportError("nvidia-smi compute process has no executable name")
        parsed_pid = _positive_int(pid, field="process pid")
        parsed_memory = _positive_int(
            used_memory, field="process memory", allow_zero=True
        )
        if parsed_pid == current_pid:
            reporter_process_present = True
            reporter_used_memory_mib += parsed_memory
        else:
            other_process_count += 1
            other_used_memory_mib += parsed_memory
    return {
        "compute_capability": capability,
        "device_name": name,
        "driver_version": driver,
        "memory_free_mib": _positive_int(free, field="free memory", allow_zero=True),
        "memory_total_mib": _positive_int(total, field="total memory"),
        "other_compute_process_count": other_process_count,
        "other_compute_used_memory_mib": other_used_memory_mib,
        "physical_index": 0,
        "reporter_compute_process_present": reporter_process_present,
        "reporter_used_memory_mib": reporter_used_memory_mib,
    }


def cuda_report() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise EnvironmentReportError(
            "environment-cuda0 requires CUDA_VISIBLE_DEVICES to equal exactly '0'"
        )
    versions = installed_versions()
    assert_expected_versions("tf4576", versions)
    torch, transformers = _runtime_versions()
    if str(torch.__version__) != EXPECTED_TORCH_RUNTIME["tf4576"]:
        raise EnvironmentReportError("environment-cuda0 requires the tf4576 torch runtime")
    if str(transformers.__version__) != versions["transformers"]:
        raise EnvironmentReportError("environment-cuda0 requires the tf4576 Transformers runtime")
    if not bool(torch.cuda.is_available()):
        raise EnvironmentReportError("CUDA is unavailable")
    if int(torch.cuda.device_count()) != 1:
        raise EnvironmentReportError("CUDA_VISIBLE_DEVICES=0 must expose one logical device")
    if not bool(torch.cuda.is_bf16_supported()):
        raise EnvironmentReportError("logical cuda:0 does not support bf16")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    total_memory = int(properties.total_memory)
    if total_memory <= 0:
        raise EnvironmentReportError("CUDA device reports non-positive total memory")
    try:
        free_memory, runtime_total_memory = (
            int(value) for value in torch.cuda.mem_get_info(0)
        )
        allocated_memory = int(torch.cuda.memory_allocated(0))
        reserved_memory = int(torch.cuda.memory_reserved(0))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise EnvironmentReportError("cannot record the CUDA memory snapshot") from exc
    if (
        runtime_total_memory != total_memory
        or not 0 <= free_memory <= total_memory
        or not 0 <= allocated_memory <= total_memory
        or not 0 <= reserved_memory <= total_memory
    ):
        raise EnvironmentReportError("CUDA memory snapshot is internally inconsistent")
    nvidia_smi = _nvidia_smi_snapshot()
    if (
        nvidia_smi["device_name"] != str(properties.name)
        or nvidia_smi["compute_capability"]
        != f"{int(capability[0])}.{int(capability[1])}"
    ):
        raise EnvironmentReportError("torch and nvidia-smi selected different devices")
    return {
        "cuda": {
            "allocated_memory_bytes": allocated_memory,
            "bf16_supported": True,
            "capability": [int(capability[0]), int(capability[1])],
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": 1,
            "device_name": str(properties.name),
            "free_memory_bytes": free_memory,
            "reserved_memory_bytes": reserved_memory,
            "runtime_version": str(torch.version.cuda),
            "total_memory_bytes": total_memory,
        },
        "lane": "cuda0",
        "nvidia_smi": nvidia_smi,
        "packages": versions,
        "python": python_identity(),
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "cuda_visible_devices": "0",
            "logical_device": "cuda:0",
        },
        "status": "passed",
        "tool_version": REPORTER_VERSION,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("tf4576", "tf5141", "cuda0"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = cuda_report() if args.lane == "cuda0" else compatibility_report(args.lane)
    except EnvironmentReportError as exc:
        print(f"environment report failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
