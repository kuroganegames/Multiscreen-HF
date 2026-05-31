"""Helpers for torch.compile setup, especially on Windows/MSVC."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path


def find_msvc_cl() -> str | None:
    """Find MSVC ``cl.exe`` for Triton/torch.compile on Windows."""

    if os.environ.get("CC"):
        return os.environ["CC"]

    bases = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio"),
    ]
    for base in bases:
        if not base.exists():
            continue
        for vs in sorted(base.iterdir(), reverse=True):
            pattern = str(
                vs / "BuildTools" / "VC" / "Tools" / "MSVC" / "*" / "bin" / "Hostx64" / "x64" / "cl.exe"
            )
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]
    return None


def _find_vcvarsall() -> Path | None:
    bases = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio"),
    ]
    for base in bases:
        if not base.exists():
            continue
        for vs in sorted(base.iterdir(), reverse=True):
            candidate = vs / "BuildTools" / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
            if candidate.exists():
                return candidate
    return None


def load_vcvars_env() -> bool:
    """Load the full MSVC build environment into ``os.environ`` on Windows."""

    if sys.platform != "win32":
        return False
    if os.environ.get("VSCMD_VER"):
        return True

    vcvarsall = _find_vcvarsall()
    if vcvarsall is None:
        return False

    try:
        result = subprocess.run(
            f'"{vcvarsall}" x64 >nul && set',
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False

    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value
    return True


def setup_compile_env() -> str | None:
    """Auto-detect MSVC and set ``CC`` for ``torch.compile`` when needed."""

    if sys.platform == "win32":
        load_vcvars_env()
        if os.environ.get("CC"):
            return os.environ["CC"]
        cl_path = find_msvc_cl()
        if cl_path:
            os.environ["CC"] = cl_path
        return cl_path
    return os.environ.get("CC")
