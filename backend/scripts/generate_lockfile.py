#!/usr/bin/env python3
"""
Generate a canonical, cross-platform `backend/requirements.lock` from `backend/pyproject.toml`.

Resolves runtime dependencies across Python 3.11 and 3.12 on both Linux (CI/Docker)
and Windows (local dev), emitting clean PEP 508 environment markers wherever versions
legitimately diverge (e.g. NumPy on Python 3.11 vs 3.12).

Also rewrites `backend/requirements.txt` as `-r requirements.lock` plus dev tools so
`backend/pyproject.toml` remains the single source of truth.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = BACKEND_DIR / "pyproject.toml"
LOCKFILE_PATH = BACKEND_DIR / "requirements.lock"
REQUIREMENTS_TXT_PATH = BACKEND_DIR / "requirements.txt"

TargetSpec = Tuple[str, List[str]]


def _resolve_target(deps: List[str], extra_pip_args: List[str]) -> Dict[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)
    try:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--report",
            str(report_path),
            *extra_pip_args,
            *deps,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"pip dependency resolution failed for {extra_pip_args}:\n{proc.stderr}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            item["metadata"]["name"].lower(): f"{item['metadata']['name']}=={item['metadata']['version']}"
            for item in report.get("install", [])
        }
    finally:
        report_path.unlink(missing_ok=True)


def _format_marker(platforms: set[str]) -> str:
    """Collapse resolved target matrix cells into concise PEP 508 expressions."""
    if platforms == {"linux_311", "linux_312", "win_311", "win_312"}:
        return ""
    if platforms == {"linux_311", "win_311"}:
        return 'python_version < "3.12"'
    if platforms == {"linux_312", "win_312"}:
        return 'python_version >= "3.12"'
    if platforms == {"linux_311", "linux_312"}:
        return 'sys_platform == "linux"'
    if platforms == {"win_311", "win_312"}:
        return 'sys_platform == "win32"'

    clauses = []
    if {"linux_311", "linux_312"} <= platforms:
        clauses.append('sys_platform == "linux"')
        platforms = platforms - {"linux_311", "linux_312"}
    if {"win_311", "win_312"} <= platforms:
        clauses.append('sys_platform == "win32"')
        platforms = platforms - {"win_311", "win_312"}
    for cell in sorted(platforms):
        os_name, py_ver = cell.split("_", 1)
        sys_plat = "linux" if os_name == "linux" else "win32"
        py_cmp = '< "3.12"' if py_ver == "311" else '>= "3.12"'
        clauses.append(f'(sys_platform == "{sys_plat}" and python_version {py_cmp})')
    return " or ".join(clauses)


def main() -> int:
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    deps: List[str] = list(pyproject["project"]["dependencies"])
    dev_deps: List[str] = list(
        pyproject.get("project", {}).get("optional-dependencies", {}).get("dev", [])
    )

    targets: List[TargetSpec] = [
        (
            "linux_311",
            [
                "--python-version", "3.11",
                "--platform", "manylinux_2_28_x86_64",
                "--platform", "manylinux_2_17_x86_64",
                "--platform", "manylinux2014_x86_64",
                "--platform", "linux_x86_64",
                "--abi", "cp311", "--abi", "abi3", "--abi", "none",
                "--only-binary=:all:",
            ],
        ),
        (
            "linux_312",
            [
                "--python-version", "3.12",
                "--platform", "manylinux_2_28_x86_64",
                "--platform", "manylinux_2_17_x86_64",
                "--platform", "manylinux2014_x86_64",
                "--platform", "linux_x86_64",
                "--abi", "cp312", "--abi", "abi3", "--abi", "none",
                "--only-binary=:all:",
            ],
        ),
        (
            "win_311",
            [
                "--python-version", "3.11",
                "--platform", "win_amd64",
                "--abi", "cp311", "--abi", "abi3", "--abi", "none",
                "--only-binary=:all:",
            ],
        ),
        (
            "win_312",
            [
                "--python-version", "3.12",
                "--platform", "win_amd64",
                "--abi", "cp312", "--abi", "abi3", "--abi", "none",
                "--only-binary=:all:",
            ],
        ),
    ]

    resolved: Dict[str, Dict[str, str]] = {}
    all_keys = set()
    for name, flags in targets:
        mapping = _resolve_target(deps, flags)
        resolved[name] = mapping
        all_keys.update(mapping.keys())

    lines = [
        "# ==============================================================================",
        "# Auto-generated canonical runtime lockfile from backend/pyproject.toml",
        "# Regenerate via: python backend/scripts/generate_lockfile.py",
        "# Supports Linux and Windows across Python 3.11 and Python 3.12.",
        "# ==============================================================================",
    ]

    for key in sorted(all_keys):
        by_spec: Dict[str, set[str]] = defaultdict(set)
        for target_name, _ in targets:
            spec = resolved[target_name].get(key)
            if spec:
                by_spec[spec].add(target_name)
        for spec, plat_set in sorted(by_spec.items()):
            marker = _format_marker(plat_set)
            lines.append(f"{spec} ; {marker}" if marker else spec)

    LOCKFILE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    req_lines = [
        "# Canonical test/dev requirements backed by backend/pyproject.toml + requirements.lock.",
        "-r requirements.lock",
        *dev_deps,
    ]
    REQUIREMENTS_TXT_PATH.write_text("\n".join(req_lines) + "\n", encoding="utf-8")

    print(f"Wrote {LOCKFILE_PATH} ({len(lines) - 5} locked entries)")
    print(f"Wrote {REQUIREMENTS_TXT_PATH} (-r requirements.lock + {len(dev_deps)} dev deps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
