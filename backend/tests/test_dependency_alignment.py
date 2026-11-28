"""Verify single-source dependency alignment across pyproject.toml, lockfile, Dockerfiles, and CI."""

from __future__ import annotations

import re
import tomllib
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"


def _pkg_name(spec: str) -> str:
    base = spec.split(";", 1)[0].strip()
    return re.split(r"[<>=!~;\[]", base, maxsplit=1)[0].strip().lower().replace("_", "-")


def test_pyproject_dependencies_unique_and_canonical():
    pyproject = tomllib.loads((BACKEND_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]

    dep_names = [_pkg_name(d) for d in deps]
    assert len(dep_names) == len(set(dep_names)), "Duplicate packages in [project].dependencies"
    assert "prometheus-client" in dep_names
    assert "pydantic" in dep_names

    dev_names = [_pkg_name(d) for d in dev_deps]
    assert len(dev_names) == len(set(dev_names)), "Duplicate packages in dev optional dependencies"
    assert {"pytest", "pytest-asyncio", "pytest-cov", "pip-audit", "ruff"} <= set(dev_names)


def test_lockfile_pins_all_runtime_deps_and_requirements_txt_aligns():
    pyproject = tomllib.loads((BACKEND_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_names = {_pkg_name(d) for d in pyproject["project"]["dependencies"]}

    lock_lines = [
        ln.strip()
        for ln in (BACKEND_DIR / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    seen_by_pkg: dict[str, list[str]] = defaultdict(list)
    for ln in lock_lines:
        assert "==" in ln.split(";", 1)[0], f"Lockfile entry must be strictly pinned (==): {ln}"
        seen_by_pkg[_pkg_name(ln)].append(ln)

    assert runtime_names <= set(seen_by_pkg.keys())

    # Duplicate package entries are only allowed when disambiguated by environment markers
    for pkg, entries in seen_by_pkg.items():
        if len(entries) > 1:
            assert all(";" in e for e in entries), f"Unmarked duplicate entry in requirements.lock: {pkg}"

    req_lines = [
        ln.strip()
        for ln in (BACKEND_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert req_lines[0] == "-r requirements.lock"
    req_dev_names = [_pkg_name(ln) for ln in req_lines[1:]]
    assert len(req_dev_names) == len(set(req_dev_names)), "Duplicate lines in requirements.txt"
    expected_dev = [_pkg_name(d) for d in pyproject["project"]["optional-dependencies"]["dev"]]
    assert req_dev_names == expected_dev


def test_dockerfiles_and_ci_install_from_lockfile():
    backend_df = (BACKEND_DIR / "Dockerfile").read_text(encoding="utf-8")
    target_df = (REPO_ROOT / "Dockerfile.target").read_text(encoding="utf-8")
    ci_yml = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    nightly_yml = (REPO_ROOT / ".github" / "workflows" / "nightly-redteam.yml").read_text(encoding="utf-8")

    assert "requirements.lock" in backend_df
    assert "requirements.lock" in target_df
    assert "requirements.lock" in ci_yml
    assert "requirements.txt" in nightly_yml
