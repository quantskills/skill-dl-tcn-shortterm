"""Stable code and file identities used by immutable experiment runs."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def code_identity(project_root: Path) -> dict[str, object]:
    """Fingerprint executable project sources and record their Git state."""

    digest = hashlib.sha256()
    source_root = project_root / "src"
    if source_root.is_dir():
        identity_root = project_root
        paths = sorted(source_root.rglob("*.py"))
        pyproject = project_root / "pyproject.toml"
        if pyproject.is_file():
            paths.append(pyproject)
        origin = "source-tree"
    else:
        package_root = Path(__file__).resolve().parent
        identity_root = package_root.parent
        paths = sorted(package_root.rglob("*.py"))
        paths.extend(sorted((package_root / "resources").rglob("*.json")))
        origin = "installed-package"
    for path in paths:
        relative = path.relative_to(identity_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    try:
        revision = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        dirty: bool | None = bool(dirty_output.strip())
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
        dirty = None
    return {
        "origin": origin,
        "revision": revision,
        "dirty": dirty,
        "source_sha256": digest.hexdigest(),
    }
