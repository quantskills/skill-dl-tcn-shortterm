from __future__ import annotations

import hashlib
from pathlib import Path

from skill_dl_tcn_shortterm.integrity import code_identity


def test_code_identity_falls_back_to_installed_package(tmp_path: Path) -> None:
    identity = code_identity(tmp_path)

    assert identity["origin"] == "installed-package"
    assert identity["source_sha256"] != hashlib.sha256(b"").hexdigest()
    assert len(str(identity["source_sha256"])) == 64
