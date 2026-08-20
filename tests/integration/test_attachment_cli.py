"""End-to-end attachment CLI behavior."""

from __future__ import annotations

import binascii
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

from unmark.core.errors import ExitCode


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + kind + payload + crc.to_bytes(4, "big")


def _png() -> bytes:
    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"tEXt", b"Software\x00Claude")
        + _chunk(b"IDAT", zlib.compress(b"\x00\x01\x02\x03"))
        + _chunk(b"IEND", b"")
    )


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    source_root = str(Path(__file__).parents[2] / "src")
    environment["PYTHONPATH"] = source_root
    return subprocess.run(
        [sys.executable, "-m", "unmark.cli.app", *args],
        cwd=cwd,
        capture_output=True,
        check=False,
        env=environment,
    )


def test_inspect_routes_on_bytes_not_misleading_suffix(tmp_path: Path) -> None:
    source = tmp_path / "download.txt"
    source.write_bytes(_png())
    result = _run("attachment", "inspect", str(source), "--format", "json", cwd=tmp_path)
    assert result.returncode == ExitCode.SUCCESS
    payload = json.loads(result.stdout)
    assert payload["source"]["media_type"] == "image/png"
    assert payload["state"] == "unsigned_ai_metadata"


def test_clean_writes_verified_sibling_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "download.png"
    original = _png()
    source.write_bytes(original)
    result = _run("attachment", "clean", str(source), "--format", "json", cwd=tmp_path)
    assert result.returncode == ExitCode.SUCCESS
    output = tmp_path / "download.unmark.png"
    assert output.exists()
    assert b"Claude" not in output.read_bytes()
    assert source.read_bytes() == original
    assert json.loads(result.stdout)["state"] == "removed_verified"


def test_clean_failure_publishes_no_output(tmp_path: Path) -> None:
    source = tmp_path / "mixed.png"
    data = _png().replace(b"Software\x00Claude", b"Copyright\x00Claude")
    # Replacement changes the chunk length/CRC, so create the valid mixed file directly.
    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"tEXt", b"Copyright\x00Licensed output made with Claude")
        + _chunk(b"IDAT", zlib.compress(b"\x00\x01\x02\x03"))
        + _chunk(b"IEND", b"")
    )
    source.write_bytes(data)
    result = _run("attachment", "clean", str(source), cwd=tmp_path)
    assert result.returncode == ExitCode.VALIDATION_OR_WRITE_FAILED
    assert not (tmp_path / "mixed.unmark.png").exists()
