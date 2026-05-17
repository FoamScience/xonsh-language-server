"""Phase 3: typed-stub preamble injected for the Python backend.

Tests the generator that converts XONSH_MAGIC_VARS into a TypedDict, and the
proxy's per-document preamble that picks up user-defined env vars from
``$X = …`` assignments. Type-check round-trip through Pyright is gated on the
binary being present.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from xonsh_lsp.lsp_proxy_backend import _build_xonsh_preamble
from xonsh_lsp.preprocessing import preprocess_with_mapping
from xonsh_lsp.xonsh_types import build_xonsh_env_typed_dict_lines


def test_typed_dict_includes_registry_entries():
    lines = build_xonsh_env_typed_dict_lines()
    body = "\n".join(lines)
    assert "class __xonsh_env_dict__(__xonsh_typing__.TypedDict, total=False):" in body
    # Known entries from XONSH_MAGIC_VARS
    assert "AUTO_CD: bool" in body
    assert "XONSH_DEBUG: int" in body
    assert "HISTCONTROL: set[str]" in body  # normalized from "set"


def test_typed_dict_skips_dunder_names():
    # XONSH_MAGIC_VARS contains "__xonsh__" — it's a Python identifier, not an
    # env-var key. Don't emit it as a TypedDict field.
    lines = build_xonsh_env_typed_dict_lines()
    body = "\n".join(lines)
    assert "__xonsh__:" not in body


def test_typed_dict_appends_user_vars_as_any():
    lines = build_xonsh_env_typed_dict_lines(user_defined={"MY_VAR", "ANOTHER"})
    body = "\n".join(lines)
    assert "MY_VAR: __xonsh_typing__.Any" in body
    assert "ANOTHER: __xonsh_typing__.Any" in body


def test_typed_dict_does_not_duplicate_registry_vars():
    # AUTO_CD already in the registry as bool — passing it as user-defined must
    # not produce a second entry (would be a duplicate-key error in Pyright).
    lines = build_xonsh_env_typed_dict_lines(user_defined={"AUTO_CD"})
    body = "\n".join(lines)
    assert body.count("AUTO_CD:") == 1
    assert "AUTO_CD: bool" in body


def test_user_env_vars_discovered_from_assignments():
    source = '$MY_VAR = "hello"\n$ANOTHER = 1\n'
    result = preprocess_with_mapping(source)
    assert "MY_VAR" in result.user_env_vars
    assert "ANOTHER" in result.user_env_vars


def test_proxy_preamble_includes_user_vars():
    preamble, line_count = _build_xonsh_preamble({"MY_VAR"})
    assert "MY_VAR: __xonsh_typing__.Any" in preamble
    assert line_count == len(preamble.rstrip("\n").split("\n"))


def test_proxy_preamble_baseline_no_user_vars():
    preamble, line_count = _build_xonsh_preamble(set())
    assert "class __xonsh_env_dict__" in preamble
    assert "__xonsh_env__: __xonsh_env_dict__" in preamble
    assert line_count > 5


# ---------------------------------------------------------------------------
# End-to-end: feed the preamble + a tiny preprocessed snippet to Pyright and
# assert it reports the right type for __xonsh_env__["HOME"]. Skipped if
# pyright isn't installed (CI gates this separately).
# ---------------------------------------------------------------------------

PYRIGHT = shutil.which("pyright")


@pytest.mark.skipif(PYRIGHT is None, reason="pyright not installed")
def test_pyright_types_known_env_var_correctly(tmp_path: Path):
    preamble, _ = _build_xonsh_preamble(set())
    snippet = preamble + 'reveal_type(__xonsh_env__["AUTO_CD"])\n'
    src = tmp_path / "check.py"
    src.write_text(snippet)

    out = subprocess.run(
        [PYRIGHT, "--outputjson", str(src)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # `reveal_type` output appears in the JSON report regardless of exit code.
    assert "bool" in out.stdout, out.stdout


@pytest.mark.skipif(PYRIGHT is None, reason="pyright not installed")
def test_pyright_types_user_env_var_as_any(tmp_path: Path):
    preamble, _ = _build_xonsh_preamble({"MY_VAR"})
    snippet = preamble + 'reveal_type(__xonsh_env__["MY_VAR"])\n'
    src = tmp_path / "check.py"
    src.write_text(snippet)

    out = subprocess.run(
        [PYRIGHT, "--outputjson", str(src)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "Any" in out.stdout, out.stdout
