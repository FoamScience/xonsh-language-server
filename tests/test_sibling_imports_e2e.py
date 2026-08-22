"""End-to-end check for issue #8: backends must resolve sibling .xsh imports."""

import asyncio
import shutil

import pytest
from lsprotocol import types as lsp

from xonsh_lsp.lsp_proxy_backend import KNOWN_BACKENDS, LspProxyBackend


async def _diagnostics_for(backend_name, tmp_path, source_name):
    published: dict[str, list[lsp.Diagnostic]] = {}

    def on_diagnostics(uri, diagnostics):
        published[uri] = diagnostics

    backend = LspProxyBackend(
        KNOWN_BACKENDS[backend_name], on_diagnostics=on_diagnostics
    )
    await backend.start(str(tmp_path))
    try:
        assert backend._started, f"{backend_name} failed to start"
        tool = tmp_path / source_name
        await backend.get_diagnostics(tool.read_text(), str(tool))
        for _ in range(60):
            await asyncio.sleep(0.25)
            if published:
                break
        return [d.message for ds in published.values() for d in ds]
    finally:
        await backend.stop()


@pytest.mark.parametrize("backend_name", ["ty", "pyright"])
@pytest.mark.asyncio
async def test_sibling_xsh_import_resolves(backend_name, tmp_path):
    if shutil.which(KNOWN_BACKENDS[backend_name][0]) is None:
        pytest.skip(f"requires the {backend_name} binary")

    scripts = tmp_path / "tools" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "git_utils.xsh").write_text(
        "def git_dir():\n    return 'x'\n\n\ndef list_submodules():\n    return []\n"
    )
    (scripts / "tool.xsh").write_text(
        "#!/usr/bin/env xonsh\nfrom git_utils import git_dir, list_submodules\n"
        "git_dir()\nlist_submodules()\n"
    )
    messages = await _diagnostics_for(
        backend_name, tmp_path, "tools/scripts/tool.xsh"
    )
    assert not [m for m in messages if "git_utils" in m], messages


@pytest.mark.parametrize("backend_name", ["ty", "pyright"])
@pytest.mark.asyncio
async def test_sibling_created_after_startup_resolves(backend_name, tmp_path):
    """A .xsh sibling appearing after startup must reach the child."""
    if shutil.which(KNOWN_BACKENDS[backend_name][0]) is None:
        pytest.skip(f"requires the {backend_name} binary")

    tool = tmp_path / "tool.xsh"
    tool.write_text("from git_utils import git_dir\ngit_dir()\n")

    published: dict[str, list[lsp.Diagnostic]] = {}

    def on_diagnostics(uri, diagnostics):
        published[uri] = diagnostics

    backend = LspProxyBackend(
        KNOWN_BACKENDS[backend_name], on_diagnostics=on_diagnostics
    )
    await backend.start(str(tmp_path))
    try:
        await backend.get_diagnostics(tool.read_text(), str(tool))
        for _ in range(60):
            await asyncio.sleep(0.25)
            if published:
                break
        assert [
            d for ds in published.values() for d in ds if "git_utils" in d.message
        ], "expected the unresolved-import error before the sibling exists"

        sibling = tmp_path / "git_utils.xsh"
        sibling.write_text("def git_dir():\n    return 'x'\n")
        published.clear()
        backend.refresh_shadow(str(sibling))
        await asyncio.sleep(0.5)
        await backend.get_diagnostics(tool.read_text() + "\n", str(tool))
        for _ in range(60):
            await asyncio.sleep(0.25)
            if published:
                break
        messages = [d.message for ds in published.values() for d in ds]
        assert not [m for m in messages if "git_utils" in m], messages
    finally:
        await backend.stop()
