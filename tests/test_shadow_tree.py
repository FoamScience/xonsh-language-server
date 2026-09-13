"""Tests for the .xsh -> .py shadow tree and Jedi's use of it (issue #8)."""

from pathlib import Path

import pytest

from xonsh_lsp.jedi_backend import JediBackend
from xonsh_lsp.shadow_tree import ShadowTree

GIT_UTILS = "def git_dir():\n    return $(git rev-parse --show-toplevel)\n"
MAIN = "from git_utils import git_dir\n\ngit_dir()\n"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "git_utils.xsh").write_text(GIT_UTILS, encoding="utf-8")
    (tmp_path / "main.xsh").write_text(MAIN, encoding="utf-8")
    return tmp_path


def test_build_mirrors_xsh_preserving_layout(workspace: Path):
    nested = workspace / "tools" / "scripts"
    nested.mkdir(parents=True)
    (nested / "helper.xsh").write_text(GIT_UTILS, encoding="utf-8")
    tree = ShadowTree(str(workspace))
    try:
        assert tree.build() == 3
        assert (tree.root / "git_utils.py").exists()
        assert (tree.root / "tools" / "scripts" / "helper.py").exists()
        # Bare-name imports need every directory on the path, not just the root.
        assert str(tree.root) in tree.search_paths
        assert str(tree.root / "tools" / "scripts") in tree.search_paths
        # Xonsh-only syntax is preprocessed, not skipped.
        assert "$(" not in (tree.root / "git_utils.py").read_text()
        # The workspace itself is never touched.
        assert not (workspace / "git_utils.py").exists()
    finally:
        tree.close()
    assert tree.root is None


def test_real_py_module_is_not_shadowed(workspace: Path):
    (workspace / "git_utils.py").write_text("real = 1\n", encoding="utf-8")
    tree = ShadowTree(str(workspace))
    try:
        tree.build()
        assert not (tree.root / "git_utils.py").exists()
    finally:
        tree.close()


def test_source_of_maps_shadow_back_to_xsh(workspace: Path):
    tree = ShadowTree(str(workspace))
    try:
        tree.build()
        assert tree.source_of(tree.root / "git_utils.py") == workspace / "git_utils.xsh"
        assert tree.source_of(tree.root / "absent.py") is None
        assert tree.source_of(Path("/elsewhere/x.py")) is None
    finally:
        tree.close()


def test_remove_drops_the_shadow(workspace: Path):
    tree = ShadowTree(str(workspace))
    try:
        tree.build()
        target = tree.remove(workspace / "git_utils.xsh")
        assert target is not None and not target.exists()
        assert tree.remove(workspace / "git_utils.xsh") is None
    finally:
        tree.close()


@pytest.mark.asyncio
async def test_jedi_resolves_sibling_xsh_import(workspace: Path):
    backend = JediBackend()
    await backend.start(str(workspace))
    main = str(workspace / "main.xsh")
    expected = (workspace / "git_utils.xsh").as_uri()
    try:
        # The module name and the name imported from it both resolve to the
        # sibling; without the shadow tree Jedi returns nothing for either.
        for col in (5, 22):
            locations = await backend.get_definitions(MAIN, 0, col, path=main)
            assert locations, f"sibling .xsh import did not resolve at col {col}"
            # Navigation lands on the real .xsh, not the temporary shadow module.
            assert locations[0].uri == expected
            # Preprocessing is line-preserving, so the def is on line 0.
            assert locations[0].range.start.line == 0
        # The sibling's signature reaches hover too.
        assert "git_dir" in (await backend.get_hover(MAIN, 2, 0, path=main) or "")
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_sibling_import_unresolved_without_the_shadow_tree(workspace: Path):
    """The gap this closes: a bare JediBackend cannot see the sibling."""
    backend = JediBackend()
    await backend.start(None)
    try:
        locations = await backend.get_definitions(
            MAIN, 0, 5, path=str(workspace / "main.xsh")
        )
        assert locations == []
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_jedi_refresh_picks_up_a_new_sibling(workspace: Path):
    backend = JediBackend()
    await backend.start(str(workspace))
    try:
        late = workspace / "tools" / "late.xsh"
        late.parent.mkdir()
        late.write_text("def later():\n    return 1\n", encoding="utf-8")
        backend.refresh_shadow(str(late))
        assert (backend._shadow.root / "tools" / "late.py").exists()
        # A new directory joined the search paths, so Jedi needs a new Project.
        assert str(backend._shadow.root / "tools") in backend._project.added_sys_path
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_jedi_without_workspace_still_works(tmp_path: Path):
    backend = JediBackend()
    await backend.start(None)
    try:
        assert backend._project is None
        assert await backend.get_definitions("x = 1\nx\n", 1, 0) == []
    finally:
        await backend.stop()
