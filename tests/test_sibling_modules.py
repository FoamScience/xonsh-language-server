"""Tests for sibling ``.xsh`` module mirroring (issue #8)."""

from pathlib import Path

from xonsh_lsp.sibling_modules import (
    SiblingModuleMirror,
    is_valid_python,
    module_name_from_stem,
)

VALID = "def git_dir():\n    return '.'\n\n\ndef list_submodules():\n    return []\n"
INVALID = "echo $HOME\nx = $(date)\n"


def test_module_name_from_stem():
    assert module_name_from_stem("git_utils") == "git_utils"
    assert module_name_from_stem("_private") == "_private"
    assert module_name_from_stem("with space") == ""
    assert module_name_from_stem("9bad") == ""
    assert module_name_from_stem("") == ""


def test_is_valid_python():
    assert is_valid_python(VALID)
    assert not is_valid_python(INVALID)
    assert not is_valid_python("def f(:\n")


def test_mirrors_valid_sibling(tmp_path: Path):
    (tmp_path / "git_utils.xsh").write_text(VALID, encoding="utf-8")
    mirror = SiblingModuleMirror()
    try:
        mirror.refresh([tmp_path / "git_utils.xsh"])
        mirrored = mirror.root / "git_utils.py"
        assert mirrored.exists()
        assert mirrored.read_text(encoding="utf-8") == VALID
        # The source directory is never polluted by the mirror.
        assert not (tmp_path / "git_utils.py").exists()
    finally:
        mirror.close()
    # The temporary tree is removed on close.
    assert not mirror.root.exists()


def test_skips_invalid_python(tmp_path: Path):
    (tmp_path / "noisy.xsh").write_text(INVALID, encoding="utf-8")
    with SiblingModuleMirror() as mirror:
        mirror.refresh([tmp_path / "noisy.xsh"])
        assert not (mirror.root / "noisy.py").exists()


def test_real_py_sibling_is_left_alone(tmp_path: Path):
    (tmp_path / "foo.xsh").write_text(VALID, encoding="utf-8")
    (tmp_path / "foo.py").write_text("real = 1\n", encoding="utf-8")
    with SiblingModuleMirror() as mirror:
        mirror.refresh([tmp_path / "foo.xsh"])
        assert not (mirror.root / "foo.py").exists()


def test_search_path_reports_temp_root():
    with SiblingModuleMirror() as mirror:
        mirror.refresh([])
        assert mirror.search_path == [str(mirror.root)]


def test_preferred_sibling_wins_collision(tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "utils.xsh").write_text("other = 1\n", encoding="utf-8")
    (tmp_path / "utils.xsh").write_text("mine = 1\n", encoding="utf-8")
    with SiblingModuleMirror() as mirror:
        mirror.refresh(
            [other / "utils.xsh", tmp_path / "utils.xsh"],
            preferred=tmp_path / "tool.xsh",
        )
        assert (mirror.root / "utils.py").read_text(encoding="utf-8") == "mine = 1\n"
