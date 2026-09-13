"""Mirror sibling ``.xsh`` modules so type-checker backends can resolve them.

Type checkers such as :mod:`ty` and Pyright resolve imports against real files
on disk.  A xonsh script that does ``from git_utils import git_dir`` therefore
cannot find a sibling ``git_utils.xsh`` even though xonsh itself imports it
without issue (issue #8).

This module mirrors workspace ``.xsh`` files into a temporary tree of ``.py``
modules and exposes that tree so the Python backend can register it as an
extra module search path.  Only files that are already valid Python are
mirrored; files that rely on xonsh-only syntax are skipped rather than written
out as broken ``.py`` stubs the backend would fail to parse.
"""

from __future__ import annotations

import ast
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


def module_name_from_stem(stem: str) -> str:
    """Return the import name for a ``.xsh`` file stem.

    ``git_utils.xsh`` next to the analysed file is imported as ``git_utils``,
    so the stem is the module name.  Stems that are not valid Python
    identifiers cannot be imported and map to an empty string.
    """
    if stem and stem.isidentifier():
        return stem
    return ""


def is_valid_python(source: str) -> bool:
    """Return ``True`` if *source* parses as a Python module."""
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return True


class SiblingModuleMirror:
    """Mirror ``.xsh`` modules into a temporary ``.py`` tree.

    The mirror is flat: every mirrored module is written at the top level of
    the temporary directory so that ``from <name> import ...`` resolves for a
    type checker that treats the directory as a module search path.
    """

    def __init__(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="xonsh-lsp-xsh-"))
        self._sources: dict[str, Path] = {}

    @property
    def root(self) -> Path:
        """Temporary directory to register as an extra search path."""
        return self._root

    @property
    def search_path(self) -> list[str]:
        """Absolute path(s) to pass to the backend as extra search paths."""
        return [str(self._root)]

    def refresh(
        self,
        xsh_paths: Iterable[Path],
        preferred: Path | None = None,
    ) -> Path:
        """(Re)build the mirror from *xsh_paths* and return the search root.

        *preferred* is the document currently being analysed; modules in the
        same directory as it take priority on name collisions, so a sibling of
        the open file wins over a same-named module elsewhere in the workspace.
        """
        self._clear()
        preferred_dir = preferred.parent if preferred is not None else None
        candidates = [Path(p) for p in xsh_paths]
        if preferred_dir is not None:
            # Sort so the current file's siblings are written last and win.
            candidates.sort(key=lambda p: p.parent == preferred_dir)

        for path in candidates:
            if path.suffix.lower() != ".xsh":
                continue
            name = module_name_from_stem(path.stem)
            if not name:
                continue
            if self._has_real_module(path, name):
                continue
            source = self._read_source(path)
            if source is None or not is_valid_python(source):
                continue
            (self._root / f"{name}.py").write_text(source, encoding="utf-8")
            self._sources[name] = path

        return self._root

    def _has_real_module(self, xsh_path: Path, name: str) -> bool:
        """Return ``True`` if a real module already shadows this name."""
        for candidate in (
            xsh_path.with_name(f"{name}.py"),
            xsh_path.with_name(f"{name}.pyi"),
            xsh_path.with_name(name),
        ):
            if candidate.exists():
                return True
        return False

    @staticmethod
    def _read_source(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _clear(self) -> None:
        for child in self._root.iterdir():
            if child.is_file():
                child.unlink()

    def close(self) -> None:
        """Remove the temporary tree."""
        shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> "SiblingModuleMirror":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
