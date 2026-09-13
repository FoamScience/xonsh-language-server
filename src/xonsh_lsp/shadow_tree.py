"""Mirror workspace ``.xsh`` files as importable ``.py`` modules.

Python analysis backends resolve imports against the file system, not against
open-document overlays, so a sibling ``git_utils.xsh`` is invisible to them and
``from git_utils import git_dir`` fails to resolve (issue #8).  Mirroring every
workspace ``.xsh`` file as a preprocessed ``.py`` file in a temporary tree, and
handing that tree to the backend as an extra module search path, makes the
import resolve without writing anything into the user's workspace.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from xonsh_lsp.preprocessing import preprocess_with_mapping

logger = logging.getLogger(__name__)

SHADOW_IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", ".env", "node_modules",
    "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox",
})


def iter_xsh_files(root: Path):
    """Yield .xsh files under *root*, skipping VCS/venv/cache directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SHADOW_IGNORE_DIRS]
        for name in filenames:
            if name.endswith(".xsh"):
                yield Path(dirpath) / name


class ShadowTree:
    """A temporary ``.py`` mirror of the workspace's ``.xsh`` files.

    The mirror keeps the workspace's directory layout so that packages and
    relative imports resolve the same way they do in the real tree.
    """

    def __init__(self, workspace_root: str | None) -> None:
        self._workspace_root = workspace_root
        self.root: Path | None = None
        self.search_paths: list[str] = []

    @property
    def workspace_root(self) -> str | None:
        """The workspace this tree mirrors, or None if there is none."""
        return self._workspace_root

    def build(self) -> int:
        """Mirror every workspace .xsh file. Returns the number written."""
        if self._workspace_root is None:
            return 0
        root = Path(self._workspace_root)
        if not root.is_dir():
            return 0
        self.root = Path(tempfile.mkdtemp(prefix="xonsh-lsp-shadow-"))
        self.search_paths = [str(self.root)]
        count = 0
        for path in iter_xsh_files(root):
            if self.write(path):
                count += 1
        logger.info(f"shadowed {count} .xsh file(s) into {self.root}")
        return count

    def write(self, path: Path) -> Path | None:
        """Write one preprocessed .xsh file into the tree.

        Returns the shadow file that was written, or None if *path* was skipped.
        """
        if self.root is None or self._workspace_root is None:
            return None
        if path.suffix != ".xsh":
            return None
        if path.with_suffix(".py").exists():
            # A real module already owns this name; shadowing it would replace
            # the real file's contents for the whole workspace analysis.
            logger.debug(f"not shadowing {path} (real .py sibling exists)")
            return None
        try:
            rel = path.relative_to(Path(self._workspace_root))
            source = path.read_text(encoding="utf-8")
        except (ValueError, OSError) as e:
            logger.debug(f"cannot shadow {path}: {e}")
            return None

        target = self.root / rel.with_suffix(".py")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # No xonsh preamble here: preprocessing is line-preserving, so
            # shadow line numbers match the real .xsh and go-to-definition
            # results remap to the right line.
            target.write_text(preprocess_with_mapping(source).source, encoding="utf-8")
        except OSError as e:
            logger.debug(f"cannot write shadow for {path}: {e}")
            return None

        # Siblings are imported by bare name, so each directory needs to be a
        # search path of its own, not just the shadow root.
        parent = str(target.parent)
        if parent not in self.search_paths:
            self.search_paths.append(parent)
        return target

    def target(self, path: Path) -> Path | None:
        """Existing shadow file for *path*, or None if it was never written."""
        if self.root is None or self._workspace_root is None:
            return None
        try:
            rel = path.relative_to(Path(self._workspace_root))
        except ValueError:
            return None
        target = self.root / rel.with_suffix(".py")
        return target if target.exists() else None

    def source_of(self, target: Path) -> Path | None:
        """The real .xsh file a shadow module was written from."""
        if self.root is None or self._workspace_root is None:
            return None
        try:
            rel = Path(target).relative_to(self.root)
        except ValueError:
            return None
        source = Path(self._workspace_root) / rel.with_suffix(".xsh")
        return source if source.exists() else None

    def remove(self, path: Path) -> Path | None:
        """Drop the shadow module for a deleted .xsh file."""
        target = self.target(path)
        if target is None:
            return None
        try:
            target.unlink()
        except OSError as e:
            logger.debug(f"cannot remove shadow {target}: {e}")
            return None
        return target

    def close(self) -> None:
        """Remove the temporary tree."""
        if self.root is None:
            return
        shutil.rmtree(self.root, ignore_errors=True)
        self.root = None
        self.search_paths = []
