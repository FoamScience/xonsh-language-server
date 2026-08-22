"""Tests for rename support."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from lsprotocol import types as lsp

from xonsh_lsp import server as server_module


def _build_workspace_edit(uri: str, ranges: list[lsp.Range], new_name: str) -> lsp.WorkspaceEdit:
    return lsp.WorkspaceEdit(
        changes={uri: [lsp.TextEdit(range=r, new_text=new_name) for r in ranges]}
    )


class TestServerRename:
    """Test textDocument/rename support."""

    @pytest.fixture
    def mock_doc(self):
        doc = MagicMock()
        doc.source = "value = 1\nprint(value)\nvalue = value + 1\n"
        doc.path = "/test/file.xsh"
        return doc

    @pytest.fixture
    def patched_server(self, monkeypatch, mock_doc):
        """Patch the module-level server with a mock backend.rename."""
        uri = "file:///test/file.xsh"
        backend_edit = _build_workspace_edit(
            uri,
            [
                lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=5),
                ),
                lsp.Range(
                    start=lsp.Position(line=1, character=6),
                    end=lsp.Position(line=1, character=11),
                ),
                lsp.Range(
                    start=lsp.Position(line=2, character=0),
                    end=lsp.Position(line=2, character=5),
                ),
                lsp.Range(
                    start=lsp.Position(line=2, character=8),
                    end=lsp.Position(line=2, character=13),
                ),
            ],
            new_name="renamed",
        )
        mock_server = MagicMock()
        mock_server.get_document.return_value = mock_doc
        mock_server.parser = server_module.XonshParser()
        mock_server.parse_document.return_value = mock_server.parser.parse(
            mock_doc.source
        )
        mock_server.python_delegate.rename = AsyncMock(return_value=backend_edit)
        monkeypatch.setattr(server_module, "server", mock_server)
        return mock_server

    @pytest.mark.asyncio
    async def test_delegates_to_backend(self, patched_server):
        """Server forwards the rename request to the backend and returns its edit."""
        params = lsp.RenameParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
            new_name="renamed",
        )

        result = await server_module.rename(params)

        assert result is not None
        assert result.changes is not None
        edits = result.changes["file:///test/file.xsh"]
        assert len(edits) == 4
        assert all(edit.new_text == "renamed" for edit in edits)
        patched_server.python_delegate.rename.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejects_invalid_python_identifier(self, patched_server):
        """Invalid Python identifiers are rejected before reaching the backend."""
        params = lsp.RenameParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
            new_name="not-valid",
        )

        result = await server_module.rename(params)

        assert result is None
        patched_server.python_delegate.rename.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_renames_env_var_without_backend(self, monkeypatch):
        """Env vars are renamed via the xonsh parse tree, not the backend."""
        doc = MagicMock()
        doc.source = "$VALUE = 'x'\nprint($VALUE)\necho ${VALUE}\n"
        doc.path = "/test/file.xsh"
        mock_server = MagicMock()
        mock_server.get_document.return_value = doc
        mock_server.parser = server_module.XonshParser()
        mock_server.parse_document.return_value = mock_server.parser.parse(doc.source)
        mock_server.python_delegate.rename = AsyncMock(return_value=None)
        monkeypatch.setattr(server_module, "server", mock_server)

        params = lsp.RenameParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
            new_name="RENAMED",
        )

        result = await server_module.rename(params)

        assert result is not None
        edits = result.changes["file:///test/file.xsh"]
        starts = sorted(
            (e.range.start.line, e.range.start.character) for e in edits
        )
        # Name spans only: after `$` on lines 0-1, after `${` on line 2.
        assert starts == [(0, 1), (1, 7), (2, 7)]
        assert all(e.new_text == "RENAMED" for e in edits)
        mock_server.python_delegate.rename.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_when_backend_returns_none(self, monkeypatch, mock_doc):
        """A backend that refuses to rename surfaces as None."""
        mock_server = MagicMock()
        mock_server.get_document.return_value = mock_doc
        mock_server.parser = server_module.XonshParser()
        mock_server.parse_document.return_value = mock_server.parser.parse(
            mock_doc.source
        )
        mock_server.python_delegate.rename = AsyncMock(return_value=None)
        monkeypatch.setattr(server_module, "server", mock_server)

        params = lsp.RenameParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
            new_name="renamed",
        )

        result = await server_module.rename(params)

        assert result is None
