"""Tests for rename support."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from lsprotocol import types as lsp

from xonsh_lsp import server as server_module


class TestRename:
    """Test textDocument/rename support."""

    @pytest.fixture
    def mock_doc(self):
        """Create a mock document."""
        doc = MagicMock()
        doc.source = "value = 1\nprint(value)\nvalue = value + 1\n"
        doc.path = "/test/file.xsh"
        return doc

    @pytest.fixture
    def patched_server(self, monkeypatch, mock_doc):
        """Patch the module-level server used by the handler."""
        mock_server = MagicMock()
        mock_server.get_document.return_value = mock_doc
        mock_server.python_delegate.get_references = AsyncMock(
            return_value=[
                lsp.Location(
                    uri="file:///test/file.xsh",
                    range=lsp.Range(
                        start=lsp.Position(line=0, character=0),
                        end=lsp.Position(line=0, character=5),
                    ),
                ),
                lsp.Location(
                    uri="file:///test/file.xsh",
                    range=lsp.Range(
                        start=lsp.Position(line=1, character=6),
                        end=lsp.Position(line=1, character=11),
                    ),
                ),
                lsp.Location(
                    uri="file:///test/file.xsh",
                    range=lsp.Range(
                        start=lsp.Position(line=2, character=0),
                        end=lsp.Position(line=2, character=5),
                    ),
                ),
                lsp.Location(
                    uri="file:///test/file.xsh",
                    range=lsp.Range(
                        start=lsp.Position(line=2, character=8),
                        end=lsp.Position(line=2, character=13),
                    ),
                ),
            ]
        )
        monkeypatch.setattr(server_module, "server", mock_server)
        return mock_server

    @pytest.mark.asyncio
    async def test_renames_python_identifier(self, patched_server):
        """Test renaming Python identifier references."""
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

    @pytest.mark.asyncio
    async def test_rejects_invalid_python_identifier(self, patched_server):
        """Test that invalid Python identifiers are rejected."""
        params = lsp.RenameParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
            new_name="not-valid",
        )

        result = await server_module.rename(params)

        assert result is None
        patched_server.python_delegate.get_references.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_rename_env_var(self, monkeypatch):
        """Test that env vars are outside Python identifier rename scope."""
        doc = MagicMock()
        doc.source = "$VALUE = 'x'\nprint($VALUE)\n"
        doc.path = "/test/file.xsh"
        mock_server = MagicMock()
        mock_server.get_document.return_value = doc
        mock_server.python_delegate.get_references = AsyncMock(return_value=[])
        monkeypatch.setattr(server_module, "server", mock_server)

        params = lsp.RenameParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
            new_name="RENAMED",
        )

        result = await server_module.rename(params)

        assert result is None
        mock_server.python_delegate.get_references.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deduplicates_reference_ranges(self, monkeypatch, mock_doc):
        """Test that duplicate backend references produce one edit."""
        range_ = lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=0, character=5),
        )
        mock_server = MagicMock()
        mock_server.get_document.return_value = mock_doc
        mock_server.python_delegate.get_references = AsyncMock(
            return_value=[
                lsp.Location(uri="file:///test/file.xsh", range=range_),
                lsp.Location(uri="file:///test/file.xsh", range=range_),
            ]
        )
        monkeypatch.setattr(server_module, "server", mock_server)

        params = lsp.RenameParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
            new_name="renamed",
        )

        result = await server_module.rename(params)

        assert result is not None
        assert result.changes is not None
        assert len(result.changes["file:///test/file.xsh"]) == 1
