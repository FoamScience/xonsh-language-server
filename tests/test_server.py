from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lsprotocol import types as lsp

import xonsh_lsp.server as server_module


@pytest.fixture
def document():
    return SimpleNamespace(source="MyClass()\nls -la\n", path="/test/file.xsh")


@pytest.fixture
def backend():
    backend = SimpleNamespace()
    backend.get_semantic_tokens = AsyncMock(
        return_value=lsp.SemanticTokens(data=[0, 0, 7, 5, 0])
    )
    backend.get_semantic_tokens_range = AsyncMock(
        return_value=lsp.SemanticTokens(data=[0, 0, 7, 5, 0])
    )
    return backend


@pytest.mark.asyncio
async def test_semantic_tokens_full_uses_python_backend(
    monkeypatch,
    document,
    backend,
):
    # Given
    monkeypatch.setattr(server_module.server, "get_document", lambda uri: document)
    monkeypatch.setattr(server_module.server, "python_backend", backend)

    # When
    result = await server_module.semantic_tokens_full(
        lsp.SemanticTokensParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh")
        )
    )

    # Then
    backend.get_semantic_tokens.assert_awaited_once_with(document.source, document.path)
    assert result == lsp.SemanticTokens(data=[0, 0, 7, 5, 0])


@pytest.mark.asyncio
async def test_semantic_tokens_range_uses_python_backend(
    monkeypatch,
    document,
    backend,
):
    # Given
    monkeypatch.setattr(server_module.server, "get_document", lambda uri: document)
    monkeypatch.setattr(server_module.server, "python_backend", backend)
    range_ = lsp.Range(
        start=lsp.Position(line=0, character=0),
        end=lsp.Position(line=0, character=9),
    )

    # When
    result = await server_module.semantic_tokens_range(
        lsp.SemanticTokensRangeParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            range=range_,
        )
    )

    # Then
    backend.get_semantic_tokens_range.assert_awaited_once_with(
        document.source,
        0,
        0,
        0,
        9,
        document.path,
    )
    assert result == lsp.SemanticTokens(data=[0, 0, 7, 5, 0])
