"""Tests for the definition provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from lsprotocol import types as lsp
from xonsh_lsp.definition import XonshDefinitionProvider, XonshReferenceProvider


class TestXonshDefinitionProvider:
    """Test the XonshDefinitionProvider class."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server."""
        server = MagicMock()
        server.python_delegate = MagicMock()
        server.python_delegate.get_definitions = AsyncMock(return_value=[])
        return server

    @pytest.fixture
    def provider(self, mock_server):
        """Create a definition provider."""
        return XonshDefinitionProvider(mock_server)

    def test_find_env_var_definition_dollar(self, provider):
        """Test finding $VAR = ... definition."""
        source = """
$MY_VAR = 'value'
print($MY_VAR)
"""
        uri = "file:///test/file.xsh"
        location = provider._find_env_var_definition(source, "MY_VAR", uri)
        assert location is not None
        assert location.uri == uri
        assert location.range.start.line == 1

    def test_find_env_var_definition_braced(self, provider):
        """Test finding ${VAR} = ... definition."""
        source = """
${MY_VAR} = 'value'
print(${MY_VAR})
"""
        uri = "file:///test/file.xsh"
        location = provider._find_env_var_definition(source, "MY_VAR", uri)
        assert location is not None
        assert location.uri == uri

    def test_find_env_var_definition_os_environ(self, provider):
        """Test finding os.environ['VAR'] = ... definition."""
        source = """
import os
os.environ['MY_VAR'] = 'value'
"""
        uri = "file:///test/file.xsh"
        location = provider._find_env_var_definition(source, "MY_VAR", uri)
        assert location is not None
        assert location.uri == uri
        assert location.range.start.line == 2

    def test_find_env_var_definition_not_found(self, provider):
        """Test env var definition not found."""
        source = "print($OTHER_VAR)"
        uri = "file:///test/file.xsh"
        location = provider._find_env_var_definition(source, "MY_VAR", uri)
        assert location is None

    def test_find_alias_definition(self, provider):
        """Test finding alias definition."""
        source = """
aliases['ll'] = 'ls -la'
ll
"""
        uri = "file:///test/file.xsh"
        location = provider._find_alias_definition(source, "ll", uri)
        assert location is not None
        assert location.uri == uri
        assert location.range.start.line == 1

    def test_find_alias_definition_not_found(self, provider):
        """Test alias definition not found."""
        source = "print('hello')"
        uri = "file:///test/file.xsh"
        location = provider._find_alias_definition(source, "ll", uri)
        assert location is None

    def test_find_function_definition(self, provider):
        """Test finding function definition."""
        source = """
def my_func():
    pass

my_func()
"""
        uri = "file:///test/file.xsh"
        location = provider._find_function_definition(source, "my_func", uri)
        assert location is not None
        assert location.uri == uri
        assert location.range.start.line == 1

    def test_find_function_definition_not_found(self, provider):
        """Test function definition not found."""
        source = "print('hello')"
        uri = "file:///test/file.xsh"
        location = provider._find_function_definition(source, "my_func", uri)
        assert location is None

    def test_get_word_at_position(self, provider):
        """Test word extraction at position."""
        source = "my_variable = 42"
        word = provider._get_word_at_position(source, 0, 5)
        assert word == "my_variable"

    def test_get_word_at_position_env_var(self, provider):
        """Test word extraction for env var."""
        source = "$MY_VAR = 1"
        word = provider._get_word_at_position(source, 0, 3)
        assert word is not None
        assert "MY_VAR" in word

    def test_get_word_at_position_out_of_range(self, provider):
        """Test word extraction out of range."""
        source = "x = 1"
        word = provider._get_word_at_position(source, 100, 0)
        assert word is None

    @pytest.mark.asyncio
    async def test_full_definition_flow(self, provider, mock_server):
        """Test full definition flow."""
        mock_doc = MagicMock()
        mock_doc.source = """
def greet():
    pass

greet()
"""
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        params = lsp.DefinitionParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=4, character=2),  # On 'greet()'
        )

        result = await provider.get_definition(params)
        # Should find the function definition
        assert result is not None

    @pytest.mark.asyncio
    async def test_python_definition_delegation(self, provider, mock_server):
        """Test that Python definitions are delegated."""
        mock_doc = MagicMock()
        mock_doc.source = "import os\nos.path"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        # Set up mock Python definition
        mock_server.python_delegate.get_definitions = AsyncMock(return_value=[
            lsp.Location(
                uri="file:///usr/lib/python/os.py",
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=10),
                ),
            )
        ])

        params = lsp.DefinitionParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=1, character=3),
        )

        result = await provider.get_definition(params)
        # Should include Python definitions
        assert result is not None


class TestXonshReferenceProvider:
    """Test the XonshReferenceProvider class."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server."""
        server = MagicMock()
        server.python_delegate = MagicMock()
        server.python_delegate.get_references.return_value = []
        return server

    @pytest.fixture
    def provider(self, mock_server):
        """Create a reference provider."""
        return XonshReferenceProvider(mock_server)

    def test_get_references(self, provider, mock_server):
        """Test finding references."""
        mock_doc = MagicMock()
        mock_doc.source = """
my_var = 1
print(my_var)
my_var = my_var + 1
"""
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        params = lsp.ReferenceParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=1, character=2),
            context=lsp.ReferenceContext(include_declaration=True),
        )

        result = provider.get_references(params)
        # Should find multiple references to my_var
        assert result is not None
        assert len(result) >= 3  # At least 3 occurrences

    def test_get_references_env_var(self, provider, mock_server):
        """Test finding references to env vars."""
        mock_doc = MagicMock()
        mock_doc.source = """
$MY_VAR = 'value'
print($MY_VAR)
echo ${MY_VAR}
"""
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        params = lsp.ReferenceParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=1, character=3),  # On MY_VAR
            context=lsp.ReferenceContext(include_declaration=True),
        )

        result = provider.get_references(params)
        assert result is not None

    def test_get_references_no_document(self, provider, mock_server):
        """Test references with no document."""
        mock_server.get_document.return_value = None

        params = lsp.ReferenceParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=0),
            context=lsp.ReferenceContext(include_declaration=True),
        )

        result = provider.get_references(params)
        assert result is None

    def test_get_word_at_position(self, provider):
        """Test word extraction."""
        source = "variable_name = 42"
        word = provider._get_word_at_position(source, 0, 5)
        assert word == "variable_name"
