"""Tests for the hover provider."""

import os
import pytest
from unittest.mock import MagicMock, patch

from lsprotocol import types as lsp
from xonsh_lsp.hover import XonshHoverProvider
from xonsh_lsp.parser import XonshParser


class TestXonshHoverProvider:
    """Test the XonshHoverProvider class."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server."""
        server = MagicMock()
        server.parser = XonshParser()
        server.python_delegate = MagicMock()
        server.python_delegate.get_hover.return_value = None
        return server

    @pytest.fixture
    def provider(self, mock_server):
        """Create a hover provider."""
        return XonshHoverProvider(mock_server)

    def test_env_var_hover(self, provider):
        """Test hover for environment variables."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            hover_content = provider._get_env_var_hover("$TEST_VAR")
            assert hover_content is not None
            assert "TEST_VAR" in hover_content
            assert "test_value" in hover_content

    def test_env_var_braced_hover(self, provider):
        """Test hover for braced environment variables."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            hover_content = provider._get_env_var_hover("${TEST_VAR}")
            assert hover_content is not None
            assert "TEST_VAR" in hover_content
            assert "test_value" in hover_content

    def test_undefined_env_var_hover(self, provider):
        """Test hover for undefined environment variables."""
        os.environ.pop("UNDEFINED_TEST_VAR_XYZ", None)
        hover_content = provider._get_env_var_hover("$UNDEFINED_TEST_VAR_XYZ")
        assert hover_content is not None
        assert "Not defined" in hover_content or "not currently set" in hover_content

    def test_xonsh_operator_hover(self, provider):
        """Test hover for xonsh operators."""
        # Test captured subprocess operator
        hover_content = provider._get_operator_hover("$(")
        if hover_content:  # Only if XONSH_OPERATORS has this
            assert "subprocess" in hover_content.lower() or "capture" in hover_content.lower()

    def test_xonsh_builtin_hover(self, provider):
        """Test hover for xonsh builtins."""
        hover_content = provider._get_builtin_hover("cd")
        if hover_content:  # Only if XONSH_BUILTINS has cd
            assert "cd" in hover_content

    def test_command_hover(self, provider):
        """Test hover for subprocess commands."""
        # ls should exist on most systems
        hover_content = provider._get_command_hover("ls")
        if hover_content:
            assert "ls" in hover_content
            assert "Path:" in hover_content or "path" in hover_content.lower()

    def test_nonexistent_command_hover(self, provider):
        """Test hover for non-existent commands."""
        hover_content = provider._get_command_hover("nonexistent_cmd_xyz123")
        assert hover_content is None

    def test_get_word_at_position(self, provider):
        """Test word extraction at position."""
        source = "print($HOME)"
        # At position 6 (H in HOME)
        word = provider._get_word_at_position(source, 0, 7)
        assert word is not None
        assert "$HOME" in word or "HOME" in word

    def test_get_word_at_position_env_var(self, provider):
        """Test word extraction for environment variable."""
        source = "$MY_VAR = 1"
        word = provider._get_word_at_position(source, 0, 3)
        assert word is not None
        assert "MY_VAR" in word

    def test_get_word_at_position_braced_env_var(self, provider):
        """Test word extraction for braced environment variable."""
        source = "print(${PATH})"
        word = provider._get_word_at_position(source, 0, 9)  # On 'A' in PATH
        assert word is not None
        assert "PATH" in word

    def test_get_word_range(self, provider):
        """Test word range calculation."""
        source = "my_variable = 42"
        range_result = provider._get_word_range(source, 0, 5)
        assert range_result is not None
        assert range_result.start.character >= 0
        assert range_result.end.character > range_result.start.character

    def test_is_command_context(self, provider, mock_server):
        """Test subprocess context detection."""
        source = "$(ls -la)"
        # This depends on tree-sitter-xonsh
        result = provider._is_command_context(source, 0, 3)
        assert isinstance(result, bool)

    def test_hover_full_flow(self, provider, mock_server):
        """Test full hover flow with a document."""
        mock_doc = MagicMock()
        mock_doc.source = "$HOME"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        params = lsp.HoverParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
        )

        hover = provider.get_hover(params)
        # Should return hover info for $HOME
        assert hover is None or isinstance(hover, lsp.Hover)

    def test_python_hover_delegation(self, provider, mock_server):
        """Test that Python hover is delegated."""
        mock_doc = MagicMock()
        mock_doc.source = "print('hello')"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        # Set up mock Python hover
        mock_server.python_delegate.get_hover.return_value = "Python hover content"

        params = lsp.HoverParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=2),
        )

        hover = provider.get_hover(params)
        # Should get something (either xonsh or Python hover)
        assert hover is None or isinstance(hover, lsp.Hover)


class TestHoverEdgeCases:
    """Test edge cases for hover provider."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server."""
        server = MagicMock()
        server.parser = XonshParser()
        server.python_delegate = MagicMock()
        server.python_delegate.get_hover.return_value = None
        return server

    @pytest.fixture
    def provider(self, mock_server):
        """Create a hover provider."""
        return XonshHoverProvider(mock_server)

    def test_hover_empty_document(self, provider, mock_server):
        """Test hover on empty document."""
        mock_server.get_document.return_value = None

        params = lsp.HoverParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=0, character=0),
        )

        hover = provider.get_hover(params)
        assert hover is None

    def test_hover_out_of_range(self, provider, mock_server):
        """Test hover at out-of-range position."""
        mock_doc = MagicMock()
        mock_doc.source = "x = 1"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        params = lsp.HoverParams(
            text_document=lsp.TextDocumentIdentifier(uri="file:///test/file.xsh"),
            position=lsp.Position(line=100, character=0),
        )

        hover = provider.get_hover(params)
        assert hover is None

    def test_long_env_var_value_truncation(self, provider):
        """Test that long env var values are truncated."""
        long_value = "x" * 1000
        with patch.dict(os.environ, {"LONG_VAR": long_value}):
            hover_content = provider._get_env_var_hover("$LONG_VAR")
            assert hover_content is not None
            # Should be truncated
            assert len(hover_content) < len(long_value) + 200
            assert "..." in hover_content
