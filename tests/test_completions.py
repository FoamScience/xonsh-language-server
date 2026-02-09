"""Tests for the completion provider."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lsprotocol import types as lsp
from xonsh_lsp.completions import XonshCompletionProvider


class TestXonshCompletionProvider:
    """Test the XonshCompletionProvider class."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server."""
        server = MagicMock()
        server.parser = MagicMock()
        server.python_delegate = MagicMock()
        server.python_delegate.get_completions = AsyncMock(return_value=[])
        return server

    @pytest.fixture
    def provider(self, mock_server):
        """Create a completion provider."""
        return XonshCompletionProvider(mock_server)

    def test_env_completions(self, provider):
        """Test environment variable completions."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            completions = provider._get_env_completions("$TEST")
            labels = [c.label for c in completions]
            assert any("TEST_VAR" in label for label in labels)

    def test_command_completions(self, provider):
        """Test command completions from PATH."""
        # This test depends on system having common commands
        completions = provider._get_command_completions("ls")
        # Should find some commands
        assert isinstance(completions, list)

    def test_path_completions(self, provider):
        """Test path completions."""
        completions = provider._get_path_completions("/tmp/")
        assert isinstance(completions, list)
        # /tmp usually exists and has files
        # But we can't guarantee specific files

    def test_glob_completions(self, provider):
        """Test glob pattern completions."""
        completions = provider._get_glob_completions()
        assert len(completions) > 0
        labels = [c.label for c in completions]
        assert "*" in labels
        assert "**" in labels

    def test_xonsh_builtin_completions(self, provider):
        """Test xonsh builtin completions."""
        completions = provider._get_xonsh_builtin_completions("cd")
        labels = [c.label for c in completions]
        assert "cd" in labels

    def test_is_path_context(self, provider):
        """Test path context detection."""
        assert provider._is_path_context("/home/")
        assert provider._is_path_context("~/")
        assert provider._is_path_context("./file")
        assert provider._is_path_context('p"')
        assert not provider._is_path_context("regular text")


class TestCompletionResolve:
    """Test completion item resolution."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_server):
        """Create a completion provider."""
        return XonshCompletionProvider(mock_server)

    def test_resolve_env_var(self, provider):
        """Test resolving environment variable completion."""
        with patch.dict(os.environ, {"MY_VAR": "my_value"}):
            item = lsp.CompletionItem(
                label="$MY_VAR",
                data={"type": "env_var", "name": "MY_VAR"},
            )
            resolved = provider.resolve_completion(item)
            assert resolved.documentation is not None

    def test_resolve_xonsh_builtin(self, provider):
        """Test resolving xonsh builtin completion."""
        item = lsp.CompletionItem(
            label="cd",
            data={"type": "xonsh_builtin", "name": "cd"},
        )
        resolved = provider.resolve_completion(item)
        assert resolved.documentation is not None
