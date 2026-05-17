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

    def test_at_object_first_level(self, provider):
        """`@.` lists at-object attributes (env, imp, ...)."""
        labels = [c.label for c in provider._get_at_object_completions("@.")]
        assert "imp" in labels
        assert "env" in labels
        assert "history" in labels
        assert "debug" in labels
        # Removed: not part of XonshSessionInterface
        assert "aliases" not in labels
        assert "builtins" not in labels

    def test_at_object_filter_by_prefix(self, provider):
        """`@.im` narrows to matching at-objects."""
        labels = [c.label for c in provider._get_at_object_completions("@.im")]
        assert "imp" in labels
        assert "env" not in labels

    def test_at_imp_completes_modules(self, provider):
        """`@.imp.` completes top-level Python module names."""
        items = provider._get_at_object_completions("@.imp.")
        labels = [c.label for c in items]
        assert "os" in labels
        assert "sys" in labels
        assert "json" in labels
        assert all(c.kind == lsp.CompletionItemKind.Module for c in items)

    def test_at_imp_module_prefix(self, provider):
        """`@.imp.js` filters modules by prefix."""
        labels = [c.label for c in provider._get_at_object_completions("@.imp.js")]
        assert "json" in labels
        assert "os" not in labels

    def test_at_imp_deeper_no_completion(self, provider):
        """`@.imp.json.` is deeper than supported — return nothing here."""
        # Deeper module attribute completion is delegated to the Python backend;
        # this helper should not return at-object items at this depth.
        items = provider._get_at_object_completions("@.imp.json.")
        assert items == []

    def test_at_other_object_no_module_list(self, provider):
        """`@.env` is a regular object — no special attribute completions."""
        # Env is a dict-like; LSP can't statically know its keys, so we return
        # nothing rather than guessing properties like `.HOME`.
        assert provider._get_at_object_completions("@.env.") == []
        assert provider._get_at_object_completions("@.lastcmd.") == []

    def test_imp_module_cache_reuse(self, provider):
        """Module list is cached across calls."""
        first = provider._get_top_level_modules()
        second = provider._get_top_level_modules()
        assert first is second

    @pytest.mark.asyncio
    async def test_imp_attribute_forwarded_to_backend(self, provider, mock_server):
        """`@.imp.json.lo` calls the Python backend with synthetic source."""
        sentinel = [lsp.CompletionItem(label="loads")]
        mock_server.python_delegate.get_completions = AsyncMock(return_value=sentinel)

        result = await provider._get_imp_attribute_completions("@.imp.json.lo", "/tmp/x.xsh")

        assert result is sentinel
        mock_server.python_delegate.get_completions.assert_awaited_once()
        args = mock_server.python_delegate.get_completions.await_args.args
        synthetic, line, col, _ = args
        assert synthetic == "import json\njson.lo"
        assert line == 1
        assert col == len("json.lo")

    @pytest.mark.asyncio
    async def test_imp_attribute_dotted_module(self, provider, mock_server):
        """`@.imp.os.path.j` imports `os.path` and asks for `j` completions."""
        mock_server.python_delegate.get_completions = AsyncMock(return_value=[])
        await provider._get_imp_attribute_completions("@.imp.os.path.j", None)
        synthetic, line, col, _ = mock_server.python_delegate.get_completions.await_args.args
        assert synthetic == "import os.path\nos.path.j"
        assert line == 1
        assert col == len("os.path.j")

    @pytest.mark.asyncio
    async def test_imp_attribute_trailing_dot(self, provider, mock_server):
        """`@.imp.json.` — empty prefix still asks the backend."""
        mock_server.python_delegate.get_completions = AsyncMock(return_value=[])
        await provider._get_imp_attribute_completions("@.imp.json.", None)
        synthetic, _, col, _ = mock_server.python_delegate.get_completions.await_args.args
        assert synthetic == "import json\njson."
        assert col == len("json.")

    @pytest.mark.asyncio
    async def test_imp_attribute_skipped_for_module_level(self, provider, mock_server):
        """`@.imp.js` is module-name level — backend not called."""
        mock_server.python_delegate.get_completions = AsyncMock(return_value=[])
        result = await provider._get_imp_attribute_completions("@.imp.js", None)
        assert result == []
        mock_server.python_delegate.get_completions.assert_not_called()

    @pytest.mark.asyncio
    async def test_imp_attribute_skipped_without_imp(self, provider, mock_server):
        """`@.env.HOME` does not invoke the @.imp deep path."""
        mock_server.python_delegate.get_completions = AsyncMock(return_value=[])
        result = await provider._get_imp_attribute_completions("@.env.HOME", None)
        assert result == []
        mock_server.python_delegate.get_completions.assert_not_called()

    @pytest.mark.asyncio
    async def test_imp_attribute_rejects_bad_identifiers(self, provider, mock_server):
        """Non-identifier segments are rejected without calling the backend."""
        mock_server.python_delegate.get_completions = AsyncMock(return_value=[])
        result = await provider._get_imp_attribute_completions("@.imp.123.foo", None)
        assert result == []
        mock_server.python_delegate.get_completions.assert_not_called()

    @pytest.mark.asyncio
    async def test_imp_attribute_swallows_backend_errors(self, provider, mock_server):
        """Backend exceptions don't bubble up."""
        mock_server.python_delegate.get_completions = AsyncMock(side_effect=RuntimeError("boom"))
        result = await provider._get_imp_attribute_completions("@.imp.json.lo", None)
        assert result == []


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


class TestEnvVarMethodCompletions:
    """Test $VAR. method completions."""

    @pytest.fixture
    def mock_server(self):
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_server):
        return XonshCompletionProvider(mock_server)

    def test_path_dot_returns_envpath_methods(self, provider):
        """$PATH. should return EnvPath methods like append, prepend."""
        completions = provider._get_env_var_method_completions("$PATH.")
        labels = [c.label for c in completions]
        assert "append" in labels
        assert "prepend" in labels
        assert "insert" in labels
        assert "remove" in labels
        assert "add" in labels

    def test_path_dot_partial_filters(self, provider):
        """$PATH.ap should still return completions (no filtering here)."""
        completions = provider._get_env_var_method_completions("$PATH.ap")
        # Method returns all EnvPath methods regardless of partial
        assert len(completions) > 0

    def test_non_envpath_var_returns_empty(self, provider):
        """$AUTO_CD. (bool type) should return no method completions."""
        completions = provider._get_env_var_method_completions("$AUTO_CD.")
        assert completions == []

    def test_unknown_var_returns_empty(self, provider):
        """$UNKNOWN_VAR. should return no method completions."""
        completions = provider._get_env_var_method_completions("$UNKNOWN_VAR.")
        assert completions == []

    def test_no_dollar_sign_returns_empty(self, provider):
        """Text without $ should return no method completions."""
        completions = provider._get_env_var_method_completions("foo.bar")
        assert completions == []

    def test_completion_items_are_methods(self, provider):
        """Completion items should have Method kind."""
        completions = provider._get_env_var_method_completions("$PATH.")
        for item in completions:
            assert item.kind == lsp.CompletionItemKind.Method

    def test_pathext_also_returns_methods(self, provider):
        """$PATHEXT is also EnvPath type, should return methods."""
        completions = provider._get_env_var_method_completions("$PATHEXT.")
        labels = [c.label for c in completions]
        assert "append" in labels
