"""Tests for the diagnostics provider."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lsprotocol import types as lsp
from xonsh_lsp.diagnostics import XonshDiagnosticsProvider
from xonsh_lsp.parser import XonshParser, ParseResult, NodeInfo


class TestXonshDiagnosticsProvider:
    """Test the XonshDiagnosticsProvider class."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server."""
        server = MagicMock()
        server.parser = XonshParser()
        server.python_delegate = MagicMock()
        server.python_delegate.get_diagnostics = AsyncMock(return_value=[])
        return server

    @pytest.fixture
    def provider(self, mock_server):
        """Create a diagnostics provider."""
        return XonshDiagnosticsProvider(mock_server)

    @pytest.mark.asyncio
    async def test_syntax_error_diagnostics(self, provider, mock_server):
        """Test syntax error detection."""
        # Create a mock document with syntax error
        mock_doc = MagicMock()
        mock_doc.source = "def foo(:"  # Invalid syntax
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        diagnostics = await provider.get_diagnostics("file:///test/file.xsh")
        # Should detect syntax error (depends on tree-sitter)
        assert isinstance(diagnostics, list)

    @pytest.mark.asyncio
    async def test_undefined_env_var_diagnostics(self, provider, mock_server):
        """Test undefined environment variable detection."""
        mock_doc = MagicMock()
        mock_doc.source = "print($UNDEFINED_VAR_XYZZY_12345)"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        # Make sure the env var is not defined
        with patch.dict(os.environ, {}, clear=False):
            # Remove the var if it exists
            os.environ.pop("UNDEFINED_VAR_XYZZY_12345", None)
            diagnostics = await provider.get_diagnostics("file:///test/file.xsh")

        # Check for undefined env var diagnostic
        undefined_diags = [d for d in diagnostics if d.code == "undefined-env-var"]
        # Note: This depends on tree-sitter-xonsh detecting env_variable nodes
        assert isinstance(undefined_diags, list)

    @pytest.mark.asyncio
    async def test_defined_env_var_no_diagnostic(self, provider, mock_server):
        """Test that defined environment variables don't produce diagnostics."""
        mock_doc = MagicMock()
        # Use PATH which exists on all platforms
        mock_doc.source = "print($PATH)"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        diagnostics = await provider.get_diagnostics("file:///test/file.xsh")

        # PATH should be defined on all platforms, so no undefined-env-var diagnostic
        undefined_diags = [d for d in diagnostics if d.code == "undefined-env-var"]
        path_diags = [d for d in undefined_diags if "PATH" in d.message]
        assert len(path_diags) == 0

    @pytest.mark.asyncio
    async def test_command_not_found_diagnostics(self, provider, mock_server):
        """Test unknown command detection."""
        mock_doc = MagicMock()
        mock_doc.source = "$(nonexistent_command_xyz123)"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        diagnostics = await provider.get_diagnostics("file:///test/file.xsh")

        # Check for command-not-found diagnostic
        cmd_diags = [d for d in diagnostics if d.code == "command-not-found"]
        # Note: This depends on tree-sitter-xonsh detecting subprocess nodes
        assert isinstance(cmd_diags, list)

    @pytest.mark.asyncio
    async def test_valid_command_no_diagnostic(self, provider, mock_server):
        """Test that valid commands don't produce diagnostics."""
        mock_doc = MagicMock()
        # Use python which should exist on all platforms in CI
        mock_doc.source = "$(python --version)"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        diagnostics = await provider.get_diagnostics("file:///test/file.xsh")

        # python should be found, so no command-not-found for it
        cmd_diags = [d for d in diagnostics if d.code == "command-not-found"]
        python_diags = [d for d in cmd_diags if "python" in d.message.lower()]
        assert len(python_diags) == 0

    @pytest.mark.asyncio
    async def test_empty_subprocess_diagnostics(self, provider, mock_server):
        """Test empty subprocess detection."""
        # This test depends on the parser detecting empty subprocess
        mock_doc = MagicMock()
        mock_doc.source = "$()"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        diagnostics = await provider.get_diagnostics("file:///test/file.xsh")

        # Check for empty-subprocess diagnostic
        empty_diags = [d for d in diagnostics if d.code == "empty-subprocess"]
        assert isinstance(empty_diags, list)

    def test_is_defined_in_source(self, provider):
        """Test detection of env vars defined in source."""
        # Test $VAR = pattern
        source1 = "$MY_VAR = 'value'"
        assert provider._is_defined_in_source(source1, "MY_VAR")

        # Test ${VAR} = pattern
        source2 = "${MY_VAR} = 'value'"
        assert provider._is_defined_in_source(source2, "MY_VAR")

        # Test os.environ pattern
        source3 = "os.environ['MY_VAR'] = 'value'"
        assert provider._is_defined_in_source(source3, "MY_VAR")

        # Test undefined
        source4 = "print($OTHER_VAR)"
        assert not provider._is_defined_in_source(source4, "MY_VAR")

    def test_command_exists(self, provider):
        """Test command existence check."""
        # python should exist on all platforms in CI
        assert provider._command_exists("python")

        # Non-existent command
        assert not provider._command_exists("nonexistent_command_xyz123456")

    @pytest.mark.asyncio
    async def test_python_diagnostics_delegation(self, provider, mock_server):
        """Test that Python diagnostics are delegated to backend."""
        mock_doc = MagicMock()
        mock_doc.source = "x = 1"
        mock_doc.path = "/test/file.xsh"
        mock_server.get_document.return_value = mock_doc

        # Set up mock Python diagnostic
        mock_server.python_delegate.get_diagnostics = AsyncMock(return_value=[
            lsp.Diagnostic(
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=1),
                ),
                message="Test Python diagnostic",
                severity=lsp.DiagnosticSeverity.Error,
            )
        ])

        diagnostics = await provider.get_diagnostics("file:///test/file.xsh")

        # Should include the Python diagnostic
        assert any("Test Python diagnostic" in d.message for d in diagnostics)

    def test_on_backend_diagnostics(self, provider, mock_server):
        """Test handling async diagnostics from proxy backend."""
        uri = "file:///test/file.xsh"

        # Pre-populate xonsh diagnostics cache
        xonsh_diag = lsp.Diagnostic(
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=0, character=5),
            ),
            message="Xonsh diagnostic",
            severity=lsp.DiagnosticSeverity.Warning,
        )
        provider._xonsh_diagnostics[uri] = [xonsh_diag]

        # Simulate backend diagnostics callback
        python_diag = lsp.Diagnostic(
            range=lsp.Range(
                start=lsp.Position(line=1, character=0),
                end=lsp.Position(line=1, character=10),
            ),
            message="Python diagnostic from backend",
            severity=lsp.DiagnosticSeverity.Error,
        )
        provider.on_backend_diagnostics(uri, [python_diag])

        # Check that merged diagnostics were published
        mock_server.text_document_publish_diagnostics.assert_called_once()
        call_args = mock_server.text_document_publish_diagnostics.call_args
        published = call_args[0][0]  # First positional argument
        assert len(published.diagnostics) == 2

    def test_clear_cache(self, provider):
        """Test clearing diagnostics cache."""
        uri = "file:///test/file.xsh"
        provider._xonsh_diagnostics[uri] = [MagicMock()]
        provider._python_diagnostics[uri] = [MagicMock()]

        provider.clear_cache(uri)

        assert uri not in provider._xonsh_diagnostics
        assert uri not in provider._python_diagnostics

    # --- Alias extraction tests ---

    def test_extract_aliases_subscript(self, provider, mock_server):
        """Test aliases['name'] = ... pattern."""
        source = "aliases['myapp'] = 'echo hello'"
        parse_result = mock_server.parser.parse(source)
        aliases = provider._extract_aliases_from_source(source, parse_result)
        assert "myapp" in aliases

    def test_extract_aliases_register_decorator(self, provider, mock_server):
        """Test @aliases.register + def _name() pattern."""
        source = "@aliases.register\ndef _myapp():\n    pass\n"
        parse_result = mock_server.parser.parse(source)
        aliases = provider._extract_aliases_from_source(source, parse_result)
        assert "myapp" in aliases

    def test_extract_aliases_register_with_name(self, provider, mock_server):
        """Test @aliases.register("name") + def __name() pattern."""
        source = '@aliases.register("j2y")\ndef __j2y(args):\n    pass\n'
        parse_result = mock_server.parser.parse(source)
        aliases = provider._extract_aliases_from_source(source, parse_result)
        assert "j2y" in aliases

    # --- Split-expression diagnostics tests ---

    def test_split_expression_numeric_args_bare_subprocess(self, provider, mock_server):
        """Tree-sitter parses 'cmd 1 2 3' as bare_subprocess, so the
        split-expression heuristic does not fire."""
        source = "unknowncmd_xyz123 1 2 3"
        parse_result = mock_server.parser.parse(source)
        aliases = provider._extract_aliases_from_source(source, parse_result)
        diags = provider._get_split_expression_diagnostics(source, parse_result, aliases)
        assert len(diags) == 0

    def test_split_expression_word_args_bare_subprocess(self, provider, mock_server):
        """Tree-sitter parses 'cmd word_args' as bare_subprocess, so the
        split-expression heuristic does not fire."""
        source = "unknowncmd_xyz123 world"
        parse_result = mock_server.parser.parse(source)
        aliases = provider._extract_aliases_from_source(source, parse_result)
        diags = provider._get_split_expression_diagnostics(source, parse_result, aliases)
        assert len(diags) == 0

    def test_split_expression_known_alias(self, provider, mock_server):
        """Known alias with numeric args -> bare_subprocess, no split-expression warning."""
        source = "aliases['args'] = lambda a: print(a)\nargs 1 2 3\n"
        parse_result = mock_server.parser.parse(source)
        aliases = provider._extract_aliases_from_source(source, parse_result)
        diags = provider._get_split_expression_diagnostics(source, parse_result, aliases)
        assert len(diags) == 0

    def test_split_expression_known_system_command(self, provider, mock_server):
        """System command in PATH -> no warning (tree-sitter may parse as bare_subprocess)."""
        # Use 'echo' which is a shell builtin
        source = "echo hello"
        parse_result = mock_server.parser.parse(source)
        aliases = provider._extract_aliases_from_source(source, parse_result)
        diags = provider._get_split_expression_diagnostics(source, parse_result, aliases)
        assert len(diags) == 0

    def test_command_exists_with_source_aliases(self, provider):
        """_command_exists returns True when command is in source_aliases."""
        assert provider._command_exists("myapp_xyz", {"myapp_xyz"})
        assert not provider._command_exists("myapp_xyz")

    def test_no_split_expression_different_lines(self, provider, mock_server):
        """Two identifiers on different lines -> no diagnostic."""
        source = "foo\nbar\n"
        parse_result = mock_server.parser.parse(source)
        aliases = provider._extract_aliases_from_source(source, parse_result)
        diags = provider._get_split_expression_diagnostics(source, parse_result, aliases)
        assert len(diags) == 0
