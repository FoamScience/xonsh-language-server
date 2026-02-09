"""Tests for the LSP proxy backend."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from lsprotocol import types as lsp
from xonsh_lsp.lsp_proxy_backend import LspProxyBackend, KNOWN_BACKENDS


class TestKnownBackends:
    """Test known backend configurations."""

    def test_pyright_in_known_backends(self):
        assert "pyright" in KNOWN_BACKENDS
        assert KNOWN_BACKENDS["pyright"] == ["pyright-langserver", "--stdio"]

    def test_basedpyright_in_known_backends(self):
        assert "basedpyright" in KNOWN_BACKENDS
        assert KNOWN_BACKENDS["basedpyright"] == ["basedpyright-langserver", "--stdio"]

    def test_pylsp_in_known_backends(self):
        assert "pylsp" in KNOWN_BACKENDS
        assert KNOWN_BACKENDS["pylsp"] == ["pylsp"]

    def test_ty_in_known_backends(self):
        assert "ty" in KNOWN_BACKENDS
        assert KNOWN_BACKENDS["ty"] == ["ty", "server"]


class TestLspProxyBackendInit:
    """Test LspProxyBackend initialization."""

    def test_init_default(self):
        backend = LspProxyBackend(command=["pyright-langserver", "--stdio"])
        assert backend._command == ["pyright-langserver", "--stdio"]
        assert backend._client is None
        assert backend._on_diagnostics is None
        assert backend._backend_settings == {}
        assert backend._server is None
        assert backend._doc_versions == {}
        assert not backend._started

    def test_init_with_settings(self):
        settings = {"python": {"pythonPath": "/usr/bin/python3"}}
        on_diag = MagicMock()
        backend = LspProxyBackend(
            command=["pylsp"],
            on_diagnostics=on_diag,
            backend_settings=settings,
        )
        assert backend._backend_settings == settings
        assert backend._on_diagnostics is on_diag


class TestLspProxyBackendNotStarted:
    """Test LspProxyBackend methods when not started."""

    @pytest.fixture
    def backend(self):
        return LspProxyBackend(command=["pyright-langserver", "--stdio"])

    @pytest.mark.asyncio
    async def test_get_completions_not_started(self, backend):
        result = await backend.get_completions("x = 1", 0, 0)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_hover_not_started(self, backend):
        result = await backend.get_hover("x = 1", 0, 0)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_definitions_not_started(self, backend):
        result = await backend.get_definitions("x = 1", 0, 0)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_references_not_started(self, backend):
        result = await backend.get_references("x = 1", 0, 0)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_diagnostics_not_started(self, backend):
        result = await backend.get_diagnostics("x = 1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_document_symbols_not_started(self, backend):
        result = await backend.get_document_symbols("x = 1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_signature_help_not_started(self, backend):
        result = await backend.get_signature_help("print(", 0, 6)
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_not_started(self, backend):
        # Should not raise
        await backend.stop()

    @pytest.mark.asyncio
    async def test_update_settings_not_started(self, backend):
        # Should not raise
        await backend.update_settings({"python": {}})


class TestLspProxyBackendDocumentSync:
    """Test document synchronization logic."""

    def test_file_uri(self):
        backend = LspProxyBackend(command=["test"])
        uri = backend._file_uri("/home/user/test.py")
        assert uri.startswith("file:///")
        assert "test.py" in uri

    def test_file_uri_none(self):
        backend = LspProxyBackend(command=["test"])
        uri = backend._file_uri(None)
        assert uri == "file:///untitled.py"

    def test_sync_document_first_open(self):
        """First sync should send didOpen."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        uri, processed = backend._sync_document("x = 1", "/test/file.py")

        backend._client.text_document_did_open.assert_called_once()
        assert uri in backend._doc_versions
        assert backend._doc_versions[uri] == 0

    def test_sync_document_subsequent_change(self):
        """Subsequent sync should send didChange."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        # First sync
        uri, _ = backend._sync_document("x = 1", "/test/file.py")
        assert backend._doc_versions[uri] == 0

        # Second sync
        backend._sync_document("x = 2", "/test/file.py")
        backend._client.text_document_did_change.assert_called_once()
        assert backend._doc_versions[uri] == 1

    def test_sync_document_preprocesses_xonsh(self):
        """Sync should preprocess xonsh syntax."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        uri, processed = backend._sync_document("print($HOME)", "/test/file.xsh")

        # The didOpen call should have preprocessed source
        call_args = backend._client.text_document_did_open.call_args[0][0]
        assert "$HOME" not in call_args.text_document.text
        assert "__xonsh_env__" in call_args.text_document.text
        assert call_args.text_document.language_id == "python"


class TestLspProxyBackendDiagnostics:
    """Test diagnostics handling."""

    def test_handle_diagnostics_with_callback(self):
        """Test that diagnostics are forwarded to callback."""
        callback = MagicMock()
        backend = LspProxyBackend(
            command=["test"],
            on_diagnostics=callback,
        )

        params = lsp.PublishDiagnosticsParams(
            uri="file:///test/file.py",
            diagnostics=[
                lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(line=0, character=0),
                        end=lsp.Position(line=0, character=5),
                    ),
                    message="test error",
                    severity=lsp.DiagnosticSeverity.Error,
                )
            ],
        )

        backend._handle_diagnostics(params)
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "file:///test/file.py"
        assert len(args[1]) == 1

    def test_handle_diagnostics_without_callback(self):
        """Test that diagnostics without callback don't raise."""
        backend = LspProxyBackend(command=["test"])

        params = lsp.PublishDiagnosticsParams(
            uri="file:///test/file.py",
            diagnostics=[],
        )

        # Should not raise
        backend._handle_diagnostics(params)


class TestLspProxyBackendConfiguration:
    """Test configuration handling."""

    @pytest.mark.asyncio
    async def test_forwards_config_to_editor(self):
        """Test that config requests are forwarded to the editor."""
        mock_server = MagicMock()
        editor_response = [{"pythonPath": "/venv/bin/python", "analysis": {}}]
        mock_server.send_request_async = AsyncMock(return_value=editor_response)

        backend = LspProxyBackend(
            command=["test"],
            server=mock_server,
        )

        params = lsp.ConfigurationParams(
            items=[lsp.ConfigurationItem(section="python")]
        )

        results = await backend._handle_configuration_request(params)
        assert results == editor_response
        mock_server.send_request_async.assert_called_once_with(
            lsp.WORKSPACE_CONFIGURATION, params
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_backend_settings(self):
        """Test fallback to backendSettings when editor doesn't respond."""
        mock_server = MagicMock()
        mock_server.send_request_async = AsyncMock(side_effect=Exception("not supported"))

        settings = {
            "python": {
                "analysis": {"autoSearchPaths": True},
                "pythonPath": "/usr/bin/python3",
            }
        }
        backend = LspProxyBackend(
            command=["test"],
            backend_settings=settings,
            server=mock_server,
        )

        params = lsp.ConfigurationParams(
            items=[
                lsp.ConfigurationItem(section="python"),
                lsp.ConfigurationItem(section="python.analysis"),
            ]
        )

        results = await backend._handle_configuration_request(params)
        assert len(results) == 2
        assert results[0] == settings["python"]
        assert results[1] == settings["python"]["analysis"]

    @pytest.mark.asyncio
    async def test_falls_back_without_server(self):
        """Test fallback when no server reference is available."""
        backend = LspProxyBackend(
            command=["test"],
            backend_settings={"key": "value"},
        )

        params = lsp.ConfigurationParams(
            items=[lsp.ConfigurationItem(section="")]
        )

        results = await backend._handle_configuration_request(params)
        assert len(results) == 1
        assert results[0] == {"key": "value"}

    def test_resolve_settings_unknown_section(self):
        """Test fallback settings resolution with unknown section."""
        backend = LspProxyBackend(
            command=["test"],
            backend_settings={},
        )

        params = lsp.ConfigurationParams(
            items=[lsp.ConfigurationItem(section="unknown.section")]
        )

        results = backend._resolve_settings(params)
        assert len(results) == 1
        assert results[0] == {}


class TestLspProxyBackendNormalizeLocations:
    """Test location normalization."""

    def test_normalize_none(self):
        backend = LspProxyBackend(command=["test"])
        result = backend._normalize_locations(None, None, None)
        assert result == []

    def test_normalize_single_location(self):
        backend = LspProxyBackend(command=["test"])
        loc = lsp.Location(
            uri="file:///other/file.py",
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=0, character=5),
            ),
        )
        result = backend._normalize_locations(loc, None, None)
        assert len(result) == 1

    def test_normalize_location_list(self):
        backend = LspProxyBackend(command=["test"])
        locs = [
            lsp.Location(
                uri="file:///file1.py",
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=5),
                ),
            ),
            lsp.Location(
                uri="file:///file2.py",
                range=lsp.Range(
                    start=lsp.Position(line=1, character=0),
                    end=lsp.Position(line=1, character=5),
                ),
            ),
        ]
        result = backend._normalize_locations(locs, None, None)
        assert len(result) == 2

    def test_normalize_location_link(self):
        backend = LspProxyBackend(command=["test"])
        link = lsp.LocationLink(
            target_uri="file:///target.py",
            target_range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=0, character=5),
            ),
            target_selection_range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=0, character=5),
            ),
        )
        result = backend._normalize_locations([link], None, None)
        assert len(result) == 1
        assert result[0].uri == "file:///target.py"
