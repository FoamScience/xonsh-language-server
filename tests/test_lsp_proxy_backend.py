"""Tests for the LSP proxy backend."""

import os
import pytest
from pathlib import Path, PurePosixPath
from unittest.mock import AsyncMock, MagicMock

from lsprotocol import types as lsp
from xonsh_lsp.lsp_proxy_backend import LspProxyBackend, KNOWN_BACKENDS, _token_in_replacement


def _test_path(name: str) -> str:
    """Return an absolute path suitable for the current platform.

    On Windows ``Path(_test_path("file.py"))`` is relative (no drive letter) and
    ``Path.as_uri()`` raises ``ValueError``.  This helper builds a proper
    absolute path that works on every OS.
    """
    if os.name == "nt":
        return f"C:\\test\\{name}"
    return f"/test/{name}"


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
    async def test_get_inlay_hints_not_started_main(self, backend):
        result = await backend.get_inlay_hints("x = 1", 0, 1)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_workspace_symbols_not_started(self, backend):
        result = await backend.get_workspace_symbols("foo")
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_workspace_symbol_not_started(self, backend):
        symbol = lsp.WorkspaceSymbol(
            name="MyClass",
            kind=lsp.SymbolKind.Class,
            location=lsp.Location(
                uri="file:///test/file.py",
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=7),
                ),
            ),
        )
        result = await backend.resolve_workspace_symbol(symbol)
        assert result is symbol

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
        uri = backend._file_uri(_test_path("test.py"))
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

        uri, processed = backend._sync_document("x = 1", _test_path("file.py"))

        backend._client.text_document_did_open.assert_called_once()
        assert uri in backend._doc_versions
        assert backend._doc_versions[uri] == 0

    def test_sync_document_subsequent_change(self):
        """Subsequent sync should send didChange."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        # First sync
        uri, _ = backend._sync_document("x = 1", _test_path("file.py"))
        assert backend._doc_versions[uri] == 0

        # Second sync
        backend._sync_document("x = 2", _test_path("file.py"))
        backend._client.text_document_did_change.assert_called_once()
        assert backend._doc_versions[uri] == 1

    def test_sync_document_preprocesses_xonsh(self):
        """Sync should preprocess xonsh syntax."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        uri, processed = backend._sync_document("print($HOME)", _test_path("file.xsh"))

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
            uri=Path(_test_path("file.py")).as_uri(),
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
        assert args[0] == Path(_test_path("file.py")).as_uri()
        assert len(args[1]) == 1

    def test_handle_diagnostics_without_callback(self):
        """Test that diagnostics without callback don't raise."""
        backend = LspProxyBackend(command=["test"])

        params = lsp.PublishDiagnosticsParams(
            uri=Path(_test_path("file.py")).as_uri(),
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


class TestLspProxyBackendProgress:
    """Test progress forwarding from child to editor."""

    @pytest.mark.asyncio
    async def test_progress_create_forwarded(self):
        """Test that window/workDoneProgress/create is forwarded to editor."""
        mock_server = MagicMock()
        mock_server.work_done_progress = MagicMock()
        mock_server.work_done_progress.create_async = AsyncMock()

        backend = LspProxyBackend(command=["test"], server=mock_server)

        params = lsp.WorkDoneProgressCreateParams(token="pyright-index-1")
        await backend._handle_progress_create(params)

        mock_server.work_done_progress.create_async.assert_called_once_with(
            "pyright-index-1"
        )

    @pytest.mark.asyncio
    async def test_progress_create_no_server(self):
        """Test that progress create without server doesn't raise."""
        backend = LspProxyBackend(command=["test"])
        params = lsp.WorkDoneProgressCreateParams(token="test-token")
        await backend._handle_progress_create(params)  # Should not raise

    def test_progress_begin_forwarded(self):
        """Test that WorkDoneProgressBegin is forwarded."""
        mock_server = MagicMock()
        mock_progress = MagicMock()
        mock_server.work_done_progress = mock_progress

        backend = LspProxyBackend(command=["test"], server=mock_server)

        value = lsp.WorkDoneProgressBegin(
            title="Indexing",
            kind="begin",
            percentage=0,
        )
        params = lsp.ProgressParams(token="pyright-index-1", value=value)
        backend._handle_progress(params)

        mock_progress.begin.assert_called_once_with("pyright-index-1", value)

    def test_progress_report_forwarded(self):
        """Test that WorkDoneProgressReport is forwarded."""
        mock_server = MagicMock()
        mock_progress = MagicMock()
        mock_server.work_done_progress = mock_progress

        backend = LspProxyBackend(command=["test"], server=mock_server)

        value = lsp.WorkDoneProgressReport(
            kind="report",
            percentage=50,
            message="50% done",
        )
        params = lsp.ProgressParams(token="pyright-index-1", value=value)
        backend._handle_progress(params)

        mock_progress.report.assert_called_once_with("pyright-index-1", value)

    def test_progress_end_forwarded(self):
        """Test that WorkDoneProgressEnd is forwarded."""
        mock_server = MagicMock()
        mock_progress = MagicMock()
        mock_server.work_done_progress = mock_progress

        backend = LspProxyBackend(command=["test"], server=mock_server)

        value = lsp.WorkDoneProgressEnd(kind="end", message="Done")
        params = lsp.ProgressParams(token="pyright-index-1", value=value)
        backend._handle_progress(params)

        mock_progress.end.assert_called_once_with("pyright-index-1", value)

    def test_progress_no_server(self):
        """Test that progress without server doesn't raise."""
        backend = LspProxyBackend(command=["test"])
        value = lsp.WorkDoneProgressBegin(title="Test", kind="begin")
        params = lsp.ProgressParams(token="test", value=value)
        backend._handle_progress(params)  # Should not raise


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


class TestLspProxyBackendPreamble:
    """Test preamble stub injection."""

    def test_preamble_added_when_xonsh_placeholders_present(self):
        """Preamble should be prepended when __xonsh_env__ etc. are used."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        source = "x = $HOME"
        uri, processed = backend._sync_document(source, _test_path("file.xsh"))

        text = backend._client.text_document_did_open.call_args[0][0].text_document.text
        assert text.startswith("import typing as __xonsh_typing__")
        assert "__xonsh_env__: dict[str, __xonsh_typing__.Any]" in text
        assert "__xonsh_subproc__: __xonsh_typing__.Any" in text
        assert "__xonsh_at__: __xonsh_typing__.Any" in text

    def test_preamble_not_added_for_pure_python(self):
        """Preamble should NOT be prepended for pure Python source."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        source = "x = 1\nprint(x)"
        uri, processed = backend._sync_document(source, _test_path("file.py"))

        text = backend._client.text_document_did_open.call_args[0][0].text_document.text
        assert "__xonsh_typing__" not in text
        assert text.startswith("x = 1")

    def test_preamble_offset_stored(self):
        """Sync state should record the preamble line count."""
        from xonsh_lsp.lsp_proxy_backend import _XONSH_PREAMBLE_LINE_COUNT

        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        source = "x = $HOME"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))

        assert uri in backend._sync_state
        assert backend._sync_state[uri].preamble_lines == _XONSH_PREAMBLE_LINE_COUNT

    def test_preamble_offset_zero_for_pure_python(self):
        """Pure Python files should have zero preamble offset."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        source = "x = 1"
        uri, _ = backend._sync_document(source, _test_path("file.py"))

        assert backend._sync_state[uri].preamble_lines == 0

    def test_preamble_added_for_path_literals(self):
        """Preamble should be added when path literals are present (no $ or subproc)."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        source = "x = p'/etc/passwd'"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))

        text = backend._client.text_document_did_open.call_args[0][0].text_document.text
        assert "__xonsh_Path__" in text
        assert "class __xonsh_Path__(" in text
        assert backend._sync_state[uri].preamble_lines > 0

    def test_path_aliased_to_xonsh_path(self):
        """Path should be aliased to __xonsh_Path__ via import replacement."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        source = "with p'/tmp/dir'.mkdir().cd():\n    pass"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))

        text = backend._client.text_document_did_open.call_args[0][0].text_document.text
        # The import should be replaced with an alias
        assert "Path = __xonsh_Path__" in text
        assert "from pathlib import Path" not in text
        # Path( should stay as Path( (no column shift)
        assert "Path('/tmp/dir')" in text

    def test_sync_document_stores_masked_lines(self):
        """Sync state should record which original lines were masked (via preprocess_result)."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        source = "import os\ncd /tmp\nx = 1"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))

        state = backend._sync_state[uri]
        assert 1 in state.preprocess_result.masked_lines  # cd /tmp
        assert 0 not in state.preprocess_result.masked_lines  # import os


class TestLspProxyBackendDiagnosticsMapping:
    """Test diagnostics position mapping and filtering."""

    def test_diagnostics_preamble_offset_subtracted(self):
        """Diagnostics positions should be adjusted for preamble lines."""
        from xonsh_lsp.lsp_proxy_backend import _XONSH_PREAMBLE_LINE_COUNT
        from xonsh_lsp.preprocessing import preprocess_with_mapping

        callback = MagicMock()
        backend = LspProxyBackend(command=["test"], on_diagnostics=callback)
        backend._client = MagicMock()
        backend._started = True

        source = "x = $HOME\nprint(x)"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))

        # Simulate backend reporting a diagnostic on line 5 (preamble(4) + line 1)
        params = lsp.PublishDiagnosticsParams(
            uri=uri,
            diagnostics=[
                lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(
                            line=_XONSH_PREAMBLE_LINE_COUNT + 1, character=0
                        ),
                        end=lsp.Position(
                            line=_XONSH_PREAMBLE_LINE_COUNT + 1, character=5
                        ),
                    ),
                    message="test error",
                )
            ],
        )

        backend._handle_diagnostics(params)
        callback.assert_called_once()
        _, diags = callback.call_args[0]
        assert len(diags) == 1
        # Should be mapped back to original line 1
        assert diags[0].range.start.line == 1

    def test_diagnostics_on_preamble_lines_filtered(self):
        """Diagnostics on preamble stub lines should be discarded."""
        callback = MagicMock()
        backend = LspProxyBackend(command=["test"], on_diagnostics=callback)
        backend._client = MagicMock()
        backend._started = True

        source = "x = $HOME"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))

        # Diagnostic on preamble line 0
        params = lsp.PublishDiagnosticsParams(
            uri=uri,
            diagnostics=[
                lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(line=0, character=0),
                        end=lsp.Position(line=0, character=5),
                    ),
                    message="preamble error",
                )
            ],
        )

        backend._handle_diagnostics(params)
        callback.assert_called_once()
        _, diags = callback.call_args[0]
        assert len(diags) == 0

    def test_diagnostics_on_masked_lines_filtered(self):
        """Diagnostics on masked xonsh lines should be discarded."""
        callback = MagicMock()
        backend = LspProxyBackend(command=["test"], on_diagnostics=callback)
        backend._client = MagicMock()
        backend._started = True

        # Line 1 (cd /tmp) will be masked
        source = "import os\ncd /tmp\nx = 1"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))
        preamble = backend._sync_state[uri].preamble_lines

        # Diagnostic on masked line (cd /tmp)
        params = lsp.PublishDiagnosticsParams(
            uri=uri,
            diagnostics=[
                lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(line=preamble + 1, character=0),
                        end=lsp.Position(line=preamble + 1, character=4),
                    ),
                    message="invalid syntax on masked line",
                )
            ],
        )

        backend._handle_diagnostics(params)
        callback.assert_called_once()
        _, diags = callback.call_args[0]
        assert len(diags) == 0

    def test_diagnostics_preserves_severity_and_code(self):
        """Mapped diagnostics should preserve all fields."""
        from xonsh_lsp.lsp_proxy_backend import _XONSH_PREAMBLE_LINE_COUNT

        callback = MagicMock()
        backend = LspProxyBackend(command=["test"], on_diagnostics=callback)
        backend._client = MagicMock()
        backend._started = True

        source = "x = $HOME"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))

        params = lsp.PublishDiagnosticsParams(
            uri=uri,
            diagnostics=[
                lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(
                            line=_XONSH_PREAMBLE_LINE_COUNT, character=0
                        ),
                        end=lsp.Position(
                            line=_XONSH_PREAMBLE_LINE_COUNT, character=1
                        ),
                    ),
                    message="test warning",
                    severity=lsp.DiagnosticSeverity.Warning,
                    code="testCode",
                    source="pyright",
                )
            ],
        )

        backend._handle_diagnostics(params)
        _, diags = callback.call_args[0]
        assert len(diags) == 1
        assert diags[0].message == "test warning"
        assert diags[0].severity == lsp.DiagnosticSeverity.Warning
        assert diags[0].code == "testCode"
        assert diags[0].source == "pyright"


class TestLspProxyBackendInlayHintsNotStarted:
    """Test inlay hint methods when backend is not started."""

    @pytest.fixture
    def backend(self):
        return LspProxyBackend(command=["pyright-langserver", "--stdio"])

    @pytest.mark.asyncio
    async def test_get_inlay_hints_not_started(self, backend):
        result = await backend.get_inlay_hints("x = 1", 0, 1)
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_inlay_hint_not_started(self, backend):
        hint = lsp.InlayHint(
            position=lsp.Position(line=0, character=5),
            label=": int",
        )
        result = await backend.resolve_inlay_hint(hint)
        assert result is hint


class TestLspProxyBackendInlayHints:
    """Test inlay hint passthrough with position mapping."""

    def _make_backend(self):
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True
        return backend

    @pytest.mark.asyncio
    async def test_pure_python_passthrough(self):
        """Inlay hints on pure Python should pass through with unchanged positions."""
        backend = self._make_backend()
        backend._client.text_document_inlay_hint_async = AsyncMock(
            return_value=[
                lsp.InlayHint(
                    position=lsp.Position(line=0, character=5),
                    label=": int",
                ),
            ]
        )

        result = await backend.get_inlay_hints("x = 1\ny = 2", 0, 1, _test_path("file.py"))

        assert len(result) == 1
        assert result[0].position.line == 0
        assert result[0].position.character == 5
        assert result[0].label == ": int"

    @pytest.mark.asyncio
    async def test_preamble_offset_subtracted(self):
        """Hint positions should have preamble offset subtracted."""
        from xonsh_lsp.lsp_proxy_backend import _XONSH_PREAMBLE_LINE_COUNT

        backend = self._make_backend()

        # Source with xonsh syntax triggers preamble
        source = "x = $HOME\ny = 2"

        # Sync document to populate state
        backend._sync_document(source, _test_path("file.xsh"))
        preamble = backend._preamble_offset(backend._file_uri(_test_path("file.xsh")))
        assert preamble == _XONSH_PREAMBLE_LINE_COUNT

        # Child returns hint on line preamble+1 (original line 1)
        backend._client.text_document_inlay_hint_async = AsyncMock(
            return_value=[
                lsp.InlayHint(
                    position=lsp.Position(line=preamble + 1, character=3),
                    label=": int",
                ),
            ]
        )

        result = await backend.get_inlay_hints(source, 0, 2, _test_path("file.xsh"))

        assert len(result) == 1
        assert result[0].position.line == 1
        assert result[0].label == ": int"

    @pytest.mark.asyncio
    async def test_preamble_hints_filtered(self):
        """Hints on preamble lines should be filtered out."""
        backend = self._make_backend()

        source = "x = $HOME"
        backend._sync_document(source, _test_path("file.xsh"))

        # Child returns hint on preamble line 0
        backend._client.text_document_inlay_hint_async = AsyncMock(
            return_value=[
                lsp.InlayHint(
                    position=lsp.Position(line=0, character=0),
                    label="preamble hint",
                ),
            ]
        )

        result = await backend.get_inlay_hints(source, 0, 1, _test_path("file.xsh"))

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_masked_line_hints_filtered(self):
        """Hints on masked xonsh lines should be filtered out."""
        backend = self._make_backend()

        # Line 1 (cd /tmp) will be masked
        source = "import os\ncd /tmp\nx = 1"
        backend._sync_document(source, _test_path("file.xsh"))
        uri = backend._file_uri(_test_path("file.xsh"))
        preamble = backend._preamble_offset(uri)

        # Child returns hint on masked line
        backend._client.text_document_inlay_hint_async = AsyncMock(
            return_value=[
                lsp.InlayHint(
                    position=lsp.Position(line=preamble + 1, character=0),
                    label="masked hint",
                ),
            ]
        )

        result = await backend.get_inlay_hints(source, 0, 3, _test_path("file.xsh"))

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_none_result_returns_empty(self):
        """None result from child should return empty list."""
        backend = self._make_backend()
        backend._client.text_document_inlay_hint_async = AsyncMock(return_value=None)

        result = await backend.get_inlay_hints("x = 1", 0, 1, _test_path("file.py"))

        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_forwards_to_child(self):
        """resolve_inlay_hint should forward to child."""
        backend = self._make_backend()
        hint = lsp.InlayHint(
            position=lsp.Position(line=0, character=5),
            label=": int",
        )
        resolved = lsp.InlayHint(
            position=lsp.Position(line=0, character=5),
            label=": int",
            tooltip="Integer type",
        )
        backend._client.inlay_hint_resolve_async = AsyncMock(return_value=resolved)

        result = await backend.resolve_inlay_hint(hint)

        backend._client.inlay_hint_resolve_async.assert_called_once_with(hint)
        assert result.tooltip == "Integer type"


class TestLspProxyBackendInlayHintRefresh:
    """Test inlay hint refresh forwarding."""

    @pytest.mark.asyncio
    async def test_refresh_forwarded_to_editor(self):
        """workspace/inlayHint/refresh should be forwarded to editor."""
        mock_server = MagicMock()
        mock_server.send_request_async = AsyncMock(return_value=None)

        backend = LspProxyBackend(command=["test"], server=mock_server)

        await backend._handle_inlay_hint_refresh()

        mock_server.send_request_async.assert_called_once_with(
            lsp.WORKSPACE_INLAY_HINT_REFRESH, None
        )

    @pytest.mark.asyncio
    async def test_refresh_no_server(self):
        """Refresh without server should not raise."""
        backend = LspProxyBackend(command=["test"])
        await backend._handle_inlay_hint_refresh()  # Should not raise


class TestLspProxyBackendWorkspaceSymbolsNotStarted:
    """Test workspace symbol methods when backend is not started."""

    @pytest.fixture
    def backend(self):
        return LspProxyBackend(command=["pyright-langserver", "--stdio"])

    @pytest.mark.asyncio
    async def test_get_workspace_symbols_not_started(self, backend):
        result = await backend.get_workspace_symbols("foo")
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_workspace_symbol_not_started(self, backend):
        symbol = lsp.WorkspaceSymbol(
            name="MyClass",
            kind=lsp.SymbolKind.Class,
            location=lsp.Location(
                uri="file:///test/file.py",
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=7),
                ),
            ),
        )
        result = await backend.resolve_workspace_symbol(symbol)
        assert result is symbol


class TestLspProxyBackendWorkspaceSymbols:
    """Test workspace symbol passthrough with URI remapping."""

    def _make_backend(self):
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True
        return backend

    @pytest.mark.asyncio
    async def test_pure_passthrough(self):
        """Workspace symbols should pass through from child."""
        backend = self._make_backend()
        child_symbols = [
            lsp.WorkspaceSymbol(
                name="my_func",
                kind=lsp.SymbolKind.Function,
                location=lsp.Location(
                    uri="file:///test/file.py",
                    range=lsp.Range(
                        start=lsp.Position(line=5, character=0),
                        end=lsp.Position(line=5, character=7),
                    ),
                ),
            ),
        ]
        backend._client.workspace_symbol_async = AsyncMock(return_value=child_symbols)

        result = await backend.get_workspace_symbols("my_func")

        assert len(result) == 1
        assert result[0].name == "my_func"
        assert result[0].kind == lsp.SymbolKind.Function
        backend._client.workspace_symbol_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_uri_remapping(self):
        """Child .py URIs should be remapped back to original .xsh URIs."""
        backend = self._make_backend()

        # Simulate a document sync that populates uri_map
        py_uri = Path(_test_path("file.py")).as_uri()
        xsh_uri = Path(_test_path("file.xsh")).as_uri()
        backend._uri_map[py_uri] = xsh_uri

        child_symbols = [
            lsp.WorkspaceSymbol(
                name="my_var",
                kind=lsp.SymbolKind.Variable,
                location=lsp.Location(
                    uri=py_uri,
                    range=lsp.Range(
                        start=lsp.Position(line=0, character=0),
                        end=lsp.Position(line=0, character=6),
                    ),
                ),
            ),
        ]
        backend._client.workspace_symbol_async = AsyncMock(return_value=child_symbols)

        result = await backend.get_workspace_symbols("my_var")

        assert len(result) == 1
        assert result[0].location.uri == xsh_uri

    @pytest.mark.asyncio
    async def test_none_result_returns_empty(self):
        """None result from child should return empty list."""
        backend = self._make_backend()
        backend._client.workspace_symbol_async = AsyncMock(return_value=None)

        result = await backend.get_workspace_symbols("anything")

        assert result == []

    @pytest.mark.asyncio
    async def test_symbol_information_converted(self):
        """SymbolInformation results should be converted to WorkspaceSymbol."""
        backend = self._make_backend()

        py_uri = Path(_test_path("file.py")).as_uri()
        xsh_uri = Path(_test_path("file.xsh")).as_uri()
        backend._uri_map[py_uri] = xsh_uri

        child_results = [
            lsp.SymbolInformation(
                name="OldClass",
                kind=lsp.SymbolKind.Class,
                location=lsp.Location(
                    uri=py_uri,
                    range=lsp.Range(
                        start=lsp.Position(line=10, character=0),
                        end=lsp.Position(line=10, character=8),
                    ),
                ),
                container_name="my_module",
            ),
        ]
        backend._client.workspace_symbol_async = AsyncMock(return_value=child_results)

        result = await backend.get_workspace_symbols("OldClass")

        assert len(result) == 1
        assert isinstance(result[0], lsp.WorkspaceSymbol)
        assert result[0].name == "OldClass"
        assert result[0].kind == lsp.SymbolKind.Class
        assert result[0].location.uri == xsh_uri
        assert result[0].container_name == "my_module"

    @pytest.mark.asyncio
    async def test_resolve_forwards_to_child(self):
        """resolve_workspace_symbol should forward to child and remap URI."""
        backend = self._make_backend()

        py_uri = Path(_test_path("file.py")).as_uri()
        xsh_uri = Path(_test_path("file.xsh")).as_uri()
        backend._uri_map[py_uri] = xsh_uri

        symbol = lsp.WorkspaceSymbol(
            name="MyClass",
            kind=lsp.SymbolKind.Class,
            location=lsp.Location(
                uri=xsh_uri,
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=0),
                ),
            ),
        )
        resolved = lsp.WorkspaceSymbol(
            name="MyClass",
            kind=lsp.SymbolKind.Class,
            location=lsp.Location(
                uri=py_uri,
                range=lsp.Range(
                    start=lsp.Position(line=5, character=6),
                    end=lsp.Position(line=5, character=13),
                ),
            ),
        )
        backend._client.workspace_symbol_resolve_async = AsyncMock(return_value=resolved)

        result = await backend.resolve_workspace_symbol(symbol)

        backend._client.workspace_symbol_resolve_async.assert_called_once_with(symbol)
        assert result.location.uri == xsh_uri
        assert result.location.range.start.line == 5

    @pytest.mark.asyncio
    async def test_resolve_no_remap_needed(self):
        """resolve should work when no URI remapping is needed."""
        backend = self._make_backend()

        symbol = lsp.WorkspaceSymbol(
            name="func",
            kind=lsp.SymbolKind.Function,
            location=lsp.Location(
                uri="file:///test/plain.py",
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=4),
                ),
            ),
        )
        backend._client.workspace_symbol_resolve_async = AsyncMock(return_value=symbol)

        result = await backend.resolve_workspace_symbol(symbol)

        assert result.location.uri == "file:///test/plain.py"


class TestLspProxyBackendSemanticTokensNotStarted:
    """Test semantic token methods when backend is not started."""

    @pytest.fixture
    def backend(self):
        return LspProxyBackend(command=["pyright-langserver", "--stdio"])

    @pytest.mark.asyncio
    async def test_get_semantic_tokens_not_started(self, backend):
        result = await backend.get_semantic_tokens("x = 1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_semantic_tokens_range_not_started(self, backend):
        result = await backend.get_semantic_tokens_range("x = 1", 0, 0, 0, 5)
        assert result is None


class TestLspProxyBackendSemanticTokens:
    """Test semantic token passthrough with position mapping and legend remap."""

    def _make_backend(self):
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True
        return backend

    @pytest.mark.asyncio
    async def test_pure_python_passthrough(self):
        """Semantic tokens on pure Python should pass through with unchanged positions."""
        backend = self._make_backend()

        # Delta-encoded tokens: (line=0, char=0, len=1, type=0, mods=0),
        #                        (line=0, char=4, len=1, type=0, mods=0)
        child_data = [0, 0, 1, 0, 0, 0, 4, 1, 0, 0]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens("x = 1", _test_path("file.py"))

        assert result is not None
        assert result.data == child_data

    @pytest.mark.asyncio
    async def test_preamble_offset_subtracted(self):
        """Token positions should have preamble offset subtracted."""
        from xonsh_lsp.lsp_proxy_backend import _XONSH_PREAMBLE_LINE_COUNT
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES

        backend = self._make_backend()

        source = "x = $HOME\ny = 2"
        backend._sync_document(source, _test_path("file.xsh"))
        uri = backend._file_uri(_test_path("file.xsh"))
        preamble = backend._preamble_offset(uri)
        assert preamble == _XONSH_PREAMBLE_LINE_COUNT

        # Token on child line preamble+1, char=0, len=1
        child_data = [preamble + 1, 0, 1, 0, 0]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens(source, _test_path("file.xsh"))

        assert result is not None
        # 2 tokens: synthetic $HOME (line 0) + y (line 1)
        assert len(result.data) == 10
        # First token: synthetic $HOME at line 0, col 4
        var_idx = SEMANTIC_TOKEN_TYPES.index("variable")
        assert result.data[0] == 0  # line 0
        assert result.data[1] == 4  # col 4
        assert result.data[2] == 5  # len("$HOME")
        assert result.data[3] == var_idx
        # Second token: y at line 1
        assert result.data[5] == 1  # delta_line = 1

    @pytest.mark.asyncio
    async def test_preamble_tokens_filtered(self):
        """Tokens on preamble lines should be filtered out, but synthetic tokens still emitted."""
        backend = self._make_backend()
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES

        source = "x = $HOME"
        backend._sync_document(source, _test_path("file.xsh"))

        # Token on preamble line 0 — should be filtered
        child_data = [0, 0, 5, 0, 0]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens(source, _test_path("file.xsh"))

        assert result is not None
        # Preamble token filtered, but synthetic $HOME token emitted
        var_idx = SEMANTIC_TOKEN_TYPES.index("variable")
        assert len(result.data) == 5
        assert result.data[0] == 0  # line 0
        assert result.data[1] == 4  # col 4 ($HOME starts at col 4)
        assert result.data[2] == 5  # len("$HOME")
        assert result.data[3] == var_idx

    @pytest.mark.asyncio
    async def test_masked_line_tokens_filtered(self):
        """Tokens on masked xonsh lines should be filtered out."""
        backend = self._make_backend()

        # Line 1 (cd /tmp) will be masked
        source = "import os\ncd /tmp\nx = 1"
        backend._sync_document(source, _test_path("file.xsh"))
        uri = backend._file_uri(_test_path("file.xsh"))
        preamble = backend._preamble_offset(uri)

        # Token on masked line (preamble + 1)
        child_data = [preamble + 1, 0, 4, 0, 0]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens(source, _test_path("file.xsh"))

        assert result is not None
        assert result.data == []

    @pytest.mark.asyncio
    async def test_none_result_returns_none(self):
        """None result from child should return None."""
        backend = self._make_backend()
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=None
        )

        result = await backend.get_semantic_tokens("x = 1", _test_path("file.py"))

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_data_returns_empty(self):
        """Empty data from child should return empty data."""
        backend = self._make_backend()
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=[])
        )

        result = await backend.get_semantic_tokens("x = 1", _test_path("file.py"))

        assert result is not None
        assert result.data == []

    @pytest.mark.asyncio
    async def test_delta_encoding_output(self):
        """Output should be correctly re-encoded in delta format."""
        backend = self._make_backend()

        # Two tokens on different lines: (line=0, char=0) and (line=2, char=4)
        # Delta: [0,0,1,0,0, 2,4,1,0,0]
        child_data = [0, 0, 1, 0, 0, 2, 4, 1, 0, 0]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens("x = 1\n\ny = 2", _test_path("file.py"))

        assert result is not None
        # Should preserve delta encoding
        assert result.data == [0, 0, 1, 0, 0, 2, 4, 1, 0, 0]

    @pytest.mark.asyncio
    async def test_delta_encoding_same_line(self):
        """Tokens on the same line should have delta_char relative to previous."""
        backend = self._make_backend()

        # Two tokens on same line: (line=0, char=0, len=1) and (line=0, char=4, len=1)
        # Input delta: [0,0,1,0,0, 0,4,1,0,0]
        child_data = [0, 0, 1, 0, 0, 0, 4, 1, 0, 0]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens("x = y", _test_path("file.py"))

        assert result is not None
        assert result.data == [0, 0, 1, 0, 0, 0, 4, 1, 0, 0]

    @pytest.mark.asyncio
    async def test_legend_type_remapping(self):
        """Child token type indices should be remapped to our legend."""
        backend = self._make_backend()

        # Set up a remap where child type 0 maps to our type 5
        backend._semantic_type_remap = [5, 2, 0]

        # Token with type=0 in child's legend
        child_data = [0, 0, 3, 0, 0]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens("foo", _test_path("file.py"))

        assert result is not None
        # Type should be remapped from 0 -> 5
        assert result.data[3] == 5

    @pytest.mark.asyncio
    async def test_legend_modifier_remapping(self):
        """Child modifier bitmask should be remapped to our legend."""
        backend = self._make_backend()

        # Child bit 0 -> our bit 2, child bit 1 -> our bit 0
        backend._semantic_modifier_remap = [2, 0]

        # Token with modifier bitmask 0b01 (child bit 0 set)
        child_data = [0, 0, 3, 0, 1]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens("foo", _test_path("file.py"))

        assert result is not None
        # Bit 0 in child -> bit 2 in ours = 0b100 = 4
        assert result.data[4] == 4

    @pytest.mark.asyncio
    async def test_modifier_remap_multiple_bits(self):
        """Multiple modifier bits should be remapped independently."""
        backend = self._make_backend()

        # Child bit 0 -> our bit 2, child bit 1 -> our bit 0
        backend._semantic_modifier_remap = [2, 0]

        # Both bits set: 0b11 = 3
        child_data = [0, 0, 3, 0, 3]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens("foo", _test_path("file.py"))

        assert result is not None
        # bit0->bit2 (4) + bit1->bit0 (1) = 5
        assert result.data[4] == 5

    @pytest.mark.asyncio
    async def test_replacement_region_gets_synthetic_token(self):
        """Tokens in replacement regions should be replaced by a single synthetic token."""
        backend = self._make_backend()

        # $PATH.append('/tmp')
        # Preprocessed: __xonsh_env__["PATH"].append('/tmp')
        source = "$PATH.append('/tmp')"
        backend._sync_document(source, _test_path("file.xsh"))

        # __xonsh_env__ at col 0, len 13 — inside replacement
        # "PATH" at col 14, len 6 — inside replacement
        # append at col 22, len 6 — outside replacement
        # '/tmp' at col 29, len 6 — outside replacement
        from xonsh_lsp.lsp_proxy_backend import _XONSH_PREAMBLE_LINE_COUNT
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES
        uri = backend._file_uri(_test_path("file.xsh"))
        preamble = backend._preamble_offset(uri)
        child_data = [
            preamble, 0, 13, 8, 0,     # __xonsh_env__ (in replacement)
            0, 14, 6, 9, 0,            # "PATH" (in replacement, delta=14)
            0, 8, 6, 6, 0,             # append (delta=22-14=8)
            0, 7, 6, 9, 0,             # '/tmp' (delta=29-22=7)
        ]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens(source, _test_path("file.xsh"))

        assert result is not None
        # Should have 3 tokens: synthetic $PATH + append + '/tmp'
        assert len(result.data) == 15  # 3 tokens * 5 values each

        # First token: synthetic $PATH at col 0, length 5
        var_idx = SEMANTIC_TOKEN_TYPES.index("variable")
        assert result.data[0] == 0  # delta line
        assert result.data[1] == 0  # delta char (col 0)
        assert result.data[2] == 5  # length of "$PATH"
        assert result.data[3] == var_idx  # "variable" type

        # Second token: append at original col 6
        assert result.data[5] == 0  # same line
        assert result.data[6] == 6  # delta char from 0 → 6
        assert result.data[7] == 6  # length of "append"

    @pytest.mark.asyncio
    async def test_synthetic_tokens_for_multiple_replacements(self):
        """Multiple xonsh constructs on one line each get a synthetic token."""
        backend = self._make_backend()

        # x = $HOME + $PATH
        source = "x = $HOME + $PATH"
        backend._sync_document(source, _test_path("file.xsh"))

        from xonsh_lsp.lsp_proxy_backend import _XONSH_PREAMBLE_LINE_COUNT
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES
        uri = backend._file_uri(_test_path("file.xsh"))
        preamble = backend._preamble_offset(uri)

        # Child sends only replacement tokens (pyright might highlight __xonsh_env__)
        child_data = [
            preamble, 4, 13, 8, 0,   # __xonsh_env__ for $HOME (in replacement)
            0, 28, 13, 8, 0,          # __xonsh_env__ for $PATH (in replacement)
        ]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens(source, _test_path("file.xsh"))

        assert result is not None
        var_idx = SEMANTIC_TOKEN_TYPES.index("variable")

        # Should have 2 synthetic tokens: $HOME and $PATH
        assert len(result.data) == 10
        # First: $HOME at col 4, len 5
        assert result.data[1] == 4
        assert result.data[2] == 5
        assert result.data[3] == var_idx
        # Second: $PATH at col 12, len 5
        assert result.data[6] == 8  # delta_char = 12 - 4 = 8
        assert result.data[7] == 5
        assert result.data[8] == var_idx

    @pytest.mark.asyncio
    async def test_path_string_highlighting_correct_columns(self):
        """p'/etc/passwd'.read_text().find('root') should have correct token positions."""
        backend = self._make_backend()
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES

        source = "p'/etc/passwd'.read_text().find('root')"
        backend._sync_document(source, _test_path("file.xsh"))
        uri = backend._file_uri(_test_path("file.xsh"))
        preamble = backend._preamble_offset(uri)
        pp = backend._sync_state[uri].preprocess_result

        # Verify no column shift: child sees Path( not __xonsh_Path__(
        text = backend._client.text_document_did_open.call_args[0][0].text_document.text
        child_lines = text.split("\n")
        source_line = child_lines[preamble + pp.added_lines_before]
        assert source_line.startswith("Path(")  # NOT __xonsh_Path__(

        # Child tokens (based on Path('/etc/passwd').read_text().find('root')):
        # Path at col 0 (in replacement), read_text at col 20, find at col 32, 'root' at col 37
        child_line = preamble + pp.added_lines_before
        child_data = [
            child_line, 0, 4, 0, 0,     # Path (in replacement)
            0, 5, 13, 18, 0,            # '/etc/passwd' at col 5, len 13, type string
            0, 15, 9, 13, 0,            # read_text at col 20, len 9, type method (delta=15)
            0, 12, 4, 13, 0,            # find at col 32, len 4, type method (delta=12)
            0, 5, 6, 18, 0,             # 'root' at col 37, len 6, type string (delta=5)
        ]
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens(source, _test_path("file.xsh"))

        assert result is not None
        string_idx = SEMANTIC_TOKEN_TYPES.index("string")
        method_idx = SEMANTIC_TOKEN_TYPES.index("method")

        # Decode tokens to check positions
        tokens = []
        line, char = 0, 0
        for i in range(0, len(result.data), 5):
            dl, dc, length, tt, tm = result.data[i:i+5]
            if dl > 0:
                line += dl
                char = dc
            else:
                char += dc
            tokens.append((line, char, length, tt))

        # Should have: synthetic path_string + read_text + find + 'root'
        # (Path and '/etc/passwd' are in replacement → filtered from child,
        #  but synthetic token emitted for p'/etc/passwd')
        assert len(tokens) >= 3

        # Synthetic token for p'/etc/passwd': line 0, col 0, len 14, type string
        assert tokens[0] == (0, 0, 14, string_idx)

        # read_text: should be at original col 15 (.read_text starts at col 14, r at 15)
        assert tokens[1][0] == 0  # same line
        assert tokens[1][1] == 15  # original col of read_text
        assert tokens[1][2] == 9  # length
        assert tokens[1][3] == method_idx

        # find: should be at original col 27 (.find starts at col 26, f at 27)
        assert tokens[2][0] == 0
        assert tokens[2][1] == 27
        assert tokens[2][2] == 4
        assert tokens[2][3] == method_idx

        # 'root': should be at original col 32
        assert tokens[3][0] == 0
        assert tokens[3][1] == 32
        assert tokens[3][2] == 6
        assert tokens[3][3] == string_idx

    @pytest.mark.asyncio
    async def test_token_length_remapped(self):
        """Token lengths should be recomputed in original coordinates."""
        backend = self._make_backend()

        # Pure python: positions are 1:1 so length stays the same
        child_data = [0, 0, 5, 0, 0]  # 5-char token at col 0
        backend._client.text_document_semantic_tokens_full_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens("hello = 1", _test_path("file.py"))

        assert result is not None
        assert result.data[2] == 5  # length preserved for 1:1 mapping


class TestLspProxyBackendSemanticTokensRange:
    """Test semantic token range request."""

    def _make_backend(self):
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True
        return backend

    @pytest.mark.asyncio
    async def test_range_positions_mapped(self):
        """Range positions should be mapped to preprocessed coordinates."""
        backend = self._make_backend()

        child_data = [0, 0, 1, 0, 0]
        backend._client.text_document_semantic_tokens_range_async = AsyncMock(
            return_value=lsp.SemanticTokens(data=child_data)
        )

        result = await backend.get_semantic_tokens_range(
            "x = 1\ny = 2", 0, 0, 1, 5, _test_path("file.py")
        )

        assert result is not None
        backend._client.text_document_semantic_tokens_range_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_range_none_result(self):
        """None result from child range request should return None."""
        backend = self._make_backend()
        backend._client.text_document_semantic_tokens_range_async = AsyncMock(
            return_value=None
        )

        result = await backend.get_semantic_tokens_range(
            "x = 1", 0, 0, 0, 5, _test_path("file.py")
        )

        assert result is None


class TestLspProxyBackendSemanticTokensRefresh:
    """Test semantic tokens refresh forwarding."""

    @pytest.mark.asyncio
    async def test_refresh_forwarded_to_editor(self):
        """workspace/semanticTokens/refresh should be forwarded to editor."""
        mock_server = MagicMock()
        mock_server.send_request_async = AsyncMock(return_value=None)

        backend = LspProxyBackend(command=["test"], server=mock_server)

        await backend._handle_semantic_tokens_refresh()

        mock_server.send_request_async.assert_called_once_with(
            lsp.WORKSPACE_SEMANTIC_TOKENS_REFRESH, None
        )

    @pytest.mark.asyncio
    async def test_refresh_no_server(self):
        """Refresh without server should not raise."""
        backend = LspProxyBackend(command=["test"])
        await backend._handle_semantic_tokens_refresh()  # Should not raise


class TestLspProxyBackendSemanticLegendRemap:
    """Test semantic token legend remap building."""

    def test_matching_legend_is_identity(self):
        """When child legend matches ours, remap should be identity."""
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES, SEMANTIC_TOKEN_MODIFIERS

        backend = LspProxyBackend(command=["test"])

        init_result = lsp.InitializeResult(
            capabilities=lsp.ServerCapabilities(
                semantic_tokens_provider=lsp.SemanticTokensOptions(
                    legend=lsp.SemanticTokensLegend(
                        token_types=SEMANTIC_TOKEN_TYPES,
                        token_modifiers=SEMANTIC_TOKEN_MODIFIERS,
                    ),
                    full=True,
                ),
            ),
        )

        backend._build_semantic_legend_remap(init_result)

        # Identity: each index maps to itself
        assert backend._semantic_type_remap == list(range(len(SEMANTIC_TOKEN_TYPES)))
        assert backend._semantic_modifier_remap == list(range(len(SEMANTIC_TOKEN_MODIFIERS)))

    def test_subset_legend_remaps_correctly(self):
        """When child has a subset of types, they should map to correct indices."""
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES

        backend = LspProxyBackend(command=["test"])

        child_types = ["function", "variable", "class"]
        init_result = lsp.InitializeResult(
            capabilities=lsp.ServerCapabilities(
                semantic_tokens_provider=lsp.SemanticTokensOptions(
                    legend=lsp.SemanticTokensLegend(
                        token_types=child_types,
                        token_modifiers=[],
                    ),
                    full=True,
                ),
            ),
        )

        backend._build_semantic_legend_remap(init_result)

        assert len(backend._semantic_type_remap) == 3
        assert backend._semantic_type_remap[0] == SEMANTIC_TOKEN_TYPES.index("function")
        assert backend._semantic_type_remap[1] == SEMANTIC_TOKEN_TYPES.index("variable")
        assert backend._semantic_type_remap[2] == SEMANTIC_TOKEN_TYPES.index("class")

    def test_no_provider_gives_empty_arrays(self):
        """When child has no semantic tokens provider, remap arrays should be empty."""
        backend = LspProxyBackend(command=["test"])

        init_result = lsp.InitializeResult(
            capabilities=lsp.ServerCapabilities(),
        )

        backend._build_semantic_legend_remap(init_result)

        assert backend._semantic_type_remap == []
        assert backend._semantic_modifier_remap == []

    def test_unknown_types_default_to_zero(self):
        """Unknown type names in child legend should default to index 0."""
        backend = LspProxyBackend(command=["test"])

        init_result = lsp.InitializeResult(
            capabilities=lsp.ServerCapabilities(
                semantic_tokens_provider=lsp.SemanticTokensOptions(
                    legend=lsp.SemanticTokensLegend(
                        token_types=["unknownCustomType"],
                        token_modifiers=["unknownCustomMod"],
                    ),
                    full=True,
                ),
            ),
        )

        backend._build_semantic_legend_remap(init_result)

        assert backend._semantic_type_remap == [0]
        assert backend._semantic_modifier_remap == [0]


class TestTokenInReplacement:
    """Test _token_in_replacement helper."""

    def test_token_in_replacement_region(self):
        """Token within a non-1:1 segment should be detected."""
        # Mapping: orig 0-4 -> proc 0-20, orig 5-20 -> proc 21-36
        mapping = [(0, 0), (5, 21), (20, 36)]
        # Token at proc 0-13 (within the 0-21 replacement)
        assert _token_in_replacement(mapping, 0, 13) is True

    def test_token_in_identity_region(self):
        """Token within a 1:1 segment should not be flagged."""
        # Mapping: orig 0-4 -> proc 0-20, orig 5-20 -> proc 21-36
        mapping = [(0, 0), (5, 21), (20, 36)]
        # Token at proc 22-28 (within the 1:1 region)
        assert _token_in_replacement(mapping, 22, 28) is False

    def test_empty_mapping(self):
        """Empty mapping should return False."""
        assert _token_in_replacement([], 0, 5) is False

    def test_token_spanning_segments(self):
        """Token spanning across segments should return False (not fully in one)."""
        mapping = [(0, 0), (5, 21), (20, 36)]
        # Token from proc 15 to 25 (spans replacement and identity)
        assert _token_in_replacement(mapping, 15, 25) is False
