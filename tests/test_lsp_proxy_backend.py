"""Tests for the LSP proxy backend."""

import os
import pytest
from pathlib import Path, PurePosixPath
from unittest.mock import AsyncMock, MagicMock

from lsprotocol import types as lsp
from xonsh_lsp.lsp_proxy_backend import LspProxyBackend, KNOWN_BACKENDS


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
        assert "class __xonsh_Path__:" in text
        assert backend._sync_state[uri].preamble_lines > 0

    def test_path_replaced_with_xonsh_path(self):
        """Path( calls should be replaced with __xonsh_Path__( in sync output."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True

        source = "with p'/tmp/dir'.mkdir().cd():\n    pass"
        uri, _ = backend._sync_document(source, _test_path("file.xsh"))

        text = backend._client.text_document_did_open.call_args[0][0].text_document.text
        assert "__xonsh_Path__('/tmp/dir')" in text
        # The from pathlib import Path should NOT be replaced
        assert "from pathlib import Path" in text

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
