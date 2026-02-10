"""
LSP proxy backend for xonsh LSP.

This module implements the PythonBackend protocol by delegating to a child
LSP server (e.g. Pyright, basedpyright, pylsp). It spawns the child process,
manages document synchronization, forwards requests with position mapping,
and handles asynchronous diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from lsprotocol import types as lsp
from pygls.lsp.client import LanguageClient

from xonsh_lsp.preprocessing import (
    PreprocessResult,
    map_position_from_processed,
    map_position_to_processed,
    preprocess_with_mapping,
)

if TYPE_CHECKING:
    from pygls.lsp.server import LanguageServer

logger = logging.getLogger(__name__)

# Known backend shortcuts
KNOWN_BACKENDS: dict[str, list[str]] = {
    "pyright": ["pyright-langserver", "--stdio"],
    "basedpyright": ["basedpyright-langserver", "--stdio"],
    "pylsp": ["pylsp"],
    "ty": ["ty", "server"],
}

# Stub declarations prepended to preprocessed source so the child backend
# recognises xonsh placeholder variables (__xonsh_env__, etc.) and xonsh
# Path extensions (.cd(), .mkdir() returning self, etc.).
_XONSH_PREAMBLE_LINES = [
    "import typing as __xonsh_typing__",
    "__xonsh_env__: dict[str, __xonsh_typing__.Any] = {}",
    "__xonsh_subproc__: __xonsh_typing__.Any = None",
    "__xonsh_at__: __xonsh_typing__.Any = None",
    "class __xonsh_Path__:",
    "    def __init__(self, *a: __xonsh_typing__.Any) -> None: ...",
    "    def __getattr__(self, name: str) -> __xonsh_typing__.Any: ...",
    "    def __truediv__(self, o: __xonsh_typing__.Any) -> '__xonsh_Path__': ...",
    "    def __enter__(self) -> '__xonsh_Path__': ...",
    "    def __exit__(self, *a: __xonsh_typing__.Any) -> None: ...",
]
_XONSH_PREAMBLE = "\n".join(_XONSH_PREAMBLE_LINES) + "\n"
_XONSH_PREAMBLE_LINE_COUNT = len(_XONSH_PREAMBLE_LINES)


@dataclass
class _SyncState:
    """Per-document synchronization state stored after each sync."""

    preprocess_result: PreprocessResult
    preamble_lines: int = 0


class LspProxyBackend:
    """Python analysis backend that delegates to a child LSP server.

    Implements the PythonBackend protocol by spawning a child LSP process
    (e.g. pyright-langserver) and forwarding preprocessed Python requests.
    """

    def __init__(
        self,
        command: list[str],
        on_diagnostics: Callable[[str, list[lsp.Diagnostic]], None] | None = None,
        backend_settings: dict[str, Any] | None = None,
        server: "LanguageServer | None" = None,
    ) -> None:
        """Initialize the proxy backend.

        Args:
            command: Command to spawn the child LSP server (e.g. ["pyright-langserver", "--stdio"]).
            on_diagnostics: Callback for asynchronous diagnostics from the backend.
                Called with (uri, diagnostics) when the backend publishes diagnostics.
            backend_settings: Fallback settings for when the editor doesn't respond
                to workspace/configuration requests.
            server: The parent LanguageServer instance, used to forward
                workspace/configuration requests from the child to the editor.
        """
        self._command = command
        self._client: LanguageClient | None = None
        self._on_diagnostics = on_diagnostics
        self._backend_settings = backend_settings or {}
        self._server: LanguageServer | None = server
        self._doc_versions: dict[str, int] = {}
        self._uri_map: dict[str, str] = {}  # py_uri -> original_uri
        self._workspace_root: str | None = None
        self._started = False
        self._sync_state: dict[str, _SyncState] = {}  # uri -> per-doc state

    async def start(self, workspace_root: str | None = None) -> None:
        """Start the child LSP server.

        Spawns the child process, sends initialize/initialized, and registers
        handlers for notifications from the child.
        """
        self._workspace_root = workspace_root
        logger.info(f"PROXY: starting child: command={self._command}, workspace={workspace_root}")

        # Create and start the language client
        self._client = LanguageClient("xonsh-lsp-proxy", "0.1.4")
        _patch_converter(self._client.protocol._converter)

        # Register diagnostics handler before starting
        @self._client.feature(lsp.TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS)
        def on_publish_diagnostics(params: lsp.PublishDiagnosticsParams) -> None:
            self._handle_diagnostics(params)

        # Register workspace/configuration handler — forwards to editor
        @self._client.feature(lsp.WORKSPACE_CONFIGURATION)
        async def on_workspace_configuration(
            params: lsp.ConfigurationParams,
        ) -> list[Any]:
            return await self._handle_configuration_request(params)

        # Forward progress notifications from child to editor
        @self._client.feature(lsp.WINDOW_WORK_DONE_PROGRESS_CREATE)
        async def on_work_done_progress_create(
            params: lsp.WorkDoneProgressCreateParams,
        ) -> None:
            await self._handle_progress_create(params)

        @self._client.feature(lsp.PROGRESS)
        def on_progress(params: lsp.ProgressParams) -> None:
            self._handle_progress(params)

        # Forward window/logMessage from child (avoids "unknown method" warnings)
        @self._client.feature(lsp.WINDOW_LOG_MESSAGE)
        def on_log_message(params: lsp.LogMessageParams) -> None:
            self._handle_log_message(params)

        try:
            await self._client.start_io(*self._command)
            logger.info("PROXY: child process spawned")
        except Exception as e:
            logger.error(f"PROXY: Failed to start backend: {self._command}: {e}")
            self._client = None
            return

        # Send initialize
        workspace_uri = Path(workspace_root).as_uri() if workspace_root else None
        workspace_folders = None
        if workspace_root:
            workspace_folders = [
                lsp.WorkspaceFolder(
                    uri=workspace_uri,
                    name=Path(workspace_root).name,
                )
            ]

        try:
            result = await self._client.initialize_async(
                lsp.InitializeParams(
                    capabilities=lsp.ClientCapabilities(
                        text_document=lsp.TextDocumentClientCapabilities(
                            completion=lsp.CompletionClientCapabilities(
                                completion_item=lsp.ClientCompletionItemOptions(
                                    snippet_support=True,
                                ),
                            ),
                            hover=lsp.HoverClientCapabilities(),
                            definition=lsp.DefinitionClientCapabilities(),
                            references=lsp.ReferenceClientCapabilities(),
                            signature_help=lsp.SignatureHelpClientCapabilities(),
                            publish_diagnostics=lsp.PublishDiagnosticsClientCapabilities(),
                            document_symbol=lsp.DocumentSymbolClientCapabilities(),
                        ),
                        workspace=lsp.WorkspaceClientCapabilities(
                            configuration=True,
                            workspace_folders=True,
                        ),
                        window=lsp.WindowClientCapabilities(
                            work_done_progress=True,
                        ),
                    ),
                    root_uri=workspace_uri,
                    workspace_folders=workspace_folders,
                    initialization_options=self._backend_settings if self._backend_settings else None,
                )
            )
            server_info = getattr(result, 'server_info', None)
            logger.info(f"PROXY: backend initialized: {server_info}")
        except Exception as e:
            logger.error(f"PROXY: initialization failed: {e}")
            await self._try_stop_client()
            return

        # Send initialized notification
        self._client.initialized(lsp.InitializedParams())

        # Nudge the backend to request workspace/configuration.
        # Some backends (e.g. Pyright) don't send workspace/configuration
        # after initialized unless prompted by didChangeConfiguration.
        self._client.workspace_did_change_configuration(
            lsp.DidChangeConfigurationParams(settings={})
        )

        self._started = True
        logger.info(f"PROXY: backend ready: {' '.join(self._command)}")

    async def stop(self) -> None:
        """Stop the child LSP server."""
        await self._try_stop_client()
        self._started = False

    async def _try_stop_client(self) -> None:
        """Attempt to gracefully stop the client."""
        if self._client is None:
            return
        try:
            await self._client.shutdown_async(None)
            self._client.exit(None)
        except Exception as e:
            logger.debug(f"Error during backend shutdown: {e}")
        try:
            await self._client.stop()
        except Exception:
            pass
        self._client = None

    async def update_settings(self, settings: Any) -> None:
        """Forward settings changes to the child backend."""
        if not self._started or self._client is None:
            return
        try:
            self._client.workspace_did_change_configuration(
                lsp.DidChangeConfigurationParams(settings=settings)
            )
        except Exception as e:
            logger.debug(f"Failed to forward settings: {e}")

    def _file_uri(self, path: str | None) -> str:
        """Convert a file path to a URI with .py extension.

        Backends like Pyright ignore files without a .py extension,
        so we rewrite .xsh/.xonshrc URIs to .py for the child.
        """
        if path is None:
            return "file:///untitled.py"
        p = Path(path)
        if p.suffix in (".xsh", ".xonshrc") or p.name in (".xonshrc", "xonshrc"):
            p = p.with_suffix(".py")
        return p.as_uri()

    def _sync_document(self, source: str, path: str | None) -> tuple[str, str]:
        """Synchronize document content with the child LSP server.

        Preprocesses xonsh source to Python, masks xonsh-only lines,
        prepends stub declarations, and sends didOpen/didChange to the child.

        Returns:
            Tuple of (uri, final_source_sent_to_child).
        """
        if self._client is None:
            return ("", "")

        preprocess_result = preprocess_with_mapping(source)
        masked_source = preprocess_result.source

        # Replace Path( with __xonsh_Path__( so xonsh Path extensions
        # (.cd(), .mkdir() returning self, etc.) don't cause false errors.
        has_xonsh_path = "Path(" in masked_source
        if has_xonsh_path:
            masked_source = masked_source.replace("Path(", "__xonsh_Path__(")

        # Prepend stub declarations when xonsh placeholders or Path are present
        preamble_lines = 0
        if has_xonsh_path or any(
            p in preprocess_result.source
            for p in ("__xonsh_env__", "__xonsh_subproc__", "__xonsh_at__")
        ):
            masked_source = _XONSH_PREAMBLE + masked_source
            preamble_lines = _XONSH_PREAMBLE_LINE_COUNT

        uri = self._file_uri(path)

        # Store per-document state for position mapping in diagnostics
        self._sync_state[uri] = _SyncState(
            preprocess_result=preprocess_result,
            preamble_lines=preamble_lines,
        )

        # Track mapping from .py URI back to original URI
        if path is not None:
            original_uri = Path(path).as_uri()
            if uri != original_uri:
                self._uri_map[uri] = original_uri

        if uri not in self._doc_versions:
            # First time seeing this document - send didOpen
            self._doc_versions[uri] = 0
            self._client.text_document_did_open(
                lsp.DidOpenTextDocumentParams(
                    text_document=lsp.TextDocumentItem(
                        uri=uri,
                        language_id="python",
                        version=self._doc_versions[uri],
                        text=masked_source,
                    )
                )
            )
        else:
            # Document already open - send didChange with full content
            self._doc_versions[uri] += 1
            self._client.text_document_did_change(
                lsp.DidChangeTextDocumentParams(
                    text_document=lsp.VersionedTextDocumentIdentifier(
                        uri=uri,
                        version=self._doc_versions[uri],
                    ),
                    content_changes=[
                        lsp.TextDocumentContentChangeWholeDocument(
                            text=masked_source,
                        )
                    ],
                )
            )

        return uri, masked_source

    def _preamble_offset(self, uri: str) -> int:
        """Return the number of preamble lines added for a document."""
        state = self._sync_state.get(uri)
        return state.preamble_lines if state else 0

    async def get_completions(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> list[lsp.CompletionItem]:
        """Get completions from the child LSP server."""
        if not self._started or self._client is None:
            return []

        try:
            preprocess_result = preprocess_with_mapping(source)
            uri = self._file_uri(path)
            self._sync_document(source, path)

            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )
            mapped_line += self._preamble_offset(uri)

            result = await self._client.text_document_completion_async(
                lsp.CompletionParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                    position=lsp.Position(line=mapped_line, character=mapped_col),
                )
            )

            if result is None:
                return []

            items: list[lsp.CompletionItem] = []
            if isinstance(result, lsp.CompletionList):
                items = result.items
            elif isinstance(result, list):
                items = result

            # Adjust sort text to be lower priority than xonsh items
            for item in items:
                if item.sort_text is None:
                    item.sort_text = f"5_{item.label}"

            return items

        except Exception as e:
            logger.debug(f"PROXY: completion error: {e}")
            return []

    async def get_hover(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> str | None:
        """Get hover information from the child LSP server."""
        if not self._started or self._client is None:
            return None

        try:
            preprocess_result = preprocess_with_mapping(source)
            uri = self._file_uri(path)
            self._sync_document(source, path)

            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )
            mapped_line += self._preamble_offset(uri)

            result = await self._client.text_document_hover_async(
                lsp.HoverParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                    position=lsp.Position(line=mapped_line, character=mapped_col),
                )
            )

            if result is None:
                return None

            # Extract markdown content
            if isinstance(result.contents, lsp.MarkupContent):
                return result.contents.value
            elif isinstance(result.contents, str):
                return result.contents
            elif isinstance(result.contents, list):
                parts = []
                for item in result.contents:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, lsp.MarkedString_Type1):
                        parts.append(f"```{item.language}\n{item.value}\n```")
                return "\n\n".join(parts) if parts else None

            return None

        except Exception as e:
            logger.debug(f"Proxy hover error: {e}")
            return None

    async def get_definitions(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> list[lsp.Location]:
        """Get definitions from the child LSP server."""
        if not self._started or self._client is None:
            return []

        try:
            preprocess_result = preprocess_with_mapping(source)
            uri = self._file_uri(path)
            self._sync_document(source, path)

            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )
            mapped_line += self._preamble_offset(uri)

            result = await self._client.text_document_definition_async(
                lsp.DefinitionParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                    position=lsp.Position(line=mapped_line, character=mapped_col),
                )
            )

            return self._normalize_locations(result, preprocess_result, path)

        except Exception as e:
            logger.debug(f"Proxy definition error: {e}")
            return []

    async def get_references(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> list[lsp.Location]:
        """Get references from the child LSP server."""
        if not self._started or self._client is None:
            return []

        try:
            preprocess_result = preprocess_with_mapping(source)
            uri = self._file_uri(path)
            self._sync_document(source, path)

            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )
            mapped_line += self._preamble_offset(uri)

            result = await self._client.text_document_references_async(
                lsp.ReferenceParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                    position=lsp.Position(line=mapped_line, character=mapped_col),
                    context=lsp.ReferenceContext(include_declaration=True),
                )
            )

            if result is None:
                return []

            return self._normalize_locations(result, preprocess_result, path)

        except Exception as e:
            logger.debug(f"Proxy references error: {e}")
            return []

    async def get_diagnostics(
        self, source: str, path: str | None = None
    ) -> list[lsp.Diagnostic]:
        """Sync document with backend. Diagnostics come asynchronously via callback.

        Returns an empty list — Python diagnostics are delivered via the
        on_diagnostics callback when the backend publishes them.
        """
        if not self._started or self._client is None:
            return []

        try:
            self._sync_document(source, path)
        except Exception as e:
            logger.debug(f"Proxy diagnostics sync error: {e}")

        # Diagnostics arrive asynchronously via textDocument/publishDiagnostics
        return []

    async def get_document_symbols(
        self, source: str, path: str | None = None
    ) -> list[lsp.DocumentSymbol]:
        """Get document symbols from the child LSP server."""
        if not self._started or self._client is None:
            return []

        try:
            uri = self._file_uri(path)
            self._sync_document(source, path)

            result = await self._client.text_document_document_symbol_async(
                lsp.DocumentSymbolParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                )
            )

            if result is None:
                return []

            # Result can be DocumentSymbol[] or SymbolInformation[]
            symbols: list[lsp.DocumentSymbol] = []
            for item in result:
                if isinstance(item, lsp.DocumentSymbol):
                    symbols.append(item)
                elif isinstance(item, lsp.SymbolInformation):
                    # Convert SymbolInformation to DocumentSymbol
                    symbols.append(
                        lsp.DocumentSymbol(
                            name=item.name,
                            kind=item.kind,
                            range=item.location.range,
                            selection_range=item.location.range,
                            detail=item.container_name,
                        )
                    )

            return symbols

        except Exception as e:
            logger.debug(f"Proxy document symbols error: {e}")
            return []

    async def get_signature_help(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> lsp.SignatureHelp | None:
        """Get signature help from the child LSP server."""
        if not self._started or self._client is None:
            return None

        try:
            preprocess_result = preprocess_with_mapping(source)
            uri = self._file_uri(path)
            self._sync_document(source, path)

            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )
            mapped_line += self._preamble_offset(uri)

            result = await self._client.text_document_signature_help_async(
                lsp.SignatureHelpParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                    position=lsp.Position(line=mapped_line, character=mapped_col),
                )
            )

            return result

        except Exception as e:
            logger.debug(f"Proxy signature help error: {e}")
            return None

    def _normalize_locations(
        self,
        result: Any,
        preprocess_result: Any,
        path: str | None,
    ) -> list[lsp.Location]:
        """Normalize definition/reference results to a list of Locations.

        Maps positions back from preprocessed to original coordinates for
        locations in the same file, accounting for the preamble offset.
        """
        if result is None:
            return []

        uri = self._file_uri(path)
        preamble = self._preamble_offset(uri)

        locations: list[lsp.Location] = []
        items = result if isinstance(result, list) else [result]

        for item in items:
            if isinstance(item, lsp.Location):
                loc = item
            elif isinstance(item, lsp.LocationLink):
                loc = lsp.Location(
                    uri=item.target_uri,
                    range=item.target_range,
                )
            else:
                continue

            # Map positions back if this is the same file
            if loc.uri == uri and preprocess_result is not None:
                start_line, start_col = map_position_from_processed(
                    preprocess_result,
                    loc.range.start.line - preamble,
                    loc.range.start.character,
                )
                end_line, end_col = map_position_from_processed(
                    preprocess_result,
                    loc.range.end.line - preamble,
                    loc.range.end.character,
                )
                loc = lsp.Location(
                    uri=loc.uri,
                    range=lsp.Range(
                        start=lsp.Position(line=start_line, character=start_col),
                        end=lsp.Position(line=end_line, character=end_col),
                    ),
                )

            locations.append(loc)

        return locations

    def _handle_diagnostics(self, params: lsp.PublishDiagnosticsParams) -> None:
        """Handle diagnostics published by the child LSP server.

        Maps diagnostic positions back from preprocessed to original coordinates,
        filters out diagnostics on masked (xonsh-only) lines, and forwards to
        the callback.
        """
        if self._on_diagnostics is None:
            return

        # Map .py URI back to original .xsh URI
        original_uri = self._uri_map.get(params.uri, params.uri)
        raw_diagnostics = params.diagnostics or []

        state = self._sync_state.get(params.uri)
        if state is None:
            # No sync state — forward as-is (shouldn't normally happen)
            self._on_diagnostics(original_uri, raw_diagnostics)
            return

        preamble = state.preamble_lines
        pp = state.preprocess_result
        masked = pp.masked_lines

        adjusted: list[lsp.Diagnostic] = []
        for diag in raw_diagnostics:
            # Subtract preamble offset
            start_line = diag.range.start.line - preamble
            end_line = diag.range.end.line - preamble
            if start_line < 0:
                continue  # diagnostic is on a preamble stub line — skip

            # Map from preprocessed coordinates back to original
            orig_start_line, orig_start_col = map_position_from_processed(
                pp, start_line, diag.range.start.character
            )
            orig_end_line, orig_end_col = map_position_from_processed(
                pp, end_line, diag.range.end.character
            )

            # Skip diagnostics whose start line falls on a masked xonsh line
            if orig_start_line in masked:
                continue

            adjusted.append(
                lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(line=orig_start_line, character=orig_start_col),
                        end=lsp.Position(line=orig_end_line, character=orig_end_col),
                    ),
                    message=diag.message,
                    severity=diag.severity,
                    code=diag.code,
                    source=diag.source,
                    related_information=diag.related_information,
                    tags=diag.tags,
                    code_description=diag.code_description,
                    data=diag.data,
                )
            )

        self._on_diagnostics(original_uri, adjusted)

    def _handle_log_message(self, params: lsp.LogMessageParams) -> None:
        """Forward window/logMessage from child to parent logger."""
        level_map = {
            lsp.MessageType.Error: logging.ERROR,
            lsp.MessageType.Warning: logging.WARNING,
            lsp.MessageType.Info: logging.INFO,
            lsp.MessageType.Log: logging.DEBUG,
            lsp.MessageType.Debug: logging.DEBUG,
        }
        level = level_map.get(params.type, logging.DEBUG)
        logger.log(level, f"[backend] {params.message}")

    async def _handle_progress_create(
        self, params: lsp.WorkDoneProgressCreateParams
    ) -> None:
        """Forward window/workDoneProgress/create from child to editor."""
        if self._server is None:
            return
        try:
            token = params.token
            logger.debug(f"PROXY: forwarding progress create: token={token}")
            await self._server.work_done_progress.create_async(token)
        except Exception as e:
            logger.debug(f"PROXY: progress create forward failed: {e}")

    def _handle_progress(self, params: lsp.ProgressParams) -> None:
        """Forward $/progress notifications from child to editor."""
        if self._server is None:
            return
        try:
            token = params.token
            value = params.value
            logger.debug(f"PROXY: forwarding progress: token={token}")
            progress = self._server.work_done_progress
            if isinstance(value, lsp.WorkDoneProgressBegin):
                progress.begin(token, value)
            elif isinstance(value, lsp.WorkDoneProgressReport):
                progress.report(token, value)
            elif isinstance(value, lsp.WorkDoneProgressEnd):
                progress.end(token, value)
            else:
                # Raw dict from JSON — forward as-is via low-level notify
                self._server.protocol.notify(
                    lsp.PROGRESS,
                    lsp.ProgressParams(token=token, value=value),
                )
        except Exception as e:
            logger.debug(f"PROXY: progress forward failed: {e}")

    async def _handle_configuration_request(
        self, params: lsp.ConfigurationParams
    ) -> list[Any]:
        """Handle workspace/configuration requests from the child backend.

        Forwards the request to the editor via the parent server. Falls back
        to backendSettings if the editor doesn't respond.
        """
        # Try forwarding to the editor
        if self._server is not None:
            try:
                logger.debug(f"PROXY: forwarding config request to editor: {[i.section for i in params.items]}")
                result = await self._server.send_request_async(
                    lsp.WORKSPACE_CONFIGURATION, params
                )
                logger.debug(f"PROXY: editor returned config: {result}")
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"PROXY: editor config request failed, using fallback: {e}")

        # Fallback: resolve from backendSettings
        return self._resolve_settings(params)

    def _resolve_settings(self, params: lsp.ConfigurationParams) -> list[Any]:
        """Resolve configuration from backendSettings (fallback)."""
        results = []
        for item in params.items:
            section = item.section or ""
            value = self._backend_settings
            if section:
                for part in section.split("."):
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = {}
                        break
            results.append(value)
        return results


# ---------------------------------------------------------------------------
# Workaround for lsprotocol issue #430: the cattrs converter is missing a
# structure hook for the Optional variant of the notebook document filter
# union.  Backends like ty advertise notebookDocumentSync capabilities that
# trigger this.  Register the missing hook on any cattrs Converter.
# ---------------------------------------------------------------------------
_NotebookFilterUnion = Optional[
    Union[
        str,
        lsp.NotebookDocumentFilterNotebookType,
        lsp.NotebookDocumentFilterScheme,
        lsp.NotebookDocumentFilterPattern,
    ]
]


def _patch_converter(converter: Any) -> None:
    """Register missing lsprotocol cattrs hooks on *converter*."""

    def _notebook_filter_hook(obj: Any, _: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        if "notebookType" in obj:
            return converter.structure(obj, lsp.NotebookDocumentFilterNotebookType)
        if "scheme" in obj:
            return converter.structure(obj, lsp.NotebookDocumentFilterScheme)
        return converter.structure(obj, lsp.NotebookDocumentFilterPattern)

    converter.register_structure_hook(_NotebookFilterUnion, _notebook_filter_hook)
