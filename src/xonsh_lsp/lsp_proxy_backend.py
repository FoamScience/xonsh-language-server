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
    "import pathlib as __xonsh_pathlib__",
    "__xonsh_env__: dict[str, __xonsh_typing__.Any] = {}",
    "__xonsh_subproc__: __xonsh_typing__.Any = None",
    "__xonsh_at__: __xonsh_typing__.Any = None",
    "class __xonsh_Path__(__xonsh_pathlib__.Path):",
    "    def cd(self) -> '__xonsh_Path__': ...",
    "    def mkdir(self, *a: __xonsh_typing__.Any, **kw: __xonsh_typing__.Any) -> '__xonsh_Path__': ...",
    "    def __truediv__(self, o: __xonsh_typing__.Any) -> '__xonsh_Path__': ...",
    "    def __enter__(self) -> '__xonsh_Path__': ...",
    "    def __exit__(self, *a: __xonsh_typing__.Any) -> None: ...",
]
_XONSH_PREAMBLE = "\n".join(_XONSH_PREAMBLE_LINES) + "\n"
_XONSH_PREAMBLE_LINE_COUNT = len(_XONSH_PREAMBLE_LINES)


# Mapping from xonsh node types to semantic token type names.
# Used to emit synthetic tokens for xonsh constructs that replace child tokens.
_XONSH_SEMANTIC_TOKEN_TYPES: dict[str, str] = {
    "env_variable": "variable",
    "env_variable_braced": "variable",
    "captured_subprocess": "macro",
    "captured_subprocess_object": "macro",
    "uncaptured_subprocess": "macro",
    "uncaptured_subprocess_object": "macro",
    "tokenized_substitution": "macro",
    "python_evaluation": "variable",
    "at_object": "variable",
    "path_string": "string",
    "macro_call": "function",
    "regex_glob": "string",
    "glob_pattern": "string",
    "formatted_glob": "string",
    "custom_function_glob": "string",
    "glob_path": "string",
    "regex_path_glob": "string",
}


@dataclass
class _SyncState:
    """Per-document synchronization state stored after each sync."""

    preprocess_result: PreprocessResult
    preamble_lines: int = 0


def _token_in_replacement(
    mapping: list[tuple[int, int]], proc_start: int, proc_end: int
) -> bool:
    """Check if a token falls entirely within a non-1:1 mapping segment.

    Returns True when both proc_start and proc_end lie in the same segment
    whose processed range differs from its original range (i.e. a replacement).
    Tokens in such segments have no meaningful original position.
    """
    for i in range(len(mapping) - 1):
        seg_orig_start = mapping[i][0]
        seg_proc_start = mapping[i][1]
        seg_orig_end = mapping[i + 1][0]
        seg_proc_end = mapping[i + 1][1]

        if seg_proc_start <= proc_start and proc_end <= seg_proc_end:
            # Token is within this segment
            return (seg_proc_end - seg_proc_start) != (seg_orig_end - seg_orig_start)

    return False


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
        self._semantic_type_remap: list[int] = []
        self._semantic_modifier_remap: list[int] = []

    async def start(self, workspace_root: str | None = None) -> None:
        """Start the child LSP server.

        Spawns the child process, sends initialize/initialized, and registers
        handlers for notifications from the child.
        """
        self._workspace_root = workspace_root
        logger.info(f"PROXY: starting child: command={self._command}, workspace={workspace_root}")

        # Create and start the language client
        self._client = LanguageClient("xonsh-lsp-proxy", "0.1.7")
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

        # Forward workspace/inlayHint/refresh from child to editor
        @self._client.feature(lsp.WORKSPACE_INLAY_HINT_REFRESH)
        async def on_inlay_hint_refresh(params: None) -> None:
            await self._handle_inlay_hint_refresh()

        # Forward workspace/semanticTokens/refresh from child to editor
        @self._client.feature(lsp.WORKSPACE_SEMANTIC_TOKENS_REFRESH)
        async def on_semantic_tokens_refresh(params: None) -> None:
            await self._handle_semantic_tokens_refresh()

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
                            inlay_hint=lsp.InlayHintClientCapabilities(
                                resolve_support=lsp.ClientInlayHintResolveOptions(
                                    properties=["tooltip", "textEdits", "label.tooltip", "label.location"],
                                ),
                            ),
                            semantic_tokens=lsp.SemanticTokensClientCapabilities(
                                requests=lsp.ClientSemanticTokensRequestOptions(
                                    full=True,
                                    range=True,
                                ),
                                token_types=[t.value for t in lsp.SemanticTokenTypes],
                                token_modifiers=[m.value for m in lsp.SemanticTokenModifiers],
                                formats=[lsp.TokenFormat.Relative],
                            ),
                        ),
                        workspace=lsp.WorkspaceClientCapabilities(
                            configuration=True,
                            workspace_folders=True,
                            inlay_hint=lsp.InlayHintWorkspaceClientCapabilities(
                                refresh_support=True,
                            ),
                            semantic_tokens=lsp.SemanticTokensWorkspaceClientCapabilities(
                                refresh_support=True,
                            ),
                            symbol=lsp.WorkspaceSymbolClientCapabilities(
                                resolve_support=lsp.ClientSymbolResolveOptions(
                                    properties=["location.range"],
                                ),
                            ),
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

            # Build semantic token legend remap arrays
            self._build_semantic_legend_remap(result)
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

        # Alias Path to __xonsh_Path__ so xonsh Path extensions
        # (.cd(), .mkdir() returning self, etc.) are available without
        # shifting column positions (which would break position mapping).
        has_xonsh_path = "Path(" in masked_source
        if has_xonsh_path:
            lines = masked_source.split("\n")
            for i, line in enumerate(lines):
                if line.strip() == "from pathlib import Path":
                    lines[i] = "Path = __xonsh_Path__"
                    break
            masked_source = "\n".join(lines)

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

    async def get_inlay_hints(
        self, source: str, start_line: int, end_line: int, path: str | None = None
    ) -> list[lsp.InlayHint]:
        """Get inlay hints from the child LSP server."""
        if not self._started or self._client is None:
            return []

        try:
            preprocess_result = preprocess_with_mapping(source)
            uri = self._file_uri(path)
            self._sync_document(source, path)

            preamble = self._preamble_offset(uri)

            # Map range to preprocessed coordinates + preamble offset
            mapped_start, _ = map_position_to_processed(preprocess_result, start_line, 0)
            mapped_end, _ = map_position_to_processed(preprocess_result, end_line, 0)
            mapped_start += preamble
            mapped_end += preamble

            result = await self._client.text_document_inlay_hint_async(
                lsp.InlayHintParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                    range=lsp.Range(
                        start=lsp.Position(line=mapped_start, character=0),
                        end=lsp.Position(line=mapped_end, character=0),
                    ),
                )
            )

            if result is None:
                return []

            masked = preprocess_result.masked_lines
            hints: list[lsp.InlayHint] = []
            for hint in result:
                orig_line = hint.position.line - preamble
                if orig_line < 0:
                    continue  # preamble hint

                orig_line, orig_col = map_position_from_processed(
                    preprocess_result, orig_line, hint.position.character
                )

                if orig_line in masked:
                    continue

                hints.append(
                    lsp.InlayHint(
                        position=lsp.Position(line=orig_line, character=orig_col),
                        label=hint.label,
                        kind=hint.kind,
                        text_edits=hint.text_edits,
                        tooltip=hint.tooltip,
                        padding_left=hint.padding_left,
                        padding_right=hint.padding_right,
                        data=hint.data,
                    )
                )

            return hints

        except Exception as e:
            logger.debug(f"Proxy inlay hints error: {e}")
            return []

    async def resolve_inlay_hint(
        self, hint: lsp.InlayHint, path: str | None = None
    ) -> lsp.InlayHint:
        """Resolve additional details for an inlay hint."""
        if not self._started or self._client is None:
            return hint

        try:
            return await self._client.inlay_hint_resolve_async(hint)
        except Exception as e:
            logger.debug(f"Proxy inlay hint resolve error: {e}")
            return hint

    async def get_workspace_symbols(self, query: str) -> list[lsp.WorkspaceSymbol]:
        """Search for symbols across the workspace via the child LSP server."""
        if not self._started or self._client is None:
            return []

        try:
            result = await self._client.workspace_symbol_async(
                lsp.WorkspaceSymbolParams(query=query)
            )

            if result is None:
                return []

            symbols: list[lsp.WorkspaceSymbol] = []
            for item in result:
                if isinstance(item, lsp.WorkspaceSymbol):
                    self._remap_workspace_symbol_uri(item)
                    symbols.append(item)
                elif isinstance(item, lsp.SymbolInformation):
                    # Convert SymbolInformation to WorkspaceSymbol
                    original_uri = self._uri_map.get(
                        item.location.uri, item.location.uri
                    )
                    symbols.append(
                        lsp.WorkspaceSymbol(
                            name=item.name,
                            kind=item.kind,
                            location=lsp.Location(
                                uri=original_uri,
                                range=item.location.range,
                            ),
                            container_name=item.container_name,
                        )
                    )

            return symbols

        except Exception as e:
            logger.debug(f"Proxy workspace symbols error: {e}")
            return []

    async def resolve_workspace_symbol(
        self, symbol: lsp.WorkspaceSymbol
    ) -> lsp.WorkspaceSymbol:
        """Resolve additional details for a workspace symbol."""
        if not self._started or self._client is None:
            return symbol

        try:
            resolved = await self._client.workspace_symbol_resolve_async(symbol)
            self._remap_workspace_symbol_uri(resolved)
            return resolved
        except Exception as e:
            logger.debug(f"Proxy workspace symbol resolve error: {e}")
            return symbol

    async def get_semantic_tokens(
        self, source: str, path: str | None = None
    ) -> lsp.SemanticTokens | None:
        """Get semantic tokens for the full document from the child LSP server."""
        if not self._started or self._client is None:
            return None

        try:
            uri = self._file_uri(path)
            self._sync_document(source, path)

            result = await self._client.text_document_semantic_tokens_full_async(
                lsp.SemanticTokensParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                )
            )

            if result is None:
                return None

            state = self._sync_state.get(uri)
            if state is None:
                return result

            return self._remap_semantic_tokens(result.data, state)

        except Exception as e:
            logger.debug(f"Proxy semantic tokens error: {e}")
            return None

    async def get_semantic_tokens_range(
        self,
        source: str,
        start_line: int,
        start_char: int,
        end_line: int,
        end_char: int,
        path: str | None = None,
    ) -> lsp.SemanticTokens | None:
        """Get semantic tokens for a range from the child LSP server."""
        if not self._started or self._client is None:
            return None

        try:
            preprocess_result = preprocess_with_mapping(source)
            uri = self._file_uri(path)
            self._sync_document(source, path)

            preamble = self._preamble_offset(uri)

            # Map range to preprocessed coordinates + preamble offset
            mapped_start_line, mapped_start_char = map_position_to_processed(
                preprocess_result, start_line, start_char
            )
            mapped_end_line, mapped_end_char = map_position_to_processed(
                preprocess_result, end_line, end_char
            )
            mapped_start_line += preamble
            mapped_end_line += preamble

            result = await self._client.text_document_semantic_tokens_range_async(
                lsp.SemanticTokensRangeParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri),
                    range=lsp.Range(
                        start=lsp.Position(line=mapped_start_line, character=mapped_start_char),
                        end=lsp.Position(line=mapped_end_line, character=mapped_end_char),
                    ),
                )
            )

            if result is None:
                return None

            state = self._sync_state.get(uri)
            if state is None:
                return result

            return self._remap_semantic_tokens(result.data, state)

        except Exception as e:
            logger.debug(f"Proxy semantic tokens range error: {e}")
            return None

    def _remap_semantic_tokens(
        self, data: list[int], state: _SyncState
    ) -> lsp.SemanticTokens | None:
        """Remap semantic token data from child coordinates to original.

        Decodes delta-encoded token positions to absolute, subtracts preamble,
        maps columns back via preprocessing, filters preamble/masked tokens,
        remaps type/modifier indices, and re-encodes to delta format.
        """
        if not data:
            return lsp.SemanticTokens(data=[])

        preamble = state.preamble_lines
        pp = state.preprocess_result
        masked = pp.masked_lines

        # Decode delta-encoded groups of 5 into absolute positions
        # Format: [deltaLine, deltaStartChar, length, tokenType, tokenModifiers, ...]
        absolute_tokens: list[tuple[int, int, int, int, int]] = []
        current_line = 0
        current_char = 0
        for i in range(0, len(data), 5):
            delta_line = data[i]
            delta_char = data[i + 1]
            length = data[i + 2]
            token_type = data[i + 3]
            token_modifiers = data[i + 4]

            if delta_line > 0:
                current_line += delta_line
                current_char = delta_char
            else:
                current_char += delta_char

            absolute_tokens.append(
                (current_line, current_char, length, token_type, token_modifiers)
            )

        # Pass 1: remap non-replacement tokens from child
        remapped: list[tuple[int, int, int, int, int]] = []
        for abs_line, abs_char, length, token_type, token_modifiers in absolute_tokens:
            # Subtract preamble
            pre_line = abs_line - preamble
            if pre_line < 0:
                continue  # token is in preamble — skip

            # Map preprocessed line → original line (accounting for added import lines)
            orig_line = pre_line
            if pp.added_lines_before > 0:
                if pre_line >= pp.added_line_position + pp.added_lines_before:
                    orig_line = pre_line - pp.added_lines_before
                elif pre_line >= pp.added_line_position:
                    continue  # token on added import line — skip

            # Get line mapping for column remapping
            mapping = (
                pp.line_mappings[orig_line]
                if orig_line < len(pp.line_mappings)
                else []
            )

            # Skip tokens that fall within a replacement region (e.g.
            # __xonsh_env__["PATH"] replacing $PATH) — handled by pass 2.
            if _token_in_replacement(mapping, abs_char, abs_char + length):
                continue

            # Map start and end positions back from preprocessed coordinates
            orig_line, orig_char = map_position_from_processed(
                pp, pre_line, abs_char
            )
            _, orig_end_char = map_position_from_processed(
                pp, pre_line, abs_char + length
            )

            # Recompute length in original coordinates
            orig_length = orig_end_char - orig_char
            if orig_length <= 0:
                continue  # token collapsed

            # Filter masked lines
            if orig_line in masked:
                continue

            # Remap token type and modifiers via legend remap arrays
            if self._semantic_type_remap and token_type < len(self._semantic_type_remap):
                token_type = self._semantic_type_remap[token_type]

            if self._semantic_modifier_remap and token_modifiers != 0:
                token_modifiers = self._remap_modifier_bitmask(token_modifiers)

            remapped.append((orig_line, orig_char, orig_length, token_type, token_modifiers))

        # Pass 2: emit synthetic tokens for xonsh replacement regions
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES
        for rr_start_line, rr_start_col, rr_end_line, rr_end_col, rr_type in pp.replacement_regions:
            if rr_start_line != rr_end_line:
                continue  # skip multi-line replacements
            if rr_start_line in masked:
                continue
            token_type_name = _XONSH_SEMANTIC_TOKEN_TYPES.get(rr_type)
            if token_type_name and token_type_name in SEMANTIC_TOKEN_TYPES:
                type_idx = SEMANTIC_TOKEN_TYPES.index(token_type_name)
                remapped.append((rr_start_line, rr_start_col, rr_end_col - rr_start_col, type_idx, 0))

        # Sort by (line, char) for proper delta encoding
        remapped.sort(key=lambda t: (t[0], t[1]))

        if not remapped:
            return lsp.SemanticTokens(data=[])

        # Re-encode to delta format
        encoded: list[int] = []
        prev_line = 0
        prev_char = 0
        for line, char, length, token_type, token_modifiers in remapped:
            delta_line = line - prev_line
            if delta_line > 0:
                delta_char = char
            else:
                delta_char = char - prev_char

            encoded.extend([delta_line, delta_char, length, token_type, token_modifiers])
            prev_line = line
            prev_char = char

        return lsp.SemanticTokens(data=encoded)

    def _remap_modifier_bitmask(self, bitmask: int) -> int:
        """Remap a modifier bitmask using the pre-computed remap array."""
        result = 0
        for child_bit in range(len(self._semantic_modifier_remap)):
            if bitmask & (1 << child_bit):
                our_bit = self._semantic_modifier_remap[child_bit]
                result |= 1 << our_bit
        return result

    def _build_semantic_legend_remap(self, init_result: lsp.InitializeResult) -> None:
        """Build remap arrays from child's legend to our legend.

        If the child's legend matches ours exactly, the remap is identity.
        """
        from xonsh_lsp.server import SEMANTIC_TOKEN_TYPES, SEMANTIC_TOKEN_MODIFIERS

        caps = init_result.capabilities
        provider = getattr(caps, 'semantic_tokens_provider', None)
        if provider is None:
            self._semantic_type_remap = []
            self._semantic_modifier_remap = []
            return

        legend = provider.legend if hasattr(provider, 'legend') else None
        if legend is None:
            self._semantic_type_remap = []
            self._semantic_modifier_remap = []
            return

        # Build type remap: child_index -> our_index
        our_type_index = {name: i for i, name in enumerate(SEMANTIC_TOKEN_TYPES)}
        self._semantic_type_remap = [
            our_type_index.get(t, 0) for t in legend.token_types
        ]

        # Build modifier remap: child_bit -> our_bit
        our_mod_index = {name: i for i, name in enumerate(SEMANTIC_TOKEN_MODIFIERS)}
        self._semantic_modifier_remap = [
            our_mod_index.get(m, 0) for m in legend.token_modifiers
        ]

    def _remap_workspace_symbol_uri(self, symbol: lsp.WorkspaceSymbol) -> None:
        """Remap .py URIs back to original .xsh URIs in a workspace symbol."""
        loc = symbol.location
        if isinstance(loc, lsp.Location):
            original_uri = self._uri_map.get(loc.uri, loc.uri)
            if original_uri != loc.uri:
                symbol.location = lsp.Location(
                    uri=original_uri,
                    range=loc.range,
                )

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

    async def _handle_inlay_hint_refresh(self) -> None:
        """Forward workspace/inlayHint/refresh from child to editor."""
        if self._server is None:
            return
        try:
            await self._server.send_request_async(lsp.WORKSPACE_INLAY_HINT_REFRESH, None)
        except Exception as e:
            logger.debug(f"PROXY: inlay hint refresh forward failed: {e}")

    async def _handle_semantic_tokens_refresh(self) -> None:
        """Forward workspace/semanticTokens/refresh from child to editor."""
        if self._server is None:
            return
        try:
            await self._server.send_request_async(lsp.WORKSPACE_SEMANTIC_TOKENS_REFRESH, None)
        except Exception as e:
            logger.debug(f"PROXY: semantic tokens refresh forward failed: {e}")

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
