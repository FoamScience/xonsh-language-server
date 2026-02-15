"""
Xonsh Language Server

Main LSP server implementation using pygls.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING, Any

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from xonsh_lsp.completions import XonshCompletionProvider
from xonsh_lsp.diagnostics import XonshDiagnosticsProvider
from xonsh_lsp.hover import XonshHoverProvider
from xonsh_lsp.definition import XonshDefinitionProvider
from xonsh_lsp.parser import XonshParser, ParseResult

if TYPE_CHECKING:
    from pygls.workspace import TextDocument
    from xonsh_lsp.python_backend import PythonBackend

# Configure logging — WARNING by default to avoid flooding stderr
# (Neovim treats all stderr as [ERROR])
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Known backend shortcuts: name -> command
KNOWN_BACKENDS: dict[str, list[str]] = {
    "pyright": ["pyright-langserver", "--stdio"],
    "basedpyright": ["basedpyright-langserver", "--stdio"],
    "pylsp": ["pylsp"],
    "ty": ["ty", "server"],
}


class XonshLanguageServer(LanguageServer):
    """Language Server for xonsh files."""

    CMD_SHOW_ENV_VARS = "xonsh.showEnvVars"
    CMD_RUN_SELECTION = "xonsh.runSelection"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Initialize components
        self.parser = XonshParser()
        self.python_backend: PythonBackend | None = None
        self.completion_provider = XonshCompletionProvider(self)
        self.diagnostics_provider = XonshDiagnosticsProvider(self)
        self.hover_provider = XonshHoverProvider(self)
        self.definition_provider = XonshDefinitionProvider(self)

        # Cache for parsed documents
        self._parse_cache: dict[str, ParseResult] = {}

        # Backend configuration (set from CLI args or initializationOptions)
        self._backend_name: str = "jedi"
        self._backend_command: list[str] | None = None
        self._backend_settings: dict[str, Any] = {}
        self._workspace_root: str | None = None

    @property
    def python_delegate(self) -> PythonBackend:
        """Backwards-compatible alias for python_backend.

        Returns a no-op backend if none is configured.
        """
        if self.python_backend is None:
            # Return a no-op backend to avoid NoneType errors
            from xonsh_lsp.jedi_backend import JediBackend
            self.python_backend = JediBackend()
        return self.python_backend

    def get_document(self, uri: str) -> TextDocument | None:
        """Get a document from the workspace."""
        return self.workspace.get_text_document(uri)

    def parse_document(self, uri: str) -> ParseResult | None:
        """Parse a document and return the parse result."""
        doc = self.get_document(uri)
        if doc is None:
            return None

        # Check cache
        cache_key = f"{uri}:{doc.version}"
        if cache_key in self._parse_cache:
            return self._parse_cache[cache_key]

        # Parse and cache
        tree = self.parser.parse(doc.source)
        self._parse_cache[cache_key] = tree

        # Clean old cache entries
        old_keys = [k for k in self._parse_cache if k.startswith(f"{uri}:") and k != cache_key]
        for k in old_keys:
            del self._parse_cache[k]

        return tree


# Create server instance
server = XonshLanguageServer(
    name="xonsh-lsp",
    version="0.1.5",
)


def _create_backend(
    backend_name: str,
    backend_command: list[str] | None,
    backend_settings: dict[str, Any],
) -> "PythonBackend":
    """Create a Python backend based on configuration.

    Args:
        backend_name: Name of the backend ("jedi", "pyright", "basedpyright", "pylsp", "lsp-proxy").
        backend_command: Custom command for lsp-proxy backend.
        backend_settings: Settings to forward to the backend.

    Returns:
        A PythonBackend instance.
    """
    if backend_name == "jedi":
        from xonsh_lsp.jedi_backend import JediBackend
        return JediBackend()

    # Resolve backend name to command
    command = backend_command
    if command is None:
        command = KNOWN_BACKENDS.get(backend_name)
        if command is None and backend_name != "lsp-proxy":
            logger.warning(
                f"Unknown backend '{backend_name}', falling back to Jedi. "
                f"Known backends: {', '.join(['jedi'] + list(KNOWN_BACKENDS.keys()))}"
            )
            from xonsh_lsp.jedi_backend import JediBackend
            return JediBackend()

    if command is None:
        logger.error("No command specified for lsp-proxy backend, falling back to Jedi")
        from xonsh_lsp.jedi_backend import JediBackend
        return JediBackend()

    from xonsh_lsp.lsp_proxy_backend import LspProxyBackend

    def on_diagnostics(uri: str, diagnostics: list[lsp.Diagnostic]) -> None:
        """Handle diagnostics from the proxy backend."""
        server.diagnostics_provider.on_backend_diagnostics(uri, diagnostics)

    return LspProxyBackend(
        command=command,
        on_diagnostics=on_diagnostics,
        backend_settings=backend_settings,
        server=server,
    )


# ============================================================================
# Lifecycle Events
# ============================================================================


@server.feature(lsp.INITIALIZE)
async def initialize(params: lsp.InitializeParams) -> None:
    """Handle the initialize request - configure backend from initializationOptions."""
    opts = params.initialization_options or {}
    if isinstance(opts, dict):
        # Read backend configuration
        backend_name = opts.get("pythonBackend", server._backend_name)
        backend_command = opts.get("pythonBackendCommand", server._backend_command)
        backend_settings = opts.get("backendSettings", server._backend_settings)

        server._backend_name = backend_name
        if backend_command:
            server._backend_command = backend_command
        if backend_settings:
            server._backend_settings = backend_settings

    # Create the backend (started in `initialized` after handshake completes)
    server.python_backend = _create_backend(
        server._backend_name,
        server._backend_command,
        server._backend_settings,
    )

    # Update serverInfo.name so clients can show the active backend
    backend_label = server._backend_name
    if backend_label != "jedi":
        server.protocol.server_info = lsp.ServerInfo(
            name=f"xonsh-lsp[{backend_label}]",
            version=server.version,
        )

    # Save workspace root for use in `initialized`
    server._workspace_root = None
    if params.root_uri:
        from urllib.parse import unquote, urlparse
        parsed = urlparse(params.root_uri)
        server._workspace_root = unquote(parsed.path)
    elif params.root_path:
        server._workspace_root = params.root_path


@server.feature(lsp.INITIALIZED)
async def initialized(params: lsp.InitializedParams) -> None:
    """Handle the initialized notification — start the backend.

    Deferred from initialize so that the editor is ready to handle
    server-to-client requests (e.g. workspace/configuration forwarding).
    """
    if server.python_backend is not None:
        await server.python_backend.start(server._workspace_root)

        # Re-sync documents that were opened while the backend was starting.
        # Editors often send textDocument/didOpen before start() finishes,
        # and the proxy backend drops those (self._started is still False).
        for uri in list(server.workspace.text_documents):
            diagnostics = await server.diagnostics_provider.get_diagnostics(uri)
            server.text_document_publish_diagnostics(
                lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
            )


@server.feature(lsp.SHUTDOWN)
async def shutdown(params: Any) -> None:
    """Handle the shutdown request."""
    if server.python_backend is not None:
        await server.python_backend.stop()


# ============================================================================
# Document Events
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
async def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
    """Handle document open."""
    uri = params.text_document.uri
    logger.debug(f"Document opened: {uri}")

    # Run diagnostics
    diagnostics = await server.diagnostics_provider.get_diagnostics(uri)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
async def did_change(params: lsp.DidChangeTextDocumentParams) -> None:
    """Handle document change."""
    uri = params.text_document.uri
    logger.debug(f"Document changed: {uri}")

    # Run diagnostics
    diagnostics = await server.diagnostics_provider.get_diagnostics(uri)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
async def did_save(params: lsp.DidSaveTextDocumentParams) -> None:
    """Handle document save."""
    uri = params.text_document.uri
    logger.debug(f"Document saved: {uri}")

    # Run diagnostics
    diagnostics = await server.diagnostics_provider.get_diagnostics(uri)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
async def did_close(params: lsp.DidCloseTextDocumentParams) -> None:
    """Handle document close."""
    uri = params.text_document.uri
    logger.debug(f"Document closed: {uri}")

    # Clear diagnostics cache
    server.diagnostics_provider.clear_cache(uri)

    # Clear diagnostics
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=[])
    )


# ============================================================================
# Completion
# ============================================================================


@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION,
    lsp.CompletionOptions(
        trigger_characters=["$", "@", ".", "/", "`"],
        resolve_provider=True,
    ),
)
async def completion(params: lsp.CompletionParams) -> lsp.CompletionList | None:
    """Provide completions."""
    return await server.completion_provider.get_completions(params)


@server.feature(lsp.COMPLETION_ITEM_RESOLVE)
async def completion_resolve(item: lsp.CompletionItem) -> lsp.CompletionItem:
    """Resolve additional completion item details."""
    return server.completion_provider.resolve_completion(item)


# ============================================================================
# Hover
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
async def hover(params: lsp.HoverParams) -> lsp.Hover | None:
    """Provide hover information."""
    return await server.hover_provider.get_hover(params)


# ============================================================================
# Go to Definition
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
async def definition(
    params: lsp.DefinitionParams,
) -> lsp.Location | list[lsp.Location] | None:
    """Provide go-to-definition."""
    return await server.definition_provider.get_definition(params)


# ============================================================================
# Signature Help
# ============================================================================


@server.feature(
    lsp.TEXT_DOCUMENT_SIGNATURE_HELP,
    lsp.SignatureHelpOptions(
        trigger_characters=["(", ","],
        retrigger_characters=[","],
    ),
)
async def signature_help(params: lsp.SignatureHelpParams) -> lsp.SignatureHelp | None:
    """Provide signature help."""
    uri = params.text_document.uri
    doc = server.get_document(uri)
    if doc is None:
        return None

    return await server.python_delegate.get_signature_help(
        doc.source,
        params.position.line,
        params.position.character,
        doc.path,
    )


# ============================================================================
# Find References
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
async def references(params: lsp.ReferenceParams) -> list[lsp.Location] | None:
    """Provide find references."""
    uri = params.text_document.uri
    doc = server.get_document(uri)
    if doc is None:
        return None

    line = params.position.line
    col = params.position.character

    # Get Python references from backend
    python_refs = await server.python_delegate.get_references(
        doc.source, line, col, doc.path
    )

    # Also search for xonsh-specific references (env vars, aliases)
    from xonsh_lsp.definition import XonshReferenceProvider
    ref_provider = XonshReferenceProvider(server)
    xonsh_refs = ref_provider.get_references(params)

    all_refs = python_refs + (xonsh_refs or [])

    # Deduplicate by (uri, start_line, start_char)
    seen: set[tuple[str, int, int]] = set()
    unique_refs = []
    for ref in all_refs:
        key = (ref.uri, ref.range.start.line, ref.range.start.character)
        if key not in seen:
            seen.add(key)
            unique_refs.append(ref)

    return unique_refs if unique_refs else None


# ============================================================================
# Document Symbols
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
async def document_symbols(
    params: lsp.DocumentSymbolParams,
) -> list[lsp.DocumentSymbol] | list[lsp.SymbolInformation] | None:
    """Provide document symbols."""
    uri = params.text_document.uri
    doc = server.get_document(uri)
    if doc is None:
        return None

    # Use tree-sitter for symbol extraction (handles xonsh syntax)
    raw_symbols = server.parser.get_document_symbols(doc.source)

    # Convert to LSP DocumentSymbol format
    kind_map = {
        "function": lsp.SymbolKind.Function,
        "class": lsp.SymbolKind.Class,
        "variable": lsp.SymbolKind.Variable,
        "module": lsp.SymbolKind.Module,
    }

    symbols: list[lsp.DocumentSymbol] = []
    for sym in raw_symbols:
        kind = kind_map.get(sym["kind"], lsp.SymbolKind.Variable)
        symbols.append(
            lsp.DocumentSymbol(
                name=sym["name"],
                kind=kind,
                range=lsp.Range(
                    start=lsp.Position(line=sym["line"], character=sym["col"]),
                    end=lsp.Position(line=sym["end_line"], character=sym["end_col"]),
                ),
                selection_range=lsp.Range(
                    start=lsp.Position(line=sym["line"], character=sym["col"]),
                    end=lsp.Position(line=sym["end_line"], character=sym["end_col"]),
                ),
                detail=sym.get("detail", ""),
            )
        )

    return symbols


# ============================================================================
# Folding Ranges
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_FOLDING_RANGE)
async def folding_range(params: lsp.FoldingRangeParams) -> list[lsp.FoldingRange] | None:
    """Provide folding ranges."""
    uri = params.text_document.uri
    doc = server.get_document(uri)
    if doc is None:
        return None

    raw_ranges = server.parser.get_folding_ranges(doc.source)

    kind_map = {
        "comment": lsp.FoldingRangeKind.Comment,
        "imports": lsp.FoldingRangeKind.Imports,
        "region": lsp.FoldingRangeKind.Region,
    }

    return [
        lsp.FoldingRange(
            start_line=r["start_line"],
            end_line=r["end_line"],
            kind=kind_map.get(r["kind"]),
        )
        for r in raw_ranges
    ]


# ============================================================================
# Inlay Hints
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_INLAY_HINT)
async def inlay_hint(params: lsp.InlayHintParams) -> list[lsp.InlayHint] | None:
    """Provide inlay hints."""
    uri = params.text_document.uri
    doc = server.get_document(uri)
    if doc is None:
        return None

    hints = await server.python_delegate.get_inlay_hints(
        doc.source,
        params.range.start.line,
        params.range.end.line,
        doc.path,
    )
    return hints or None


@server.feature(lsp.INLAY_HINT_RESOLVE)
async def inlay_hint_resolve(hint: lsp.InlayHint) -> lsp.InlayHint:
    """Resolve additional inlay hint details."""
    return await server.python_delegate.resolve_inlay_hint(hint)


# ============================================================================
# Workspace Symbols
# ============================================================================


@server.feature(lsp.WORKSPACE_SYMBOL)
async def workspace_symbol(params: lsp.WorkspaceSymbolParams) -> list[lsp.WorkspaceSymbol] | None:
    """Provide workspace symbol search."""
    symbols = await server.python_delegate.get_workspace_symbols(params.query)
    return symbols or None


@server.feature(lsp.WORKSPACE_SYMBOL_RESOLVE)
async def workspace_symbol_resolve(symbol: lsp.WorkspaceSymbol) -> lsp.WorkspaceSymbol:
    """Resolve additional workspace symbol details."""
    return await server.python_delegate.resolve_workspace_symbol(symbol)


# ============================================================================
# Code Actions
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_CODE_ACTION)
async def code_action(params: lsp.CodeActionParams) -> list[lsp.CodeAction] | None:
    """Provide code actions."""
    uri = params.text_document.uri
    doc = server.get_document(uri)
    if doc is None:
        return None

    actions: list[lsp.CodeAction] = []

    # Check diagnostics for quick fixes
    for diagnostic in params.context.diagnostics:
        if diagnostic.source == "xonsh-lsp":
            # Add quick fix for undefined environment variable
            if "undefined environment variable" in (diagnostic.message or "").lower():
                var_name = diagnostic.data.get("var_name") if diagnostic.data else None
                if var_name:
                    actions.append(
                        lsp.CodeAction(
                            title=f"Define ${var_name}",
                            kind=lsp.CodeActionKind.QuickFix,
                            diagnostics=[diagnostic],
                            edit=lsp.WorkspaceEdit(
                                changes={
                                    uri: [
                                        lsp.TextEdit(
                                            range=lsp.Range(
                                                start=lsp.Position(line=0, character=0),
                                                end=lsp.Position(line=0, character=0),
                                            ),
                                            new_text=f'${var_name} = ""\n',
                                        )
                                    ]
                                }
                            ),
                        )
                    )

    return actions if actions else None


# ============================================================================
# Workspace Configuration
# ============================================================================


@server.feature(lsp.WORKSPACE_DID_CHANGE_CONFIGURATION)
async def did_change_configuration(params: lsp.DidChangeConfigurationParams) -> None:
    """Handle workspace configuration changes."""
    from xonsh_lsp.lsp_proxy_backend import LspProxyBackend
    if isinstance(server.python_backend, LspProxyBackend):
        await server.python_backend.update_settings(params.settings)


# ============================================================================
# Execute Command
# ============================================================================


@server.feature(lsp.WORKSPACE_EXECUTE_COMMAND)
async def execute_command(params: lsp.ExecuteCommandParams) -> object | None:
    """Execute a command."""
    if params.command == XonshLanguageServer.CMD_SHOW_ENV_VARS:
        # Show environment variables (informational)
        env_vars = dict(os.environ)
        return {"env_vars": env_vars}

    elif params.command == XonshLanguageServer.CMD_RUN_SELECTION:
        # This would require xonsh execution - placeholder
        return {"status": "not_implemented"}

    return None


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for the language server."""
    parser = argparse.ArgumentParser(
        description="Xonsh Language Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stdio",
        action="store_true",
        default=True,
        help="Use stdio for communication (default)",
    )
    parser.add_argument(
        "--tcp",
        action="store_true",
        help="Use TCP for communication",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2087,
        help="TCP port (default: 2087)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="xonsh-lsp 0.1.5",
    )
    parser.add_argument(
        "--python-backend",
        choices=["jedi", "lsp-proxy"] + list(KNOWN_BACKENDS.keys()),
        default="jedi",
        help="Python analysis backend (default: jedi)",
    )
    parser.add_argument(
        "--backend-command",
        nargs="+",
        help='Command to start the backend LSP server (e.g. "pyright-langserver --stdio")',
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Store backend configuration for use during initialization
    server._backend_name = args.python_backend
    server._backend_command = args.backend_command

    if args.tcp:
        logger.info(f"Starting xonsh-lsp in TCP mode on {args.host}:{args.port}")
        server.start_tcp(args.host, args.port)
    else:
        logger.info("Starting xonsh-lsp in stdio mode")
        server.start_io()


if __name__ == "__main__":
    main()
