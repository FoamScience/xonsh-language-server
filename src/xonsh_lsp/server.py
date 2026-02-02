"""
Xonsh Language Server

Main LSP server implementation using pygls.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from xonsh_lsp.completions import XonshCompletionProvider
from xonsh_lsp.diagnostics import XonshDiagnosticsProvider
from xonsh_lsp.hover import XonshHoverProvider
from xonsh_lsp.definition import XonshDefinitionProvider
from xonsh_lsp.parser import XonshParser, ParseResult
from xonsh_lsp.python_delegate import PythonDelegate

if TYPE_CHECKING:
    from pygls.workspace import TextDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class XonshLanguageServer(LanguageServer):
    """Language Server for xonsh files."""

    CMD_SHOW_ENV_VARS = "xonsh.showEnvVars"
    CMD_RUN_SELECTION = "xonsh.runSelection"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize components
        self.parser = XonshParser()
        self.python_delegate = PythonDelegate()
        self.completion_provider = XonshCompletionProvider(self)
        self.diagnostics_provider = XonshDiagnosticsProvider(self)
        self.hover_provider = XonshHoverProvider(self)
        self.definition_provider = XonshDefinitionProvider(self)

        # Cache for parsed documents
        self._parse_cache: dict[str, ParseResult] = {}

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
    version="0.1.0",
)


# ============================================================================
# Lifecycle Events
# ============================================================================


@server.feature(lsp.INITIALIZED)
def initialized(params: lsp.InitializedParams) -> None:
    """Handle the initialized notification."""
    logger.info("xonsh-lsp initialized")


# ============================================================================
# Document Events
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
    """Handle document open."""
    uri = params.text_document.uri
    logger.debug(f"Document opened: {uri}")

    # Run diagnostics
    diagnostics = server.diagnostics_provider.get_diagnostics(uri)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(params: lsp.DidChangeTextDocumentParams) -> None:
    """Handle document change."""
    uri = params.text_document.uri
    logger.debug(f"Document changed: {uri}")

    # Run diagnostics
    diagnostics = server.diagnostics_provider.get_diagnostics(uri)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
def did_save(params: lsp.DidSaveTextDocumentParams) -> None:
    """Handle document save."""
    uri = params.text_document.uri
    logger.debug(f"Document saved: {uri}")

    # Run diagnostics
    diagnostics = server.diagnostics_provider.get_diagnostics(uri)
    server.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def did_close(params: lsp.DidCloseTextDocumentParams) -> None:
    """Handle document close."""
    uri = params.text_document.uri
    logger.debug(f"Document closed: {uri}")

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
def completion(params: lsp.CompletionParams) -> lsp.CompletionList | None:
    """Provide completions."""
    return server.completion_provider.get_completions(params)


@server.feature(lsp.COMPLETION_ITEM_RESOLVE)
def completion_resolve(item: lsp.CompletionItem) -> lsp.CompletionItem:
    """Resolve additional completion item details."""
    return server.completion_provider.resolve_completion(item)


# ============================================================================
# Hover
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(params: lsp.HoverParams) -> lsp.Hover | None:
    """Provide hover information."""
    return server.hover_provider.get_hover(params)


# ============================================================================
# Go to Definition
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
def definition(
    params: lsp.DefinitionParams,
) -> lsp.Location | list[lsp.Location] | None:
    """Provide go-to-definition."""
    return server.definition_provider.get_definition(params)


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
def signature_help(params: lsp.SignatureHelpParams) -> lsp.SignatureHelp | None:
    """Provide signature help."""
    uri = params.text_document.uri
    doc = server.get_document(uri)
    if doc is None:
        return None

    return server.python_delegate.get_signature_help(
        doc.source,
        params.position.line,
        params.position.character,
        doc.path,
    )


# ============================================================================
# Find References
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
def references(params: lsp.ReferenceParams) -> list[lsp.Location] | None:
    """Provide find references."""
    uri = params.text_document.uri
    doc = server.get_document(uri)
    if doc is None:
        return None

    line = params.position.line
    col = params.position.character

    # Get Python references from Jedi
    python_refs = server.python_delegate.get_references(
        doc.source, line, col, doc.path
    )

    # Also search for xonsh-specific references (env vars, aliases)
    from xonsh_lsp.definition import XonshReferenceProvider
    ref_provider = XonshReferenceProvider(server)
    xonsh_refs = ref_provider.get_references(params)

    all_refs = python_refs + (xonsh_refs or [])

    # Deduplicate by (uri, start_line, start_char)
    seen = set()
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
def document_symbols(
    params: lsp.DocumentSymbolParams,
) -> list[lsp.DocumentSymbol] | list[lsp.SymbolInformation] | None:
    """Provide document symbols."""
    uri = params.text_document.uri
    logger.info(f"Document symbols requested for {uri}")
    doc = server.get_document(uri)
    if doc is None:
        logger.warning(f"No document found for {uri}")
        return None

    # Use tree-sitter for symbol extraction (handles xonsh syntax)
    raw_symbols = server.parser.get_document_symbols(doc.source)
    logger.info(f"Found {len(raw_symbols)} symbols from tree-sitter")

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
# Code Actions
# ============================================================================


@server.feature(lsp.TEXT_DOCUMENT_CODE_ACTION)
def code_action(params: lsp.CodeActionParams) -> list[lsp.CodeAction] | None:
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
# Execute Command
# ============================================================================


@server.feature(lsp.WORKSPACE_EXECUTE_COMMAND)
def execute_command(params: lsp.ExecuteCommandParams) -> object | None:
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
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="xonsh-lsp 0.1.0",
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.tcp:
        logger.info(f"Starting xonsh-lsp in TCP mode on {args.host}:{args.port}")
        server.start_tcp(args.host, args.port)
    else:
        logger.info("Starting xonsh-lsp in stdio mode")
        server.start_io()


if __name__ == "__main__":
    main()
