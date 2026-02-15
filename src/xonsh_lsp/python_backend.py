"""
Python backend protocol for xonsh LSP.

Defines the interface that all Python analysis backends must implement.
Backends include JediBackend (built-in) and LspProxyBackend (delegates to
an external LSP server like Pyright, basedpyright, or pylsp).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lsprotocol import types as lsp


@runtime_checkable
class PythonBackend(Protocol):
    """Protocol for Python analysis backends.

    All methods are async to support both synchronous backends (Jedi)
    and asynchronous ones (LSP proxy). Synchronous backends can simply
    return their results directly from async methods.
    """

    async def start(self, workspace_root: str | None = None) -> None:
        """Start the backend. Called once during server initialization.

        Args:
            workspace_root: Path to the workspace root directory.
        """
        ...

    async def stop(self) -> None:
        """Stop the backend. Called during server shutdown."""
        ...

    async def get_completions(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> list[lsp.CompletionItem]:
        """Get Python completions at the given position.

        Args:
            source: The original xonsh source code.
            line: 0-based line number.
            col: 0-based column number.
            path: Optional file path for context.

        Returns:
            List of completion items.
        """
        ...

    async def get_hover(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> str | None:
        """Get hover information at the given position.

        Args:
            source: The original xonsh source code.
            line: 0-based line number.
            col: 0-based column number.
            path: Optional file path for context.

        Returns:
            Markdown hover content, or None.
        """
        ...

    async def get_definitions(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> list[lsp.Location]:
        """Get definition locations for symbol at position.

        Args:
            source: The original xonsh source code.
            line: 0-based line number.
            col: 0-based column number.
            path: Optional file path for context.

        Returns:
            List of definition locations.
        """
        ...

    async def get_references(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> list[lsp.Location]:
        """Get reference locations for symbol at position.

        Args:
            source: The original xonsh source code.
            line: 0-based line number.
            col: 0-based column number.
            path: Optional file path for context.

        Returns:
            List of reference locations.
        """
        ...

    async def get_diagnostics(
        self, source: str, path: str | None = None
    ) -> list[lsp.Diagnostic]:
        """Get Python diagnostics for the source.

        Args:
            source: The original xonsh source code.
            path: Optional file path for context.

        Returns:
            List of diagnostics.
        """
        ...

    async def get_document_symbols(
        self, source: str, path: str | None = None
    ) -> list[lsp.DocumentSymbol]:
        """Get document symbols from the source.

        Args:
            source: The original xonsh source code.
            path: Optional file path for context.

        Returns:
            List of document symbols.
        """
        ...

    async def get_signature_help(
        self, source: str, line: int, col: int, path: str | None = None
    ) -> lsp.SignatureHelp | None:
        """Get signature help at the given position.

        Args:
            source: The original xonsh source code.
            line: 0-based line number.
            col: 0-based column number.
            path: Optional file path for context.

        Returns:
            Signature help, or None.
        """
        ...

    async def get_inlay_hints(
        self,
        source: str,
        start_line: int,
        end_line: int,
        path: str | None = None,
    ) -> list[lsp.InlayHint]:
        """Get inlay hints for the given line range.

        Args:
            source: The original xonsh source code.
            start_line: 0-based start line of the range.
            end_line: 0-based end line of the range.
            path: Optional file path for context.

        Returns:
            List of inlay hints.
        """
        ...

    async def resolve_inlay_hint(
        self,
        hint: lsp.InlayHint,
        path: str | None = None,
    ) -> lsp.InlayHint:
        """Resolve additional details for an inlay hint.

        Args:
            hint: The inlay hint to resolve.
            path: Optional file path for context.

        Returns:
            The resolved inlay hint.
        """
        ...
