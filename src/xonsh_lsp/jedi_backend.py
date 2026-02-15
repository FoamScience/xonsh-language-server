"""
Jedi-based Python backend for xonsh LSP.

This module provides Python-specific language features by delegating to Jedi,
It implements the PythonBackend protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lsprotocol import types as lsp

from xonsh_lsp.preprocessing import (
    PreprocessResult,
    map_position_from_processed,
    map_position_to_processed,
    preprocess_source,
    preprocess_with_mapping,
)

try:
    import jedi
    from jedi import Script
    from jedi.api.classes import Completion, Name

    JEDI_AVAILABLE = True
except ImportError:
    JEDI_AVAILABLE = False
    Script = object  # type: ignore
    Completion = object  # type: ignore
    Name = object  # type: ignore

logger = logging.getLogger(__name__)


class JediBackend:
    """Python analysis backend using Jedi.

    Implements the PythonBackend protocol. All public methods are async
    but call Jedi synchronously internally.
    """

    def __init__(self) -> None:
        if not JEDI_AVAILABLE:
            logger.warning("Jedi not available - Python features will be limited")

    async def start(self, workspace_root: str | None = None) -> None:
        """Start the backend (no-op for Jedi)."""
        pass

    async def stop(self) -> None:
        """Stop the backend (no-op for Jedi)."""
        pass

    async def get_completions(
        self,
        source: str,
        line: int,
        col: int,
        path: str | None = None,
    ) -> list[lsp.CompletionItem]:
        """Get Python completions using Jedi."""
        if not JEDI_AVAILABLE:
            return []

        try:
            # Preprocess xonsh syntax to valid Python with position mapping
            preprocess_result = preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )

            # Jedi uses 1-based line numbers
            script = Script(preprocess_result.source, path=path)
            completions = script.complete(mapped_line + 1, mapped_col)

            items = []
            for c in completions:
                kind = self._jedi_type_to_lsp_kind(c.type)
                items.append(
                    lsp.CompletionItem(
                        label=c.name,
                        kind=kind,
                        detail=c.description,
                        documentation=self._get_completion_docs(c),
                        insert_text=c.name,
                        sort_text=f"5_{c.name}",  # Lower priority than xonsh items
                    )
                )

            return items

        except Exception as e:
            logger.debug(f"Jedi completion error: {e}")
            return []

    async def get_hover(
        self,
        source: str,
        line: int,
        col: int,
        path: str | None = None,
    ) -> str | None:
        """Get Python hover information using Jedi."""
        if not JEDI_AVAILABLE:
            return None

        try:
            # Preprocess xonsh syntax to valid Python with position mapping
            preprocess_result = preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )

            script = Script(preprocess_result.source, path=path)
            names = script.infer(mapped_line + 1, mapped_col)

            if not names:
                # Try goto instead
                names = script.goto(mapped_line + 1, mapped_col)

            if not names:
                return None

            name = names[0]
            hover_parts = []

            # Add signature if available
            if name.type == "function":
                try:
                    sigs = script.get_signatures(mapped_line + 1, mapped_col)
                    if sigs:
                        sig = sigs[0]
                        hover_parts.append(f"```python\n{sig.to_string()}\n```")
                except Exception:
                    pass

            # Add type/description
            if name.description:
                hover_parts.append(f"**{name.description}**")

            # Add docstring
            docstring = name.docstring()
            if docstring:
                hover_parts.append(f"\n{docstring}")

            return "\n\n".join(hover_parts) if hover_parts else None

        except Exception as e:
            logger.debug(f"Jedi hover error: {e}")
            return None

    async def get_definitions(
        self,
        source: str,
        line: int,
        col: int,
        path: str | None = None,
    ) -> list[lsp.Location]:
        """Get Python definitions using Jedi."""
        if not JEDI_AVAILABLE:
            return []

        try:
            # Preprocess xonsh syntax to valid Python with position mapping
            preprocess_result = preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )

            script = Script(preprocess_result.source, path=path)
            names = script.goto(mapped_line + 1, mapped_col)

            locations = []
            for name in names:
                if name.module_path:
                    result_line = name.line - 1 if name.line else 0
                    result_col = name.column or 0

                    # Map positions back if this is the same file
                    if path and str(name.module_path) == path:
                        _, result_col = map_position_from_processed(
                            preprocess_result, result_line, result_col
                        )

                    locations.append(
                        lsp.Location(
                            uri=Path(name.module_path).as_uri(),
                            range=lsp.Range(
                                start=lsp.Position(
                                    line=result_line,
                                    character=result_col,
                                ),
                                end=lsp.Position(
                                    line=result_line,
                                    character=result_col + len(name.name),
                                ),
                            ),
                        )
                    )

            return locations

        except Exception as e:
            logger.debug(f"Jedi definition error: {e}")
            return []

    async def get_references(
        self,
        source: str,
        line: int,
        col: int,
        path: str | None = None,
    ) -> list[lsp.Location]:
        """Get Python references using Jedi."""
        if not JEDI_AVAILABLE:
            return []

        try:
            # Preprocess xonsh syntax to valid Python with position mapping
            preprocess_result = preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )

            script = Script(preprocess_result.source, path=path)
            names = script.get_references(mapped_line + 1, mapped_col)

            locations = []
            for name in names:
                if name.module_path:
                    result_line = name.line - 1 if name.line else 0
                    result_col = name.column or 0

                    # Map positions back if this is the same file
                    if path and str(name.module_path) == path:
                        _, result_col = map_position_from_processed(
                            preprocess_result, result_line, result_col
                        )

                    locations.append(
                        lsp.Location(
                            uri=Path(name.module_path).as_uri(),
                            range=lsp.Range(
                                start=lsp.Position(
                                    line=result_line,
                                    character=result_col,
                                ),
                                end=lsp.Position(
                                    line=result_line,
                                    character=result_col + len(name.name),
                                ),
                            ),
                        )
                    )

            return locations

        except Exception as e:
            logger.debug(f"Jedi references error: {e}")
            return []

    async def get_diagnostics(
        self,
        source: str,
        path: str | None = None,
    ) -> list[lsp.Diagnostic]:
        """Get Python diagnostics using Jedi."""
        if not JEDI_AVAILABLE:
            return []

        try:
            # Preprocess xonsh syntax to valid Python
            preprocess_result = preprocess_with_mapping(source)
            script = Script(preprocess_result.source, path=path)
            errors = script.get_syntax_errors()

            xonsh_lines = preprocess_result.xonsh_lines

            diagnostics = []
            for error in errors:
                error_line = error.line - 1 if error.line else 0

                # Skip errors on lines that had xonsh syntax in the original
                # Also check nearby lines since preprocessing can cause errors
                # on subsequent lines (e.g., unbalanced parens)
                start_check = max(0, error_line - 10)
                if any(ln in xonsh_lines for ln in range(start_check, error_line + 1)):
                    continue

                diagnostics.append(
                    lsp.Diagnostic(
                        range=lsp.Range(
                            start=lsp.Position(
                                line=error_line,
                                character=error.column or 0,
                            ),
                            end=lsp.Position(
                                line=error.until_line - 1 if error.until_line else error_line,
                                character=error.until_column or (error.column or 0) + 1,
                            ),
                        ),
                        message=str(error),
                        severity=lsp.DiagnosticSeverity.Error,
                        source="jedi",
                        code="python-syntax-error",
                    )
                )

            return diagnostics

        except Exception as e:
            logger.debug(f"Jedi diagnostics error: {e}")
            return []

    async def get_document_symbols(
        self,
        source: str,
        path: str | None = None,
    ) -> list[lsp.DocumentSymbol]:
        """Get Python document symbols using Jedi."""
        if not JEDI_AVAILABLE:
            return []

        try:
            # Preprocess xonsh syntax to valid Python
            processed_source = preprocess_source(source)
            script = Script(processed_source, path=path)
            names = script.get_names(all_scopes=True, definitions=True)

            symbols = []
            for name in names:
                kind = self._jedi_type_to_lsp_symbol_kind(name.type)
                if name.line is None:
                    continue

                symbol = lsp.DocumentSymbol(
                    name=name.name,
                    kind=kind,
                    range=lsp.Range(
                        start=lsp.Position(
                            line=name.line - 1,
                            character=name.column or 0,
                        ),
                        end=lsp.Position(
                            line=name.line - 1,
                            character=(name.column or 0) + len(name.name),
                        ),
                    ),
                    selection_range=lsp.Range(
                        start=lsp.Position(
                            line=name.line - 1,
                            character=name.column or 0,
                        ),
                        end=lsp.Position(
                            line=name.line - 1,
                            character=(name.column or 0) + len(name.name),
                        ),
                    ),
                    detail=name.description,
                )
                symbols.append(symbol)

            return symbols

        except Exception as e:
            logger.debug(f"Jedi symbols error: {e}")
            return []

    async def get_signature_help(
        self,
        source: str,
        line: int,
        col: int,
        path: str | None = None,
    ) -> lsp.SignatureHelp | None:
        """Get Python signature help using Jedi."""
        if not JEDI_AVAILABLE:
            return None

        try:
            # Preprocess xonsh syntax to valid Python with position mapping
            preprocess_result = preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = map_position_to_processed(
                preprocess_result, line, col
            )

            script = Script(preprocess_result.source, path=path)
            signatures = script.get_signatures(mapped_line + 1, mapped_col)

            if not signatures:
                return None

            lsp_signatures = []
            active_signature = 0
            active_parameter = 0

            for i, sig in enumerate(signatures):
                params = []
                for param in sig.params:
                    params.append(
                        lsp.ParameterInformation(
                            label=param.name,
                            documentation=param.description if hasattr(param, 'description') else None,
                        )
                    )

                lsp_sig = lsp.SignatureInformation(
                    label=sig.to_string(),
                    documentation=sig.docstring(),
                    parameters=params,
                )
                lsp_signatures.append(lsp_sig)

                if sig.index is not None:
                    active_signature = i
                    active_parameter = sig.index

            return lsp.SignatureHelp(
                signatures=lsp_signatures,
                active_signature=active_signature,
                active_parameter=active_parameter,
            )

        except Exception as e:
            logger.debug(f"Jedi signature help error: {e}")
            return None

    async def get_inlay_hints(
        self,
        source: str,
        start_line: int,
        end_line: int,
        path: str | None = None,
    ) -> list[lsp.InlayHint]:
        return []

    async def resolve_inlay_hint(
        self,
        hint: lsp.InlayHint,
        path: str | None = None,
    ) -> lsp.InlayHint:
        return hint

    async def get_workspace_symbols(self, query: str) -> list[lsp.WorkspaceSymbol]:
        return []

    async def resolve_workspace_symbol(
        self, symbol: lsp.WorkspaceSymbol
    ) -> lsp.WorkspaceSymbol:
        return symbol

    def _jedi_type_to_lsp_kind(self, jedi_type: str) -> lsp.CompletionItemKind:
        """Convert Jedi completion type to LSP completion item kind."""
        mapping = {
            "module": lsp.CompletionItemKind.Module,
            "class": lsp.CompletionItemKind.Class,
            "instance": lsp.CompletionItemKind.Variable,
            "function": lsp.CompletionItemKind.Function,
            "param": lsp.CompletionItemKind.Variable,
            "path": lsp.CompletionItemKind.File,
            "keyword": lsp.CompletionItemKind.Keyword,
            "property": lsp.CompletionItemKind.Property,
            "statement": lsp.CompletionItemKind.Variable,
        }
        return mapping.get(jedi_type, lsp.CompletionItemKind.Text)

    def _jedi_type_to_lsp_symbol_kind(self, jedi_type: str) -> lsp.SymbolKind:
        """Convert Jedi type to LSP symbol kind."""
        mapping = {
            "module": lsp.SymbolKind.Module,
            "class": lsp.SymbolKind.Class,
            "instance": lsp.SymbolKind.Variable,
            "function": lsp.SymbolKind.Function,
            "param": lsp.SymbolKind.Variable,
            "property": lsp.SymbolKind.Property,
            "statement": lsp.SymbolKind.Variable,
        }
        return mapping.get(jedi_type, lsp.SymbolKind.Variable)

    def _get_completion_docs(self, completion: Completion) -> str | None:
        """Get documentation for a completion item."""
        try:
            docstring = completion.docstring()
            if docstring:
                # Truncate very long docstrings
                if len(docstring) > 1000:
                    docstring = docstring[:1000] + "..."
                return docstring
        except Exception:
            pass
        return None
