"""
Python language features delegate using Jedi.

This module provides Python-specific language features by delegating to Jedi,
a popular Python static analysis library.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lsprotocol import types as lsp

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

# Patterns that must be processed before balanced delimiter matching
PRE_BALANCED_PATTERNS = [
    # Macro call: func!(args) -> func("args") - MUST be before !(cmd) balanced pattern
    # Args are converted to a string since they may not be valid Python
    (re.compile(r'(\w+)!\(([^)]*)\)'), r'\1("\2")'),
]

# Simple regex patterns for xonsh syntax (patterns that don't need balanced matching)
# NOTE: Order matters! More specific patterns must come before general ones.
SIMPLE_PATTERNS = [
    # At object access: @.imp -> __xonsh_at__.imp (lazy import)
    (re.compile(r'@\.'), '__xonsh_at__.'),
    # Braced env var: ${expr} -> __xonsh_env__[expr] (expr is evaluated as Python)
    (re.compile(r'\$\{([^}]*)\}'), r'__xonsh_env__[\1]'),
    # Simple env var: $VAR -> __xonsh_env__["VAR"] (but not $( or $[ or ${)
    (re.compile(r'\$([a-zA-Z_][a-zA-Z0-9_]*)(?![(\[{])'), r'__xonsh_env__["\1"]'),
    # Glob pattern: g`pattern` -> "__glob__" - MUST be before regex glob
    (re.compile(r'g`[^`]*`'), '"__glob__"'),
    # Formatted glob: f`pattern` -> "__glob__"
    (re.compile(r'f`[^`]*`'), '"__glob__"'),
    # Regex glob: `pattern` -> "__glob__"
    (re.compile(r'`[^`]*`'), '"__glob__"'),
    # Path literals: p"path" -> Path("path"), pf"path" -> Path(f"path"), etc.
    # This allows Jedi to understand these are Path objects for method completion
    (re.compile(r'p(["\'])([^"\']*)\1'), r'Path(\1\2\1)'),
    (re.compile(r'pf(["\'])([^"\']*)\1'), r'Path(f\1\2\1)'),
    (re.compile(r'pr(["\'])([^"\']*)\1'), r'Path(r\1\2\1)'),
    (re.compile(r'pb(["\'])([^"\']*)\1'), r'Path(b\1\2\1)'),
    # Handle triple-quoted path literals
    (re.compile(r'p(""".*?"""|\'\'\'.*?\'\'\')'), r'Path(\1)'),
    (re.compile(r'pf(""".*?"""|\'\'\'.*?\'\'\')'), r'Path(f\1)'),
    (re.compile(r'pr(""".*?"""|\'\'\'.*?\'\'\')'), r'Path(r\1)'),
    (re.compile(r'pb(""".*?"""|\'\'\'.*?\'\'\')'), r'Path(b\1)'),
    # Xontrib statement: xontrib load ... (including continuations) -> pass
    (re.compile(r'^xontrib\s+load\s+.*?(?:\\\n.*?)*(?=\n|$)', re.MULTILINE), 'pass  # xontrib'),
]

# Patterns that need balanced delimiter matching (prefix, open_char, close_char, replacement)
BALANCED_PATTERNS = [
    # @$(cmd) - tokenized substitution (must be before $())
    ('@$', '(', ')', '__xonsh_subproc__()'),
    # $(cmd) - captured subprocess
    ('$', '(', ')', '__xonsh_subproc__()'),
    # !(cmd) - captured subprocess object
    ('!', '(', ')', '__xonsh_subproc__()'),
    # $[cmd] - uncaptured subprocess
    ('$', '[', ']', '__xonsh_subproc__()'),
    # ![cmd] - uncaptured subprocess object
    ('!', '[', ']', '__xonsh_subproc__()'),
    # @(expr) - Python evaluation in subprocess
    ('@', '(', ')', '__xonsh_pyeval__'),
]


@dataclass
class PreprocessResult:
    """Result of preprocessing xonsh source."""

    source: str
    """The preprocessed Python source."""

    line_mappings: list[list[tuple[int, int]]]
    """Per-line mappings from original column to preprocessed column.

    Each entry is a list of (original_col, preprocessed_col) pairs for that line,
    representing the column positions AFTER each xonsh construct on the line.
    """

    added_lines_before: int = 0
    """Number of lines added at the beginning (e.g., for imports)."""

    added_line_position: int = 0
    """Line position where lines were added."""


def _build_line_mapping(original_line: str, processed_line: str) -> list[tuple[int, int]]:
    """Build a column mapping between original and processed line.

    Returns a list of (original_col, processed_col) anchor points where
    the mapping changes due to xonsh syntax preprocessing.
    """
    # Find positions where changes occurred by looking at the diff
    # This is a simplified approach - we track key anchor points

    mapping: list[tuple[int, int]] = [(0, 0)]

    # Find common prefix
    common_prefix_len = 0
    min_len = min(len(original_line), len(processed_line))
    while common_prefix_len < min_len:
        if original_line[common_prefix_len] != processed_line[common_prefix_len]:
            break
        common_prefix_len += 1

    if common_prefix_len > 0:
        mapping.append((common_prefix_len, common_prefix_len))

    # Find common suffix
    common_suffix_len = 0
    while common_suffix_len < min_len - common_prefix_len:
        if original_line[-(common_suffix_len + 1)] != processed_line[-(common_suffix_len + 1)]:
            break
        common_suffix_len += 1

    if common_suffix_len > 0:
        orig_suffix_start = len(original_line) - common_suffix_len
        proc_suffix_start = len(processed_line) - common_suffix_len
        mapping.append((orig_suffix_start, proc_suffix_start))

    # Add end points
    mapping.append((len(original_line), len(processed_line)))

    return mapping


def _map_column(col: int, mapping: list[tuple[int, int]], to_processed: bool = True) -> int:
    """Map a column position using line mapping.

    Args:
        col: Column position to map
        mapping: List of (original_col, processed_col) anchor points
        to_processed: If True, map original->processed; otherwise processed->original

    Returns:
        Mapped column position
    """
    if not mapping:
        return col

    src_idx = 0 if to_processed else 1
    dst_idx = 1 if to_processed else 0

    # Find the segment this column falls into
    for i in range(len(mapping) - 1):
        start = mapping[i]
        end = mapping[i + 1]

        if start[src_idx] <= col < end[src_idx]:
            # Linear interpolation within segment
            src_range = end[src_idx] - start[src_idx]
            dst_range = end[dst_idx] - start[dst_idx]

            if src_range == 0:
                return start[dst_idx]

            offset = col - start[src_idx]
            ratio = offset / src_range
            return int(start[dst_idx] + ratio * dst_range)

    # Past the end - use last mapping
    last = mapping[-1]
    return last[dst_idx]


def _find_balanced_end(source: str, start: int, open_char: str, close_char: str) -> int:
    """Find the position of the matching closing delimiter, handling nesting.

    Args:
        source: The source string
        start: Position right after the opening delimiter
        open_char: The opening delimiter character
        close_char: The closing delimiter character

    Returns:
        Position of the matching closing delimiter, or -1 if not found
    """
    depth = 1
    i = start
    in_string = None
    escape_next = False

    while i < len(source) and depth > 0:
        c = source[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if c == '\\':
            escape_next = True
            i += 1
            continue

        # Track string boundaries
        if in_string:
            if c == in_string:
                in_string = None
        else:
            if c in ('"', "'"):
                # Check for triple quotes
                if i + 2 < len(source) and source[i:i+3] == c * 3:
                    # Find matching triple quote
                    end = source.find(c * 3, i + 3)
                    if end == -1:
                        return -1
                    i = end + 3
                    continue
                in_string = c
            elif c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return i

        i += 1

    return -1


def _replace_balanced_patterns(source: str) -> str:
    """Replace patterns that need balanced delimiter matching."""
    result = source

    # Process patterns in order (longer prefixes first for @$ vs $)
    for prefix, open_char, close_char, replacement in BALANCED_PATTERNS:
        new_result = []
        i = 0
        while i < len(result):
            # Look for the prefix followed by the open char
            pattern_start = prefix + open_char
            if result[i:i+len(pattern_start)] == pattern_start:
                # Find the matching close
                content_start = i + len(pattern_start)
                close_pos = _find_balanced_end(result, content_start, open_char, close_char)
                if close_pos != -1:
                    # For @(expr), we want to preserve the expression
                    if prefix == '@' and open_char == '(':
                        content = result[content_start:close_pos]
                        new_result.append(f'({content})')
                    else:
                        new_result.append(replacement)
                    i = close_pos + 1
                    continue
            new_result.append(result[i])
            i += 1
        result = ''.join(new_result)

    return result


class PythonDelegate:
    """Delegate Python language features to Jedi."""

    def __init__(self):
        if not JEDI_AVAILABLE:
            logger.warning("Jedi not available - Python features will be limited")

    def _preprocess_source(self, source: str) -> str:
        """Preprocess xonsh source to make it valid Python for Jedi.

        Replaces xonsh-specific syntax with Python placeholders so Jedi
        can parse and analyze the code without errors.
        """
        return self._preprocess_with_mapping(source).source

    def _preprocess_with_mapping(self, source: str) -> PreprocessResult:
        """Preprocess xonsh source and build position mapping.

        Returns the preprocessed source along with line-by-line column mappings
        that can be used to convert positions between original and preprocessed.
        """
        result = source

        # First, apply patterns that must run before balanced matching (e.g., macros)
        for pattern, replacement in PRE_BALANCED_PATTERNS:
            result = pattern.sub(replacement, result)

        # Then handle patterns that need balanced delimiter matching
        result = _replace_balanced_patterns(result)

        # Finally apply simple regex patterns
        for pattern, replacement in SIMPLE_PATTERNS:
            result = pattern.sub(replacement, result)

        # Track if we need to add Path import (affects line numbering)
        path_import_added = False
        path_import_line = 0

        # Add Path import if path literals were used (Path() appears in result)
        if 'Path(' in result and 'from pathlib import Path' not in result:
            # Add import at the beginning, after any existing imports or docstrings
            lines = result.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Skip docstrings, comments, and empty lines at the start
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Find end of docstring
                    if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        insert_pos = i + 1
                    else:
                        for j in range(i + 1, len(lines)):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                insert_pos = j + 1
                                break
                    break
                elif stripped.startswith('#') or not stripped:
                    insert_pos = i + 1
                elif stripped.startswith('import ') or stripped.startswith('from '):
                    insert_pos = i + 1
                else:
                    break
            lines.insert(insert_pos, 'from pathlib import Path')
            result = '\n'.join(lines)
            path_import_added = True
            path_import_line = insert_pos

        # Build line mappings
        original_lines = source.splitlines()
        processed_lines = result.splitlines()

        # Build mappings accounting for added import lines
        line_mappings: list[list[tuple[int, int]]] = []
        added_lines = 1 if path_import_added else 0

        for i in range(len(original_lines)):
            orig_line = original_lines[i]
            # Map to the corresponding preprocessed line (offset by added imports)
            proc_idx = i + added_lines if i >= path_import_line else i
            if path_import_added and i >= path_import_line:
                proc_idx = i + 1
            proc_line = processed_lines[proc_idx] if proc_idx < len(processed_lines) else ""
            line_mappings.append(_build_line_mapping(orig_line, proc_line))

        return PreprocessResult(
            source=result,
            line_mappings=line_mappings,
            added_lines_before=added_lines,
            added_line_position=path_import_line if path_import_added else 0,
        )

    def map_position_to_processed(
        self, preprocess_result: PreprocessResult, line: int, col: int
    ) -> tuple[int, int]:
        """Map a position from original source to preprocessed source."""
        # Adjust line number for added imports
        proc_line = line
        if preprocess_result.added_lines_before > 0:
            if line >= preprocess_result.added_line_position:
                proc_line = line + preprocess_result.added_lines_before

        if line >= len(preprocess_result.line_mappings):
            return (proc_line, col)
        mapping = preprocess_result.line_mappings[line]
        return (proc_line, _map_column(col, mapping, to_processed=True))

    def map_position_from_processed(
        self, preprocess_result: PreprocessResult, line: int, col: int
    ) -> tuple[int, int]:
        """Map a position from preprocessed source back to original source."""
        # Adjust line number for added imports
        orig_line = line
        if preprocess_result.added_lines_before > 0:
            if line >= preprocess_result.added_line_position + preprocess_result.added_lines_before:
                orig_line = line - preprocess_result.added_lines_before
            elif line >= preprocess_result.added_line_position:
                # This line is the added import - map to the insert position
                orig_line = preprocess_result.added_line_position

        if orig_line >= len(preprocess_result.line_mappings):
            return (orig_line, col)
        mapping = preprocess_result.line_mappings[orig_line]
        return (orig_line, _map_column(col, mapping, to_processed=False))

    def _has_xonsh_syntax(self, line: str) -> bool:
        """Check if a line contains xonsh-specific syntax."""
        xonsh_indicators = [
            '$', '!(', '![', '@(', '@$(',  # Subprocess and env vars
            '`',  # Globs
            '@.',  # At object access (lazy imports)
        ]
        # Check for xontrib statement
        if line.strip().startswith('xontrib '):
            return True
        # Check for macro call pattern: identifier!(
        if re.search(r'\w+!\(', line):
            return True
        # Check for bare subprocess indicators
        # Shell logical operators && and ||
        if '&&' in line or '||' in line:
            return True
        # Shell pipe operator (single |, not ||)
        if re.search(r'(?<!\|)\|(?!\|)', line):
            return True
        # Shell flags: -x or --flag
        if re.search(r'\s-[a-zA-Z]|\s--[a-zA-Z]', line):
            return True
        # Path-like command at start: ./cmd, /usr/bin/cmd, ~/bin/cmd
        stripped = line.lstrip()
        if stripped.startswith('./') or stripped.startswith('/') or stripped.startswith('~/'):
            return True
        # Shell redirects: >, >>, <, 2>, &>, etc.
        if re.search(r'(?<![=!<>])>|>>|<(?![=<])|2>|&>', line):
            return True
        return any(indicator in line for indicator in xonsh_indicators)

    def get_completions(
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
            preprocess_result = self._preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = self.map_position_to_processed(
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

    def get_hover(
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
            preprocess_result = self._preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = self.map_position_to_processed(
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

    def get_definitions(
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
            preprocess_result = self._preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = self.map_position_to_processed(
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
                        _, result_col = self.map_position_from_processed(
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

    def get_references(
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
            preprocess_result = self._preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = self.map_position_to_processed(
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
                        _, result_col = self.map_position_from_processed(
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

    def get_diagnostics(
        self,
        source: str,
        path: str | None = None,
    ) -> list[lsp.Diagnostic]:
        """Get Python diagnostics using Jedi."""
        if not JEDI_AVAILABLE:
            return []

        try:
            # Preprocess xonsh syntax to valid Python
            processed_source = self._preprocess_source(source)
            script = Script(processed_source, path=path)
            errors = script.get_syntax_errors()

            # Get original source lines to check for xonsh syntax
            original_lines = source.splitlines()

            diagnostics = []
            for error in errors:
                error_line = error.line - 1 if error.line else 0

                # Skip errors on lines that had xonsh syntax in the original
                # Also check nearby lines since preprocessing can cause errors
                # on subsequent lines (e.g., unbalanced parens)
                if error_line < len(original_lines):
                    # Check current line and up to 10 lines before for xonsh syntax
                    start_check = max(0, error_line - 10)
                    context_lines = original_lines[start_check:error_line + 1]
                    if any(self._has_xonsh_syntax(line) for line in context_lines):
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

    def get_document_symbols(
        self,
        source: str,
        path: str | None = None,
    ) -> list[lsp.DocumentSymbol]:
        """Get Python document symbols using Jedi."""
        if not JEDI_AVAILABLE:
            return []

        try:
            # Preprocess xonsh syntax to valid Python
            processed_source = self._preprocess_source(source)
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

    def get_signature_help(
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
            preprocess_result = self._preprocess_with_mapping(source)

            # Map the position to the preprocessed source
            mapped_line, mapped_col = self.map_position_to_processed(
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
