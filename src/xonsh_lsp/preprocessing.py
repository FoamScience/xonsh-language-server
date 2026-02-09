"""
Shared preprocessing utilities for xonsh-to-Python source transformation.

This module provides the core preprocessing pipeline that converts xonsh syntax
to valid Python, along with position mapping utilities for translating between
original and preprocessed source coordinates.

Used by both JediBackend and LspProxyBackend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


def preprocess_source(source: str) -> str:
    """Preprocess xonsh source to make it valid Python.

    Replaces xonsh-specific syntax with Python placeholders.
    """
    return preprocess_with_mapping(source).source


def preprocess_with_mapping(source: str) -> PreprocessResult:
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
    preprocess_result: PreprocessResult, line: int, col: int
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
    preprocess_result: PreprocessResult, line: int, col: int
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


def has_xonsh_syntax(line: str) -> bool:
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
