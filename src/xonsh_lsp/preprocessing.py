"""
Shared preprocessing utilities for xonsh-to-Python source transformation.

This module provides the core preprocessing pipeline that converts xonsh syntax
to valid Python, along with position mapping utilities for translating between
original and preprocessed source coordinates.

Uses tree-sitter-xonsh to parse xonsh source into an AST and walks the tree
to produce byte-range replacements. This replaces the previous regex-based
pipeline and correctly handles xonsh syntax inside string literals, nested
constructs, and compound glob prefixes.

Used by both JediBackend and LspProxyBackend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xonsh_lsp.parser import XonshParser

# Lazy singleton parser
_parser: XonshParser | None = None


def _get_parser() -> XonshParser:
    global _parser
    if _parser is None:
        _parser = XonshParser()
    return _parser


# ---------------------------------------------------------------------------
# Node type classification
# ---------------------------------------------------------------------------

# Node types where the byte range is replaced with a Python equivalent
_REPLACEABLE_TYPES = {
    "env_variable",
    "env_variable_braced",
    "captured_subprocess",
    "captured_subprocess_object",
    "uncaptured_subprocess",
    "uncaptured_subprocess_object",
    "tokenized_substitution",
    "python_evaluation",
    "at_object",
    "regex_glob",
    "glob_pattern",
    "formatted_glob",
    "custom_function_glob",
    "glob_path",
    "regex_path_glob",
    "path_string",
    "macro_call",
}

# Node types where the entire line(s) are replaced with `pass`
_MASKABLE_TYPES = {
    "bare_subprocess",
    "background_command",
    "help_expression",
    "super_help_expression",
    "env_scoped_command",
    "xontrib_statement",
}

# Transparent wrapper types — descend into children
_TRANSPARENT_TYPES = {
    "xonsh_expression",
    "xonsh_statement",
    "env_assignment",
    "env_deletion",
}


# ---------------------------------------------------------------------------
# Replacement computation
# ---------------------------------------------------------------------------

def _compute_replacement(node, source_bytes: bytes) -> str:
    """Compute the Python replacement text for a replaceable node."""
    ntype = node.type

    if ntype == "env_variable":
        # $HOME -> __xonsh_env__["HOME"]
        # children: $ identifier
        for child in node.children:
            if child.type == "identifier":
                name = source_bytes[child.start_byte:child.end_byte].decode()
                return f'__xonsh_env__["{name}"]'
        # fallback: extract after $
        text = source_bytes[node.start_byte:node.end_byte].decode()
        return f'__xonsh_env__["{text[1:]}"]'

    if ntype == "env_variable_braced":
        # ${expr} -> __xonsh_env__[expr]
        expr_node = node.child_by_field_name("expression")
        if expr_node:
            expr_text = source_bytes[expr_node.start_byte:expr_node.end_byte].decode()
            return f"__xonsh_env__[{expr_text}]"
        # fallback
        text = source_bytes[node.start_byte:node.end_byte].decode()
        inner = text[2:-1]  # strip ${ and }
        return f"__xonsh_env__[{inner}]"

    if ntype in (
        "captured_subprocess",
        "captured_subprocess_object",
        "uncaptured_subprocess",
        "uncaptured_subprocess_object",
        "tokenized_substitution",
    ):
        return "__xonsh_subproc__()"

    if ntype == "python_evaluation":
        # @(expr) -> (expr)
        expr_node = node.child_by_field_name("expression")
        if expr_node:
            expr_text = source_bytes[expr_node.start_byte:expr_node.end_byte].decode()
            return f"({expr_text})"
        # fallback
        text = source_bytes[node.start_byte:node.end_byte].decode()
        return f"({text[2:-1]})"

    if ntype == "at_object":
        # @.imp -> __xonsh_at__.imp
        attr_node = node.child_by_field_name("attribute")
        if attr_node:
            attr_text = source_bytes[attr_node.start_byte:attr_node.end_byte].decode()
            return f"__xonsh_at__.{attr_text}"
        return "__xonsh_at__"

    if ntype in ("glob_path", "regex_path_glob"):
        # gp`*.*`, rp`.*\.py` -> [Path("")]
        return '[Path("")]'

    if ntype in ("regex_glob", "glob_pattern", "formatted_glob", "custom_function_glob"):
        # `pattern`, g`pattern`, f`pattern`, @func`pattern` -> [""]
        return '[""]'

    if ntype == "path_string":
        # p"/home/user" -> Path("/home/user")
        # pf"/home/{user}" -> Path(f"/home/{user}")
        # pr"\home\user" -> Path(r"\home\user")
        prefix_node = node.child_by_field_name("prefix")
        string_node = node.child_by_field_name("string")
        if prefix_node and string_node:
            prefix_text = source_bytes[prefix_node.start_byte:prefix_node.end_byte].decode()
            string_text = source_bytes[string_node.start_byte:string_node.end_byte].decode()
            # Strip leading 'p' from prefix to get Python string prefix
            py_prefix = prefix_text[1:] if prefix_text.startswith("p") else prefix_text
            return f"Path({py_prefix}{string_text})"
        # fallback
        text = source_bytes[node.start_byte:node.end_byte].decode()
        return f"Path({text})"

    if ntype == "macro_call":
        # func!(args) -> func("args")
        name_node = node.child_by_field_name("name")
        arg_node = node.child_by_field_name("argument")
        name_text = source_bytes[name_node.start_byte:name_node.end_byte].decode() if name_node else "macro"
        arg_text = source_bytes[arg_node.start_byte:arg_node.end_byte].decode() if arg_node else ""
        return f'{name_text}("{arg_text}")'

    # Should not reach here
    return source_bytes[node.start_byte:node.end_byte].decode()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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

    masked_lines: set[int] = field(default_factory=set)
    """Set of original line numbers that were masked (replaced with ``pass``)."""

    xonsh_lines: set[int] = field(default_factory=set)
    """Set of original line numbers that contained any xonsh syntax."""

    replacement_regions: list[tuple[int, int, int, int, str]] = field(default_factory=list)
    """(start_line, start_col, end_line, end_col, node_type) for each replaced xonsh construct.

    Coordinates are in original source. Used by semantic tokens to emit
    synthetic tokens for xonsh constructs."""


# ---------------------------------------------------------------------------
# Line mapping utilities (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_source(source: str) -> str:
    """Preprocess xonsh source to make it valid Python.

    Replaces xonsh-specific syntax with Python placeholders.
    """
    return preprocess_with_mapping(source).source


def preprocess_with_mapping(source: str) -> PreprocessResult:
    """Preprocess xonsh source and build position mapping.

    Uses tree-sitter to parse the source into an AST, collects byte-range
    replacements for convertible xonsh constructs, and masks unconvertible
    lines with ``pass``.

    Returns the preprocessed source along with line-by-line column mappings
    that can be used to convert positions between original and preprocessed.
    """
    parser = _get_parser()
    parse_result = parser.parse(source)
    source_bytes = source.encode("utf-8")

    replacements: list[tuple[int, int, str]] = []  # (start_byte, end_byte, replacement)
    replacement_regions: list[tuple[int, int, int, int, str]] = []  # (start_line, start_col, end_line, end_col, node_type)
    masked_lines: set[int] = set()
    xonsh_lines: set[int] = set()

    if parse_result.tree is not None:
        def visit(node):
            if node.type in _REPLACEABLE_TYPES:
                repl = _compute_replacement(node, source_bytes)
                replacements.append((node.start_byte, node.end_byte, repl))
                # Store region for synthetic semantic tokens
                replacement_regions.append((
                    node.start_point[0],   # start_line
                    node.start_point[1],   # start_col
                    node.end_point[0],     # end_line
                    node.end_point[1],     # end_col
                    node.type,
                ))
                for ln in range(node.start_point[0], node.end_point[0] + 1):
                    xonsh_lines.add(ln)
                return  # don't descend into replaced nodes
            if node.type in _MASKABLE_TYPES:
                for ln in range(node.start_point[0], node.end_point[0] + 1):
                    masked_lines.add(ln)
                    xonsh_lines.add(ln)
                return
            if node.type in _TRANSPARENT_TYPES:
                # Descend into children (e.g. env_assignment -> env_variable + right side)
                for child in node.children:
                    visit(child)
                return
            # For all other nodes, recurse
            for child in node.children:
                visit(child)

        visit(parse_result.tree.root_node)

    # Apply byte-range replacements end→start to preserve earlier offsets
    buf = bytearray(source_bytes)
    for start, end, repl in sorted(replacements, reverse=True, key=lambda r: r[0]):
        buf[start:end] = repl.encode("utf-8")
    result = buf.decode("utf-8")

    # Apply line masking
    if masked_lines:
        lines = result.split("\n")
        orig_lines = source.split("\n")
        for ln in masked_lines:
            if ln < len(lines) and ln < len(orig_lines):
                indent = len(orig_lines[ln]) - len(orig_lines[ln].lstrip()) if orig_lines[ln].strip() else 0
                lines[ln] = " " * indent + "pass"
        result = "\n".join(lines)

    # Track if we need to add Path import (affects line numbering)
    path_import_added = False
    path_import_line = 0

    # Add Path import if path literals were used (Path() appears in result)
    if 'Path(' in result and 'from pathlib import Path' not in result:
        lines = result.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
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

    line_mappings: list[list[tuple[int, int]]] = []
    added_lines = 1 if path_import_added else 0

    for i in range(len(original_lines)):
        orig_line = original_lines[i]
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
        masked_lines=masked_lines,
        xonsh_lines=xonsh_lines,
        replacement_regions=replacement_regions,
    )


def map_position_to_processed(
    preprocess_result: PreprocessResult, line: int, col: int
) -> tuple[int, int]:
    """Map a position from original source to preprocessed source."""
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
    orig_line = line
    if preprocess_result.added_lines_before > 0:
        if line >= preprocess_result.added_line_position + preprocess_result.added_lines_before:
            orig_line = line - preprocess_result.added_lines_before
        elif line >= preprocess_result.added_line_position:
            orig_line = preprocess_result.added_line_position

    if orig_line >= len(preprocess_result.line_mappings):
        return (orig_line, col)
    mapping = preprocess_result.line_mappings[orig_line]
    return (orig_line, _map_column(col, mapping, to_processed=False))
