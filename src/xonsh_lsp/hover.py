"""
Hover provider for xonsh LSP.

Provides hover information for:
- Environment variables (shows current value)
- Xonsh operators (documentation)
- Subprocess commands (man page summary, path)
- Python symbols (via backend - Jedi or LSP proxy)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from lsprotocol import types as lsp

from xonsh_lsp.xonsh_builtins import (
    XONSH_BUILTINS,
    XONSH_ALIASES,
    XONSH_MAGIC_VARS,
    XONSH_OPERATORS,
)

if TYPE_CHECKING:
    from xonsh_lsp.server import XonshLanguageServer

logger = logging.getLogger(__name__)


class XonshHoverProvider:
    """Provides hover information for xonsh files."""

    def __init__(self, server: XonshLanguageServer):
        self.server = server

    async def get_hover(self, params: lsp.HoverParams) -> lsp.Hover | None:
        """Get hover information at the given position."""
        uri = params.text_document.uri
        doc = self.server.get_document(uri)
        if doc is None:
            return None

        line = params.position.line
        col = params.position.character
        source = doc.source

        # Get the word at position
        word = self._get_word_at_position(source, line, col)
        if not word:
            return None

        # Check for xonsh-specific constructs
        hover_content = None

        # Check for environment variable
        # But NOT if we're inside ${...} - that's a Python expression
        if word.startswith("$") and not self._is_inside_braced_env_var(source, line, col):
            hover_content = self._get_env_var_hover(word)

        # Check if we're hovering over a Python expression inside ${...}
        elif self._is_inside_braced_env_var(source, line, col):
            hover_content = await self._get_braced_env_var_hover(source, line, col, word)

        # Check for path literal (p"...", pf"...", pr"...", pb"...")
        elif self._is_path_literal_context(source, line, col):
            hover_content = self._get_path_literal_hover(source, line, col)

        # Check for xonsh operator
        elif word in XONSH_OPERATORS:
            hover_content = self._get_operator_hover(word)

        # Check for xonsh builtin
        elif word in XONSH_BUILTINS:
            hover_content = self._get_builtin_hover(word)

        # Check for xonsh alias
        elif word in XONSH_ALIASES:
            hover_content = self._get_alias_hover(word)

        # Check if in subprocess context for command hover
        elif self._is_command_context(source, line, col):
            hover_content = self._find_source_alias_hover(source, word)
            if hover_content is None:
                hover_content = self._get_command_hover(word)

        # Fall back to Python hover
        if hover_content is None:
            hover_content = await self.server.python_delegate.get_hover(
                source, line, col, doc.path
            )

        if hover_content:
            return lsp.Hover(
                contents=lsp.MarkupContent(
                    kind=lsp.MarkupKind.Markdown,
                    value=hover_content,
                ),
                range=self._get_word_range(source, line, col),
            )

        return None

    def _get_word_at_position(
        self, source: str, line: int, col: int
    ) -> str | None:
        """Get the word at the given position."""
        lines = source.splitlines()
        if line >= len(lines):
            return None

        line_text = lines[line]
        if col > len(line_text):
            return None

        # Find word boundaries
        # Include $ for env vars and special chars for operators
        word_chars = re.compile(r"[\w$@!`<>|&]")

        start = col
        while start > 0 and word_chars.match(line_text[start - 1]):
            start -= 1

        end = col
        while end < len(line_text) and word_chars.match(line_text[end]):
            end += 1

        word = line_text[start:end]

        # Check for ${expr} syntax - if we're inside braces after $
        # In this case, return just the identifier (Python expression), not the full ${expr}
        # because ${expr} evaluates expr as Python, not as an env var name
        if start >= 2 and line_text[start - 2 : start] == "${":
            # We're inside ${...}, just return the word (Python identifier)
            # Don't include the ${ prefix
            pass  # Fall through to return word

        # Extend for multi-char operators
        for op in ["$(", "!(", "$[", "![", "@(", "@$(", ">>", "2>", "&>", "||", "&&"]:
            if op in line_text[max(0, start - 2) : end + 2]:
                # Check if we're hovering over this operator
                op_start = line_text.find(op, max(0, start - 2))
                if op_start != -1 and op_start <= col < op_start + len(op):
                    return op

        return word if word else None

    def _get_word_range(
        self, source: str, line: int, col: int
    ) -> lsp.Range | None:
        """Get the range of the word at the given position."""
        lines = source.splitlines()
        if line >= len(lines):
            return None

        line_text = lines[line]
        word_chars = re.compile(r"[\w$@!`]")

        start = col
        while start > 0 and word_chars.match(line_text[start - 1]):
            start -= 1

        end = col
        while end < len(line_text) and word_chars.match(line_text[end]):
            end += 1

        return lsp.Range(
            start=lsp.Position(line=line, character=start),
            end=lsp.Position(line=line, character=end),
        )

    def _get_env_var_hover(self, word: str) -> str | None:
        """Get hover content for an environment variable."""
        # Extract variable name
        if word.startswith("${") and word.endswith("}"):
            var_name = word[2:-1]
        elif word.startswith("$"):
            var_name = word[1:]
        else:
            return None

        # Check environment
        if var_name in os.environ:
            value = os.environ[var_name]
            # Truncate long values
            if len(value) > 500:
                value = value[:500] + "..."

            return f"""## Environment Variable: `${var_name}`

**Value:**
```
{value}
```
"""

        # Check xonsh magic vars
        if var_name in XONSH_MAGIC_VARS:
            info = XONSH_MAGIC_VARS[var_name]
            return f"""## Xonsh Magic Variable: `${var_name}`

**Type:** `{info.get('type', 'unknown')}`

{info.get('doc', '')}
"""

        return f"""## Environment Variable: `${var_name}`

**Status:** Not defined

This environment variable is not currently set.
"""

    def _get_operator_hover(self, operator: str) -> str | None:
        """Get hover content for a xonsh operator."""
        if operator not in XONSH_OPERATORS:
            return None

        info = XONSH_OPERATORS[operator]
        examples = info.get("examples", [])
        examples_md = "\n".join(f"- `{ex}`" for ex in examples) if examples else ""

        return f"""## Xonsh Operator: `{operator}`

{info.get('doc', '')}

**Syntax:** `{info.get('syntax', operator)}`

**Examples:**
{examples_md}
"""

    def _get_builtin_hover(self, name: str) -> str | None:
        """Get hover content for a xonsh builtin."""
        if name not in XONSH_BUILTINS:
            return None

        info = XONSH_BUILTINS[name]
        return f"""## Xonsh Builtin: `{name}`

**Signature:** `{info.get('signature', name + '()')}`

{info.get('doc', '')}
"""

    def _get_alias_hover(self, name: str) -> str | None:
        """Get hover content for a xonsh alias."""
        if name not in XONSH_ALIASES:
            return None

        info = XONSH_ALIASES[name]
        return f"""## Xonsh Alias: `{name}`

{info.get('doc', info.get('description', ''))}
"""

    def _get_command_hover(self, command: str) -> str | None:
        """Get hover content for a subprocess command."""
        path = shutil.which(command)
        if not path:
            return None

        # Get basic info
        hover = f"""## Command: `{command}`

**Path:** `{path}`

"""

        # Try to get --help output (first few lines)
        try:
            result = subprocess.run(
                [command, "--help"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            help_text = result.stdout or result.stderr
            if help_text:
                # Get first 10 lines
                lines = help_text.strip().split("\n")[:10]
                help_preview = "\n".join(lines)
                hover += f"""**Usage:**
```
{help_preview}
```
"""
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass

        return hover

    def _find_source_alias_hover(self, source: str, name: str) -> str | None:
        """Find a source-defined alias and return hover markdown.

        Detects three patterns:
        - Pattern A: aliases['name'] = value
        - Pattern B: @aliases.register + def _name()
        - Pattern C: @aliases.register("name") + def _name()
        """
        parse_result = self.server.parser.parse(source)
        tree = parse_result.tree
        if tree is None:
            return None

        def _get_text(node) -> str:
            return source[node.start_byte : node.end_byte]

        def _is_aliases_attr(node) -> bool:
            if node.type != "attribute":
                return False
            obj = node.child_by_field_name("object")
            attr = node.child_by_field_name("attribute")
            return (
                obj is not None
                and attr is not None
                and _get_text(obj) == "aliases"
                and _get_text(attr) == "register"
            )

        def _extract_string_content(node) -> str | None:
            if node.type == "string":
                for child in node.children:
                    if child.type == "string_content":
                        return _get_text(child)
            return None

        result: str | None = None

        def visit(node) -> None:
            nonlocal result
            if result is not None:
                return

            # Pattern A: aliases['name'] = value
            if node.type == "assignment":
                left = node.child_by_field_name("left")
                if left is not None and left.type == "subscript":
                    value_node = left.child_by_field_name("value")
                    subscript = left.child_by_field_name("subscript")
                    if (
                        value_node is not None
                        and _get_text(value_node) == "aliases"
                        and subscript is not None
                    ):
                        alias_name = _extract_string_content(subscript)
                        if alias_name == name:
                            rhs = node.child_by_field_name("right")
                            rhs_text = _get_text(rhs) if rhs is not None else "..."
                            line_num = node.start_point[0] + 1
                            result = (
                                f"## Alias: `{name}`\n\n"
                                f"**Defined at:** line {line_num}\n\n"
                                f"**Value:**\n```\n{rhs_text}\n```\n"
                            )
                            return

            # Patterns B & C: @aliases.register / @aliases.register("name")
            elif node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "decorator":
                        for deco_child in child.children:
                            if deco_child.type == "comment" or _get_text(deco_child) == "@":
                                continue
                            matched = False
                            # Pattern B: @aliases.register (bare)
                            if _is_aliases_attr(deco_child):
                                func_def = node.child_by_field_name("definition")
                                if func_def is None:
                                    for c in node.children:
                                        if c.type == "function_definition":
                                            func_def = c
                                            break
                                if func_def is not None:
                                    name_node = func_def.child_by_field_name("name")
                                    if name_node is not None:
                                        fname = _get_text(name_node).lstrip("_")
                                        if fname == name:
                                            matched = True
                            # Pattern C: @aliases.register("name") (call)
                            elif deco_child.type == "call":
                                func = deco_child.child_by_field_name("function")
                                if func is not None and _is_aliases_attr(func):
                                    args = deco_child.child_by_field_name("arguments")
                                    if args is not None:
                                        for arg in args.children:
                                            arg_name = _extract_string_content(arg)
                                            if arg_name == name:
                                                matched = True
                                                break

                            if matched:
                                line_num = node.start_point[0] + 1
                                func_text = _get_text(node)
                                result = (
                                    f"## Alias: `{name}`\n\n"
                                    f"**Defined at:** line {line_num}\n\n"
                                    f"**Implementation:**\n```python\n{func_text}\n```\n"
                                )
                                return

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return result

    def _is_command_context(self, source: str, line: int, col: int) -> bool:
        """Check if the position is in a subprocess command context."""
        tree = self.server.parser.parse(source).tree
        if tree is None:
            # Fallback: check if line looks like subprocess
            lines = source.splitlines()
            if line < len(lines):
                line_text = lines[line]
                subprocess_markers = ["$(", "!(", "$[", "!["]
                return any(marker in line_text for marker in subprocess_markers)
            return False

        return self.server.parser.is_in_subprocess_context(tree, line, col)

    def _is_path_literal_context(self, source: str, line: int, col: int) -> bool:
        """Check if the position is within a path literal (p"...", pf"...", etc.)."""
        lines = source.splitlines()
        if line >= len(lines):
            return False

        line_text = lines[line]

        # Look for path literal patterns before the cursor position
        path_prefixes = ['p"', "p'", 'pf"', "pf'", 'pr"', "pr'", 'pb"', "pb'",
                         'P"', "P'", 'PF"', "PF'", 'PR"', "PR'", 'PB"', "PB'"]

        text_before = line_text[:col]
        for prefix in path_prefixes:
            idx = text_before.rfind(prefix)
            if idx != -1:
                # Check if we're inside the string (quote not closed yet, or we're past it)
                quote_char = prefix[-1]
                after_prefix = text_before[idx + len(prefix):]
                # If there's no closing quote, or cursor is before it, we're in the path literal
                if quote_char not in after_prefix:
                    return True
                # Check if we're right after the closing quote (for method calls)
                close_idx = after_prefix.find(quote_char)
                if close_idx != -1 and idx + len(prefix) + close_idx < col:
                    return True

        return False

    def _is_inside_braced_env_var(self, source: str, line: int, col: int) -> bool:
        """Check if position is inside a ${...} braced environment variable expression."""
        lines = source.splitlines()
        if line >= len(lines):
            return False

        line_text = lines[line]
        if col > len(line_text):
            return False

        # Look backwards for ${ and forwards for }
        text_before = line_text[:col]
        text_after = line_text[col:]

        # Find the last ${ before cursor
        brace_start = text_before.rfind("${")
        if brace_start == -1:
            return False

        # Check if there's a } between ${ and cursor (meaning we're outside)
        between = text_before[brace_start + 2:]
        if "}" in between:
            return False

        # Check if there's a } after cursor (meaning we're inside)
        if "}" in text_after:
            return True

        return False

    async def _get_braced_env_var_hover(
        self, source: str, line: int, col: int, word: str
    ) -> str | None:
        """Get hover for content inside ${...} - this is a Python expression."""
        lines = source.splitlines()
        if line >= len(lines):
            return None

        line_text = lines[line]

        # Find the ${...} boundaries
        text_before = line_text[:col]
        brace_start = text_before.rfind("${")
        if brace_start == -1:
            return None

        brace_end = line_text.find("}", brace_start)
        if brace_end == -1:
            return None

        expr = line_text[brace_start + 2:brace_end]

        # Calculate the position of 'word' within the expression
        expr_start = brace_start + 2
        word_start_in_expr = col - expr_start

        # Build a synthetic source where we can query the backend for the expression
        # Keep the lines before to preserve variable definitions
        synthetic_lines = lines[:line]
        # Add a line that just has the expression so the backend can analyze it
        synthetic_lines.append(expr)
        synthetic_source = "\n".join(synthetic_lines)

        # Query backend for hover on the word within the expression
        python_hover = await self.server.python_delegate.get_hover(
            synthetic_source, line, word_start_in_expr, None
        )

        hover_parts = [
            f"## Braced Environment Variable: `${{{expr}}}`",
            "",
            "**Syntax:** `${expr}` evaluates `expr` as Python, then uses the result as an environment variable name.",
            "",
            f"**Expression:** `{expr}`",
        ]

        if python_hover:
            hover_parts.extend([
                "",
                "---",
                "",
                f"**Python info for `{word}`:**",
                "",
                python_hover,
            ])
        else:
            # If no Python hover, explain what the expression does
            hover_parts.extend([
                "",
                f"The expression `{expr}` will be evaluated as Python code.",
                f"If it evaluates to `'USER'`, then `${{{expr}}}` is equivalent to `$USER`.",
            ])

        return "\n".join(hover_parts)

    def _get_path_literal_hover(self, source: str, line: int, col: int) -> str | None:
        """Get hover content for a path literal."""
        lines = source.splitlines()
        if line >= len(lines):
            return None

        line_text = lines[line]

        # Extract the path literal
        path_prefixes = [
            ('pf', 'Formatted path string (f-string interpolation)'),
            ('pr', 'Raw path string (no escape processing)'),
            ('pb', 'Bytes path string'),
            ('p', 'Path string'),
            ('PF', 'Formatted path string (f-string interpolation)'),
            ('PR', 'Raw path string (no escape processing)'),
            ('PB', 'Bytes path string'),
            ('P', 'Path string'),
        ]

        for prefix, description in path_prefixes:
            for quote in ['"', "'"]:
                pattern = prefix + quote
                idx = line_text.find(pattern)
                if idx != -1 and idx < col:
                    # Find the closing quote
                    start = idx + len(pattern)
                    end = line_text.find(quote, start)
                    if end != -1:
                        path_value = line_text[start:end]
                        return f"""## Path Literal: `{prefix}"{path_value}"`

**Type:** `pathlib.Path`

{description}

**Value:** `{path_value}`

Path literals in xonsh create `pathlib.Path` objects, providing convenient access to file system operations.

**Common methods:**
- `.read_text()` - Read file contents as string
- `.read_bytes()` - Read file contents as bytes
- `.write_text(data)` - Write string to file
- `.exists()` - Check if path exists
- `.is_file()` / `.is_dir()` - Check path type
- `.parent` - Get parent directory
- `.name` / `.stem` / `.suffix` - Get path components
- `.glob(pattern)` - Find matching files
"""

        return None
