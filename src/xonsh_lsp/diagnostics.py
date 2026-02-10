"""
Diagnostic providers for xonsh LSP.

Provides diagnostics for:
- Syntax errors from tree-sitter
- Undefined environment variables
- Invalid subprocess syntax
- Python diagnostics (via backend - Jedi or LSP proxy)

Supports two-phase diagnostics merging:
- Phase 1: xonsh-specific diagnostics (synchronous, returned immediately)
- Phase 2: Python diagnostics (async from backend callback, merged and re-published)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import TYPE_CHECKING

from lsprotocol import types as lsp

from xonsh_lsp.xonsh_builtins import XONSH_MAGIC_VARS

if TYPE_CHECKING:
    from xonsh_lsp.server import XonshLanguageServer

logger = logging.getLogger(__name__)


class XonshDiagnosticsProvider:
    """Provides diagnostics for xonsh files."""

    def __init__(self, server: XonshLanguageServer):
        self.server = server
        # Diagnostics caches for two-phase merging
        self._xonsh_diagnostics: dict[str, list[lsp.Diagnostic]] = {}
        self._python_diagnostics: dict[str, list[lsp.Diagnostic]] = {}

    async def get_diagnostics(self, uri: str) -> list[lsp.Diagnostic]:
        """Get all diagnostics for a document.

        For JediBackend: returns both xonsh and Python diagnostics synchronously.
        For LspProxyBackend: returns xonsh diagnostics immediately; Python
        diagnostics arrive asynchronously via on_backend_diagnostics().
        """
        doc = self.server.get_document(uri)
        if doc is None:
            return []

        diagnostics: list[lsp.Diagnostic] = []

        # Get tree-sitter parse result
        parse_result = self.server.parser.parse(doc.source)

        # Syntax errors from tree-sitter
        diagnostics.extend(self._get_syntax_errors(parse_result))

        # Environment variable diagnostics
        diagnostics.extend(
            self._get_env_var_diagnostics(doc.source, parse_result)
        )

        # Extract source-defined aliases
        source_aliases = self._extract_aliases_from_source(doc.source, parse_result)

        # Subprocess diagnostics
        diagnostics.extend(
            self._get_subprocess_diagnostics(doc.source, parse_result, source_aliases)
        )

        # Split-expression diagnostics (e.g. `myapp world` parsed as two expressions)
        diagnostics.extend(
            self._get_split_expression_diagnostics(doc.source, parse_result, source_aliases)
        )

        # Cache xonsh diagnostics
        self._xonsh_diagnostics[uri] = list(diagnostics)

        # Python diagnostics (from backend)
        python_diagnostics = await self.server.python_delegate.get_diagnostics(
            doc.source, doc.path
        )
        if python_diagnostics:
            self._python_diagnostics[uri] = python_diagnostics
            diagnostics.extend(python_diagnostics)
        elif uri in self._python_diagnostics:
            # For proxy backend, include previously cached Python diagnostics
            diagnostics.extend(self._python_diagnostics[uri])

        return diagnostics

    def on_backend_diagnostics(self, uri: str, diagnostics: list[lsp.Diagnostic]) -> None:
        """Handle asynchronous diagnostics from the proxy backend.

        Caches the Python diagnostics and re-publishes merged diagnostics
        (xonsh + Python) to the editor.
        """
        self._python_diagnostics[uri] = diagnostics

        # Merge with cached xonsh diagnostics and re-publish
        xonsh_diags = self._xonsh_diagnostics.get(uri, [])
        merged = xonsh_diags + diagnostics

        self.server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=merged)
        )

    def clear_cache(self, uri: str) -> None:
        """Clear cached diagnostics for a URI."""
        self._xonsh_diagnostics.pop(uri, None)
        self._python_diagnostics.pop(uri, None)

    def _get_syntax_errors(self, parse_result) -> list[lsp.Diagnostic]:
        """Get syntax error diagnostics from tree-sitter."""
        diagnostics = []

        for error in parse_result.errors:
            diagnostics.append(
                lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(
                            line=error.start_point[0],
                            character=error.start_point[1],
                        ),
                        end=lsp.Position(
                            line=error.end_point[0],
                            character=error.end_point[1],
                        ),
                    ),
                    message=f"Syntax error: unexpected '{error.text[:20]}...'"
                    if len(error.text) > 20
                    else f"Syntax error: unexpected '{error.text}'",
                    severity=lsp.DiagnosticSeverity.Error,
                    source="xonsh-lsp",
                    code="syntax-error",
                )
            )

        return diagnostics

    def _get_env_var_diagnostics(
        self, source: str, parse_result
    ) -> list[lsp.Diagnostic]:
        """Get diagnostics for environment variable usage."""
        diagnostics = []

        for env_var in parse_result.env_variables:
            var_text = env_var.text

            # Extract variable name
            if var_text.startswith("${") and var_text.endswith("}"):
                var_name = var_text[2:-1]
            elif var_text.startswith("$"):
                var_name = var_text[1:]
            else:
                continue

            # Skip if it's an expression (contains non-identifier chars)
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var_name):
                continue

            # Check if defined in environment or xonsh magic vars
            if var_name not in os.environ and var_name not in XONSH_MAGIC_VARS:
                # Check if it might be defined in the file itself
                if not self._is_defined_in_source(source, var_name):
                    diagnostics.append(
                        lsp.Diagnostic(
                            range=lsp.Range(
                                start=lsp.Position(
                                    line=env_var.start_point[0],
                                    character=env_var.start_point[1],
                                ),
                                end=lsp.Position(
                                    line=env_var.end_point[0],
                                    character=env_var.end_point[1],
                                ),
                            ),
                            message=f"Undefined environment variable: ${var_name}",
                            severity=lsp.DiagnosticSeverity.Warning,
                            source="xonsh-lsp",
                            code="undefined-env-var",
                            data={"var_name": var_name},
                        )
                    )

        return diagnostics

    def _get_subprocess_diagnostics(
        self, source: str, parse_result, source_aliases: set[str] | None = None
    ) -> list[lsp.Diagnostic]:
        """Get diagnostics for subprocess constructs."""
        diagnostics = []

        for subprocess in parse_result.subprocesses:
            # Check for empty subprocess
            body_text = subprocess.text

            # Remove outer markers to get body
            for prefix, suffix in [("$(", ")"), ("!(", ")"), ("$[", "]"), ("![", "]")]:
                if body_text.startswith(prefix) and body_text.endswith(suffix):
                    body = body_text[len(prefix) : -len(suffix)].strip()
                    break
            else:
                body = body_text

            if not body:
                diagnostics.append(
                    lsp.Diagnostic(
                        range=lsp.Range(
                            start=lsp.Position(
                                line=subprocess.start_point[0],
                                character=subprocess.start_point[1],
                            ),
                            end=lsp.Position(
                                line=subprocess.end_point[0],
                                character=subprocess.end_point[1],
                            ),
                        ),
                        message="Empty subprocess",
                        severity=lsp.DiagnosticSeverity.Warning,
                        source="xonsh-lsp",
                        code="empty-subprocess",
                    )
                )
                continue

            # Check if command exists
            parts = body.split()
            if parts:
                cmd = parts[0]
                # Skip if it's a variable or Python expression
                if cmd.startswith("$") or cmd.startswith("@"):
                    continue

                # Check if command exists
                if not self._command_exists(cmd, source_aliases):
                    diagnostics.append(
                        lsp.Diagnostic(
                            range=lsp.Range(
                                start=lsp.Position(
                                    line=subprocess.start_point[0],
                                    character=subprocess.start_point[1],
                                ),
                                end=lsp.Position(
                                    line=subprocess.end_point[0],
                                    character=subprocess.end_point[1],
                                ),
                            ),
                            message=f"Command not found: {cmd}",
                            severity=lsp.DiagnosticSeverity.Hint,
                            source="xonsh-lsp",
                            code="command-not-found",
                        )
                    )

        return diagnostics

    def _is_defined_in_source(self, source: str, var_name: str) -> bool:
        """Check if an environment variable is defined in the source."""
        # Check for $VAR = ... or ${VAR} = ... patterns
        patterns = [
            rf"\${var_name}\s*=",
            rf"\$\{{{var_name}\}}\s*=",
            rf'os\.environ\[[\'"]{var_name}[\'"]\]',
            rf'\$ENV\[[\'"]{var_name}[\'"]\]',
        ]

        for pattern in patterns:
            if re.search(pattern, source):
                return True

        return False

    # Shell builtins that won't be found by shutil.which()
    SHELL_BUILTINS = {
        # Xonsh builtins
        "cd", "pushd", "popd", "dirs", "jobs", "fg", "bg", "disown",
        "source", "history", "replay", "trace", "timeit", "xonfig",
        "aliases", "abbrevs", "completer", "xontrib",
        # Common shell builtins
        "echo", "printf", "read", "test", "[", "true", "false",
        "exit", "return", "break", "continue", "shift",
        "export", "unset", "set", "shopt", "enable",
        "eval", "exec", "builtin", "command", "type", "which", "hash",
        "ulimit", "umask", "wait", "kill", "trap",
        "pwd", "times", "logout", "help",
    }

    def _command_exists(self, cmd: str, source_aliases: set[str] | None = None) -> bool:
        """Check if a command exists in PATH, is a shell builtin, or a source alias."""
        if cmd in self.SHELL_BUILTINS:
            return True
        if source_aliases and cmd in source_aliases:
            return True
        return shutil.which(cmd) is not None

    def _extract_aliases_from_source(self, source: str, parse_result) -> set[str]:
        """Extract alias names defined in source by walking the tree-sitter AST.

        Detects three patterns:
        - Pattern A: aliases['name'] = ...
        - Pattern B: @aliases.register + def _name()
        - Pattern C: @aliases.register("name") + def __name()
        """
        aliases: set[str] = set()
        tree = parse_result.tree
        if tree is None:
            return aliases

        def _get_text(node) -> str:
            return source[node.start_byte:node.end_byte]

        def _is_aliases_attr(node) -> bool:
            """Check if node is `aliases.register` attribute access."""
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
            """Extract plain string content from a string node."""
            if node.type == "string":
                for child in node.children:
                    if child.type == "string_content":
                        return _get_text(child)
            return None

        def visit(node) -> None:
            # Pattern A: aliases['name'] = ...
            if node.type == "assignment":
                left = node.child_by_field_name("left")
                if left is not None and left.type == "subscript":
                    value = left.child_by_field_name("value")
                    subscript = left.child_by_field_name("subscript")
                    if (
                        value is not None
                        and _get_text(value) == "aliases"
                        and subscript is not None
                    ):
                        name = _extract_string_content(subscript)
                        if name:
                            aliases.add(name)

            # Patterns B & C: @aliases.register / @aliases.register("name")
            elif node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "decorator":
                        # The decorator's expression is its child (skip the '@')
                        for deco_child in child.children:
                            if deco_child.type == "comment" or _get_text(deco_child) == "@":
                                continue
                            # Pattern B: @aliases.register (bare attribute)
                            if _is_aliases_attr(deco_child):
                                # Get function name, strip leading _
                                func_def = node.child_by_field_name("definition")
                                if func_def is None:
                                    for c in node.children:
                                        if c.type == "function_definition":
                                            func_def = c
                                            break
                                if func_def is not None:
                                    name_node = func_def.child_by_field_name("name")
                                    if name_node is not None:
                                        fname = _get_text(name_node)
                                        aliases.add(fname.lstrip("_"))
                            # Pattern C: @aliases.register("name") (call)
                            elif deco_child.type == "call":
                                func = deco_child.child_by_field_name("function")
                                if func is not None and _is_aliases_attr(func):
                                    args = deco_child.child_by_field_name("arguments")
                                    if args is not None:
                                        for arg in args.children:
                                            name = _extract_string_content(arg)
                                            if name:
                                                aliases.add(name)
                                                break

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return aliases

    def _get_split_expression_diagnostics(
        self, source: str, parse_result, source_aliases: set[str]
    ) -> list[lsp.Diagnostic]:
        """Detect bare identifier followed by more tokens on the same line.

        Tree-sitter may parse ``cmd args`` as either:
        - ``ERROR(identifier) + expression_statement`` (short identifiers), or
        - ``expression_statement(identifier) + expression_statement`` (longer ones).

        When the first identifier is not a known command or source-defined alias
        we emit a warning.
        """
        diagnostics: list[lsp.Diagnostic] = []
        tree = parse_result.tree
        if tree is None:
            return diagnostics

        def _leading_identifier_name(node) -> str | None:
            """Return identifier text if *node* is a single bare identifier."""
            if node.type == "expression_statement":
                children = [c for c in node.children if c.type not in ("comment", "newline")]
                if len(children) == 1 and children[0].type == "identifier":
                    return source[children[0].start_byte:children[0].end_byte]
            return None

        def check_siblings(children_list) -> None:
            """Check a list of sibling nodes for split-expression patterns."""
            i = 0
            while i < len(children_list):
                node = children_list[i]

                # Recurse into compound statements to check their blocks
                if node.type in (
                    "function_definition", "class_definition",
                    "if_statement", "for_statement", "while_statement",
                    "try_statement", "with_statement",
                ):
                    for child in node.children:
                        if child.type == "block":
                            check_siblings(list(child.children))

                name = _leading_identifier_name(node)
                if name is None:
                    i += 1
                    continue

                # Look ahead for more nodes on the same line
                line = node.start_point[0]
                j = i + 1
                while j < len(children_list):
                    sibling = children_list[j]
                    if sibling.start_point[0] != line:
                        break
                    if sibling.type not in ("expression_statement", "ERROR"):
                        break
                    j += 1

                if j > i + 1:
                    if not self._command_exists(name, source_aliases):
                        last = children_list[j - 1]
                        diagnostics.append(
                            lsp.Diagnostic(
                                range=lsp.Range(
                                    start=lsp.Position(
                                        line=node.start_point[0],
                                        character=node.start_point[1],
                                    ),
                                    end=lsp.Position(
                                        line=last.end_point[0],
                                        character=last.end_point[1],
                                    ),
                                ),
                                message=(
                                    f"'{name}' is not a known command or alias. "
                                    f"Use $[{name} ...] for subprocess invocation."
                                ),
                                severity=lsp.DiagnosticSeverity.Warning,
                                source="xonsh-lsp",
                                code=DiagnosticCode.UNKNOWN_COMMAND_OR_ALIAS,
                            )
                        )
                    i = j
                else:
                    i += 1

        check_siblings(list(tree.root_node.children))
        return diagnostics


class DiagnosticCode:
    """Diagnostic codes for xonsh-lsp."""

    SYNTAX_ERROR = "syntax-error"
    UNDEFINED_ENV_VAR = "undefined-env-var"
    COMMAND_NOT_FOUND = "command-not-found"
    EMPTY_SUBPROCESS = "empty-subprocess"
    INVALID_GLOB = "invalid-glob"
    PYTHON_ERROR = "python-error"
    UNKNOWN_COMMAND_OR_ALIAS = "unknown-command-or-alias"
