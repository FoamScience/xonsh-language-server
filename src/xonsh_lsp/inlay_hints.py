"""Native inlay hints for xonsh-specific syntax.

Emits hints for:
  - Type annotations after xonsh expressions (``$HOME : str``, ``$() : str``,
    ``!() : CommandPipeline``, ``p"…" : Path``, etc.) — on by default.
  - Env-var values after ``$VAR`` (``$HOME = /home/user``) — opt-in.
  - Alias resolution at unknown subprocess commands — opt-in (TODO).

The Python backend's hints are returned separately by the proxy and merged
in the server handler; this module produces only the xonsh-native portion.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lsprotocol import types as lsp

from .xonsh_types import envvar_type, expression_type

if TYPE_CHECKING:
    from tree_sitter import Node

    from .parser import ParseResult

logger = logging.getLogger(__name__)


# Node types that produce a value worth annotating. env_variable is handled
# separately because its type depends on the variable name.
_TYPED_EXPR_NODES = frozenset({
    "captured_subprocess",
    "captured_subprocess_object",
    "uncaptured_subprocess",
    "uncaptured_subprocess_object",
    "env_variable_braced",
    "regex_glob",
    "glob_pattern",
    "formatted_glob",
    "glob_path",
    "regex_path_glob",
    "custom_function_glob",
    "path_string",
    "tokenized_substitution",
})

# Suppress hints when the node lives inside one of these contexts — the type
# would be redundant clutter (subprocess args are obviously strings to xonsh).
_SUPPRESS_INSIDE = frozenset({
    "subprocess_argument",
    "subprocess_command",
    "subprocess_body",
    "subprocess_redirect",
})


@dataclass
class InlayHintConfig:
    """Per-document inlay hint preferences (read from initializationOptions)."""

    xonsh_types: bool = True
    env_var_values: bool = False
    alias_resolution: bool = False

    @classmethod
    def from_options(cls, opts: dict | None) -> "InlayHintConfig":
        if not opts:
            return cls()
        section = opts.get("inlayHints") or {}
        if not isinstance(section, dict):
            return cls()
        return cls(
            xonsh_types=bool(section.get("xonshTypes", True)),
            env_var_values=bool(section.get("envVarValues", False)),
            alias_resolution=bool(section.get("aliasResolution", False)),
        )


class XonshInlayHintProvider:
    """Produces inlay hints for xonsh-specific syntax."""

    def __init__(self, config: InlayHintConfig | None = None) -> None:
        self.config = config or InlayHintConfig()

    def get_hints(
        self,
        parse_result: "ParseResult | None",
        range_: lsp.Range,
    ) -> list[lsp.InlayHint]:
        if parse_result is None or parse_result.tree is None:
            return []
        if not (self.config.xonsh_types or self.config.env_var_values):
            return []

        hints: list[lsp.InlayHint] = []
        for node in _walk(parse_result.tree.root_node):
            if not _in_range(node, range_):
                continue
            if _is_suppressed(node):
                continue

            if node.type in {"env_variable", "env_variable_braced"}:
                self._emit_env_var(node, hints)
            elif node.type in _TYPED_EXPR_NODES and self.config.xonsh_types:
                t = expression_type(node.type)
                if t and t != "None":
                    _emit_type(hints, node, t)
        return hints

    def _emit_env_var(self, node: "Node", hints: list[lsp.InlayHint]) -> None:
        name = _env_var_name(node)
        if self.config.xonsh_types:
            t = envvar_type(name) if name else None
            # Unknown vars default to str (xonsh's coercion behavior); braced
            # forms always evaluate to str regardless of inner expression.
            label = t or ("str" if node.type == "env_variable_braced" else "str")
            _emit_type(hints, node, label)

        if self.config.env_var_values and name and name in os.environ:
            value = os.environ[name]
            if len(value) > 60:
                value = value[:57] + "..."
            row, col = node.end_point
            hints.append(
                lsp.InlayHint(
                    position=lsp.Position(line=row, character=col),
                    label=f" = {value}",
                    kind=lsp.InlayHintKind.Parameter,
                    padding_left=True,
                )
            )


def _emit_type(hints: list[lsp.InlayHint], node: "Node", type_str: str) -> None:
    row, col = node.end_point
    hints.append(
        lsp.InlayHint(
            position=lsp.Position(line=row, character=col),
            label=f": {type_str}",
            kind=lsp.InlayHintKind.Type,
            padding_left=True,
        )
    )


def _env_var_name(node: "Node") -> str | None:
    """Extract the variable name from an env_variable / env_variable_braced node."""
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace")
    return None


def _walk(node: "Node"):
    yield node
    for child in node.children:
        yield from _walk(child)


def _is_suppressed(node: "Node") -> bool:
    """True when this node is structurally inside a subprocess argument list,
    where adding ``: str`` after every $VAR / $() would be pure noise."""
    parent = node.parent
    while parent is not None:
        if parent.type in _SUPPRESS_INSIDE:
            return True
        parent = parent.parent
    return False


def _in_range(node: "Node", range_: lsp.Range) -> bool:
    start_line, _ = node.start_point
    end_line, _ = node.end_point
    return not (end_line < range_.start.line or start_line > range_.end.line)
