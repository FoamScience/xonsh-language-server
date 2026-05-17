"""Type registry for xonsh-specific constructs.

Shared by inlay hints (display) and the typed stub injected into the Python
backend (Pyright/ty). Sources env-var types from xonsh_builtins.XONSH_MAGIC_VARS
and hard-codes return types for xonsh expression forms ($(), !(), p"...", etc.).
"""

from __future__ import annotations

from .xonsh_builtins import XONSH_MAGIC_VARS

# Tree-sitter node type -> raw Python type string for xonsh expression forms.
# `None` means "delegate to surrounding context" (e.g. python_evaluation).
_EXPR_TYPES: dict[str, str | None] = {
    "captured_subprocess":        "str",                # $(cmd)
    "captured_subprocess_object": "CommandPipeline",    # !(cmd)
    "uncaptured_subprocess":      "None",               # $[cmd]
    "uncaptured_subprocess_object": "CommandPipeline",  # ![cmd]
    "bare_subprocess":            "None",
    "background_command":         "None",
    "env_variable_braced":        "str",                # ${expr} always str
    "regex_glob":                 "list[str]",
    "glob_pattern":               "list[str]",
    "formatted_glob":             "list[str]",
    "glob_path":                  "list[str]",
    "regex_path_glob":            "list[str]",
    "custom_function_glob":       "list[str]",
    "path_string":                "Path",
    "python_evaluation":          None,
    "tokenized_substitution":     "list[str]",
}


def envvar_type(name: str) -> str | None:
    """Raw type annotation string for a known xonsh env var, or None if unknown."""
    entry = XONSH_MAGIC_VARS.get(name)
    if entry is None:
        return None
    return entry.get("type")


def expression_type(node_type: str) -> str | None:
    """Return type string for a xonsh expression tree-sitter node.

    Returns None for nodes that don't carry a static type (e.g. python_evaluation,
    which evaluates a Python expression of unknown type).
    """
    return _EXPR_TYPES.get(node_type)


# Crude normalizer for registry type strings -> valid Python type expressions
# for use inside the generated stub. The registry includes informal forms like
# "callable" (lowercase) and "int | tuple" that need cleaning before Pyright
# will accept them.
# Stub-friendly substitutions for registry type strings. Names that aren't
# valid Python type expressions in the generated stub (no Callable import,
# no xonsh-specific EnvPath, etc.) fall back to ``Any``.
_NORMALIZE: dict[str, str] = {
    "callable": "__xonsh_typing__.Callable[..., object]",
    "Callable": "__xonsh_typing__.Callable[..., object]",
    "set":      "set[str]",
    "list":     "list[str]",
    "tuple":    "tuple",
    "dict":     "dict[str, str]",
    "path":     "__xonsh_pathlib__.Path",
    "Path":     "__xonsh_pathlib__.Path",
    "EnvPath":  "__xonsh_typing__.Any",
    "XonshSession": "__xonsh_typing__.Any",
}


def to_python_type(type_str: str | None) -> str:
    """Best-effort conversion of a registry type string into a valid Python type
    expression suitable for the generated stub. Falls back to ``object``."""
    if not type_str:
        return "object"
    parts = [p.strip() for p in type_str.split("|")]
    normalized = [_NORMALIZE.get(p, p) for p in parts]
    return " | ".join(normalized) if normalized else "object"


def _identifier_ok(name: str) -> bool:
    return name.isidentifier() and not name.startswith("__")


def build_xonsh_env_typed_dict_lines(
    user_defined: set[str] | None = None,
    class_name: str = "__xonsh_env_dict__",
    typing_alias: str = "__xonsh_typing__",
) -> list[str]:
    """Generate TypedDict body lines for ``__xonsh_env__``.

    Combines the static xonsh env-var registry with user-defined names
    discovered in the current document (typed ``Any`` in v1). The generated
    class is fed into the Python backend so ``__xonsh_env__["HOME"]`` types
    as ``str``, ``__xonsh_env__["XONSH_DEBUG"]`` as ``int``, etc.
    """
    lines: list[str] = [f"class {class_name}({typing_alias}.TypedDict, total=False):"]
    seen: set[str] = set()
    body_added = False

    for name, entry in XONSH_MAGIC_VARS.items():
        if not _identifier_ok(name):
            continue  # e.g. "__xonsh__" — not a string env-var key
        py_type = to_python_type(entry.get("type"))
        lines.append(f"    {name}: {py_type}")
        seen.add(name)
        body_added = True

    for name in sorted(user_defined or ()):
        if name in seen or not _identifier_ok(name):
            continue
        lines.append(f"    {name}: {typing_alias}.Any")
        body_added = True

    if not body_added:
        lines.append("    pass")
    return lines
