"""
Completion providers for xonsh LSP.

Provides intelligent code completions for:
- Environment variables
- Subprocess commands (from PATH)
- Xonsh builtins
- Python completions (via backend - Jedi or LSP proxy)
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from lsprotocol import types as lsp

from xonsh_lsp.xonsh_builtins import XONSH_BUILTINS, XONSH_ALIASES, XONSH_MAGIC_VARS, XONSH_XONTRIBS, XONSH_AT_OBJECTS

if TYPE_CHECKING:
    from xonsh_lsp.server import XonshLanguageServer

logger = logging.getLogger(__name__)


class XonshCompletionProvider:
    """Provides completions for xonsh files."""

    def __init__(self, server: XonshLanguageServer):
        self.server = server
        self._command_cache: list[str] | None = None
        self._command_cache_time: float = 0

    async def get_completions(self, params: lsp.CompletionParams) -> lsp.CompletionList | None:
        """Get completions at the given position."""
        uri = params.text_document.uri
        doc = self.server.get_document(uri)
        if doc is None:
            logger.debug(f"No document found for {uri}")
            return None

        line = params.position.line
        col = params.position.character
        source = doc.source

        logger.debug(f"Completion requested at line {line}, col {col}")

        # Get current line text
        lines = source.splitlines()
        if line >= len(lines):
            current_line = ""
        else:
            current_line = lines[line]

        # Get text before cursor
        text_before = current_line[:col] if col <= len(current_line) else current_line
        logger.debug(f"Text before cursor: '{text_before}'")

        items: list[lsp.CompletionItem] = []

        # Determine context and provide appropriate completions
        trigger_char = params.context.trigger_character if params.context else None
        logger.debug(f"Trigger character: {trigger_char}")

        # Environment variable completions
        if trigger_char == "$" or "$" in text_before:
            env_items = self._get_env_completions(text_before)
            logger.debug(f"Added {len(env_items)} env completions")
            items.extend(env_items)

        # Path completions
        if trigger_char == "/" or self._is_path_context(text_before):
            path_items = self._get_path_completions(text_before)
            logger.debug(f"Added {len(path_items)} path completions")
            items.extend(path_items)

        # Glob completions
        if trigger_char == "`":
            glob_items = self._get_glob_completions()
            logger.debug(f"Added {len(glob_items)} glob completions")
            items.extend(glob_items)

        # Python evaluation completions
        if trigger_char == "@" or "@(" in text_before:
            py_eval_items = await self._get_python_eval_completions(source, line, col)
            logger.debug(f"Added {len(py_eval_items)} python eval completions")
            items.extend(py_eval_items)

        # @ object completions (@.env, @.imp, etc.)
        if "@." in text_before:
            at_obj_items = self._get_at_object_completions(text_before)
            logger.debug(f"Added {len(at_obj_items)} @ object completions")
            items.extend(at_obj_items)

        # Xontrib completions
        if "xontrib" in text_before.lower():
            xontrib_items = self._get_xontrib_completions(text_before)
            logger.debug(f"Added {len(xontrib_items)} xontrib completions")
            items.extend(xontrib_items)

        # Check if we're in subprocess context
        parse_result = self.server.parse_document(uri)
        in_subprocess = False
        if parse_result and parse_result.tree:
            in_subprocess = self.server.parser.is_in_subprocess_context(
                parse_result.tree, line, col
            )
        logger.debug(f"In subprocess context: {in_subprocess}")

        if in_subprocess:
            # Subprocess command completions
            cmd_items = self._get_command_completions(text_before)
            logger.debug(f"Added {len(cmd_items)} command completions")
            items.extend(cmd_items)
        else:
            # Python completions
            python_completions = await self.server.python_delegate.get_completions(
                source, line, col, doc.path
            )
            logger.debug(f"Added {len(python_completions)} python completions")
            items.extend(python_completions)

        # Always add xonsh builtins
        builtin_items = self._get_xonsh_builtin_completions(text_before)
        logger.debug(f"Added {len(builtin_items)} xonsh builtin completions")
        items.extend(builtin_items)

        logger.debug(f"Returning {len(items)} total completions")
        return lsp.CompletionList(is_incomplete=False, items=items)

    def resolve_completion(self, item: lsp.CompletionItem) -> lsp.CompletionItem:
        """Resolve additional completion item details."""
        # Add documentation based on item kind
        if item.data:
            data = item.data
            if isinstance(data, dict):
                if data.get("type") == "env_var":
                    var_name = data.get("name", "")
                    value = os.environ.get(var_name, "")
                    item.documentation = lsp.MarkupContent(
                        kind=lsp.MarkupKind.Markdown,
                        value=f"**Environment Variable**\n\n`${var_name}` = `{value}`",
                    )
                elif data.get("type") == "command":
                    cmd = data.get("name", "")
                    path = shutil.which(cmd)
                    item.documentation = lsp.MarkupContent(
                        kind=lsp.MarkupKind.Markdown,
                        value=f"**Command**\n\n`{cmd}`\n\nPath: `{path or 'not found'}`",
                    )
                elif data.get("type") == "xonsh_builtin":
                    name = data.get("name", "")
                    doc = XONSH_BUILTINS.get(name, {}).get("doc", "")
                    item.documentation = lsp.MarkupContent(
                        kind=lsp.MarkupKind.Markdown,
                        value=f"**Xonsh Builtin**\n\n{doc}",
                    )

        return item

    def _get_env_completions(self, text_before: str) -> list[lsp.CompletionItem]:
        """Get environment variable completions."""
        items = []

        # Get prefix after $
        prefix = ""
        if "${" in text_before:
            prefix = text_before.split("${")[-1].rstrip("}")
        elif "$" in text_before:
            prefix = text_before.split("$")[-1]

        # Add environment variables
        for name, value in sorted(os.environ.items()):
            if prefix.lower() in name.lower():
                # Truncate long values
                display_value = value if len(value) < 50 else value[:47] + "..."
                items.append(
                    lsp.CompletionItem(
                        label=name,
                        kind=lsp.CompletionItemKind.Variable,
                        detail=display_value,
                        insert_text=name,
                        filter_text=name,
                        sort_text=f"0_{name}",
                        data={"type": "env_var", "name": name},
                    )
                )

        # Add xonsh magic variables
        for name, info in XONSH_MAGIC_VARS.items():
            if prefix.lower() in name.lower():
                items.append(
                    lsp.CompletionItem(
                        label=name,
                        kind=lsp.CompletionItemKind.Variable,
                        detail=info.get("type", ""),
                        documentation=info.get("doc", ""),
                        insert_text=name,
                        filter_text=name,
                        sort_text=f"1_{name}",
                    )
                )

        return items

    def _get_command_completions(self, text_before: str) -> list[lsp.CompletionItem]:
        """Get subprocess command completions from PATH."""
        items = []

        # Get command prefix
        parts = text_before.split()
        prefix = parts[-1] if parts else ""

        # Remove subprocess markers
        for marker in ["$(", "!(", "$[", "![", "@$("]:
            if prefix.startswith(marker):
                prefix = prefix[len(marker) :]
                break

        # Get commands from PATH
        commands = self._get_available_commands()

        for cmd in commands:
            if prefix.lower() in cmd.lower():
                items.append(
                    lsp.CompletionItem(
                        label=cmd,
                        kind=lsp.CompletionItemKind.Function,
                        detail="command",
                        sort_text=f"2_{cmd}",
                        data={"type": "command", "name": cmd},
                    )
                )

        return items

    def _get_available_commands(self) -> list[str]:
        """Get list of available commands from PATH."""
        import time

        # Cache commands for 60 seconds
        now = time.time()
        if self._command_cache and (now - self._command_cache_time) < 60:
            return self._command_cache

        commands = set()
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)

        for path_dir in path_dirs:
            try:
                if os.path.isdir(path_dir):
                    for entry in os.listdir(path_dir):
                        full_path = os.path.join(path_dir, entry)
                        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                            commands.add(entry)
            except (PermissionError, OSError):
                continue

        self._command_cache = sorted(commands)
        self._command_cache_time = now

        return self._command_cache

    def _get_path_completions(self, text_before: str) -> list[lsp.CompletionItem]:
        """Get file path completions."""
        items = []

        # Extract path from text
        path_start = max(
            text_before.rfind(" "),
            text_before.rfind("("),
            text_before.rfind("["),
            text_before.rfind('"'),
            text_before.rfind("'"),
        )
        path_text = text_before[path_start + 1 :] if path_start >= 0 else text_before

        # Expand ~ and environment variables
        try:
            expanded = os.path.expanduser(os.path.expandvars(path_text))
        except Exception:
            expanded = path_text

        # Get directory and prefix
        if os.path.isdir(expanded):
            directory = expanded
            prefix = ""
        else:
            directory = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)

        # List directory contents
        try:
            for entry in os.listdir(directory):
                if entry.startswith(".") and not prefix.startswith("."):
                    continue

                if prefix.lower() in entry.lower():
                    full_path = os.path.join(directory, entry)
                    is_dir = os.path.isdir(full_path)

                    items.append(
                        lsp.CompletionItem(
                            label=entry + ("/" if is_dir else ""),
                            kind=(
                                lsp.CompletionItemKind.Folder
                                if is_dir
                                else lsp.CompletionItemKind.File
                            ),
                            detail="directory" if is_dir else "file",
                            insert_text=entry + ("/" if is_dir else ""),
                            sort_text=f"{'0' if is_dir else '1'}_{entry}",
                        )
                    )
        except (PermissionError, OSError):
            pass

        return items

    def _get_glob_completions(self) -> list[lsp.CompletionItem]:
        """Get glob pattern completions."""
        patterns = [
            ("*", "Match any characters"),
            ("**", "Match recursively"),
            ("?", "Match single character"),
            ("[abc]", "Match character set"),
            ("[!abc]", "Match inverted set"),
            ("[a-z]", "Match character range"),
            ("*.py", "Python files"),
            ("**/*.py", "All Python files recursively"),
            (".*", "Hidden files"),
        ]

        return [
            lsp.CompletionItem(
                label=pattern,
                kind=lsp.CompletionItemKind.Snippet,
                detail=description,
                insert_text=pattern,
            )
            for pattern, description in patterns
        ]

    def _get_xonsh_builtin_completions(
        self, text_before: str
    ) -> list[lsp.CompletionItem]:
        """Get xonsh builtin completions."""
        items = []

        # Get prefix
        parts = text_before.split()
        prefix = parts[-1] if parts else ""

        # Add xonsh builtins
        for name, info in XONSH_BUILTINS.items():
            if prefix.lower() in name.lower():
                items.append(
                    lsp.CompletionItem(
                        label=name,
                        kind=lsp.CompletionItemKind.Function,
                        detail=info.get("signature", ""),
                        documentation=info.get("doc", ""),
                        insert_text=info.get("snippet", name),
                        insert_text_format=lsp.InsertTextFormat.Snippet,
                        sort_text=f"3_{name}",
                        data={"type": "xonsh_builtin", "name": name},
                    )
                )

        # Add xonsh aliases
        for name, info in XONSH_ALIASES.items():
            if prefix.lower() in name.lower():
                items.append(
                    lsp.CompletionItem(
                        label=name,
                        kind=lsp.CompletionItemKind.Keyword,
                        detail=info.get("description", "alias"),
                        documentation=info.get("doc", ""),
                        sort_text=f"4_{name}",
                    )
                )

        return items

    async def _get_python_eval_completions(
        self, source: str, line: int, col: int
    ) -> list[lsp.CompletionItem]:
        """Get completions inside @() Python evaluation."""
        # Delegate to Python completions
        return await self.server.python_delegate.get_completions(source, line, col, None)

    def _is_path_context(self, text_before: str) -> bool:
        """Check if we're in a path completion context."""
        # Look for path indicators
        path_indicators = [
            "/",
            "~/",
            "./",
            "../",
            'p"',
            "p'",
            'pf"',
            "pf'",
        ]
        return any(indicator in text_before for indicator in path_indicators)

    def _get_at_object_completions(self, text_before: str) -> list[lsp.CompletionItem]:
        """Get @ object attribute completions (@.env, @.imp, etc.)."""
        items = []

        # Get prefix after @.
        prefix = ""
        if "@." in text_before:
            after_at = text_before.split("@.")[-1]
            # Handle chained access like @.imp.json
            prefix = after_at.split(".")[0] if "." in after_at else after_at

        for name, info in XONSH_AT_OBJECTS.items():
            if prefix.lower() in name.lower():
                items.append(
                    lsp.CompletionItem(
                        label=name,
                        kind=lsp.CompletionItemKind.Property,
                        detail="@ object attribute",
                        documentation=lsp.MarkupContent(
                            kind=lsp.MarkupKind.Markdown,
                            value=f"{info.get('doc', '')}\n\n**Example:** `{info.get('example', '')}`",
                        ),
                        insert_text=name,
                        sort_text=f"0_{name}",
                    )
                )

        return items

    def _get_xontrib_completions(self, text_before: str) -> list[lsp.CompletionItem]:
        """Get xontrib name completions."""
        items = []

        # Check if we're after "xontrib load"
        if "load" not in text_before.lower():
            # Just suggest "load" subcommand
            items.append(
                lsp.CompletionItem(
                    label="load",
                    kind=lsp.CompletionItemKind.Keyword,
                    detail="Load xontrib extension",
                    insert_text="load ",
                    sort_text="0_load",
                )
            )
            return items

        # Get prefix after "load"
        parts = text_before.split()
        prefix = ""
        if parts and parts[-1] not in ("load", "xontrib"):
            prefix = parts[-1]

        for name, info in XONSH_XONTRIBS.items():
            if prefix.lower() in name.lower():
                items.append(
                    lsp.CompletionItem(
                        label=name,
                        kind=lsp.CompletionItemKind.Module,
                        detail="xontrib",
                        documentation=info.get("doc", ""),
                        insert_text=name,
                        sort_text=f"1_{name}",
                    )
                )

        return items
