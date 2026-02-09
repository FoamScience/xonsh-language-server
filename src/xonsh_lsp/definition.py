"""
Go-to-definition provider for xonsh LSP.

Provides go-to-definition for:
- Python symbols (via backend - Jedi or LSP proxy)
- Xonsh aliases defined in file
- Environment variables set in file
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from lsprotocol import types as lsp

if TYPE_CHECKING:
    from xonsh_lsp.server import XonshLanguageServer

logger = logging.getLogger(__name__)


class XonshDefinitionProvider:
    """Provides go-to-definition for xonsh files."""

    def __init__(self, server: XonshLanguageServer):
        self.server = server

    async def get_definition(
        self, params: lsp.DefinitionParams
    ) -> lsp.Location | list[lsp.Location] | None:
        """Get definition location(s) for symbol at position."""
        uri = params.text_document.uri
        doc = self.server.get_document(uri)
        if doc is None:
            return None

        line = params.position.line
        col = params.position.character
        source = doc.source

        word = self._get_word_at_position(source, line, col)
        if not word:
            return None

        locations: list[lsp.Location] = []

        # Check for environment variable definition
        if word.startswith("$"):
            var_name = word[1:] if word.startswith("$") else word
            if word.startswith("${") and word.endswith("}"):
                var_name = word[2:-1]

            env_def = self._find_env_var_definition(source, var_name, uri)
            if env_def:
                locations.append(env_def)

        # Check for alias definition
        alias_def = self._find_alias_definition(source, word, uri)
        if alias_def:
            locations.append(alias_def)

        # Check for xonsh function definition
        func_def = self._find_function_definition(source, word, uri)
        if func_def:
            locations.append(func_def)

        # Get Python definitions from backend
        python_defs = await self.server.python_delegate.get_definitions(
            source, line, col, doc.path
        )
        locations.extend(python_defs)

        if not locations:
            return None
        elif len(locations) == 1:
            return locations[0]
        else:
            return locations

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
        word_chars = re.compile(r"[\w$]")

        start = col
        while start > 0 and word_chars.match(line_text[start - 1]):
            start -= 1

        end = col
        while end < len(line_text) and word_chars.match(line_text[end]):
            end += 1

        return line_text[start:end] if start < end else None

    def _find_env_var_definition(
        self, source: str, var_name: str, uri: str
    ) -> lsp.Location | None:
        """Find where an environment variable is defined in source."""
        lines = source.splitlines()

        # Patterns for environment variable assignment
        patterns = [
            # $VAR = value
            rf"^\s*\${re.escape(var_name)}\s*=",
            # ${VAR} = value
            rf"^\s*\$\{{{re.escape(var_name)}\}}\s*=",
            # os.environ['VAR'] = value
            rf"os\.environ\[[\'\"]" + re.escape(var_name) + r"[\'\"]\]\s*=",
            # $ENV['VAR'] = value
            rf"\$ENV\[[\'\"]" + re.escape(var_name) + r"[\'\"]\]\s*=",
        ]

        for line_num, line_text in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line_text)
                if match:
                    return lsp.Location(
                        uri=uri,
                        range=lsp.Range(
                            start=lsp.Position(
                                line=line_num,
                                character=match.start(),
                            ),
                            end=lsp.Position(
                                line=line_num,
                                character=match.end(),
                            ),
                        ),
                    )

        return None

    def _find_alias_definition(
        self, source: str, name: str, uri: str
    ) -> lsp.Location | None:
        """Find where an alias is defined in source."""
        lines = source.splitlines()

        # Patterns for alias definition
        patterns = [
            # aliases['name'] = ...
            rf"aliases\[[\'\"]" + re.escape(name) + r"[\'\"]\]\s*=",
            # aliases.register('name', ...)
            rf"aliases\.register\s*\(\s*[\'\"]" + re.escape(name) + r"[\'\"]",
        ]

        for line_num, line_text in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line_text)
                if match:
                    return lsp.Location(
                        uri=uri,
                        range=lsp.Range(
                            start=lsp.Position(
                                line=line_num,
                                character=match.start(),
                            ),
                            end=lsp.Position(
                                line=line_num,
                                character=match.end(),
                            ),
                        ),
                    )

        return None

    def _find_function_definition(
        self, source: str, name: str, uri: str
    ) -> lsp.Location | None:
        """Find where a function is defined in source."""
        lines = source.splitlines()

        # Pattern for function definition
        pattern = rf"^\s*def\s+{re.escape(name)}\s*\("

        for line_num, line_text in enumerate(lines):
            match = re.search(pattern, line_text)
            if match:
                return lsp.Location(
                    uri=uri,
                    range=lsp.Range(
                        start=lsp.Position(
                            line=line_num,
                            character=match.start(),
                        ),
                        end=lsp.Position(
                            line=line_num,
                            character=match.end(),
                        ),
                    ),
                )

        return None


class XonshReferenceProvider:
    """Provides find-references for xonsh files."""

    def __init__(self, server: XonshLanguageServer):
        self.server = server

    def get_references(
        self, params: lsp.ReferenceParams
    ) -> list[lsp.Location] | None:
        """Get all references to symbol at position."""
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

        locations: list[lsp.Location] = []

        # Find all occurrences in source
        lines = source.splitlines()
        for line_num, line_text in enumerate(lines):
            # Find all occurrences of word in line
            pattern = rf"\b{re.escape(word)}\b"
            for match in re.finditer(pattern, line_text):
                locations.append(
                    lsp.Location(
                        uri=uri,
                        range=lsp.Range(
                            start=lsp.Position(
                                line=line_num,
                                character=match.start(),
                            ),
                            end=lsp.Position(
                                line=line_num,
                                character=match.end(),
                            ),
                        ),
                    )
                )

        # Also check for $VAR references
        if not word.startswith("$"):
            var_patterns = [rf"\${re.escape(word)}\b", rf"\$\{{{re.escape(word)}\}}"]
            for line_num, line_text in enumerate(lines):
                for pattern in var_patterns:
                    for match in re.finditer(pattern, line_text):
                        locations.append(
                            lsp.Location(
                                uri=uri,
                                range=lsp.Range(
                                    start=lsp.Position(
                                        line=line_num,
                                        character=match.start(),
                                    ),
                                    end=lsp.Position(
                                        line=line_num,
                                        character=match.end(),
                                    ),
                                ),
                            )
                        )

        return locations if locations else None

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

        word_chars = re.compile(r"\w")

        start = col
        while start > 0 and word_chars.match(line_text[start - 1]):
            start -= 1

        end = col
        while end < len(line_text) and word_chars.match(line_text[end]):
            end += 1

        return line_text[start:end] if start < end else None
