"""Tests for semantic token generation."""

import pytest
from lsprotocol import types as lsp
from xonsh_lsp.parser import XonshParser


def decode_tokens(tokens: lsp.SemanticTokens, source: str) -> list[dict]:
    """Decode delta-encoded semantic tokens into readable dicts."""
    type_names = [t.value for t in lsp.SemanticTokenTypes]
    mod_names = [m.value for m in lsp.SemanticTokenModifiers]
    lines = source.splitlines()
    result = []
    prev_line = prev_col = 0
    data = tokens.data
    for i in range(0, len(data), 5):
        dl, dc, length, ti, mod_bits = data[i:i+5]
        line = prev_line + dl
        col = dc if dl else prev_col + dc
        text = lines[line][col:col+length] if line < len(lines) else ""
        modifiers = [mod_names[j] for j in range(len(mod_names)) if mod_bits & (1 << j)]
        result.append({
            "line": line, "col": col, "length": length,
            "type": type_names[ti] if ti < len(type_names) else str(ti),
            "modifiers": modifiers,
            "text": text,
        })
        prev_line, prev_col = line, col
    return result


def tokens_of_type(decoded: list[dict], token_type: str) -> list[dict]:
    return [t for t in decoded if t["type"] == token_type]


class TestParserSemanticTokens:

    @pytest.fixture
    def parser(self):
        return XonshParser()

    def test_empty_source(self, parser):
        result = parser.get_semantic_tokens("")
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        assert result.data == []

    def test_returns_none_without_tree_sitter(self):
        p = XonshParser()
        p._initialized = False
        assert p.get_semantic_tokens("x = 1") is None

    def test_python_keyword(self, parser):
        result = parser.get_semantic_tokens("def foo(): pass")
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, "def foo(): pass")
        keywords = [t["text"] for t in tokens_of_type(decoded, "keyword")]
        assert "def" in keywords
        assert "pass" in keywords

    def test_function_definition(self, parser):
        result = parser.get_semantic_tokens("def greet(name): pass")
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, "def greet(name): pass")
        assert any(t["text"] == "greet" and t["type"] == "function" for t in decoded)
        assert any(t["text"] == "name" and t["type"] == "parameter" for t in decoded)

    def test_import(self, parser):
        source = "import os"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        assert any(t["text"] == "import" and t["type"] == "keyword" for t in decoded)
        assert any(t["text"] == "os" and t["type"] == "namespace" for t in decoded)

    def test_string_no_duplicates(self, parser):
        source = "x = 'hello'"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        strings = tokens_of_type(decoded, "string")
        # outer string node and inner string_content must not both appear
        assert len(strings) == 1
        assert strings[0]["text"] == "'hello'"

    def test_number(self, parser):
        source = "x = 42"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        assert any(t["text"] == "42" and t["type"] == "number" for t in decoded)

    def test_operator(self, parser):
        source = "x = 1 + 2"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        ops = [t["text"] for t in tokens_of_type(decoded, "operator")]
        assert "+" in ops
        assert "=" in ops

    def test_env_variable(self, parser):
        source = "$HOME"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        assert any(t["text"] == "HOME" and t["type"] == "variable" for t in decoded)
        home = next(t for t in decoded if t["text"] == "HOME")
        assert "defaultLibrary" in home["modifiers"]

    def test_subprocess_command(self, parser):
        source = "ls -la"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        assert any(t["text"] == "ls" and t["type"] == "function" for t in decoded)

    def test_subprocess_flag_is_parameter(self, parser):
        source = "ls -la"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        assert any(t["text"] == "-la" and t["type"] == "parameter" for t in decoded)

    def test_tokens_sorted_by_position(self, parser):
        source = "import os\nx = 1\n"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        positions = [(t["line"], t["col"]) for t in decoded]
        assert positions == sorted(positions)

    def test_no_overlapping_tokens(self, parser):
        source = "x = 'hello world'\n"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        for i in range(len(decoded) - 1):
            a, b = decoded[i], decoded[i + 1]
            if a["line"] == b["line"]:
                assert a["col"] + a["length"] <= b["col"], (
                    f"overlapping tokens: {a} and {b}"
                )

    def test_range_subset(self, parser):
        source = "import os\nx = 1\nls -la\n"
        full = parser.get_semantic_tokens(source)
        ranged = parser.get_semantic_tokens(source, start_line=1, end_line=1)
        if full is None or ranged is None:
            pytest.skip("tree-sitter-xonsh not available")
        full_decoded = decode_tokens(full, source)
        range_decoded = decode_tokens(ranged, source)
        range_lines = {t["line"] for t in range_decoded}
        assert range_lines <= {1}
        full_on_line1 = [t for t in full_decoded if t["line"] == 1]
        assert range_decoded == full_on_line1

    def test_builtin_function_has_default_library_modifier(self, parser):
        source = "print('hi')"
        result = parser.get_semantic_tokens(source)
        if result is None:
            pytest.skip("tree-sitter-xonsh not available")
        decoded = decode_tokens(result, source)
        print_tok = next((t for t in decoded if t["text"] == "print"), None)
        assert print_tok is not None
        assert print_tok["type"] == "function"
        assert "defaultLibrary" in print_tok["modifiers"]
