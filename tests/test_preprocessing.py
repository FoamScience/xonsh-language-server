"""Tests for the preprocessing module."""

import pytest
from xonsh_lsp.preprocessing import (
    PreprocessResult,
    _build_line_mapping,
    _find_balanced_end,
    _map_column,
    _replace_balanced_patterns,
    has_xonsh_syntax,
    map_position_from_processed,
    map_position_to_processed,
    preprocess_source,
    preprocess_with_mapping,
)


class TestPreprocessSource:
    """Test the preprocess_source function."""

    def test_simple_env_var(self):
        result = preprocess_source("print($HOME)")
        assert "__xonsh_env__" in result
        assert "$HOME" not in result

    def test_braced_env_var(self):
        result = preprocess_source("print(${PATH})")
        assert "__xonsh_env__[PATH]" in result

    def test_braced_env_var_string_key(self):
        result = preprocess_source("print(${'PATH'})")
        assert "__xonsh_env__['PATH']" in result

    def test_captured_subprocess(self):
        result = preprocess_source("x = $(ls -la)")
        assert "__xonsh_subproc__" in result
        assert "$(ls -la)" not in result

    def test_subprocess_object(self):
        result = preprocess_source("x = !(git status)")
        assert "__xonsh_subproc__" in result

    def test_uncaptured_subprocess(self):
        result = preprocess_source("$[echo hello]")
        assert "__xonsh_subproc__" in result

    def test_uncaptured_subprocess_object(self):
        result = preprocess_source("![make build]")
        assert "__xonsh_subproc__" in result

    def test_tokenized_substitution(self):
        result = preprocess_source("@$(cmd)")
        assert "__xonsh_subproc__" in result

    def test_python_eval(self):
        result = preprocess_source("print(@(expr))")
        assert "print((expr))" in result

    def test_at_object(self):
        result = preprocess_source("@.imp")
        assert "__xonsh_at__." in result

    def test_glob(self):
        result = preprocess_source("x = `*.py`")
        assert '"__glob__"' in result

    def test_named_glob(self):
        result = preprocess_source("x = g`*.txt`")
        assert '"__glob__"' in result

    def test_formatted_glob(self):
        result = preprocess_source("x = f`*.{ext}`")
        assert '"__glob__"' in result

    def test_path_literal(self):
        result = preprocess_source('x = p"/home/user"')
        assert 'Path("/home/user")' in result
        assert "from pathlib import Path" in result

    def test_formatted_path_literal(self):
        result = preprocess_source('x = pf"/home/{user}"')
        assert 'Path(f"/home/{user}")' in result

    def test_raw_path_literal(self):
        result = preprocess_source(r'x = pr"\home\user"')
        assert r'Path(r"\home\user")' in result

    def test_macro_call(self):
        result = preprocess_source("func!(args)")
        assert 'func("args")' in result
        assert "func!(" not in result

    def test_xontrib_statement(self):
        result = preprocess_source("xontrib load vox z")
        assert "pass" in result
        assert "xontrib" not in result or "# xontrib" in result

    def test_pure_python_preserved(self):
        source = "import os\nx = 1 + 2\ndef foo(): pass"
        result = preprocess_source(source)
        assert "import os" in result
        assert "x = 1 + 2" in result
        assert "def foo(): pass" in result

    def test_nested_subprocess(self):
        result = preprocess_source("$(echo $(inner))")
        assert "__xonsh_subproc__" in result


class TestPreprocessWithMapping:
    """Test preprocess_with_mapping and position mapping."""

    def test_returns_preprocess_result(self):
        result = preprocess_with_mapping("x = 1")
        assert isinstance(result, PreprocessResult)
        assert result.source == "x = 1"
        assert result.added_lines_before == 0

    def test_path_literal_adds_import(self):
        result = preprocess_with_mapping('x = p"/home"')
        assert "from pathlib import Path" in result.source
        assert result.added_lines_before == 1

    def test_line_mappings_exist(self):
        result = preprocess_with_mapping("print($HOME)\nx = 1")
        assert len(result.line_mappings) == 2

    def test_map_position_identity(self):
        """Positions in pure Python should map to themselves."""
        result = preprocess_with_mapping("x = 1 + 2")
        mapped = map_position_to_processed(result, 0, 5)
        assert mapped == (0, 5)

    def test_map_position_with_import(self):
        """Positions after added import should be offset."""
        result = preprocess_with_mapping('x = p"/home"\ny = 1')
        assert result.added_lines_before == 1
        # Line 1 in original should map to line 2 in processed
        mapped = map_position_to_processed(result, 1, 0)
        assert mapped[0] == 2

    def test_map_position_from_processed(self):
        """Reverse mapping should work."""
        result = preprocess_with_mapping('x = p"/home"\ny = 1')
        # Line 2 in processed should map back to line 1 in original
        mapped = map_position_from_processed(result, 2, 0)
        assert mapped[0] == 1


class TestBuildLineMapping:
    """Test the _build_line_mapping function."""

    def test_identical_lines(self):
        mapping = _build_line_mapping("hello", "hello")
        assert len(mapping) >= 2  # At least start and end

    def test_different_lines(self):
        mapping = _build_line_mapping("$HOME", '__xonsh_env__["HOME"]')
        assert len(mapping) >= 2

    def test_empty_lines(self):
        mapping = _build_line_mapping("", "")
        assert (0, 0) in mapping


class TestMapColumn:
    """Test the _map_column function."""

    def test_identity_mapping(self):
        mapping = [(0, 0), (10, 10)]
        assert _map_column(5, mapping) == 5

    def test_empty_mapping(self):
        assert _map_column(5, []) == 5

    def test_past_end(self):
        mapping = [(0, 0), (5, 10)]
        assert _map_column(10, mapping) == 10


class TestFindBalancedEnd:
    """Test the _find_balanced_end function."""

    def test_simple_parens(self):
        assert _find_balanced_end("(hello)", 1, "(", ")") == 6

    def test_nested_parens(self):
        assert _find_balanced_end("(a(b)c)", 1, "(", ")") == 6

    def test_brackets(self):
        assert _find_balanced_end("[hello]", 1, "[", "]") == 6

    def test_unbalanced(self):
        assert _find_balanced_end("(hello", 1, "(", ")") == -1

    def test_with_strings(self):
        assert _find_balanced_end('("hello")', 1, "(", ")") == 8

    def test_with_escaped_chars(self):
        assert _find_balanced_end(r'(a\)b)', 1, "(", ")") == 5


class TestReplaceBalancedPatterns:
    """Test the _replace_balanced_patterns function."""

    def test_captured_subprocess(self):
        result = _replace_balanced_patterns("$(ls)")
        assert result == "__xonsh_subproc__()"

    def test_subprocess_object(self):
        result = _replace_balanced_patterns("!(cmd)")
        assert result == "__xonsh_subproc__()"

    def test_python_eval_preserves_expr(self):
        result = _replace_balanced_patterns("@(expr)")
        assert result == "(expr)"

    def test_tokenized_substitution(self):
        result = _replace_balanced_patterns("@$(cmd)")
        assert result == "__xonsh_subproc__()"

    def test_mixed_text(self):
        result = _replace_balanced_patterns("x = $(ls) + 1")
        assert "__xonsh_subproc__" in result
        assert "x = " in result
        assert " + 1" in result


class TestHasXonshSyntax:
    """Test the has_xonsh_syntax function."""

    def test_env_var(self):
        assert has_xonsh_syntax("print($HOME)")

    def test_subprocess(self):
        assert has_xonsh_syntax("$(ls)")

    def test_glob(self):
        assert has_xonsh_syntax("`*.py`")

    def test_at_object(self):
        assert has_xonsh_syntax("@.imp")

    def test_xontrib(self):
        assert has_xonsh_syntax("xontrib load vox")

    def test_macro(self):
        assert has_xonsh_syntax("func!(args)")

    def test_shell_operators(self):
        assert has_xonsh_syntax("cmd1 && cmd2")
        assert has_xonsh_syntax("cmd1 || cmd2")

    def test_pipe(self):
        assert has_xonsh_syntax("cmd1 | cmd2")

    def test_flags(self):
        assert has_xonsh_syntax("cmd -v")
        assert has_xonsh_syntax("cmd --verbose")

    def test_path_like_command(self):
        assert has_xonsh_syntax("./script.sh")
        assert has_xonsh_syntax("/usr/bin/cmd")
        assert has_xonsh_syntax("~/bin/cmd")

    def test_redirects(self):
        assert has_xonsh_syntax("cmd > file")
        assert has_xonsh_syntax("cmd 2> file")
        assert has_xonsh_syntax("cmd &> file")

    def test_pure_python(self):
        assert not has_xonsh_syntax("x = 1 + 2")
        assert not has_xonsh_syntax("def foo(): pass")
        assert not has_xonsh_syntax("import os")
