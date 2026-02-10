"""Tests for the preprocessing module."""

import pytest
from xonsh_lsp.preprocessing import (
    PreprocessResult,
    _build_line_mapping,
    _map_column,
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
        assert '[""]' in result

    def test_named_glob(self):
        result = preprocess_source("x = g`*.txt`")
        assert '[""]' in result

    def test_formatted_glob(self):
        result = preprocess_source("x = f`*.{ext}`")
        assert '[""]' in result

    def test_path_glob(self):
        result = preprocess_source("for f in gp`*.*`:")
        assert '[Path("")]' in result

    def test_glob_path_compound(self):
        result = preprocess_source("for f in gp`*.*`:")
        assert '[Path("")]' in result
        assert "gPath" not in result  # should not leak prefix

    def test_regex_path_glob(self):
        result = preprocess_source("x = rp`.*\\.py`")
        assert '[Path("")]' in result

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

    def test_pure_python_preserved(self):
        source = "import os\nx = 1 + 2\ndef foo(): pass"
        result = preprocess_source(source)
        assert "import os" in result
        assert "x = 1 + 2" in result
        assert "def foo(): pass" in result

    def test_nested_subprocess(self):
        result = preprocess_source("$(echo $(inner))")
        assert "__xonsh_subproc__" in result

    def test_string_not_transformed(self):
        """Xonsh syntax inside string literals should NOT be transformed."""
        result = preprocess_source('x = "echo $HOME"')
        assert '__xonsh_env__' not in result
        assert '"echo $HOME"' in result

    def test_env_assignment(self):
        """$VAR = value should transform $VAR but preserve the assignment."""
        result = preprocess_source('$HOME = "/home/me"')
        assert '__xonsh_env__["HOME"]' in result
        assert '"/home/me"' in result

    def test_env_deletion(self):
        """del $VAR should transform $VAR."""
        result = preprocess_source('del $HOME')
        assert '__xonsh_env__["HOME"]' in result

    def test_custom_function_glob(self):
        result = preprocess_source("x = @func`pattern`")
        assert '[""]' in result


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


class TestPreprocessResultNewFields:
    """Test the new masked_lines and xonsh_lines fields."""

    def test_masked_lines_for_bare_subprocess(self):
        result = preprocess_with_mapping("import os\ncd /tmp\nx = 1")
        assert 1 in result.masked_lines
        assert 0 not in result.masked_lines
        assert 2 not in result.masked_lines

    def test_xonsh_lines_for_replaceable(self):
        result = preprocess_with_mapping("x = $HOME\ny = 1")
        assert 0 in result.xonsh_lines
        assert 1 not in result.xonsh_lines

    def test_xonsh_lines_for_masked(self):
        result = preprocess_with_mapping("cd /tmp")
        assert 0 in result.xonsh_lines
        assert 0 in result.masked_lines

    def test_env_assignment_transparency(self):
        """env_assignment should descend into children (env_variable is replaced)."""
        result = preprocess_with_mapping('$HOME = "/home/me"')
        assert '__xonsh_env__["HOME"]' in result.source
        assert 0 in result.xonsh_lines

    def test_env_deletion_transparency(self):
        """env_deletion should descend into children."""
        result = preprocess_with_mapping('del $HOME')
        assert '__xonsh_env__["HOME"]' in result.source
        assert 0 in result.xonsh_lines

    def test_multiple_constructs_per_line(self):
        result = preprocess_with_mapping("print($HOME, $PATH)")
        assert '__xonsh_env__["HOME"]' in result.source
        assert '__xonsh_env__["PATH"]' in result.source
        assert 0 in result.xonsh_lines

    def test_pure_python_no_xonsh_lines(self):
        result = preprocess_with_mapping("x = 1\ndef foo(): pass")
        assert len(result.xonsh_lines) == 0
        assert len(result.masked_lines) == 0

    def test_bare_subprocess_masked_with_indent(self):
        result = preprocess_with_mapping("if True:\n    cd /tmp\n    x = 1")
        lines = result.source.split("\n")
        assert lines[1] == "    pass"
        assert 1 in result.masked_lines

    def test_multiple_bare_subprocesses(self):
        result = preprocess_with_mapping("cd /tmp\necho hello\nls -la")
        lines = result.source.split("\n")
        assert all(line == "pass" for line in lines)
        assert result.masked_lines == {0, 1, 2}

    def test_env_scoped_command_masked(self):
        result = preprocess_with_mapping("import os\n$CONCH='snail' ls\nx = 1")
        lines = result.source.split("\n")
        assert lines[0] == "import os"
        assert lines[1] == "pass"
        assert lines[2] == "x = 1"
        assert 1 in result.masked_lines

    def test_help_expression_masked(self):
        result = preprocess_with_mapping("os.path?")
        assert 0 in result.masked_lines

    def test_super_help_expression_masked(self):
        result = preprocess_with_mapping("os.path??")
        assert 0 in result.masked_lines

    def test_xontrib_masked(self):
        result = preprocess_with_mapping("xontrib load vox z")
        assert 0 in result.masked_lines

    def test_line_count_preserved_after_masking(self):
        source = "import os\ncd /tmp\necho hello\nls -la\nx = 1"
        result = preprocess_with_mapping(source)
        assert len(result.source.split("\n")) == len(source.split("\n"))

    def test_empty_source(self):
        result = preprocess_with_mapping("")
        assert result.source == ""
        assert len(result.masked_lines) == 0
        assert len(result.xonsh_lines) == 0


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
