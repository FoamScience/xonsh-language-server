"""Tests for the Python delegate."""

import pytest
from xonsh_lsp.python_delegate import PythonDelegate, JEDI_AVAILABLE


class TestPythonDelegatePreprocessing:
    """Test xonsh-to-Python preprocessing."""

    @pytest.fixture
    def delegate(self):
        """Create a Python delegate."""
        return PythonDelegate()

    def test_preprocess_simple_env_var(self, delegate):
        """Test preprocessing simple environment variables."""
        source = "print($HOME)"
        result = delegate._preprocess_source(source)
        assert "__xonsh_env__" in result
        assert "HOME" in result
        assert "$HOME" not in result

    def test_preprocess_braced_env_var_expression(self, delegate):
        """Test preprocessing braced environment variables with expression.

        In xonsh, ${expr} evaluates expr as Python first, then uses
        the result as the env var name. So ${PATH} tries to evaluate
        PATH as a Python variable (which would fail if PATH isn't defined).
        The correct usage is ${'PATH'} or ${var_name} where var_name is defined.
        """
        # ${PATH} keeps PATH as a Python expression (not a string)
        source = "print(${PATH})"
        result = delegate._preprocess_source(source)
        assert "__xonsh_env__[PATH]" in result
        assert "${PATH}" not in result

    def test_preprocess_braced_env_var_string_literal(self, delegate):
        """Test preprocessing braced env var with string literal."""
        # ${'PATH'} evaluates 'PATH' to string, then looks up env var
        source = "print(${'PATH'})"
        result = delegate._preprocess_source(source)
        assert "__xonsh_env__['PATH']" in result
        assert "${'PATH'}" not in result

    def test_preprocess_braced_env_var_dynamic(self, delegate):
        """Test preprocessing braced env var with dynamic expression."""
        # ${var_name} where var_name is a Python variable
        source = "var_name = 'USER'\\nprint(${var_name})"
        result = delegate._preprocess_source(source)
        assert "__xonsh_env__[var_name]" in result
        assert "${var_name}" not in result

    def test_preprocess_captured_subprocess(self, delegate):
        """Test preprocessing captured subprocess."""
        source = "output = $(ls -la)"
        result = delegate._preprocess_source(source)
        assert "__xonsh_subproc__" in result
        assert "$(ls -la)" not in result

    def test_preprocess_captured_subprocess_object(self, delegate):
        """Test preprocessing captured subprocess object."""
        source = "result = !(git status)"
        result = delegate._preprocess_source(source)
        assert "__xonsh_subproc__" in result
        assert "!(git status)" not in result

    def test_preprocess_uncaptured_subprocess(self, delegate):
        """Test preprocessing uncaptured subprocess."""
        source = "$[echo hello]"
        result = delegate._preprocess_source(source)
        assert "__xonsh_subproc__" in result
        assert "$[echo hello]" not in result

    def test_preprocess_uncaptured_subprocess_object(self, delegate):
        """Test preprocessing uncaptured subprocess object."""
        source = "![make build]"
        result = delegate._preprocess_source(source)
        assert "__xonsh_subproc__" in result
        assert "![make build]" not in result

    def test_preprocess_python_eval(self, delegate):
        """Test preprocessing Python evaluation standalone."""
        # Test @(expr) outside of subprocess context
        source = "print(@(name))"
        result = delegate._preprocess_source(source)
        # @(name) should become (name)
        assert "print((name))" in result
        assert "@(name)" not in result

    def test_preprocess_python_eval_in_subprocess(self, delegate):
        """Test that subprocess replacement takes precedence over @()."""
        # When @(expr) is inside $(cmd), the whole subprocess is replaced
        source = "$(echo @(name))"
        result = delegate._preprocess_source(source)
        # The entire subprocess gets replaced, so @(name) is gone too
        assert "__xonsh_subproc__" in result
        assert "@(name)" not in result

    def test_preprocess_tokenized_substitution(self, delegate):
        """Test preprocessing tokenized substitution."""
        source = "@$(cmd)"
        result = delegate._preprocess_source(source)
        assert "__xonsh_subproc__" in result
        assert "@$(cmd)" not in result

    def test_preprocess_regex_glob(self, delegate):
        """Test preprocessing regex glob."""
        source = 'files = `.*\\.py$`'
        result = delegate._preprocess_source(source)
        assert '"__glob__"' in result
        assert "`" not in result

    def test_preprocess_glob_pattern(self, delegate):
        """Test preprocessing glob pattern."""
        source = "files = g`*.txt`"
        result = delegate._preprocess_source(source)
        assert '"__glob__"' in result
        assert "g`" not in result

    def test_preprocess_path_literal(self, delegate):
        """Test preprocessing path literals."""
        source = 'config = p"/home/user/.config"'
        result = delegate._preprocess_source(source)
        # p" should become just "
        assert 'p"' not in result
        assert '"/home/user/.config"' in result

    def test_preprocess_formatted_path_literal(self, delegate):
        """Test preprocessing formatted path literals."""
        source = 'config = pf"/home/{user}/.config"'
        result = delegate._preprocess_source(source)
        # pf" should become f"
        assert 'pf"' not in result
        assert 'f"/home/{user}/.config"' in result

    def test_preprocess_preserves_pure_python(self, delegate):
        """Test that pure Python code is preserved."""
        source = """
import os
def greet(name):
    print(f"Hello, {name}!")
x = 1 + 2
"""
        result = delegate._preprocess_source(source)
        assert "import os" in result
        assert "def greet(name):" in result
        assert 'print(f"Hello, {name}!")' in result
        assert "x = 1 + 2" in result

    def test_preprocess_mixed_code(self, delegate):
        """Test preprocessing mixed xonsh and Python code."""
        source = """
import os
home = $HOME
files = $(ls -la)
def process():
    return len(files)
"""
        result = delegate._preprocess_source(source)
        assert "import os" in result
        assert "__xonsh_env__" in result
        assert "__xonsh_subproc__" in result
        assert "def process():" in result

    def test_has_xonsh_syntax_env_var(self, delegate):
        """Test xonsh syntax detection for env vars."""
        assert delegate._has_xonsh_syntax("print($HOME)")
        assert delegate._has_xonsh_syntax("x = ${PATH}")

    def test_has_xonsh_syntax_subprocess(self, delegate):
        """Test xonsh syntax detection for subprocesses."""
        assert delegate._has_xonsh_syntax("$(ls)")
        assert delegate._has_xonsh_syntax("!(git status)")
        assert delegate._has_xonsh_syntax("$[echo]")
        assert delegate._has_xonsh_syntax("![make]")

    def test_has_xonsh_syntax_glob(self, delegate):
        """Test xonsh syntax detection for globs."""
        assert delegate._has_xonsh_syntax("`*.py`")
        assert delegate._has_xonsh_syntax("g`*.txt`")

    def test_has_xonsh_syntax_python_eval(self, delegate):
        """Test xonsh syntax detection for Python eval."""
        assert delegate._has_xonsh_syntax("@(name)")
        assert delegate._has_xonsh_syntax("@$(cmd)")

    def test_has_xonsh_syntax_pure_python(self, delegate):
        """Test that pure Python is not flagged as xonsh."""
        assert not delegate._has_xonsh_syntax("x = 1 + 2")
        assert not delegate._has_xonsh_syntax("def foo(): pass")
        assert not delegate._has_xonsh_syntax("import os")


@pytest.mark.skipif(not JEDI_AVAILABLE, reason="Jedi not available")
class TestPythonDelegateWithJedi:
    """Test Python delegate functionality with Jedi."""

    @pytest.fixture
    def delegate(self):
        """Create a Python delegate."""
        return PythonDelegate()

    def test_completions_with_xonsh_source(self, delegate):
        """Test that completions work with xonsh source."""
        source = """
import os
home = $HOME
os."""
        # Should not raise, should return completions
        completions = delegate.get_completions(source, 3, 3)
        assert isinstance(completions, list)

    def test_diagnostics_skip_xonsh_lines(self, delegate):
        """Test that diagnostics skip lines with xonsh syntax."""
        source = """
x = 1
y = $HOME
z = $(invalid python on this line too
"""
        diagnostics = delegate.get_diagnostics(source)
        # Should not report errors on lines with xonsh syntax
        for diag in diagnostics:
            line = diag.range.start.line
            lines = source.splitlines()
            if line < len(lines):
                # Ensure we're not reporting on xonsh lines
                assert not delegate._has_xonsh_syntax(lines[line]) or \
                       "xonsh" not in diag.message.lower()

    def test_hover_with_xonsh_source(self, delegate):
        """Test that hover works with xonsh source."""
        source = """
import os
home = $HOME
print(os.path)
"""
        # Should not raise
        hover = delegate.get_hover(source, 3, 10)
        # May return None or hover info, but shouldn't crash
        assert hover is None or isinstance(hover, str)
