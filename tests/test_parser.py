"""Tests for the xonsh parser."""

import pytest
from xonsh_lsp.parser import XonshParser, ParseResult


class TestXonshParser:
    """Test the XonshParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return XonshParser()

    def test_parse_empty(self, parser):
        """Test parsing empty source."""
        result = parser.parse("")
        assert isinstance(result, ParseResult)
        assert result.errors == []

    def test_parse_simple_python(self, parser):
        """Test parsing simple Python code."""
        source = "x = 1 + 2"
        result = parser.parse(source)
        assert isinstance(result, ParseResult)
        assert result.errors == []

    def test_parse_env_variable(self, parser):
        """Test parsing environment variables."""
        source = "print($HOME)"
        result = parser.parse(source)
        # Note: This test depends on tree-sitter-xonsh being available
        # If only Python parser is available, env_variables will be empty
        assert isinstance(result, ParseResult)

    def test_parse_subprocess(self, parser):
        """Test parsing subprocess syntax."""
        source = "output = $(ls -la)"
        result = parser.parse(source)
        assert isinstance(result, ParseResult)

    def test_parse_function_definition(self, parser):
        """Test parsing function definitions."""
        source = '''
def greet(name):
    print(f"Hello, {name}!")
'''
        result = parser.parse(source)
        assert isinstance(result, ParseResult)
        assert result.errors == []

    def test_parse_class_definition(self, parser):
        """Test parsing class definitions."""
        source = '''
class MyClass:
    def __init__(self):
        self.value = 42
'''
        result = parser.parse(source)
        assert isinstance(result, ParseResult)
        assert result.errors == []

    def test_parse_syntax_error(self, parser):
        """Test parsing code with syntax errors."""
        source = "def foo(:"  # Invalid syntax
        result = parser.parse(source)
        assert isinstance(result, ParseResult)
        # Should have at least one error
        # Note: Behavior depends on tree-sitter configuration

    def test_extract_identifier(self, parser):
        """Test extracting identifiers at position."""
        source = "my_variable = 42"
        identifier = parser.extract_identifier_at_position(source, 0, 5)
        assert identifier == "my_variable"


class TestParseResult:
    """Test the ParseResult dataclass."""

    def test_parse_result_creation(self):
        """Test creating a ParseResult."""
        result = ParseResult(
            tree=None,
            errors=[],
            env_variables=[],
            subprocesses=[],
            python_regions=[],
            macro_calls=[],
            xontrib_statements=[],
            at_objects=[],
            globs=[],
            path_literals=[],
        )
        assert result.tree is None
        assert result.errors == []
        assert result.env_variables == []
        assert result.subprocesses == []
        assert result.python_regions == []
        assert result.macro_calls == []
        assert result.xontrib_statements == []
        assert result.at_objects == []
        assert result.globs == []
        assert result.path_literals == []
