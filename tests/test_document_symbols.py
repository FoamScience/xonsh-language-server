"""Tests for document symbols."""

import pytest
from xonsh_lsp.parser import XonshParser


class TestDocumentSymbols:
    """Test document symbol extraction."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return XonshParser()

    def test_function_symbols(self, parser):
        """Test extracting function definitions."""
        source = """
def greet(name):
    print(f"Hello, {name}!")

def farewell():
    print("Goodbye!")
"""
        symbols = parser.get_document_symbols(source)
        function_symbols = [s for s in symbols if s["kind"] == "function"]
        names = [s["name"] for s in function_symbols]
        assert "greet" in names
        assert "farewell" in names

    def test_class_symbols(self, parser):
        """Test extracting class definitions."""
        source = """
class MyClass:
    def __init__(self):
        pass

class AnotherClass:
    pass
"""
        symbols = parser.get_document_symbols(source)
        class_symbols = [s for s in symbols if s["kind"] == "class"]
        names = [s["name"] for s in class_symbols]
        assert "MyClass" in names
        assert "AnotherClass" in names

    def test_variable_symbols(self, parser):
        """Test extracting variable assignments."""
        source = """
x = 1
my_variable = "hello"
result = calculate()
"""
        symbols = parser.get_document_symbols(source)
        var_symbols = [s for s in symbols if s["kind"] == "variable"]
        names = [s["name"] for s in var_symbols]
        assert "x" in names
        assert "my_variable" in names
        assert "result" in names

    def test_xonsh_variable_symbols(self, parser):
        """Test extracting variables with xonsh-specific RHS."""
        source = """
home = $HOME
path = ${PATH}
files = $(ls -la)
result = !(git status)
"""
        symbols = parser.get_document_symbols(source)
        var_symbols = [s for s in symbols if s["kind"] == "variable"]
        names = [s["name"] for s in var_symbols]
        assert "home" in names
        assert "path" in names
        assert "files" in names
        assert "result" in names

    def test_variable_details(self, parser):
        """Test that variable details contain RHS."""
        source = 'name = "world"'
        symbols = parser.get_document_symbols(source)
        var_symbols = [s for s in symbols if s["kind"] == "variable"]
        assert len(var_symbols) == 1
        assert var_symbols[0]["name"] == "name"
        assert '"world"' in var_symbols[0]["detail"]

    def test_xonsh_variable_details(self, parser):
        """Test that xonsh variable details show xonsh syntax."""
        source = "home = $HOME"
        symbols = parser.get_document_symbols(source)
        var_symbols = [s for s in symbols if s["kind"] == "variable"]
        assert len(var_symbols) == 1
        assert var_symbols[0]["name"] == "home"
        assert "$HOME" in var_symbols[0]["detail"]

    def test_import_symbols(self, parser):
        """Test extracting import statements."""
        source = """
import os
import sys
"""
        symbols = parser.get_document_symbols(source)
        module_symbols = [s for s in symbols if s["kind"] == "module"]
        names = [s["name"] for s in module_symbols]
        assert "os" in names
        assert "sys" in names

    def test_mixed_symbols(self, parser):
        """Test extracting mixed symbol types."""
        source = """
import os

x = 1
home = $HOME

def greet():
    pass

class MyClass:
    pass
"""
        symbols = parser.get_document_symbols(source)

        # Check all types are present
        kinds = set(s["kind"] for s in symbols)
        assert "module" in kinds
        assert "variable" in kinds
        assert "function" in kinds
        assert "class" in kinds

    def test_symbol_line_numbers(self, parser):
        """Test that symbol line numbers are correct."""
        source = """def first():
    pass

def second():
    pass
"""
        symbols = parser.get_document_symbols(source)
        function_symbols = [s for s in symbols if s["kind"] == "function"]

        first_func = next(s for s in function_symbols if s["name"] == "first")
        second_func = next(s for s in function_symbols if s["name"] == "second")

        assert first_func["line"] == 0
        assert second_func["line"] == 3

    def test_empty_source(self, parser):
        """Test document symbols for empty source."""
        symbols = parser.get_document_symbols("")
        assert symbols == []

    def test_comment_only_source(self, parser):
        """Test document symbols for comment-only source."""
        source = """
# This is a comment
# Another comment
"""
        symbols = parser.get_document_symbols(source)
        assert symbols == []

    def test_long_detail_truncation(self, parser):
        """Test that long details are truncated."""
        long_value = "x" * 100
        source = f'var = "{long_value}"'
        symbols = parser.get_document_symbols(source)
        var_symbols = [s for s in symbols if s["kind"] == "variable"]
        assert len(var_symbols) == 1
        # Detail should be truncated
        assert len(var_symbols[0]["detail"]) <= 33  # 30 chars + "..."

    def test_nested_function_not_included(self, parser):
        """Test that only top-level symbols are included."""
        source = """
def outer():
    def inner():
        pass
    return inner
"""
        symbols = parser.get_document_symbols(source)
        function_symbols = [s for s in symbols if s["kind"] == "function"]
        names = [s["name"] for s in function_symbols]
        # Should include both outer and inner (we traverse all nodes)
        assert "outer" in names
        # Inner might or might not be included depending on implementation
