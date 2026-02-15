"""Tests for folding ranges."""

import pytest
from xonsh_lsp.parser import XonshParser


class TestFoldingRanges:
    """Test folding range extraction."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return XonshParser()

    def test_function_folding(self, parser):
        """Test that multi-line functions produce a fold."""
        source = "def greet(name):\n    print(name)\n    return name\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r["start_line"] == 0 and r["end_line"] == 2 for r in folds)

    def test_single_line_function_no_fold(self, parser):
        """Test that single-line constructs produce no fold."""
        source = "def f(): pass\n"
        ranges = parser.get_folding_ranges(source)
        func_folds = [
            r for r in ranges
            if r["kind"] == "region" and r["start_line"] == 0
        ]
        assert func_folds == []

    def test_class_folding(self, parser):
        """Test that multi-line classes produce a fold."""
        source = "class MyClass:\n    x = 1\n    y = 2\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r["start_line"] == 0 for r in folds)

    def test_nested_functions(self, parser):
        """Test that nested functions both produce folds."""
        source = "def outer():\n    def inner():\n        pass\n    return inner\n"
        ranges = parser.get_folding_ranges(source)
        region_folds = [r for r in ranges if r["kind"] == "region"]
        # Both outer and inner should fold
        starts = {r["start_line"] for r in region_folds}
        assert 0 in starts  # outer
        assert 1 in starts  # inner

    def test_if_statement(self, parser):
        """Test if/elif/else folding."""
        source = (
            "if x:\n"
            "    a = 1\n"
            "elif y:\n"
            "    b = 2\n"
            "else:\n"
            "    c = 3\n"
        )
        ranges = parser.get_folding_ranges(source)
        region_folds = [r for r in ranges if r["kind"] == "region"]
        # if_statement itself spans all lines
        assert any(r["start_line"] == 0 for r in region_folds)

    def test_for_loop(self, parser):
        """Test for loop folding."""
        source = "for i in range(10):\n    print(i)\n    print(i + 1)\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r["start_line"] == 0 for r in folds)

    def test_while_loop(self, parser):
        """Test while loop folding."""
        source = "while True:\n    break\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r["start_line"] == 0 for r in folds)

    def test_with_statement(self, parser):
        """Test with statement folding."""
        source = "with open('f') as fp:\n    data = fp.read()\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r["start_line"] == 0 for r in folds)

    def test_try_except(self, parser):
        """Test try/except/finally folding."""
        source = (
            "try:\n"
            "    x = 1\n"
            "except Exception:\n"
            "    x = 2\n"
            "finally:\n"
            "    x = 3\n"
        )
        ranges = parser.get_folding_ranges(source)
        region_folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r["start_line"] == 0 for r in region_folds)

    def test_multiline_string(self, parser):
        """Test multi-line string folding with comment kind."""
        source = 'x = """\nline1\nline2\n"""\n'
        ranges = parser.get_folding_ranges(source)
        string_folds = [r for r in ranges if r["kind"] == "comment"]
        assert any(r["start_line"] == 0 for r in string_folds)

    def test_multiline_list(self, parser):
        """Test multi-line list folding."""
        source = "items = [\n    1,\n    2,\n    3,\n]\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r for r in folds)

    def test_multiline_dict(self, parser):
        """Test multi-line dict folding."""
        source = "d = {\n    'a': 1,\n    'b': 2,\n}\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r for r in folds)

    def test_multiline_tuple(self, parser):
        """Test multi-line tuple folding."""
        source = "t = (\n    1,\n    2,\n)\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r for r in folds)

    def test_multiline_import(self, parser):
        """Test multi-line import folding with imports kind."""
        source = "from os import (\n    path,\n    getcwd,\n)\n"
        ranges = parser.get_folding_ranges(source)
        import_folds = [r for r in ranges if r["kind"] == "imports"]
        assert len(import_folds) >= 1
        assert import_folds[0]["start_line"] == 0

    def test_decorated_function(self, parser):
        """Test that decorated definitions produce a fold."""
        source = "@decorator\ndef func():\n    pass\n"
        ranges = parser.get_folding_ranges(source)
        region_folds = [r for r in ranges if r["kind"] == "region"]
        # decorated_definition should fold from line 0
        assert any(r["start_line"] == 0 for r in region_folds)

    def test_consecutive_comments(self, parser):
        """Test that consecutive comment lines produce a single fold."""
        source = "# line 1\n# line 2\n# line 3\nx = 1\n"
        ranges = parser.get_folding_ranges(source)
        comment_folds = [r for r in ranges if r["kind"] == "comment"]
        assert any(
            r["start_line"] == 0 and r["end_line"] == 2
            for r in comment_folds
        )

    def test_single_comment_no_fold(self, parser):
        """Test that a single comment line does not produce a fold."""
        source = "# just one comment\nx = 1\n"
        ranges = parser.get_folding_ranges(source)
        comment_folds = [r for r in ranges if r["kind"] == "comment"]
        assert comment_folds == []

    def test_empty_source(self, parser):
        """Test no folds for empty source."""
        ranges = parser.get_folding_ranges("")
        assert ranges == []

    def test_no_foldable_regions(self, parser):
        """Test no folds when source has only single-line statements."""
        source = "x = 1\ny = 2\nz = 3\n"
        ranges = parser.get_folding_ranges(source)
        assert ranges == []

    def test_block_macro_statement(self, parser):
        """Test xonsh block macro folding."""
        source = "with! ctx:\n    body_line1\n    body_line2\n"
        ranges = parser.get_folding_ranges(source)
        region_folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r["start_line"] == 0 for r in region_folds)

    def test_multiline_argument_list(self, parser):
        """Test multi-line function call arguments fold."""
        source = "func(\n    a,\n    b,\n    c,\n)\n"
        ranges = parser.get_folding_ranges(source)
        folds = [r for r in ranges if r["kind"] == "region"]
        assert any(r for r in folds)

    def test_multiple_foldable_regions(self, parser):
        """Test source with multiple foldable regions."""
        source = (
            "def func1():\n"
            "    pass\n"
            "\n"
            "def func2():\n"
            "    pass\n"
        )
        ranges = parser.get_folding_ranges(source)
        region_folds = [r for r in ranges if r["kind"] == "region"]
        starts = {r["start_line"] for r in region_folds}
        assert 0 in starts
        assert 3 in starts
