import pytest


class TestJediBackend:

    @pytest.mark.asyncio
    async def test_jedi_rename_produces_expected_edits(self):
        from xonsh_lsp.jedi_backend import JEDI_AVAILABLE, JediBackend

        if not JEDI_AVAILABLE:
            pytest.skip("jedi not installed")

        backend = JediBackend()
        src = "value = 1\nprint(value)\nvalue = value + 1\n"
        edit = await backend.rename(src, 0, 2, "renamed", path=None)

        assert edit is not None
        assert edit.changes is not None
        edits = next(iter(edit.changes.values()))
        ranges = sorted(
            (e.range.start.line, e.range.start.character) for e in edits
        )
        assert ranges == [(0, 0), (1, 6), (2, 0), (2, 8)]
        assert all(e.new_text == "renamed" for e in edits)
