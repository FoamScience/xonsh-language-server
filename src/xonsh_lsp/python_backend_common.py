from lsprotocol import types as lsp

from xonsh_lsp.preprocessing import PreprocessResult, map_position_from_processed


def remap_text_edit(
    pp: PreprocessResult,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    new_text: str,
    *,
    is_current: bool,
    preamble_offset: int = 0,
) -> lsp.TextEdit | None:
    """Build a TextEdit in original xonsh coordinates.

    When ``is_current`` is True, the input range is interpreted in
    processed-source coordinates and mapped back to the original xonsh
    source. Edits falling above the preamble or starting on a masked
    (xonsh-only) line are dropped (returns None).

    When ``is_current`` is False (edit targets a different file), the
    coordinates are used as-is.
    """
    if is_current:
        start_line -= preamble_offset
        end_line -= preamble_offset
        if start_line < 0:
            return None
        orig_start_line, orig_start_col = map_position_from_processed(
            pp, start_line, start_col
        )
        orig_end_line, orig_end_col = map_position_from_processed(
            pp, end_line, end_col
        )
        if orig_start_line in pp.masked_lines:
            return None
        start_line, start_col = orig_start_line, orig_start_col
        end_line, end_col = orig_end_line, orig_end_col

    return lsp.TextEdit(
        range=lsp.Range(
            start=lsp.Position(line=start_line, character=start_col),
            end=lsp.Position(line=end_line, character=end_col),
        ),
        new_text=new_text,
    )
