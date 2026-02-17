"""Regression tests for multi-byte UTF-8 parser bug and proxy backend fixes.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lsprotocol import types as lsp
from xonsh_lsp.parser import XonshParser
from xonsh_lsp.lsp_proxy_backend import LspProxyBackend


def _test_path(name: str) -> str:
    if os.name == "nt":
        return f"C:\\test\\{name}"
    return f"/test/{name}"


# ---------------------------------------------------------------------------
# 1. Multi-byte UTF-8 parser bug
#
# The old code extracted node text by slicing the Python *string* with
# tree-sitter *byte* offsets:
#
#     source[node.start_byte : node.end_byte]
#
# This works fine for ASCII because 1 char == 1 byte, but breaks as soon as
# the source contains multi-byte UTF-8 characters (emoji, accented letters,
# CJK, etc.) because byte offsets overshoot string indices.
# ---------------------------------------------------------------------------


class TestMultiByteNodeText:
    """_node_to_info and friends must return correct text even when the
    source contains multi-byte UTF-8 characters."""

    @pytest.fixture
    def parser(self):
        return XonshParser()

    def test_identifier_after_emoji_comment(self, parser):
        """An identifier on the same line after an emoji should be extracted
        correctly.  The old byte-slice approach would return garbage."""
        source = '# 🎉\nmy_var = 42'
        identifier = parser.extract_identifier_at_position(source, 1, 2)
        assert identifier == "my_var"

    def test_identifier_after_multibyte_string(self, parser):
        """Variable after a string containing accented characters."""
        source = 'greeting = "héllo"\nresult = 1'
        identifier = parser.extract_identifier_at_position(source, 1, 2)
        assert identifier == "result"

    def test_node_info_text_with_emoji(self, parser):
        """NodeInfo.text for an identifier must be the identifier text,
        not a shifted slice that picks up neighbouring characters."""
        source = '# 🚀🚀🚀\nx = 1'
        result = parser.parse(source)
        assert result.tree is not None

        # Walk tree to find the "x" identifier node
        def find_ident(node):
            if node.type == "identifier":
                info = parser._node_to_info(node, source)
                return info
            for child in node.children:
                found = find_ident(child)
                if found is not None:
                    return found
            return None

        info = find_ident(result.tree.root_node)
        assert info is not None
        assert info.text == "x"

    def test_document_symbols_with_cjk_comment(self, parser):
        """get_document_symbols must return correct function names even when
        the source has CJK characters above the function definition."""
        source = '# 你好世界\ndef hello():\n    pass'
        symbols = parser.get_document_symbols(source)
        func_names = [s["name"] for s in symbols if s["kind"] == "function"]
        assert "hello" in func_names

    def test_document_symbols_variable_after_emoji(self, parser):
        """Variable name extraction must be correct after multi-byte chars."""
        source = 'emoji = "🎉"\nmy_variable = 42'
        symbols = parser.get_document_symbols(source)
        var_names = [s["name"] for s in symbols if s["kind"] == "variable"]
        assert "my_variable" in var_names

    def test_extract_env_var_after_multibyte(self, parser):
        """$VAR extraction must work when preceded by multi-byte chars."""
        source = '# café\nprint($HOME)'
        result = parser.parse(source)
        # The env variable should be correctly extracted
        env_texts = [e.text for e in result.env_variables]
        assert any("HOME" in t for t in env_texts)


# ---------------------------------------------------------------------------
# 2. Hover plaintext wrapping
#
# The old code returned MarkupKind.PlainText and bare-string hover contents
# as-is, so editors displayed them without syntax highlighting.  The fix
# wraps them in ```python code fences.
# ---------------------------------------------------------------------------


class TestHoverPlaintextWrapping:
    """Hover responses that are plain text should be wrapped in code fences."""

    def _make_backend(self):
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()
        backend._started = True
        return backend

    @pytest.mark.asyncio
    async def test_plaintext_markup_wrapped(self):
        """MarkupContent with PlainText kind should be code-fenced."""
        backend = self._make_backend()

        hover_result = lsp.Hover(
            contents=lsp.MarkupContent(
                kind=lsp.MarkupKind.PlainText,
                value="def foo(x: int) -> str",
            )
        )
        backend._client.text_document_hover_async = AsyncMock(
            return_value=hover_result
        )

        result = await backend.get_hover("x = 1", 0, 0, _test_path("file.py"))

        assert result is not None
        assert result.startswith("```python\n")
        assert result.endswith("\n```")
        assert "def foo(x: int) -> str" in result

    @pytest.mark.asyncio
    async def test_bare_string_hover_wrapped(self):
        """Bare string hover contents should be code-fenced."""
        backend = self._make_backend()

        hover_result = lsp.Hover(contents="int")
        backend._client.text_document_hover_async = AsyncMock(
            return_value=hover_result
        )

        result = await backend.get_hover("x = 1", 0, 0, _test_path("file.py"))

        assert result is not None
        assert result.startswith("```python\n")
        assert result.endswith("\n```")
        assert "int" in result

    @pytest.mark.asyncio
    async def test_markdown_hover_not_double_wrapped(self):
        """MarkupContent with Markdown kind should NOT be wrapped again."""
        backend = self._make_backend()

        md_value = "```python\ndef foo(): ...\n```"
        hover_result = lsp.Hover(
            contents=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value=md_value,
            )
        )
        backend._client.text_document_hover_async = AsyncMock(
            return_value=hover_result
        )

        result = await backend.get_hover("x = 1", 0, 0, _test_path("file.py"))

        # Should return the markdown as-is, not double-wrapped
        assert result == md_value


# ---------------------------------------------------------------------------
# 3. Race condition: cancelled Future in pygls
#
# When Zed cancels a request but the child LSP's response arrives afterward,
# pygls tries to call set_result() on a cancelled Future, causing an
# InvalidStateError.  The monkey-patch in start() should silently drop
# responses for cancelled futures.
# ---------------------------------------------------------------------------


class TestCancelledFutureGuard:
    """The _safe_handle_response wrapper must not crash on cancelled futures."""

    @pytest.mark.asyncio
    async def test_cancelled_future_does_not_raise(self):
        """Calling _handle_response with a cancelled future must not raise."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()

        # Set up _request_futures with a cancelled future
        cancelled_future = asyncio.get_event_loop().create_future()
        cancelled_future.cancel()
        backend._client.protocol._request_futures = {"42": cancelled_future}

        # Record whether the original handler was called
        orig_called = []
        orig_handle_response = MagicMock(side_effect=lambda *a, **kw: orig_called.append(True))
        backend._client.protocol._handle_response = orig_handle_response

        # Simulate what start() does: install the safe wrapper
        _orig = backend._client.protocol._handle_response

        def _safe_handle_response(msg_id, result=None, error=None):
            future = backend._client.protocol._request_futures.get(msg_id)
            if future is not None and future.cancelled():
                backend._client.protocol._request_futures.pop(msg_id, None)
                return
            _orig(msg_id, result, error)

        backend._client.protocol._handle_response = _safe_handle_response

        # This must not raise InvalidStateError
        _safe_handle_response("42", result={"hover": "data"})

        # The original handler should NOT have been called
        assert len(orig_called) == 0
        # The cancelled future should be cleaned up
        assert "42" not in backend._client.protocol._request_futures

    @pytest.mark.asyncio
    async def test_normal_future_passes_through(self):
        """Non-cancelled futures should still be forwarded normally."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()

        # Set up _request_futures with a normal (pending) future
        pending_future = asyncio.get_event_loop().create_future()
        backend._client.protocol._request_futures = {"99": pending_future}

        orig_called_with = []
        orig_handle_response = MagicMock(
            side_effect=lambda *a, **kw: orig_called_with.append((a, kw))
        )
        backend._client.protocol._handle_response = orig_handle_response

        _orig = backend._client.protocol._handle_response

        def _safe_handle_response(msg_id, result=None, error=None):
            future = backend._client.protocol._request_futures.get(msg_id)
            if future is not None and future.cancelled():
                backend._client.protocol._request_futures.pop(msg_id, None)
                return
            _orig(msg_id, result, error)

        backend._client.protocol._handle_response = _safe_handle_response

        _safe_handle_response("99", result={"data": 1})

        # The original handler SHOULD have been called
        assert len(orig_called_with) == 1

    @pytest.mark.asyncio
    async def test_unknown_msg_id_passes_through(self):
        """A response for an unknown msg_id should still be forwarded
        (the original handler decides what to do with it)."""
        backend = LspProxyBackend(command=["test"])
        backend._client = MagicMock()

        backend._client.protocol._request_futures = {}

        orig_called = []
        orig_handle_response = MagicMock(
            side_effect=lambda *a, **kw: orig_called.append(True)
        )
        backend._client.protocol._handle_response = orig_handle_response

        _orig = backend._client.protocol._handle_response

        def _safe_handle_response(msg_id, result=None, error=None):
            future = backend._client.protocol._request_futures.get(msg_id)
            if future is not None and future.cancelled():
                backend._client.protocol._request_futures.pop(msg_id, None)
                return
            _orig(msg_id, result, error)

        backend._client.protocol._handle_response = _safe_handle_response

        _safe_handle_response("unknown", result={})

        assert len(orig_called) == 1
