import os

import pytest
from lsprotocol import types as lsp

from xonsh_lsp.inlay_hints import InlayHintConfig, XonshInlayHintProvider
from xonsh_lsp.parser import XonshParser


@pytest.fixture
def parser():
    p = XonshParser()
    if not p._initialized:
        pytest.skip("tree-sitter-xonsh not available")
    return p


@pytest.fixture
def full_range():
    return lsp.Range(
        start=lsp.Position(line=0, character=0),
        end=lsp.Position(line=1000, character=0),
    )


def _hint_at(hints, line, char):
    return [h for h in hints if h.position.line == line and h.position.character == char]


def test_captured_subprocess_emits_str(parser, full_range):
    pr = parser.parse("x = $(date)\n")
    hints = XonshInlayHintProvider().get_hints(pr, full_range)
    matches = _hint_at(hints, 0, 11)  # end of $(date)
    assert any(h.label == ": str" for h in matches)


def test_captured_subprocess_object_emits_commandpipeline(parser, full_range):
    pr = parser.parse("x = !(git status)\n")
    hints = XonshInlayHintProvider().get_hints(pr, full_range)
    matches = _hint_at(hints, 0, 17)
    assert any(h.label == ": CommandPipeline" for h in matches)


def test_path_string_emits_path(parser, full_range):
    pr = parser.parse('p = p"/tmp"\n')
    hints = XonshInlayHintProvider().get_hints(pr, full_range)
    labels = [h.label for h in hints]
    assert ": Path" in labels


def test_env_var_known_emits_registry_type(parser, full_range):
    # AUTO_CD is registered as bool in xonsh_builtins.XONSH_MAGIC_VARS.
    pr = parser.parse("x = $AUTO_CD\n")
    hints = XonshInlayHintProvider().get_hints(pr, full_range)
    matches = _hint_at(hints, 0, 12)
    assert any(h.label == ": bool" for h in matches)


def test_env_var_unknown_defaults_to_str(parser, full_range):
    pr = parser.parse("x = $MY_RANDOM_VAR\n")
    hints = XonshInlayHintProvider().get_hints(pr, full_range)
    assert any(h.label == ": str" for h in hints)


def test_nested_in_subprocess_is_suppressed(parser, full_range):
    # echo $HOME — $HOME lives inside subprocess_argument; suppressed.
    pr = parser.parse("echo $HOME\n")
    hints = XonshInlayHintProvider().get_hints(pr, full_range)
    assert hints == []


def test_xonsh_types_disabled_suppresses_type_hints(parser, full_range):
    pr = parser.parse("x = $(date)\n")
    cfg = InlayHintConfig(xonsh_types=False, env_var_values=False)
    hints = XonshInlayHintProvider(cfg).get_hints(pr, full_range)
    assert hints == []


def test_env_var_values_opt_in(parser, full_range, monkeypatch):
    monkeypatch.setenv("XONSHLSP_TEST_VAR", "hello")
    pr = parser.parse("x = $XONSHLSP_TEST_VAR\n")
    cfg = InlayHintConfig(xonsh_types=False, env_var_values=True)
    hints = XonshInlayHintProvider(cfg).get_hints(pr, full_range)
    assert any("hello" in h.label for h in hints)


def test_range_filtering(parser):
    src = "x = $(a)\ny = $(b)\nz = $(c)\n"
    pr = parser.parse(src)
    rng = lsp.Range(
        start=lsp.Position(line=1, character=0),
        end=lsp.Position(line=1, character=100),
    )
    hints = XonshInlayHintProvider().get_hints(pr, rng)
    assert all(h.position.line == 1 for h in hints)


def test_config_from_options_defaults():
    cfg = InlayHintConfig.from_options(None)
    assert cfg.xonsh_types is True
    assert cfg.env_var_values is False
    assert cfg.alias_resolution is False


def test_config_from_options_override():
    cfg = InlayHintConfig.from_options({
        "inlayHints": {"xonshTypes": False, "envVarValues": True}
    })
    assert cfg.xonsh_types is False
    assert cfg.env_var_values is True
