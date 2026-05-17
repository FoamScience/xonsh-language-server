from xonsh_lsp.xonsh_types import envvar_type, expression_type, to_python_type


def test_envvar_type_known():
    assert envvar_type("XONSH_DEBUG") == "int"
    assert envvar_type("AUTO_CD") == "bool"
    assert envvar_type("COMPLETIONS_MENU_ROWS") == "int"


def test_envvar_type_unknown():
    assert envvar_type("DEFINITELY_NOT_A_VAR") is None


def test_expression_type_subprocess_forms():
    assert expression_type("captured_subprocess") == "str"
    assert expression_type("captured_subprocess_object") == "CommandPipeline"
    assert expression_type("uncaptured_subprocess") == "None"
    assert expression_type("uncaptured_subprocess_object") == "CommandPipeline"


def test_expression_type_misc():
    assert expression_type("path_string") == "Path"
    assert expression_type("regex_glob") == "list[str]"
    assert expression_type("env_variable_braced") == "str"


def test_expression_type_delegated():
    # python_evaluation: type comes from the inner Python expression, not the node
    assert expression_type("python_evaluation") is None


def test_expression_type_unknown_node():
    assert expression_type("nonexistent_node_type") is None


def test_to_python_type_normalizes_callable():
    # Registry stores "callable" (lowercase) which isn't a valid type expr.
    # Generated stub references typing.Callable via the __xonsh_typing__ alias.
    assert to_python_type("callable") == "__xonsh_typing__.Callable[..., object]"
    assert (
        to_python_type("str | callable")
        == "str | __xonsh_typing__.Callable[..., object]"
    )


def test_to_python_type_widens_bare_collections():
    assert to_python_type("set") == "set[str]"
    assert to_python_type("list") == "list[str]"


def test_to_python_type_preserves_valid():
    assert to_python_type("bool") == "bool"
    assert to_python_type("int | tuple") == "int | tuple"


def test_to_python_type_fallback():
    assert to_python_type(None) == "object"
    assert to_python_type("") == "object"
