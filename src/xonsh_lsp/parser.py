"""
Tree-sitter integration for xonsh parsing.

This module provides parsing capabilities using tree-sitter-xonsh grammar.
Falls back to basic parsing if xonsh grammar is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

try:
    from tree_sitter import Language, Parser, Node, Tree

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Node = object  # type: ignore
    Tree = object  # type: ignore
    Language = object  # type: ignore
    Parser = object  # type: ignore

try:
    import tree_sitter_xonsh

    XONSH_GRAMMAR_AVAILABLE = True
except ImportError:
    XONSH_GRAMMAR_AVAILABLE = False
    tree_sitter_xonsh = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a parsed node."""

    type: str
    text: str
    start_point: tuple[int, int]  # (row, col)
    end_point: tuple[int, int]  # (row, col)
    start_byte: int
    end_byte: int
    children: list[NodeInfo]


@dataclass
class ParseResult:
    """Result of parsing a document."""

    tree: Tree | None
    errors: list[NodeInfo]
    env_variables: list[NodeInfo]
    subprocesses: list[NodeInfo]
    python_regions: list[tuple[int, int]]  # (start_byte, end_byte)
    macro_calls: list[NodeInfo]
    xontrib_statements: list[NodeInfo]
    at_objects: list[NodeInfo]
    globs: list[NodeInfo]
    path_literals: list[NodeInfo]


class XonshParser:
    """Parser for xonsh files using tree-sitter."""

    # Node types for xonsh-specific constructs
    ENV_VAR_TYPES = {"env_variable", "env_variable_braced"}
    SUBPROCESS_TYPES = {
        "captured_subprocess",
        "captured_subprocess_object",
        "uncaptured_subprocess",
        "uncaptured_subprocess_object",
        "background_command",
        "bare_subprocess",  # Scanner-detected bare subprocess commands
    }
    PYTHON_EVAL_TYPES = {"python_evaluation", "tokenized_substitution"}
    GLOB_TYPES = {"regex_glob", "glob_pattern", "formatted_glob"}
    PATH_TYPES = {"path_string"}
    XONSH_STATEMENT_TYPES = {
        "env_assignment",
        "env_deletion",
        "xonsh_statement",
        "xontrib_statement",
    }
    # @ object access: @.env, @.imp, @.lastcmd, etc.
    AT_OBJECT_TYPES = {"at_object"}
    # Macro calls: func!(args)
    MACRO_TYPES = {"macro_call"}
    # Help expressions: expr? and expr??
    HELP_TYPES = {"help_expression", "super_help_expression"}
    # All xonsh expression types combined
    XONSH_EXPRESSION_TYPES = {"xonsh_expression"}

    def __init__(self):
        self._parser: Parser | None = None
        self._language: Language | None = None
        self._initialized = False

        if TREE_SITTER_AVAILABLE:
            self._init_parser()

    def _init_parser(self) -> None:
        """Initialize the tree-sitter parser."""
        if self._initialized:
            return

        try:
            self._parser = Parser()

            # Load xonsh language from tree-sitter-xonsh package
            if XONSH_GRAMMAR_AVAILABLE and tree_sitter_xonsh is not None:
                self._language = Language(tree_sitter_xonsh.language())
                logger.info("Loaded tree-sitter-xonsh from Python bindings")
            else:
                logger.warning(
                    "tree-sitter-xonsh not available, install with: pip install tree-sitter-xonsh"
                )
                return

            self._parser.language = self._language
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize parser: {e}")
            self._parser = None
            self._language = None

    def parse(self, source: str) -> ParseResult:
        """Parse xonsh source code."""
        if not TREE_SITTER_AVAILABLE or self._parser is None:
            return ParseResult(
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

        try:
            tree = self._parser.parse(source.encode("utf-8"))
            return self._analyze_tree(tree, source)
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return ParseResult(
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

    def _analyze_tree(self, tree: Tree, source: str) -> ParseResult:
        """Analyze the parsed tree and extract xonsh constructs."""
        errors: list[NodeInfo] = []
        env_variables: list[NodeInfo] = []
        subprocesses: list[NodeInfo] = []
        python_regions: list[tuple[int, int]] = []
        macro_calls: list[NodeInfo] = []
        xontrib_statements: list[NodeInfo] = []
        at_objects: list[NodeInfo] = []
        globs: list[NodeInfo] = []
        path_literals: list[NodeInfo] = []

        def visit(node: Node) -> None:
            node_info = self._node_to_info(node, source)

            # Check for errors
            if node.type == "ERROR" or node.is_missing:
                errors.append(node_info)

            # Check for environment variables
            elif node.type in self.ENV_VAR_TYPES:
                env_variables.append(node_info)

            # Check for subprocess constructs
            elif node.type in self.SUBPROCESS_TYPES:
                subprocesses.append(node_info)

            # Check for macro calls: func!(args)
            elif node.type in self.MACRO_TYPES:
                macro_calls.append(node_info)

            # Check for xontrib statements: xontrib load ...
            elif node.type == "xontrib_statement":
                xontrib_statements.append(node_info)

            # Check for @ object access: @.env, @.imp, etc.
            elif node.type in self.AT_OBJECT_TYPES:
                at_objects.append(node_info)

            # Check for glob patterns
            elif node.type in self.GLOB_TYPES:
                globs.append(node_info)

            # Check for path literals: p"...", pf"...", etc.
            elif node.type in self.PATH_TYPES:
                path_literals.append(node_info)

            # Track Python regions (everything that's not subprocess)
            elif node.type in {"expression_statement", "assignment", "function_definition", "class_definition"}:
                # Mark as Python region if not inside subprocess
                if not self._is_inside_subprocess(node):
                    python_regions.append((node.start_byte, node.end_byte))

            # Recurse
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        return ParseResult(
            tree=tree,
            errors=errors,
            env_variables=env_variables,
            subprocesses=subprocesses,
            python_regions=python_regions,
            macro_calls=macro_calls,
            xontrib_statements=xontrib_statements,
            at_objects=at_objects,
            globs=globs,
            path_literals=path_literals,
        )

    def _node_to_info(self, node: Node, source: str) -> NodeInfo:
        """Convert a tree-sitter node to NodeInfo."""
        return NodeInfo(
            type=node.type,
            text=source[node.start_byte : node.end_byte],
            start_point=(node.start_point[0], node.start_point[1]),
            end_point=(node.end_point[0], node.end_point[1]),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            children=[self._node_to_info(c, source) for c in node.children],
        )

    def _is_inside_subprocess(self, node: Node) -> bool:
        """Check if a node is inside a subprocess construct."""
        current = node.parent
        while current is not None:
            if current.type in self.SUBPROCESS_TYPES:
                return True
            current = current.parent
        return False

    def get_node_at_position(
        self, tree: Tree, row: int, col: int
    ) -> Node | None:
        """Get the smallest node at a given position."""
        if tree is None:
            return None

        def find_node(node: Node) -> Node | None:
            # Check if position is within this node
            if not (
                (node.start_point[0], node.start_point[1])
                <= (row, col)
                <= (node.end_point[0], node.end_point[1])
            ):
                return None

            # Check children for more specific match
            for child in node.children:
                result = find_node(child)
                if result is not None:
                    return result

            return node

        return find_node(tree.root_node)

    def get_nodes_by_type(
        self, tree: Tree, node_types: set[str]
    ) -> Iterator[Node]:
        """Get all nodes of specific types."""
        if tree is None:
            return

        def visit(node: Node) -> Iterator[Node]:
            if node.type in node_types:
                yield node
            for child in node.children:
                yield from visit(child)

        yield from visit(tree.root_node)

    def extract_identifier_at_position(
        self, source: str, row: int, col: int
    ) -> str | None:
        """Extract the identifier at a given position."""
        result = self.parse(source)
        if result.tree is None:
            return None

        node = self.get_node_at_position(result.tree, row, col)
        if node is None:
            return None

        # Walk up to find identifier or env variable
        while node is not None:
            if node.type == "identifier":
                return source[node.start_byte : node.end_byte]
            elif node.type in self.ENV_VAR_TYPES:
                # Extract variable name from env variable
                text = source[node.start_byte : node.end_byte]
                if text.startswith("${") and text.endswith("}"):
                    return text[2:-1]
                elif text.startswith("$"):
                    return text[1:]
                return text
            node = node.parent

        return None

    def is_in_subprocess_context(self, tree: Tree, row: int, col: int) -> bool:
        """Check if position is inside a subprocess context."""
        if tree is None:
            return False

        node = self.get_node_at_position(tree, row, col)
        return self._is_inside_subprocess(node) if node else False

    def get_parent_context(self, tree: Tree, row: int, col: int) -> str | None:
        """Get the context type at position (python, subprocess, etc.)."""
        if tree is None:
            return None

        node = self.get_node_at_position(tree, row, col)
        if node is None:
            return None

        # Walk up to find context
        current = node
        while current is not None:
            if current.type in self.SUBPROCESS_TYPES:
                return "subprocess"
            elif current.type in self.PYTHON_EVAL_TYPES:
                return "python_eval"
            elif current.type in self.GLOB_TYPES:
                return "glob"
            elif current.type in self.MACRO_TYPES:
                return "macro"
            elif current.type in self.AT_OBJECT_TYPES:
                return "at_object"
            elif current.type in self.PATH_TYPES:
                return "path_literal"
            elif current.type == "xontrib_statement":
                return "xontrib"
            elif current.type == "function_definition":
                return "function"
            elif current.type == "class_definition":
                return "class"
            current = current.parent

        return "module"

    def get_env_var_name(self, node_info: NodeInfo) -> str | None:
        """Extract the environment variable name from a node."""
        text = node_info.text
        if text.startswith("${") and text.endswith("}"):
            # ${expr} - for now just extract simple identifiers
            inner = text[2:-1].strip()
            if inner.isidentifier() or inner.startswith('"') or inner.startswith("'"):
                return inner.strip("'\"")
            return None
        elif text.startswith("$"):
            return text[1:]
        return None

    def is_xonsh_language(self) -> bool:
        """Check if xonsh language is loaded (vs fallback to Python)."""
        if self._language is None:
            return False
        try:
            return self._language.name == "xonsh"
        except Exception:
            return False

    def get_document_symbols(self, source: str) -> list[dict]:
        """Extract document symbols from source code.

        Returns a list of symbol dicts with keys:
        - name: symbol name
        - kind: symbol kind (variable, function, class, etc.)
        - line: 0-based line number
        - col: 0-based column
        - end_line: 0-based end line
        - end_col: 0-based end column
        - detail: optional detail string
        """
        result = self.parse(source)
        if result.tree is None:
            return []

        symbols = []

        def visit(node: Node) -> None:
            # Function definitions
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append({
                        "name": source[name_node.start_byte:name_node.end_byte],
                        "kind": "function",
                        "line": node.start_point[0],
                        "col": node.start_point[1],
                        "end_line": node.end_point[0],
                        "end_col": node.end_point[1],
                        "detail": "function",
                    })

            # Class definitions
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append({
                        "name": source[name_node.start_byte:name_node.end_byte],
                        "kind": "class",
                        "line": node.start_point[0],
                        "col": node.start_point[1],
                        "end_line": node.end_point[0],
                        "end_col": node.end_point[1],
                        "detail": "class",
                    })

            # Assignments (variable definitions)
            elif node.type == "assignment":
                left = node.child_by_field_name("left")
                if left and left.type == "identifier":
                    name = source[left.start_byte:left.end_byte]
                    # Get right side for detail
                    right = node.child_by_field_name("right")
                    detail = ""
                    if right:
                        right_text = source[right.start_byte:right.end_byte]
                        # Truncate long values
                        if len(right_text) > 30:
                            right_text = right_text[:27] + "..."
                        detail = right_text
                    symbols.append({
                        "name": name,
                        "kind": "variable",
                        "line": left.start_point[0],
                        "col": left.start_point[1],
                        "end_line": left.end_point[0],
                        "end_col": left.end_point[1],
                        "detail": detail,
                    })

            # Import statements
            elif node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        name = source[child.start_byte:child.end_byte]
                        symbols.append({
                            "name": name,
                            "kind": "module",
                            "line": child.start_point[0],
                            "col": child.start_point[1],
                            "end_line": child.end_point[0],
                            "end_col": child.end_point[1],
                            "detail": "import",
                        })

            # Import from statements
            elif node.type == "import_from_statement":
                # Get imported names
                for child in node.children:
                    if child.type == "dotted_name":
                        name = source[child.start_byte:child.end_byte]
                        symbols.append({
                            "name": name,
                            "kind": "module",
                            "line": child.start_point[0],
                            "col": child.start_point[1],
                            "end_line": child.end_point[0],
                            "end_col": child.end_point[1],
                            "detail": "from import",
                        })

            # Environment variable assignments: $VAR = value
            elif node.type == "env_assignment":
                left = node.child_by_field_name("left")
                if left:
                    name = source[left.start_byte:left.end_byte]
                    right = node.child_by_field_name("right")
                    detail = ""
                    if right:
                        right_text = source[right.start_byte:right.end_byte]
                        if len(right_text) > 30:
                            right_text = right_text[:27] + "..."
                        detail = right_text
                    symbols.append({
                        "name": name,
                        "kind": "variable",
                        "line": left.start_point[0],
                        "col": left.start_point[1],
                        "end_line": left.end_point[0],
                        "end_col": left.end_point[1],
                        "detail": f"env: {detail}" if detail else "env",
                    })

            # Xontrib statements: xontrib load name1 name2 ...
            elif node.type == "xontrib_statement":
                for child in node.children:
                    if child.type == "xontrib_name":
                        name = source[child.start_byte:child.end_byte]
                        symbols.append({
                            "name": f"xontrib:{name}",
                            "kind": "module",
                            "line": child.start_point[0],
                            "col": child.start_point[1],
                            "end_line": child.end_point[0],
                            "end_col": child.end_point[1],
                            "detail": "xontrib",
                        })

            # Macro calls: func!(args)
            elif node.type == "macro_call":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte]
                    arg_node = node.child_by_field_name("argument")
                    detail = ""
                    if arg_node:
                        arg_text = source[arg_node.start_byte:arg_node.end_byte]
                        if len(arg_text) > 30:
                            arg_text = arg_text[:27] + "..."
                        detail = f"!({arg_text})"
                    symbols.append({
                        "name": f"{name}!",
                        "kind": "function",
                        "line": node.start_point[0],
                        "col": node.start_point[1],
                        "end_line": node.end_point[0],
                        "end_col": node.end_point[1],
                        "detail": f"macro{detail}",
                    })

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(result.tree.root_node)
        return symbols

    def get_macro_name(self, node_info: NodeInfo) -> str | None:
        """Extract the macro function name from a macro_call node."""
        # macro_call format: name!(argument)
        text = node_info.text
        if "!" in text:
            return text.split("!")[0]
        return None

    def get_xontrib_names(self, node_info: NodeInfo) -> list[str]:
        """Extract xontrib names from a xontrib_statement node."""
        # xontrib_statement format: xontrib load name1 name2 ...
        names = []
        for child in node_info.children:
            if child.type == "xontrib_name":
                names.append(child.text)
        return names

    def get_at_object_attribute(self, node_info: NodeInfo) -> str | None:
        """Extract the attribute name from an at_object node."""
        # at_object format: @.attribute
        text = node_info.text
        if text.startswith("@."):
            return text[2:]
        return None

    def is_in_macro_context(self, tree: Tree, row: int, col: int) -> bool:
        """Check if position is inside a macro call."""
        if tree is None:
            return False

        node = self.get_node_at_position(tree, row, col)
        if node is None:
            return False

        current = node
        while current is not None:
            if current.type in self.MACRO_TYPES:
                return True
            current = current.parent
        return False

    def is_in_xontrib_context(self, tree: Tree, row: int, col: int) -> bool:
        """Check if position is inside a xontrib statement."""
        if tree is None:
            return False

        node = self.get_node_at_position(tree, row, col)
        if node is None:
            return False

        current = node
        while current is not None:
            if current.type == "xontrib_statement":
                return True
            current = current.parent
        return False

    def is_in_path_literal(self, tree: Tree, row: int, col: int) -> bool:
        """Check if position is inside a path literal."""
        if tree is None:
            return False

        node = self.get_node_at_position(tree, row, col)
        if node is None:
            return False

        current = node
        while current is not None:
            if current.type in self.PATH_TYPES:
                return True
            current = current.parent
        return False
