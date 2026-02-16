"""
Xonsh Language Server

A Language Server Protocol implementation for xonsh,
providing intelligent code completion, diagnostics, and more.
"""

__version__ = "0.2.0"
__author__ = "Mohammed Elwardi Fadeli"

# Import on demand to avoid import errors
def get_server():
    from xonsh_lsp.server import XonshLanguageServer
    return XonshLanguageServer

__all__ = ["get_server", "__version__"]
