"""
Xonsh Language Server

A Language Server Protocol implementation for xonsh,
providing intelligent code completion, diagnostics, and more.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("xonsh-lsp")
except PackageNotFoundError:  # not installed (e.g. running from source tree)
    __version__ = "0.0.0+local"

__author__ = "Mohammed Elwardi Fadeli"

# Import on demand to avoid import errors
def get_server():
    from xonsh_lsp.server import XonshLanguageServer
    return XonshLanguageServer

__all__ = ["get_server", "__version__"]
