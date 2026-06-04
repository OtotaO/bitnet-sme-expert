"""Middleware components.

Authorization is no longer middleware — it's per-route dependencies in
``app/auth.py`` (``Depends(require_role(...))``), which avoids the path-prefix
matching bypasses.
"""

from .cors_middleware import setup_cors
from .error_middleware import setup_error_handling
from .logging_middleware import LoggingMiddleware

__all__ = [
    "LoggingMiddleware",
    "setup_cors",
    "setup_error_handling",
]
