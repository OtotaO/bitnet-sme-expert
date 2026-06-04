"""Shared SlowAPI limiter.

Defined in its own module (not in ``main.py``) so routers can apply
``@limiter.limit(...)`` to their endpoints without importing ``main`` — which
would be circular, since ``main`` imports the routers. ``main`` attaches this
instance to ``app.state.limiter`` and registers the 429 handler.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import settings

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])
