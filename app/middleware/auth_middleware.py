"""Authentication and authorization middleware for sensitive endpoints.

Uses PyJWT (the maintained successor to ``python-jose``, which has unresolved
CVEs and is effectively abandoned).
"""

from __future__ import annotations

from typing import Iterable

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..config import settings


class AuthzMiddleware(BaseHTTPMiddleware):
    """Enforce role-based access controls on sensitive routes."""

    PROTECTED_ROUTES: dict[str, frozenset[str]] = {
        "/cache/clear": frozenset({"admin"}),
        "/api/v1/train": frozenset({"admin"}),
        "/api/v1/fine-tune": frozenset({"admin"}),
        "/api/v1/training/status/": frozenset({"admin", "operator"}),
        "/api/v1/training/jobs": frozenset({"admin", "operator"}),
    }

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        required = self._required_roles(request.url.path)
        if not required:
            return await call_next(request)

        # In dev with auth disabled, bypass; in production it must be enabled.
        if not (settings.ENABLE_AUTHENTICATION or settings.is_production):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})
        token = auth_header.removeprefix("Bearer ").strip()
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})

        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
        except jwt.PyJWTError:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid authentication token"}
            )

        if not self._has_required_role(payload, required):
            return JSONResponse(
                status_code=403, content={"detail": "Insufficient role for this endpoint"}
            )

        return await call_next(request)

    def _required_roles(self, path: str) -> frozenset[str]:
        for route, roles in self.PROTECTED_ROUTES.items():
            if route.endswith("/") and path.startswith(route):
                return roles
            if path == route:
                return roles
        return frozenset()

    @staticmethod
    def _has_required_role(payload: dict, required: Iterable[str]) -> bool:
        token_roles: set[str] = set()
        role = payload.get("role")
        if isinstance(role, str) and role:
            token_roles.add(role)
        roles = payload.get("roles", [])
        if isinstance(roles, list):
            token_roles.update(r for r in roles if isinstance(r, str) and r)
        return bool(token_roles.intersection(required))
