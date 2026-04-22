"""Authentication and authorization middleware for sensitive endpoints."""

from typing import Dict, Set

from fastapi import Request
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import settings


class AuthzMiddleware(BaseHTTPMiddleware):
    """Enforce role-based access controls on sensitive routes."""

    protected_routes: Dict[str, Set[str]] = {
        "/cache/clear": {"admin"},
        "/api/v1/train": {"admin"},
        "/api/v1/fine-tune": {"admin"},
        "/api/v1/training/status/": {"admin", "operator"},
        "/api/v1/training/jobs": {"admin", "operator"},
    }

    async def dispatch(self, request: Request, call_next):
        required_roles = self._get_required_roles(request.url.path)
        if not required_roles:
            return await call_next(request)

        if not (settings.ENABLE_AUTHENTICATION or settings.is_production):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})

        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
        except JWTError:
            return JSONResponse(status_code=401, content={"detail": "Invalid authentication token"})

        role = payload.get("role")
        roles = payload.get("roles", [])
        token_roles = {role} if isinstance(role, str) and role else set()
        if isinstance(roles, list):
            token_roles.update({r for r in roles if isinstance(r, str) and r})

        if not token_roles.intersection(required_roles):
            return JSONResponse(
                status_code=403,
                content={"detail": "Insufficient role for this endpoint"},
            )

        return await call_next(request)

    def _get_required_roles(self, path: str) -> Set[str]:
        """Get required roles for an endpoint path."""
        for route, roles in self.protected_routes.items():
            if route.endswith("/") and path.startswith(route):
                return roles
            if path == route:
                return roles
        return set()

