"""Dependency-based authentication & authorization (PyJWT).

Replaces the old path-string ``AuthzMiddleware``. Authorization now attaches to
resolved FastAPI routes via ``Depends(require_role(...))``, which is immune to
the trailing-slash and route-ordering bypasses that prefix matching suffers
(and which removes the dead ``/api/v1/train`` entry the middleware carried).

JWT validation is hardened: the algorithm is pinned to a single value, ``exp``
is required (a token with no expiry is rejected), and a small leeway absorbs
clock skew.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import jwt
from fastapi import Depends, HTTPException, Request, status

from .config import settings

_BEARER = {"WWW-Authenticate": "Bearer"}


def _auth_enforced() -> bool:
    """Auth is enforced when explicitly enabled, or always in production.

    In development with ``ENABLE_AUTHENTICATION=False`` the guards are bypassed
    so local work doesn't need a token — matching the prior middleware's
    behavior, but the production validator still forbids shipping with auth off.
    """
    return settings.ENABLE_AUTHENTICATION or settings.is_production


def _decode(token: str) -> dict:
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],  # pinned — never read alg from the token
        leeway=30,  # seconds, for clock skew
        options={"require": ["exp"], "verify_signature": True, "verify_exp": True},
    )


async def get_current_claims(request: Request) -> dict:
    """Validate the bearer token and return its claims (or a dev-bypass admin)."""
    if not _auth_enforced():
        return {"sub": "dev-bypass", "role": "admin"}
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers=_BEARER)
    token = header.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers=_BEARER)
    try:
        return _decode(token)
    except jwt.PyJWTError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid authentication token", headers=_BEARER
        ) from None


def claim_roles(claims: dict) -> set[str]:
    """Collect roles from a ``role`` string and/or a ``roles`` list claim."""
    roles: set[str] = set()
    role = claims.get("role")
    if isinstance(role, str) and role:
        roles.add(role)
    listed = claims.get("roles", [])
    if isinstance(listed, list):
        roles.update(r for r in listed if isinstance(r, str) and r)
    return roles


def require_role(*allowed: str) -> Callable[..., Awaitable[dict]]:
    """Dependency factory: require any of ``allowed`` roles on the caller."""

    async def checker(claims: dict = Depends(get_current_claims)) -> dict:
        if not _auth_enforced():
            return claims
        if not claim_roles(claims).intersection(allowed):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient role for this endpoint")
        return claims

    return checker
