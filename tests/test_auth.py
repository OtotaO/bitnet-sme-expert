"""Tests for dependency-based auth (app/auth.py) — no network, no LM."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import auth
from app.config import settings


def _request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw})


def _token(claims: dict, *, exp_in: int | None = 60) -> str:
    payload = dict(claims)
    if exp_in is not None:
        payload["exp"] = int(time.time()) + exp_in
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def test_claim_roles_extraction() -> None:
    assert auth.claim_roles({"role": "admin"}) == {"admin"}
    assert auth.claim_roles({"roles": ["a", "b", ""]}) == {"a", "b"}
    assert auth.claim_roles({"role": "x", "roles": ["y"]}) == {"x", "y"}
    assert auth.claim_roles({}) == set()


def test_decode_requires_exp() -> None:
    # A token with no expiry must be rejected (options={"require": ["exp"]}).
    no_exp = _token({"role": "admin"}, exp_in=None)
    with pytest.raises(jwt.exceptions.MissingRequiredClaimError):
        auth._decode(no_exp)


def test_decode_pins_algorithm() -> None:
    good = _token({"role": "admin"})
    assert auth._decode(good)["role"] == "admin"


async def test_dev_bypass_returns_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_AUTHENTICATION", False)
    monkeypatch.setattr(type(settings), "is_production", property(lambda self: False))
    claims = await auth.get_current_claims(_request())
    assert "admin" in auth.claim_roles(claims)


async def test_enforced_missing_token_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_AUTHENTICATION", True)
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_claims(_request())
    assert exc.value.status_code == 401


async def test_enforced_valid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_AUTHENTICATION", True)
    req = _request({"Authorization": f"Bearer {_token({'role': 'operator'})}"})
    claims = await auth.get_current_claims(req)
    assert claims["role"] == "operator"


async def test_require_role_rejects_wrong_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_AUTHENTICATION", True)
    checker = auth.require_role("admin")
    with pytest.raises(HTTPException) as exc:
        await checker(claims={"role": "operator"})
    assert exc.value.status_code == 403


async def test_require_role_allows_right_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_AUTHENTICATION", True)
    checker = auth.require_role("admin", "operator")
    claims = await checker(claims={"roles": ["operator"]})
    assert claims["roles"] == ["operator"]
