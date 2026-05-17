"""
VetOnco — Clerk JWT Authentication
Validates Clerk JWTs using JWKS endpoint.
Uses python-jose for JWKS validation — no Clerk SDK required.
"""
from __future__ import annotations
import os
import time
from typing import Any

import httpx
from fastapi import HTTPException, Header
from jose import jwt, JWTError, jwk
from jose.utils import base64url_decode

CLERK_JWKS_URL = os.getenv(
    "CLERK_JWKS_URL",
    "https://saved-unicorn-80.clerk.accounts.dev/.well-known/jwks.json",
)

# Cache JWKS for 5 minutes
_jwks_cache: dict[str, Any] = {}
_jwks_fetched_at: float = 0.0
JWKS_TTL = 300.0


async def _get_jwks() -> dict:
    global _jwks_cache, _jwks_fetched_at
    now = time.time()
    if _jwks_cache and (now - _jwks_fetched_at) < JWKS_TTL:
        return _jwks_cache
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(CLERK_JWKS_URL)
        r.raise_for_status()
        _jwks_cache = r.json()
        _jwks_fetched_at = now
    return _jwks_cache


async def verify_clerk_token(token: str) -> dict:
    """Verify a Clerk JWT and return the decoded payload."""
    try:
        jwks = await _get_jwks()
        # Decode header to get kid
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")

        # Find matching key
        key_data = None
        for k in jwks.get("keys", []):
            if k.get("kid") == kid:
                key_data = k
                break

        if key_data is None:
            raise HTTPException(status_code=401, detail="JWT key not found in JWKS")

        # Construct public key and verify
        public_key = jwk.construct(key_data)
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},  # Clerk doesn't always set aud
        )
        return payload

    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth error: {str(e)}")


async def get_current_user_id(authorization: str = Header(default="")) -> str:
    """
    FastAPI dependency — extracts and validates Clerk JWT from Authorization header.
    Returns the Clerk user_id (sub claim).
    
    In development (no CLERK_JWKS_URL set), returns 'dev-user' to allow local testing.
    """
    # Allow unauthenticated access in dev mode
    if not os.getenv("CLERK_JWKS_URL") and not authorization:
        return "dev-user"

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected: Bearer <token>",
        )

    token = authorization.removeprefix("Bearer ").strip()
    payload = await verify_clerk_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
    return user_id
