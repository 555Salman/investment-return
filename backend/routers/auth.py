"""
Auth endpoints — simple JWT login for the dashboard.
Credentials are read from DEMO_USERNAME / DEMO_PASSWORD environment variables.
Leave DEMO_PASSWORD unset in production to disable the demo account entirely.
"""

from fastapi import APIRouter, HTTPException, status

from backend.core.config import settings
from backend.core.security import verify_password, create_access_token, hash_password
from backend.models.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["Auth"])


def _build_demo_users() -> dict[str, str]:
    if settings.demo_username and settings.demo_password:
        return {settings.demo_username: hash_password(settings.demo_password)}
    return {}


# Populated at startup from settings; empty dict → login always fails (safe default).
DEMO_USERS: dict[str, str] = _build_demo_users()


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    """
    Authenticate with username + password, returns a JWT access token.
    """
    hashed = DEMO_USERS.get(body.username)
    if not hashed or not verify_password(body.password, hashed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = create_access_token({"sub": body.username})
    return TokenResponse(access_token=token)
