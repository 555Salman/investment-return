"""
Auth endpoints — simple JWT login for the dashboard.
Demo credentials: admin / password123  (replace with DB-backed auth for production)
"""

from fastapi import APIRouter, HTTPException, status

from backend.core.security import verify_password, create_access_token, hash_password
from backend.models.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["Auth"])

# Demo user store — replace with database lookup in production
DEMO_USERS = {
    "admin": hash_password("password123"),
}


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
