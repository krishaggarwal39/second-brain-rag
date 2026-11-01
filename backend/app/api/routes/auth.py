from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from app.core.auth import get_user_store, create_access_token, resolve_role

router = APIRouter(prefix="/auth", tags=["Auth"])

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest):
    store = get_user_store()
    try:
        user_id = store.create_user(body.email, body.password)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    role = resolve_role(body.email)
    store.set_role(user_id, role)
    token = create_access_token(user_id, role=role)
    return TokenResponse(access_token=token)

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    store = get_user_store()
    user_id = store.authenticate(body.email, body.password)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    role = resolve_role(body.email)
    store.set_role(user_id, role)
    token = create_access_token(user_id, role=role)
    return TokenResponse(access_token=token)
