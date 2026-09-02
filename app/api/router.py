from fastapi import APIRouter
from app.api.v1 import health
from app.api.v1.echo import router as echo_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(echo_router)