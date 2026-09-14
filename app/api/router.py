from fastapi import APIRouter
from app.api.v1 import health, auth, datasource, chat
from app.api.v1.echo import router as echo_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(echo_router)
api_router.include_router(auth.router)
api_router.include_router(datasource.router)
api_router.include_router(chat.router)