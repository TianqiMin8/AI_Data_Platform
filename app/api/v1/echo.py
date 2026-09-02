from fastapi import APIRouter, Request

from app.schemas.echo import EchoRequest
from app.services.echo import echo_message


router = APIRouter(tags=["echo"])


@router.post("/echo")
async def echo(request_body: EchoRequest, request: Request):
    data = echo_message(request_body.message)

    return {
        "data": data,
        "request_id": getattr(request.state, "request_id", ""),
    }