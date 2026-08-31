from pydantic import BaseModel
from typing import Any


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str = ""


class ErrorResponse(BaseModel):
    error: ErrorDetail


class SuccessResponse(BaseModel):
    data: Any = None
    message: str = "ok"
    request_id: str = ""