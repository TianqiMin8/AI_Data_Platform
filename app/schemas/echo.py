from pydantic import BaseModel


class EchoRequest(BaseModel):
    message: str