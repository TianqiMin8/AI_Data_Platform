from datetime import datetime, timezone

from app.core.errors import ValidationError


def echo_message(message: str) -> dict:
    if not message.strip():
        raise ValidationError("message 不能为空")

    return {
        "echo": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }