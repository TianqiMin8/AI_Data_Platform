from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.auth import RegisterRequest
from app.core.security import hash_password, verify_password, create_access_token
from app.core.errors import ValidationError, NotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)


async def register_user(db: AsyncSession, req: RegisterRequest) -> User:
    # 检查邮箱是否已注册
    stmt = select(User).where(User.email == req.email)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise ValidationError("该邮箱已注册")

    user = User(
        email=req.email,
        hashed_password=hash_password(req.password),
        nickname=req.nickname or req.email.split("@")[0],
    )
    db.add(user)
    await db.flush()

    logger.info("user.registered", user_id=user.id, email=user.email)
    return user


async def login_user(db: AsyncSession, email: str, password: str) -> str:
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise ValidationError("邮箱或密码错误")

    if not user.is_active:
        raise ValidationError("账号已被禁用")

    token = create_access_token(user.id)
    logger.info("user.logged_in", user_id=user.id)
    return token