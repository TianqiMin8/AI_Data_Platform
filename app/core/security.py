from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet, MultiFernet

from app.config.settings import get_settings

# ── 密码哈希 ──

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ── JWT Token ──

def create_access_token(user_id: int) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int | None:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        user_id = int(payload.get("sub", 0))
        return user_id if user_id else None
    except (JWTError, ValueError):
        return None


# ── 对称加密（用于数据库密码） ──

def _get_fernet() -> MultiFernet:
    settings = get_settings()

    keys = [settings.encryption_key]

    if settings.encryption_old_keys:
        old_keys = [
            key.strip()
            for key in settings.encryption_old_keys.split(",")
            if key.strip()
        ]
        keys.extend(old_keys)

    fernets = [
        Fernet(key.encode())
        for key in keys
    ]

    return MultiFernet(fernets)


def encrypt_value(plain_text: str) -> str:
    """
    使用当前 primary key 加密。
    MultiFernet.encrypt() 永远使用第一个 key。
    """
    if not plain_text:
        return plain_text

    f = _get_fernet()
    return f.encrypt(plain_text.encode()).decode()


def decrypt_value(encrypted_text: str) -> str:
    """
    解密时依次尝试当前 key 和旧 key。
    """
    if not encrypted_text:
        return encrypted_text

    f = _get_fernet()
    return f.decrypt(encrypted_text.encode()).decode()