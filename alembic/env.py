# alembic/env.py 关键修改
from app.models.base import Base
from app.config.settings import get_settings

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url.replace("+asyncpg", ""))
target_metadata = Base.metadata

