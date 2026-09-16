import asyncio

from sqlalchemy import select

from app.db.session import async_session_factory
from app.models.datasource import DataSource
from app.core.security import decrypt_value, encrypt_value


async def rotate_encryption_keys():
    async with async_session_factory() as db:
        result = await db.execute(select(DataSource))
        datasources = list(result.scalars().all())

        rotated = 0

        for ds in datasources:
            if not ds.encrypted_password:
                continue

            plain_password = decrypt_value(ds.encrypted_password)

            ds.encrypted_password = encrypt_value(plain_password)

            rotated += 1

        await db.commit()

        print(
            f"Key rotation completed. "
            f"Rotated {rotated} datasource passwords."
        )


if __name__ == "__main__":
    asyncio.run(rotate_encryption_keys())