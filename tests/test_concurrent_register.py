import asyncio
import httpx

URL = "http://127.0.0.1:8000/api/v1/auth/register"

payload = {
    "email": "concurrent@test.com",
    "password": "123456",
    "nickname": "Concurrent Test",
}


async def register(client):
    response = await client.post(URL, json=payload)
    print(response.status_code, response.json())


async def main():
    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            register(client),
            register(client),
        )


asyncio.run(main())