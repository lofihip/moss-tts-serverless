import asyncio
from vastai import Serverless

async def main():
    client = Serverless(api_key="b0124a2d40b86b60c293f0be29245399d12f83cc16403764511d6e3bc96f3f57")

    endpoint = await client.get_endpoint(name="MOSS-TTS")

    result = await endpoint.request("/health", {}, cost=0)
    print(result)

    await client.close()

asyncio.run(main())