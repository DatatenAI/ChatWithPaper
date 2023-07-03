import asyncio
import re

import httpx

import optimize_openai
from modules.util import print_token



chat_paper_api = optimize_openai.ChatPaperAPI(
    model_name="gpt-3.5-turbo-16k",
    # model_name="gpt-3.5-turbo",
    top_p=1,
    temperature=0.0,
    apiTimeInterval=0.02)

async def get_content_test():
    api_key = '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'
    content = "你好"
    # api_key = api_key_db.get_single_alive_key()
    async with httpx.AsyncClient() as client:
        async with client.stream(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization":
                        f"Bearer {api_key}"
                },
                json={
                    "model": 'gpt-3.5-turbo-16k',
                    "messages": [{"role": "system", "content": "you are a helpful assistant"},
                                 {"role": "user", "content": content}],
                    # kwargs
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "n": 1,
                    "user": "你好",
                    "stream": True,
                },
        ) as response:
            # if response.status_code != 200:
            #     if response.status_code == 403:
            #         await api_key_db.update_api_key_is_alive(api_key, False)
            #     raise Exception(
            #         f"Error: {response.status_code}",
            #     )
            content = []
            async for chunk in response.aiter_bytes():
                content.append(chunk.decode())
            full_response = "".join(content)
            print(full_response)
            return full_response

async def test():
    from loguru import logger
    response = await get_content_test()
    pattern = r'"content":"([^"]*)"'
    matches = re.findall(pattern, response)
    full_response: str = "".join(matches)
    logger.info(full_response)


if __name__ == "__main__":
    asyncio.run(test())
