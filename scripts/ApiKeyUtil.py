import asyncio
from typing import Union
import requests
from datetime import datetime, timedelta
import json

from pydantic import BaseModel
from loguru import logger
# 查询使用量URL
usage_url = '/v1/dashboard/billing/usage'
# 查询订阅信息URL
subscription_url = '/v1/dashboard/billing/subscription'

host = 'https://api.openai.com'

class ApiKey(BaseModel):
    key: str
    alive: bool
    amount: float
    used: float


params = {
    # start date is 90 days ago
    'start_date': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
    # end date is tomorrow
    'end_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
}


# set headers and authorization
def get_headers(key: str) -> dict:
    return {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}


import aiohttp


async def send_request(api_key: str):
    # 设置代理服务器的地址和端口
    # ChatGPT API的URL
    url = "https://api.openai.com/v1/chat/completions"

    # 请求参数
    parameters = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [{"role": "system", "content": "你是一个逻辑推理AI"},
                     {"role": "user", "content": "Hello~"}]
    }

    # 请求头
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # 发送请求
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=parameters) as response:
                # 解析响应
                if response.status == 200:
                    data = await response.json()
                    text = data["choices"][0]["message"]
                    return text
                else:
                    logger.error(response)
                    return None
    except Exception as e:
        logger.error(f"send response error: {repr(e)}")
        return None

async def query_key_info(key: str) -> Union[ApiKey, None]:
    try:
        is_text = await send_request(key)
        if is_text:
            alive = True
        else:
            alive = False
            return None

        usage_response = requests.get(host + usage_url, headers=get_headers(key), params=params)
        subscription_response = requests.get(host + subscription_url, headers=get_headers(key))

        if usage_response.status_code != 200 or subscription_response.status_code != 200:
            return None

        # get response body
        usage_result = usage_response.text
        subscription_result = subscription_response.text

        # 获取总额
        amount = round(json.loads(subscription_result)['hard_limit_usd'], 4)
        # 获取使用量
        used = round(json.loads(usage_result)['total_usage'], 4)
        # divide by 100 to convert from cents to dollars
        used = round(used / 100, 4)

        expire_timestamp = int(json.loads(subscription_result)['access_until'])
        alive = datetime.now().timestamp() < expire_timestamp

        logger.info(f"query key:{key}, alive={alive}, amount={amount}, used={used}")
        return ApiKey(key=key,
                      alive=alive,
                      amount=amount,
                      used=used)

    except Exception as e:
        logger.error(f"query error: {key}, {repr(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(query_key_info("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"))
