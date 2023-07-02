# -*- coding: utf-8 -*-
"""
更新 api_keys中key状态的的脚本

"""
import os
import asyncio

from dotenv import load_dotenv

from modules.util import split_list

if os.getenv('ENV') == 'DEV':
    load_dotenv()


from modules.database.mysql import db
from scripts.ApiKeyUtil import query_key_info, ApiKey
from loguru import logger


# 查询使用量URL
usage_url = '/v1/dashboard/billing/usage'
# 查询订阅信息URL
subscription_url = '/v1/dashboard/billing/subscription'

host = 'https://api.openai.com'




# set headers and authorization
def get_headers(key: str) -> dict:
    return {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}



async def update_key(key) -> ApiKey:

    try:
        key_state = query_key_info(key)
        is_alive = True if key_state.alive else False
        if key_state.amount - key_state.used < 1:
            is_alive = False
            key_state.live = is_alive
        task_obj = db.ApiKey.update(
            alive=key_state.alive,
            amount=key_state.amount,
            used=key_state.used
        ).where(
            db.ApiKey.key == key_state.key
        ).execute()
        logger.info(f"update key:{key_state.key}, alive={is_alive}, amount={key_state.amount}, used={key_state.used}")
        return key_state
    except Exception as e:
        logger.error(f"update error {repr(e)}")

async def update_keys(keys):
    for key in keys:
        await update_key(key)

    # return f"update api key success,{num_alive} keys alive, total amount remaining:{remain_amount}"

async def update_schedule():
    api_keys_get = db.ApiKey.select()
    api_keys = [res.key for res in api_keys_get]
    # 拆分chunks
    if len(api_keys) < 20:
        num_chunks = len(api_keys)
    else:
        num_chunks = int(len(api_keys) / 20)
    logger.info(f"split {len(api_keys)} into {num_chunks} chunks")
    split_api_keys = split_list(api_keys, num_chunks)

    api_tasks = [update_keys(res) for res in split_api_keys]
    results = await asyncio.gather(*api_tasks)

if __name__ == "__main__":
    # Check if the server is alive.
    # If the connection is closed, reconnect.
    asyncio.run(update_schedule())


    print("schedule finished.")
    logger.info("schedule finished.")

