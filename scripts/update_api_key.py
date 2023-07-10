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
        key_state = await query_key_info(key)
        if key_state:
            task_obj = db.ApiKey.update(
                alive=key_state.alive,
                amount=key_state.amount,
                used=key_state.used
            ).where(
                db.ApiKey.key == key_state.key
            ).execute()
            logger.info(f"update key:{key_state.key}, alive={key_state.alive}, amount={key_state.amount}, used={key_state.used}")
            return key_state
        else:
            task_obj = db.ApiKey.update(
                alive=False,
            ).where(
                db.ApiKey.key == key
            ).execute()
            logger.error(f"query no info key:{key}")
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
        chunk_size = len(api_keys)
    else:
        chunk_size = int(len(api_keys) / 20)+1
    logger.info(f"split {len(api_keys)} into {chunk_size} chunks")
    split_api_keys = split_list(api_keys, chunk_size)

    api_tasks = [update_keys(res) for res in split_api_keys]
    results = await asyncio.gather(*api_tasks)

if __name__ == "__main__":
    # Check if the server is alive.
    # If the connection is closed, reconnect.
    asyncio.run(update_schedule())


    print("schedule finished.")
    logger.info("schedule finished.")

