"""
批量插入APIkey的脚本
从 ../config/api_keys.txt中读取并插入 key
"""
import asyncio
import os
from dotenv import load_dotenv
from loguru import logger

if os.getenv('ENV') == 'DEV':
    load_dotenv()

from modules.database.mysql import db
from scripts.ApiKeyUtil import query_key_info


def read_non_empty_lines(file_path):
    """
    读取文件中的非空行，并返回一个列表
    """
    lines = []

    # 打开文件，逐行读取
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()  # 去除行尾的空白字符
            if line:  # 如果不是空行
                lines.append(line)  # 将该行添加到列表中

    return lines


async def test():
    local_apikeys = read_non_empty_lines('../config/api_keys.txt')
    print(local_apikeys)

    for key in local_apikeys:
        key_state = await query_key_info(key)
        if key_state:

            data_tasks = {
                'key': key_state.key,
                'alive': key_state.alive,
                'amount': key_state.amount,
                'used': key_state.used,
            }

            obj_id, key_create = db.ApiKey.get_or_create(key=key_state.key,
                                                         defaults=data_tasks)

            if key_create:
                logger.info(f"add key:{key_state.key}, amount={key_state.amount}, used={key_state.used}")
        else:
            logger.error(f"error key:{key}")
if __name__ == "__main__":
    asyncio.run(test())
