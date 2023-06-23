from typing import Union
import os
from loguru import logger
from dotenv import load_dotenv

if os.getenv('ENV') == 'DEV':
    load_dotenv()

import db
import numpy as np



def update_api_key_is_alive(api_key: str, is_alive: bool):
    db.ApiKey.update(is_alive=is_alive).where(db.ApiKey.apikey == api_key).execute()


def get_index_by_probability(probs):
    # 对概率进行归一化
    probs = np.array(probs)
    probs /= probs.sum()
    # 使用 numpy.random.choice 函数进行等比例取值
    index = np.random.choice(len(probs), p=probs)
    return index

def get_single_alive_key() -> Union[str, None]:
    try:
        # query_api_key = db.ApiKey.select().order_by(db.ApiKey.consumption.asc()).where(
        #     db.ApiKey.is_alive == True).first()
        query_api_keys = db.ApiKey.select().order_by(db.ApiKey.consumption.asc()).where(
            db.ApiKey.is_alive == True)
        keys = [apiKey.apikey for apiKey in query_api_keys]
        key_res = [apiKey.total_amount - apiKey.consumption for apiKey in query_api_keys]
        index = get_index_by_probability(key_res)
        key = keys[index]
        logger.info(f"get single api key: {key}")
        return key
        # if query_api_key:
        #     return query_api_key.apikey
        # return None
    except Exception as e:
        logger.error(f"get alive openai api key error,{e}")
        return None


def test():

    apikey = get_single_alive_key()


if __name__ == '__main__':
    test()
