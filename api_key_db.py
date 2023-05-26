from typing import Union

from loguru import logger

import db


def update_api_key_is_alive(api_key: str, is_alive: bool):
    db.ApiKey.update(is_alive=is_alive).where(db.ApiKey.apikey == api_key).execute()


def get_single_alive_key() -> Union[str, None]:
    try:
        query_api_key = db.ApiKey.select().order_by(db.ApiKey.consumption.asc()).where(
            db.ApiKey.is_alive == True).first()
        if query_api_key:
            return query_api_key.apikey
        return None
    except Exception as e:
        logger.error(f"get alive openai api key error,{e}")
        return None
