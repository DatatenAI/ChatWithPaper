import logging
import os
import uuid
from typing import Union

import fitz
import tiktoken

import db
from constants import FREE_TOKEN


def is_cost_purchased(user: db.User, token_cost: float):
    if user.token_consumed + token_cost > FREE_TOKEN and user.vip_level > 1:
        return True
    return False


def estimate_embedding_token(path: str) -> int:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    full_text = ""
    with fitz.open(path) as doc:  # type: ignore
        page_cnt = 0
        for page in doc:
            page_cnt += 1
            text = page.get_text("text")
            full_text += f"page:{page_cnt}\n" + text
    return token_str(full_text)


def token_str(content: str):
    encoding = tiktoken.get_encoding("gpt2")
    return len(encoding.encode(content))


async def retry(tries: int, function, *args,
                **kwargs) -> Union[Exception, None]:
    for attempt in range(tries):
        try:
            return await function(*args, **kwargs)
        except Exception as e:
            logging.info(f"attempt execute function error,tries: {attempt},error:{e}")
            continue


def gen_uuid():
    return str(uuid.uuid4())
