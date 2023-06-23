import os
import pickle
import uuid
from typing import Union
from loguru import logger
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
            logger.info(f"attempt execute function error,tries: {attempt},error:{e}")
            continue

def split_list(lst: list, chunk_size: int):
    """
    将list进行按照chunk_size大小拆分
    :param lst:
    :param chunk_size:
    :return:
    """
    assert len(lst)>0
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def gen_uuid():
    return str(uuid.uuid4())

def print_token(tip, result):
    logger.info(
        f"{tip} prompt used: {str(result[1])} tokens. Completion used: {str(result[2])} tokens. Totally used: {str(result[3])} tokens.")


# 保存对象列表到文件
async def save_to_file(obj_list: object, file_path: str) -> object:
    with open(file_path, 'wb') as file:
        pickle.dump(obj_list, file)


# 从文件加载对象列表
async def load_from_file(file_path: str):
    with open(file_path, 'rb') as file:
        obj_list = pickle.load(file)
    return obj_list


if __name__ == '__main__':
    # 示例代码
    pdf_meta_list = [1,2,3]  # 包含 PDFMetaInfoModel 类对象的列表

    # 保存到文件
    save_to_file(pdf_meta_list, 'pdf_meta_list.pkl')

    # 从文件加载
    loaded_pdf_meta_list = load_from_file('pdf_meta_list.pkl')
    print(loaded_pdf_meta_list)
