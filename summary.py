import asyncio
import json
import os
import re
from pathlib import Path

import flask
from dotenv import load_dotenv
from loguru import logger


load_dotenv()
import chat_db
import db
import pdf_summary
import redis_manager
import user_db
import util
from flask import Flask

logger.opt(exception=True)
PDF_SAVE_DIR = os.getenv("FILE_PATH")


def delete_wrong_file(file_path: str, file_size: int = None):
    """
    删除文件
    """
    if file_size:
        if Path(file_path).is_file() and Path(file_path).stat().st_size < file_size:
            try:
                Path(file_path).unlink()
                return True
            except Exception as e:
                raise Exception(f"delete {file_path} error:{e}")
        return False
    else:
        if Path(file_path).is_file():
            try:
                Path(file_path).unlink()
                return True
            except Exception as e:
                raise Exception(f"delete {file_path} error:{e}")
        return False

def delete_wrong_summary_res(file_hash, language, summary_temp):
    base_path = os.path.join(PDF_SAVE_DIR, file_hash)
    complete_path = f"{base_path}.complete.txt"
    format_path = f"{base_path}.formated.{language}.txt"
    first_page_path = f"{base_path}.firstpage_conclusion.txt"

    if Path(first_page_path).is_file():   # 文件小于等于100字节
        delete_wrong_file(first_page_path, file_size=100)
    if Path(complete_path).is_file():
        delete_wrong_file(complete_path, 1000)
    if Path(format_path).is_file():
        delete_wrong_file(format_path, 1500)


from pydantic import BaseModel
class SummaryData(BaseModel):
    user_type: str
    pdf_hash: str
    language: str
    summary_temp: str

class TranslateDate(BaseModel):
    user_type: str
    pdf_hash: str
    language: str
    translate_temp: str

# async def summary(summary_id: str):
async def summary(summary_data:SummaryData):
    user_type = summary_data.user_type
    file_hash = summary_data.pdf_hash
    language = summary_data.language
    summary_temp = summary_data.summary_temp

    logger.info(f"start user: {user_type},  summary {file_hash}")
    pdf_path = os.path.join(PDF_SAVE_DIR, f"{file_hash}.pdf")
    if not Path(pdf_path).is_file():
        logger.error(f'file {file_hash}.pdf  not found')
        return None

    try:
        # Generate the summary from the PDF file
        title, title_zh, basic_info, brief_intro, summary_res, token_cost_all = await pdf_summary.get_the_formatted_summary_from_pdf(
            pdf_path, language, summary_temp=summary_temp)

        return title, title_zh, basic_info, brief_intro, summary_res, token_cost_all
    except Exception as e:
        logger.error(f"generate summary error:{e}", )
        error_res = {"status": "error", "detail": str(e)}
        json.dumps(error_res, ensure_ascii=False, indent=4)     # 返回错误信息
        # Delete any previous wrong summary results associated with the summary_id
        delete_wrong_summary_res(file_hash, language, summary_temp)


async def process_summary(task_data):
    """
    summary的处理逻辑
    """
    pdf_hash = task_data['pdf_hash']
    user_type = task_data['user_type']
    # 总结逻辑
    user_id = task_data['user_id']


    logger.info(f"user id {user_id}")
    summary_data = SummaryData(
        user_type=task_data['user_type'],
        pdf_hash=task_data['pdf_hash'],
        language=task_data['language'],
        summary_temp=task_data['temp']
    )
    res = await summary(summary_data=summary_data)
    # 扣费和写表逻辑
    if user_type == 'spider':
        pass
    elif user_type == 'user':
        if res:
            points = task_data['pages']
            if util.is_cost_purchased(user, estimate_token):
                await user_db.update_token_consumed_paid(user_id, points)
        return
        pass


    # "user_type": user_type,
    # "task_id": task_id,
    # "task_type": task.type,
    # "language": task.language,
    # "pdf_hash": task.pdf_hash

    return None

def handler(event_str):
    try:
        event = os.getenv("FC_CUSTOM_CONTAINER_EVENT")
        logger.info(f"receive env: {event}")
        if event is None:
            event = event_str
        logger.info(f"receive event: {event}")
        task_data = json.loads(event)
        asyncio.run(process_summary(task_data))
    except Exception as e:
        logger.error(f"handler error: {e}")



async def testUserTask():
    pass
async def testSubTask():
    task_id = '4'
    user_type = 'spider'

    task = db.SubscribeTasks.get(db.SubscribeTasks.id == task_id)
    if task.type.lower() == 'summary':  # 总结的任务
        logger.info("begin spider summary")
        dumps = json.dumps({
            "user_type": user_type,
            "task_id": task_id,
            "user_id": 'chat-paper',    # 添加用户id
            "task_type": task.type,
            "language": task.language,
            "pages": task.pages,
            "pdf_hash": task.pdf_hash,
            "temp": 'default'
        }, ensure_ascii=False)
        task_data = json.loads(dumps)
        await process_summary(task_data)

if __name__ == '__main__':
    asyncio.run(testSubTask())
