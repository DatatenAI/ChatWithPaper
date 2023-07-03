import asyncio
import datetime
import json
import os

from dotenv import load_dotenv
from loguru import logger

import user_db

if os.getenv('ENV') == 'DEV':
    load_dotenv()

from modules.database.mysql import db
import pdf_summary

logger.opt(exception=True)
PDF_SAVE_DIR = os.path.join(os.getenv("FILE_PATH"), 'uploads')

from pydantic import BaseModel

class TranslateData(BaseModel):
    user_type: str
    pdf_hash: str
    language: str
    translate_temp: str


async def translate(translate_data: TranslateData) -> dict:
    pdf_hash = translate_data.pdf_hash
    user_type = translate_data.user_type
    language = translate_data.language
    translate_temp = translate_data.translate_temp

    try:
        summary = db.Summaries.get(db.Summaries.pdf_hash == pdf_hash)

        texts = {
            'basic info': summary.basic_info,
            'brief introduction': summary.brief_introduction,
            'first page conclusion': summary.first_page_conclusion,
            'content': summary.content,
            'medium length content': summary.medium_content,
            'short length content': summary.short_content,
        }

        translated_texts = await pdf_summary.translate(texts, language)

        new_summary = {
            'pdf_hash': summary.pdf_hash,
            'language': language,
            'title': summary.title,
            'title_zh': summary.title_zh,
            'basic_info': translated_texts['basic info'],
            'brief_introduction': translated_texts['brief introduction'],
            'first_page_conclusion': translated_texts['first page conclusion'],
            'content': translated_texts['content'],
            'medium_content': translated_texts['medium length content'],
            'short_content': translated_texts['short length content'],
        }

        return new_summary


    except Exception as e:
        logger.error(f"translate error:{repr(e)}")
        raise e


async def process_translate(task_data):
    """
    translate的处理逻辑
    """
    pdf_hash = task_data['pdf_hash']
    user_type = task_data['user_type']
    user_id = task_data['user_id']
    task_id = task_data['task_id']
    pages = task_data['pages']
    language = task_data['language']

    logger.info(f"process translate user id {user_id}")
    try:
        earliest_created_task = (
            db.UserTasks.select()
            .where(db.UserTasks.pdf_hash == pdf_hash,db.UserTasks.state == 'SUCCESS')
            .order_by(db.UserTasks.created_at)
            .first()
        )

        if earliest_created_task:
            translate_data = TranslateData(
                user_type=user_type,
                pdf_hash=pdf_hash,
                language=language,
                translate_temp='default'
            )

            new_summary = await translate(translate_data)

            _, translated_info = db.Summaries.get_or_create(**new_summary)
            if translated_info:
                logger.info(f'translate fields of summary, pdf_hash: {pdf_hash}')

    except Exception as e:
        # 还钱
        if user_type == 'spider':
            # TODO 写报错信息和改变状态
            task_obj = db.SubscribeTasks.update(state='FAIL',
                                                tokens=0,
                                                finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                                ).where(db.SubscribeTasks.id == task_id).execute()
            logger.info(f"Fail Subscribe tasks {task_obj}, pdf_hash={pdf_hash}")

        elif user_type == 'user':
            # 写报错信息
            task_obj = db.UserTasks.update(
                user_id=task_data['user_id'],
                state='FAIL',
                finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ).where(
                db.UserTasks.id == task_id
            ).execute()
            logger.info(f"Fail User:{user_id} tasks {task_obj}, pdf_hash={pdf_hash}")
            # 还钱
            try:
                await user_db.update_points_add(user_id, pages, 'TASK')
                logger.info(f"give back user {user_id}, points {pages} success")
            except Exception as e:
                logger.error(f"give back user {user_id}, points {pages} fail {repr(e)}")
                raise Exception(e)
        logger.error(f"{repr(e)}")
        raise e




async def test_process_translate():
    user_type = 'spider'

    logger.info("begin spider translation")
    dumps = json.dumps({
        "user_type": user_type,
        "user_id": 'chat-paper',  # 添加用户id
        "temp": 'default'
    }, ensure_ascii=False)
    task_data = json.loads(dumps)
    await process_translate(task_data)

async def test_translate():
    translate_data = TranslateData(
        user_type='spider',
        pdf_hash='29e507c6562a444ce50b131453324c41',
        language='English',
        translate_temp='default'
    )
    new_summary = await translate(translate_data)
    print(new_summary)
    pass


if __name__ == '__main__':
    # asyncio.run(test_translate())
    asyncio.run(test_process_translate())