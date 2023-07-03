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
    language: str  # 目标语言
    translate_temp: str


async def translate(translate_data: TranslateData) -> tuple[dict, float]:
    """
    输入的是
    """
    pdf_hash = translate_data.pdf_hash
    user_type = translate_data.user_type
    language = translate_data.language
    translate_temp = translate_data.translate_temp

    try:
        # TODO 将summary 和 paper_question都翻译了
        summary = db.Summaries.select().where(
            db.Summaries.pdf_hash == pdf_hash).order_by(
            db.Summaries.create_time)[0]

        texts = {
            'basic info': summary.basic_info,
            'brief introduction': summary.brief_introduction,
            'first page conclusion': summary.first_page_conclusion,
            'content': summary.content,
            'medium length content': summary.medium_content,
            'short length content': summary.short_content,
        }

        translated_texts, translate_tokens = await pdf_summary.translate(texts, language)

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

        return new_summary, translate_tokens


    except Exception as e:
        logger.error(f"translate error:{repr(e)}")
        raise Exception(e)


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
        if user_type == 'user':
            earliest_summary = db.Summaries.select().where(db.Summaries.pdf_hash == pdf_hash)

            if earliest_summary:
                translate_data = TranslateData(
                    user_type=user_type,
                    pdf_hash=pdf_hash,
                    language=language,
                    translate_temp='default'
                )

                new_summary, translate_tokens = await translate(translate_data)
                # 更新task 表 和 summary
                try:
                    with db.mysql_db_new.atomic():
                        _, translated_info = db.Summaries.get_or_create(pdf_hash=new_summary['pdf_hash'],
                                                                        language=new_summary['language'],
                                                                        # 后面的模板
                                                                        defaults=new_summary)
                        if translated_info:
                            logger.info(f'translate fields of summary, pdf_hash: {pdf_hash}')
                            task_obj = db.UserTasks.update(state='SUCCESS',
                                                           cost_credits=translate_tokens,
                                                           finished_at=datetime.datetime.now().strftime(
                                                               '%Y-%m-%d %H:%M:%S')
                                                           ).where(db.UserTasks.id == task_id).execute()
                            logger.info(f"translate success")
                except Exception as e:
                    logger.info(f"{e}")
            else:
                logger.info(f"no earliest summary task")
        elif user_type == 'spider':


            pass
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
            try:
                with db.mysql_db_new.atomic():
                    task_obj = db.UserTasks.update(
                        user_id=task_data['user_id'],
                        state='FAIL',
                        finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ).where(db.UserTasks.id == task_id).execute()
                    logger.info(f"Fail translate, User:{user_id} tasks {task_id}, pdf_hash={pdf_hash}")
                    await user_db.update_points_add(user_id, pages, 'TASK')
                    # 还钱
                    logger.info(f"give back user {user_id}, points {pages} success")
            except Exception as e:
                logger.error(f"give back user {user_id}, points {pages} fail {repr(e)}")
                raise Exception(e)
        logger.error(f"{repr(e)}")
        raise Exception(e)


async def test_process_translate():
    user_type = 'user'

    logger.info("begin translation")
    task = db.UserTasks.get(db.UserTasks.id == 'abcd')
    logger.info("begin spider summary")
    dumps = json.dumps({
        "user_id": task.user_id,
        "user_type": user_type,
        "task_id": task.id,
        "task_type": task.type,
        "language": task.language,
        "pages": task.pages,
        "pdf_hash": task.pdf_hash,
        "translate_temp": 'default'  # 总结模板
    }, ensure_ascii=False)
    task_data = json.loads(dumps)
    await process_translate(task_data)


async def test_translate():
    translate_data = TranslateData(
        user_type='spider',
        pdf_hash='e23f8e1adc451df11f169c9408d4f52e',
        language='English',
        translate_temp='default'
    )
    new_summary = await translate(translate_data)
    print(new_summary)
    pass


if __name__ == '__main__':
    # asyncio.run(test_translate())
    asyncio.run(test_process_translate())
