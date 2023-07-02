import asyncio
import datetime
import json
import os
from pathlib import Path
from typing import Union, Tuple

from dotenv import load_dotenv
from loguru import logger

import user_db
from modules.vectors.get_embeddings import get_embeddings_from_pdf

if os.getenv('ENV') == 'DEV':
    load_dotenv()

from modules.database.mysql import db
import pdf_summary

logger.opt(exception=True)
PDF_SAVE_DIR = os.path.join(os.getenv("FILE_PATH"), 'uploads')


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


def delete_wrong_summary_res(pdf_hash: str, language: str, summary_temp: str):
    """
    :param language: 语言
    :summary_temp: 总结模板
    """
    base_path = os.path.join(PDF_SAVE_DIR, pdf_hash)
    complete_path = f"{base_path}.complete.{language}.{summary_temp}.txt"  #
    format_path = f"{base_path}.formated.{language}.{summary_temp}.txt"
    first_page_path = f"{base_path}.firstpage_conclusion.{language}.{summary_temp}.txt"  #
    basic_info_path = f"{base_path}.basic_info.{language}.{summary_temp}.txt"
    brief_intro_path = f"{base_path}.brief.{language}.{summary_temp}.txt"
    structure_path = os.path.join(os.getenv('FILE_PATH'), f"out/{pdf_hash}_structure.json")
    pdf_vec_path = f"{os.getenv('FILE_PATH')}/out/{pdf_hash}.json"

    if Path(first_page_path).is_file():  # 文件小于等于100字节
        delete_wrong_file(first_page_path, file_size=100)
        delete_wrong_file(structure_path, file_size=1000)
        delete_wrong_file(pdf_vec_path, file_size=100)
    if Path(complete_path).is_file():
        delete_wrong_file(complete_path, 1000)
    if Path(format_path).is_file():
        delete_wrong_file(format_path, 1500)
    if Path(basic_info_path).is_file():
        delete_wrong_file(basic_info_path, 100)
    if Path(brief_intro_path).is_file():
        delete_wrong_file(brief_intro_path, 100)


from pydantic import BaseModel


class SummaryData(BaseModel):
    user_type: str
    pdf_hash: str
    language: str
    summary_temp: str


class TranslateData(BaseModel):
    user_type: str
    pdf_hash: str
    language: str
    translate_temp: str


# async def summary(summary_id: str):
async def summary(summary_data: SummaryData) -> Union[Tuple, None]:
    user_type = summary_data.user_type
    file_hash = summary_data.pdf_hash
    language = summary_data.language
    summary_temp = summary_data.summary_temp

    logger.info(f"start user: {user_type},  summary {file_hash}")
    pdf_path = os.path.join(PDF_SAVE_DIR, f"{file_hash}.pdf")
    if not Path(pdf_path).is_file():
        logger.error(f'file {file_hash}.pdf  not found')
        raise ValueError('file {file_hash}.pdf  not found')
    try:
        # 选择模板
        if summary_temp == 'default':
            # Generate the summary from the PDF file
            token_cost_all = await pdf_summary.get_the_formatted_summary_from_pdf(
                pdf_path, language, summary_temp=summary_temp)
            return token_cost_all
        # TODO 其他模板
        else:
            logger.error(f"summary temp error: no {summary_temp} temp")
            error_res = {"status": "error", "detail": f"summary temp error: no {summary_temp} temp"}
            raise json.dumps(error_res, ensure_ascii=False, indent=4)  # 返回错误信息
    except Exception as e:
        logger.error(f"generate summary error:{e}", )
        error_res = {"status": "error", "detail": str(e)}
        # Delete any previous wrong summary results associated with the summary_id
        delete_wrong_summary_res(file_hash, language, summary_temp)
        raise e


async def process_summary(task_data):
    """
    summary的处理逻辑和扣款逻辑
    """
    pdf_hash = task_data['pdf_hash']
    user_type = task_data['user_type']
    user_id = task_data['user_id']
    pages = task_data['pages']
    task_id = task_data['task_id']
    # 总结逻辑
    logger.info(f"user id {user_id}")

    summary_data = SummaryData(
        user_type=task_data['user_type'],
        pdf_hash=task_data['pdf_hash'],
        language=task_data['language'],
        summary_temp=task_data['summary_temp']
    )
    pdf_path = os.path.join(PDF_SAVE_DIR, f'{pdf_hash}.pdf')
    # TODO 检查是否处理过
    is_process = db.Summaries.get_or_none(
        pdf_hash=task_data['pdf_hash'],
        language=task_data['language'],
    )
    if is_process:
        logger.info(f"user:{user_id}, pdf hash: {pdf_hash},lang:{task_data['language']} finished")
        return None
    try:
        vec_split_task = get_embeddings_from_pdf(pdf_path, max_token=512)
        summary_task = summary(summary_data=summary_data)
        # 只需要token就够了
        embeddings_tokens, summary_tokens = await asyncio.gather(vec_split_task, summary_task)


    except Exception as e:
        logger.error(f"generate summary error:{e}", )
        error_res = {"status": "error", "detail": str(e)}
        # TODO 处理报错信息
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
                logger.error(f"give back user {user_id}, points {pages} fail {e}")
                raise Exception(e)
        raise e


    token_cost_all = embeddings_tokens + summary_tokens
    if user_type == 'spider':
        # TODO 将数据写入到SubscribeTasks任务表中和summaries表中
        # 添加任务表并传参数
        try:
            task_obj = db.SubscribeTasks.update(state='SUCCESS',
                                                tokens=token_cost_all,
                                                finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                                ).where(
                db.SubscribeTasks.pdf_hash == pdf_hash,
                db.SubscribeTasks.type == task_data['task_type'],
                db.SubscribeTasks.language == task_data['language']).execute()
            logger.info(f"finish Subscribe tasks {task_obj}, pdf_hash={pdf_hash}, tokens={token_cost_all}")

        except Exception as e:
            logger.error(f"{e}")

    elif user_type == 'user':
        # 将数据写入到任务表中和summaries表中

        try:
            task_obj = db.UserTasks.update(
                user_id=task_data['user_id'],
                state='SUCCESS',
                finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ).where(
                db.UserTasks.user_id == task_data['user_id'],
                db.UserTasks.pdf_hash == pdf_hash,
                db.UserTasks.language == task_data['language'],
                db.UserTasks.type == task_data['task_type'],
            ).execute()
            logger.info(f"finish Subscribe tasks {task_obj}, pdf_hash={pdf_hash}, tokens={token_cost_all}")

        except Exception as e:
            logger.error(f"{e}")

    return None


async def testUserTask():
    pass


async def test_SubTask():
    task_id = '107'
    user_type = 'spider'

    task = db.SubscribeTasks.get(db.SubscribeTasks.id == task_id)
    if task.type.lower() == 'summary':  # 总结的任务
        logger.info("begin spider summary")
        dumps = json.dumps({
            "user_type": user_type,
            "task_id": task_id,
            "user_id": 'chat-paper',  # 添加用户id
            "task_type": task.type,
            "language": task.language,
            "pages": task.pages,
            "pdf_hash": task.pdf_hash,
            "summary_temp": 'default'
        }, ensure_ascii=False)
        task_data = json.loads(dumps)
        res = await process_summary(task_data)
        print(res)


async def test_summary():
    summary_data = SummaryData(
        user_type='spider',
        pdf_hash='3047b38215263278f07178419489a887',
        language='中文',
        summary_temp='default1'
    )
    summary_res = await summary(summary_data)
    print(summary_res)
    pass


if __name__ == '__main__':
    asyncio.run(test_SubTask())

    # 测试 summary
    # asyncio.run(test_summary())
