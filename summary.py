import asyncio
import datetime
import json
import os
from pathlib import Path
from typing import Union, Tuple

from dotenv import load_dotenv
from loguru import logger

from modules.vectors.get_embeddings import get_embeddings_from_pdf, embed_text

if os.getenv('ENV') == 'DEV':
    load_dotenv()

import db
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


def delete_wrong_summary_res(file_hash: str, language: str, summary_temp: str):
    """
    :param language: 语言
    :summary_temp: 总结模板
    """
    base_path = os.path.join(PDF_SAVE_DIR, file_hash)
    complete_path = f"{base_path}.complete.{language}.{summary_temp}.txt"  #
    format_path = f"{base_path}.formated.{language}.{summary_temp}.txt"
    first_page_path = f"{base_path}.firstpage_conclusion.{language}.{summary_temp}.txt"  #
    basic_info_path = f"{base_path}.basic_info.{language}.{summary_temp}.txt"
    brief_intro_path = f"{base_path}.brief.{language}.{summary_temp}.txt"

    if Path(first_page_path).is_file():  # 文件小于等于100字节
        delete_wrong_file(first_page_path, file_size=100)
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


class TranslateDate(BaseModel):
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
        raise f'file {file_hash}.pdf  not found'
    try:
        # 选择模板
        if summary_temp == 'default':
            # Generate the summary from the PDF file
            title, title_zh, basic_info, brief_intro, firstpage_conclusion, summary_res, pdf_vec, token_cost_all = await pdf_summary.get_the_formatted_summary_from_pdf(
                pdf_path, language, summary_temp=summary_temp)
            return title, title_zh, basic_info, brief_intro, firstpage_conclusion, summary_res, pdf_vec, token_cost_all
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
        raise json.dumps(error_res, ensure_ascii=False, indent=4)  # 返回错误信息


async def process_summary(task_data):
    """
    summary的处理逻辑和扣款逻辑
    """
    pdf_hash = task_data['pdf_hash']
    user_type = task_data['user_type']
    user_id = task_data['user_id']
    pages = task_data['pages']
    # 总结逻辑
    logger.info(f"user id {user_id}")

    summary_data = SummaryData(
        user_type=task_data['user_type'],
        pdf_hash=task_data['pdf_hash'],
        language=task_data['language'],
        summary_temp=task_data['temp']
    )
    pdf_path = os.path.join(PDF_SAVE_DIR, f'{pdf_hash}.pdf')
    try:

        vec_task = get_embeddings_from_pdf(pdf_path, max_token=512)
        summary_task = summary(summary_data=summary_data)
        vec_data, res_data = await asyncio.gather(vec_task, summary_task)

    except Exception as e:
        logger.error(f"generate summary error:{e}", )
        error_res = {"status": "error", "detail": str(e)}
        str_error_res = json.dumps(error_res, ensure_ascii=False, indent=4)  # 返回错误信息
        # TODO 处理报错信息
        if user_type == 'spider':
            # TODO 写报错信息和改变状态
            task_obj = db.SubscribeTasks.update(state='FAIL',
                                                tokens=0,
                                                finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                                ).where(
                db.SubscribeTasks.pdf_hash == pdf_hash,
                db.SubscribeTasks.type == task_data['type'],
                db.SubscribeTasks.language == task_data['language']).execute()
            logger.info(f"Fail Subscribe tasks {task_obj}, pdf_hash={pdf_hash}")

        elif user_type == 'user':
            # 写报错信息
            task_obj = db.UserTasks.update(
                user_id=task_data['user_id'],
                state='FAIL',
                cost_credits=0,
                finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ).where(
                db.UserTasks.user_id == task_data['user_id'],
                db.UserTasks.pdf_hash == pdf_hash,
                db.UserTasks.language == task_data['language'],
                db.UserTasks.type == task_data['type'],
            ).execute()
            logger.info(f"Fail User:{user_id} tasks {task_obj}, pdf_hash={pdf_hash}")
            # TODO 还钱


            logger.info(f"give back user {user_id}, points {pages}")

        raise str_error_res

    title, title_zh, basic_info, brief_intro, firstpage_conclusion, summary_res, token_cost_all = res_data

    if user_type == 'spider':
        # TODO 将数据写入到SubscribeTasks任务表中和summaries表中
        # 添加任务表并传参数
        try:
            task_obj = db.SubscribeTasks.update(state='SUCCESS',
                                                tokens=token_cost_all,
                                                finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                                ).where(
                db.SubscribeTasks.pdf_hash == pdf_hash,
                db.SubscribeTasks.type == task_data['type'],
                db.SubscribeTasks.language == task_data['language']).execute()
            logger.info(f"finish Subscribe tasks {task_obj}, pdf_hash={pdf_hash}, tokens={token_cost_all}")
            # 添加进summaries
            summary_obg = db.Summaries.create(
                pdf_hash=pdf_hash,
                language=task_data['language'],
                title=title,
                title_zh=title_zh,
                basic_info=basic_info,
                brief_introduction=brief_intro,
                first_page_conclusion=firstpage_conclusion,
                content=summary_res,
            )
            logger.info(f"add summaries id={summary_obg}, pdf_hash={pdf_hash}")
        except Exception as e:
            logger.error(f"{e}")

    elif user_type == 'user':
        # TODO 将数据写入到任务表中和summaries表中
        points = task_data['pages']
        # if util.is_cost_purchased(user, estimate_token):
        #     await user_db.update_token_consumed_paid(user_id, points)

        try:
            task_obj = db.UserTasks.update(
                user_id=task_data['user_id'],
                state='SUCCESS',
                cost_credits=token_cost_all,
                finished_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ).where(
                db.UserTasks.user_id == task_data['user_id'],
                db.UserTasks.pdf_hash == pdf_hash,
                db.UserTasks.language == task_data['language'],
                db.UserTasks.type == task_data['type'],
                ).execute()
            logger.info(f"finish Subscribe tasks {task_obj}, pdf_hash={pdf_hash}, tokens={token_cost_all}")
            # 添加进summaries
            summary_obg = db.Summaries.create(
                pdf_hash=pdf_hash,
                language=task_data['language'],
                title=title,
                title_zh=title_zh,
                basic_info=basic_info,
                brief_introduction=brief_intro,
                first_page_conclusion=firstpage_conclusion,
                content=summary_res,
            )
            logger.info(f"add user {task_data['user_id']}, summaries id={summary_obg}, pdf_hash={pdf_hash}")
        except Exception as e:
            logger.error(f"{e}")

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


async def test_SubTask():
    task_id = '3'
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
            "temp": 'default'
        }, ensure_ascii=False)
        task_data = json.loads(dumps)
        res = await process_summary(task_data)
        print(res)

async def test_summary():
    summary_data = SummaryData(
        user_type= 'spider',
        pdf_hash= '3047b38215263278f07178419489a887',
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