import os
from loguru import logger
import json
import asyncio

from modules.database.mysql import db
from summary import process_summary
from translation import process_translate


def handler(event_str):

    try:
        event = os.getenv("FC_CUSTOM_CONTAINER_EVENT")
        logger.info(f"receive env: {event}")
        if event is None:
            event = event_str
            logger.info(f"dev: {event}")

        logger.info(f"receive event: {event}")
        task_data = json.loads(event)
        task_id = task_data["task_id"]
        user_type = task_data["user_type"]

        if user_type == 'user':
            logger.info(f"task_id:{task_id}, user_type:{user_type}")
            try:
                task = db.UserTasks.get_by_id(task_id)  # 从用户的任务表中取数据
                logger.info("begin user summary")
                dumps = {
                    "user_type": user_type,
                    "task_id": task_id,
                    "task_type": task.type,
                    "language": task.language,
                    "user_id": task.user_id,
                    "pages": task.pages,
                    "pdf_hash": task.pdf_hash,
                    "summary_temp": 'default'
                }
                if task.type.lower() == 'summary':
                    asyncio.run(process_summary(dumps))
                elif task_data['task_type'].lower() == 'translate':
                    asyncio.run(process_translate(dumps))
                return 'success'
            except Exception as e:
                logger.error(f'sql error, user task_id:{task_id} {e}')

        elif user_type == 'spider':
            logger.info(f"task_id:{task_id}, user_type:{user_type}")
            # 读取特定task_id行的数据
            try:
                task = db.SubscribeTasks.get(db.SubscribeTasks.id == task_id)
                logger.info("begin spider summary")
                dumps = json.dumps({
                    "user_id": 'chat-paper',
                    "user_type": user_type,
                    "task_id": task_id,
                    "task_type": task.type,
                    "language": task.language,
                    "pages": task.pages,
                    "pdf_hash": task.pdf_hash,
                    "summary_temp": 'default'  # 总结模板
                }, ensure_ascii=False)
                if task.type.lower() == 'summary':
                    asyncio.run(process_summary(dumps))
                elif task.type.lower() == 'translate':
                    asyncio.run(process_translate(dumps))
                return 'success'
            except Exception as e:
                logger.error(f'sql error, spider task_id:{task_id} {e}')
        else:
            return 'error user type, need user|spider'

    except Exception as e:
        logger.error(f"handler error: {e}")


