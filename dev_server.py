import os
import json
import threading
from loguru import logger
from pydantic import BaseModel, Field, validator

current_script_directory = os.path.dirname(os.path.abspath(__file__))
logger.info(f"work path:{current_script_directory}")

from dotenv import load_dotenv
load_dotenv()
import db


from fastapi import FastAPI, Request

import redis_manager
from summary import handler

app = FastAPI()


class RequestParams(BaseModel):
    task_id: str = Field(..., description='任务表的task id')
    user_type: str = Field(..., description="用户类型，可选值：user|spider")

    @validator('task_id', 'user_type')
    def validate_required_fields(cls, value):
        if not value:
            raise ValueError("该字段为必填字段")
        return value


@app.get('/invoke')
async def invoke(params: Request):
    task_id = params.query_params.get('task_id')
    user_type = params.query_params.get('user_type')
    if user_type == 'user':
        logger.info(f"task_id:{task_id}, user_type:{user_type}")
        task = db.UserTasks.get(db.UserTasks.id == task_id)     # 从用户的任务表中取数据
        logger.info("begin user summary")
        dumps = json.dumps({
            "user_type": user_type,
            "task_id": task_id,
            "task_type": task.type,
            "language": task.language,
            "user_id": task.user_id,
            "pages": task.pages,
            "pdf_hash": task.pdf_hash,
            "summary_temp": 'default'
        }, ensure_ascii=False)
        threading.Thread(target=handler, args=(dumps,)).start()
        return 'success'
    elif user_type == 'spider':
        logger.info(f"task_id:{task_id}, user_type:{user_type}")
        # 读取特定task_id行的数据
        task = db.SubscribeTasks.get(db.SubscribeTasks.id == task_id)
        if task.type.lower() == 'summary':  # 总结的任务
            logger.info("begin spider summary")
            dumps = json.dumps({
                "user_id": 'chat-paper',
                "user_type": user_type,
                "task_id": task_id,
                "task_type": task.type,
                "language": task.language,
                "pages": task.pages,
                "pdf_hash": task.pdf_hash,
                "summary_temp": 'default'   # 总结模板
            }, ensure_ascii=False)
            threading.Thread(target=handler, args=(dumps,), daemon=True).start()
            return 'success'
        elif task.type.lower() == 'translate':      # 翻译的任务
            return 'success'
        else:
            return 'success'
    else:
        return 'error user type, need user|spider'


    # 在这里执行相应的任务处理逻辑
    # 可以根据传入的参数进行相应的操作

    # 靠pdf_hash, language, type来执行

    # if summary_id is None:
    #     return 'failed'
    #
    # summary_key = f"subscribe_summary:{summary_id}"
    # # 判断是否是订阅的
    # pdf_url = await redis_manager.get(summary_key)
    # pdf_url = pdf_url.decode('utf-8')
    # if pdf_url:
    #     logger.info("begin subscribe summary")
    #     dumps = json.dumps({"summary_id": summary_id})
    #     threading.Thread(target=handler_subscribe, args=(dumps,)).start()
    #     return 'success'
    # else:
    #     dumps = json.dumps({"summary_id": summary_id})
    #     threading.Thread(target=handler, args=(dumps,)).start()
    #     return 'success'


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5555)
