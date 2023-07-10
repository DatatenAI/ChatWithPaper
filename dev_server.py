import os
import json
import threading
from loguru import logger
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, Request
from dotenv import load_dotenv
load_dotenv()

from chat_with_paper import handler

current_script_directory = os.path.dirname(os.path.abspath(__file__))
logger.info(f"work path:{current_script_directory}")


print(os.getenv('FILE_PATH'))

app = FastAPI()


class RequestParams(BaseModel):
    task_id: int = Field(..., description='任务表的task id')
    user_type: str = Field(..., description="用户类型，可选值：user|spider")

    @validator('task_id', 'user_type')
    def validate_required_fields(cls, value):
        if not value:
            raise ValueError("该字段为必填字段")
        return value


@app.get('/invoke')
async def invoke(params: Request):
    # 用户的task_id 用字符串
    # 订阅的task_id 用uuid
    task_id = params.query_params.get('task_id')
    user_type = params.query_params.get('user_type')

    dumps = json.dumps({
        "task_id": task_id,
        "user_type": user_type,
    })

    threading.Thread(target=handler, args=(dumps,)).start()
    # handler(dumps)
    return 'success'

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5555)
