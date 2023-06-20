import os
import json
import threading
from loguru import logger
from dotenv import load_dotenv

import db
from modules.download.donwload_pdf import download_pdf_from_url

load_dotenv()

from fastapi import FastAPI, Request

import redis_manager
from summary import handler, handler_subscribe

app = FastAPI()

@app.get('/invoke')
async def invoke(request: Request):
    summary_id = request.query_params.get('summary_id')
    if summary_id is None:
        return 'failed'

    summary_key = f"subscribe_summary:{summary_id}"
    # 判断是否是订阅的
    pdf_url = await redis_manager.get(summary_key)
    pdf_url = pdf_url.decode('utf-8')
    if pdf_url:
        logger.info("begin subscribe summary")
        dumps = json.dumps({"summary_id": summary_id})
        threading.Thread(target=handler_subscribe, args=(dumps,)).start()
        return 'success'
    else:
        dumps = json.dumps({"summary_id": summary_id})
        threading.Thread(target=handler, args=(dumps,)).start()
        return 'success'


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5555)
