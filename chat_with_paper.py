import os
from loguru import logger
import json
import asyncio

from summary import process_summary
from translation import process_translate


def handler(event_str):
    try:
        event = os.getenv("FC_CUSTOM_CONTAINER_EVENT")
        logger.info(f"receive env: {event}")
        if event is None:
            event = event_str
        logger.info(f"receive event: {event}")
        task_data = json.loads(event)
        if task_data['task_type'].lower() == 'summary':
            asyncio.run(process_summary(task_data))
        elif task_data['task_type'].lower() == 'translate':
            asyncio.run(process_translate(task_data))
    except Exception as e:
        logger.error(f"handler error: {e}")


