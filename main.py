import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
import chat_db
import db
import pdf_summary
import redis_manager
import user_db
import util

logger.opt(exception=True)
PDF_SAVE_DIR = os.getenv("FILE_PATH")


def delete_wrong_summary_res(summary_id):
    file_hash = summary_id.split('_')[0]
    language = summary_id.split('_')[1]
    base_path = os.path.join(PDF_SAVE_DIR, file_hash)
    title_path = f"{base_path}.title.txt"
    complete_path = f"{base_path}.complete.txt"
    format_path = f"{base_path}.formated.{language}.txt"
    first_page_path = f"{base_path}.firstpage_conclusion.txt"
    if Path(title_path).is_file() and Path(title_path).stat().st_size < 100:
        Path(title_path).unlink()
        if Path(first_page_path).is_file():
            Path(first_page_path).unlink()
    if not Path(title_path).is_file() and Path(first_page_path).is_file():
        Path(first_page_path).unlink()
    if Path(complete_path).is_file(
    ) and Path(complete_path).stat().st_size < 1000:
        Path(complete_path).unlink()
    if not Path(complete_path).is_file() or not Path(title_path).is_file():
        if Path(format_path).is_file():
            Path(format_path).unlink()
    if Path(format_path).is_file() and Path(format_path).stat().st_size < 1500:
        Path(format_path).unlink()


async def summary(summary_id: str):
    logger.info(f"start summary {summary_id}")
    user_id = await redis_manager.redis.get(summary_id)
    logger.info(f"user id {user_id}")
    if user_id is None:
        logger.error(f"redis summary_id {summary_id}  is none")
        return
    user_id = user_id.decode('utf-8')
    logger.info(f"process summary:{summary_id},user_id:{user_id}")
    user = db.User.get(db.User.user_id == user_id)
    if not user:
        error_res = {"status": "error", "detail": "Login required"}
        summary_key = f"summary_{summary_id}"
        await redis_manager.redis.set(summary_key, error_res)
        return

    # Extract file_hash and language from summary_id
    file_hash, language = summary_id.split('_')

    # Construct PDF file path and check if it exists
    pdf_path = os.path.join(PDF_SAVE_DIR, f"{file_hash}.pdf")
    estimate_token = 20000 + util.estimate_embedding_token(pdf_path) // 10
    if not Path(pdf_path).is_file():
        # Set the Redis response to be an error
        error_res = {"status": "error", "detail": "PDF file not found"}
        summary_key = f"summary_{summary_id}"
        await redis_manager.redis.set(summary_key, error_res)
        if util.is_cost_purchased(user, estimate_token):
            await user_db.update_token_consumed_paid(user_id, -estimate_token)
        else:
            await user_db.update_token_consumed_free(user_id, -estimate_token)
        return

    # Delete any previous wrong summary results associated with the summary_id
    delete_wrong_summary_res(summary_id)

    try:
        # Generate the summary from the PDF file
        res, token_cost = await pdf_summary.get_the_formatted_summary_from_pdf(
            pdf_path, language)
    except Exception as e:
        logger.error(f"generate summary error:{e}", )
        error_res = {"status": "error", "detail": str(e)}
        summary_key = f"summary_{summary_id}"
        await redis_manager.set(
            summary_key, json.dumps(error_res, ensure_ascii=False, indent=4))
        if util.is_cost_purchased(user, estimate_token):
            await user_db.update_token_consumed_paid(user_id, -estimate_token)
        else:
            await user_db.update_token_consumed_free(user_id, -estimate_token)
        return
    # Format the summary text
    summary = re.sub(r'\\+n', '\n', res)
    # Create a dictionary containing the summary, token_cost, and pdf_hash
    summary_res = {
        "status": "ok",
        "summary": summary,
        "token_cost": token_cost,
        "pdf_hash": file_hash
    }
    try:
        histories = chat_db.get_history(pdf_hash=file_hash, user_id=user_id)
        if len(histories) == 0:
            chat_db.add_history(pdf_hash=file_hash,
                                user_id=user_id,
                                query="summary this paper",
                                content=summary,
                                token_cost=token_cost,
                                pages=[])
    except Exception as e:
        logger.error(f"get chat history error:{e}")
        # Set the Redis response to be an error
        # turn the e to string
        error_res = {"status": "error", "detail": str(e)}
        summary_key = f"summary_{summary_id}"
        await redis_manager.redis.set(
            summary_key, json.dumps(error_res, ensure_ascii=False, indent=4))
        if util.is_cost_purchased(user, estimate_token):
            await user_db.update_token_consumed_paid(user_id, -estimate_token)
        else:
            await user_db.update_token_consumed_free(user_id, -estimate_token)
        return

    # Save the summary back to Redis
    summary_key = f"summary_{summary_id}"
    if util.is_cost_purchased(user, estimate_token):
        await user_db.update_token_consumed_paid(user_id,
                                                 token_cost - estimate_token)
    else:
        await user_db.update_token_consumed_free(user_id,
                                                 token_cost - estimate_token)
    await redis_manager.set(
        summary_key, json.dumps(summary_res, ensure_ascii=False, indent=4))


def handler(event_str):
    try:
        event = os.getenv("FC_CUSTOM_CONTAINER_EVENT")
        logger.info(f"receive env: {event}")
        if event is None:
            event = event_str
        logger.info(f"receive event: {event}")
        evt = json.loads(event)
        asyncio.run(summary(evt['summary_id']))
    except Exception as e:
        logger.error(e)


if __name__ == '__main__':
    logger.info(f"os env: {os.environ}")
    handler('{"summary_id":"5e13cdbfc20453746daf8e99da766a74_中文"}')
