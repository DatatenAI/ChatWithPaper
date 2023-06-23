import redis_manager
from constants import FREE_TOKEN
import db
from loguru import logger
async def update_token_consumed_paid(user_id: str,
                                     token_consumed_increase: float):
    lock_name = "consumed_token_paid" + user_id
    lock = await redis_manager.acquire_lock(lock_name)
    if not lock:
        raise Exception("Too many requests")

    try:
        result = db.UserToken.get_or_none(db.UserToken.user_id == user_id)
        if not result:
            logger.error(f"User: {user_id}, not found")
            raise Exception("User not found")
        if result.tokens_purchased < result.tokens_consumed + token_consumed_increase:
            logger.error(f"token paid consumption exceeded, user_id:{user_id}, "
                         f"tokens_consumed:{result.tokens_consumed},"
                         f"token_purchased:{result.tokens_purchased}")
            raise Exception("Token consumption exceeded")
        db.UserToken.update(tokens_consumed=db.UserToken.tokens_consumed + token_consumed_increase).where(
            db.UserToken.user_id == user_id).execute()
        logger.info(f"user_id:{user_id}, take paid {token_consumed_increase} tokens")
    finally:
        await redis_manager.release_lock(lock_name, lock)


async def update_token_consumed_free(user_id: str,
                                     token_consumed_increase: float):
    lock_name = "consumed_token_paid" + user_id
    lock = await redis_manager.acquire_lock(lock_name)
    if not lock:
        raise Exception("Too many requests")
    try:
        user = db.User.get_or_none(db.User.user_id == user_id)
        if not user:
            logger.error(f"User: {user_id}, not found")
            raise Exception("User not found")
        if user.token_consumed + token_consumed_increase > FREE_TOKEN:
            logger.error(
                f"token free consumption exceeded, user_id:{user_id}, tokens_consumed:{user.token_consumed}, "
                f"token_increase:{token_consumed_increase}")
            raise Exception("Token consumption exceeded")
        # db.User.update(token_consumed=db.User.token_consumed + token_consumed_increase).where(db.User.user_id == user_id).execute()
        # 修复token负数的bug.
        if user.token_consumed + token_consumed_increase < 0:
           db.User.update(token_consumed=FREE_TOKEN).where(
                db.User.user_id == user_id).execute()
        else:
           db.User.update(token_consumed=db.User.token_consumed + token_consumed_increase).where(
                db.User.user_id == user_id).execute()
        logger.info(f"user_id:{user_id}, take free {token_consumed_increase} tokens")
    finally:
        await redis_manager.release_lock(lock_name, lock)
