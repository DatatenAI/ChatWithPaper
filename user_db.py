import redis_manager
from constants import FREE_TOKEN
import db

async def update_token_consumed_paid(user_id: str,
                                     token_consumed_increase: float):
    lock_name = "consumed_token_paid" + user_id
    lock = await redis_manager.acquire_lock(lock_name)
    if not lock:
        raise Exception("Too many requests")

    try:
        result = db.UserToken.get_or_none(db.UserToken.user_id == user_id)
        if not result:
            raise Exception("User not found")
        if result.tokens_purchased < result.tokens_consumed + token_consumed_increase:
            raise Exception("Token consumption exceeded")
        db.UserToken.update(tokens_consumed=db.UserToken.tokens_consumed + token_consumed_increase).where(
            db.UserToken.user_id == user_id).execute()
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
            raise Exception("User not found")
        if user.token_consumed + token_consumed_increase > FREE_TOKEN:
            raise Exception("Token consumption exceeded")
        db.User.update(token_consumed=db.User.token_consumed + token_consumed_increase).where(db.User.user_id == user_id).execute()
    finally:
        await redis_manager.release_lock(lock_name, lock)
