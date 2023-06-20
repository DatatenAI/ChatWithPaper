import asyncio
import os
import time
import uuid

import aioredis
from aioredis import RedisError
from loguru import logger

redis = aioredis.from_url(os.getenv("REDIS_URL"))


async def set(key, value):
    return await redis.set(key, value)

async def get(key):
    return await redis.get(key)

async def delete(key):
    if redis is None:
        raise Exception("Redis is not connected")
    return await redis.delete(key)

async def acquire_lock(lock_name, acquire_timeout=30, lock_timeout=30):
    identifier = str(uuid.uuid4())
    lock_name = 'lock:' + lock_name
    logger.info(f"get lock: lock_name:{lock_name},identifier:{identifier}")
    end = time.time() + acquire_timeout
    while time.time() < end:
        if await redis.set(
                lock_name,
                identifier,
                ex=lock_timeout,
                nx=True):
            return identifier
        await asyncio.sleep(1)
    return False


async def release_lock(lock_name, identifier):
    lock_name = 'lock:' + lock_name
    while True:
        try:
            value = await redis.get(lock_name)
            if value.decode() == identifier:
                await redis.delete(lock_name)
                logger.info(f"release lock: lock_name:{lock_name},identifier:{identifier}")
                return True
            break
        except RedisError as e:
            logger.error(f"release lock failed:{e}")
            pass

    return False
