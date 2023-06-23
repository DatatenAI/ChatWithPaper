import asyncio
import os
from typing import List

import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from prisma import Prisma
from pydantic import BaseModel
from pymilvus import connections, Collection, utility
from loguru import logger