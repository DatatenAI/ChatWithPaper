import asyncio
import os
from typing import List

import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from prisma import Prisma
from pydantic import BaseModel
from pymilvus import connections, Collection, utility
from loguru import logger


def token_str(content: str):
    ENCODER = tiktoken.get_encoding("gpt2")
    return len(ENCODER.encode(content))

class PaperDocVector(BaseModel):
    doc_id: int
    pdf_hash: str
    sql_id: int
    summary_vector: List[float]