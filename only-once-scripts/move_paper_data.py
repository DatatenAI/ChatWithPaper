import json
import os
import asyncio
import re

from peewee import *
import datetime
import logging
from tqdm import tqdm

from modules.database.milvus.milvus_db import MilvusSinglePaperManager, MilvusPaperDocManager
import milvus_config.SinglePaperConfig as Spc
import milvus_config.PaperDocConfig as Pdc
from modules.util import gen_uuid
from modules.vectors.get_embeddings import embed_text, From_ChunkText_Get_Questions
from pdf_summary import get_title_brief_info

logger = logging.getLogger('peewee')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# 开发的时候取消注释
from dotenv import load_dotenv

load_dotenv()

from modules.database.mysql import db

milvus_PaperDocManager = MilvusPaperDocManager(host=os.getenv("MILVUS_HOST"),
                                    port=os.getenv("MILVUS_PORT"),
                                    alias="default",
                                    user=os.getenv("MILVUS_USER"),
                                    password=os.getenv("MILVUS_PASSWORD"),
                                    collection_name=Pdc.collection_name,
                                    partition_name=Pdc.partition_name,
                                    schema=Pdc.schema,
                                    field_name=Pdc.field_name,
                                    index_param=Pdc.index_param,
                                    nprobe=10)
old_manager_doc = ''


def move_paper_data():
    papers = db.Paper.select()

    for paper in tqdm(papers):
        paper_info_data = {
            'url': paper.pub_url,
            'pdf_url': paper.eprint_url,
            'pdf_hash': paper.pdf_hash,
            'year': paper.year,
            'title': paper.title,
            'venue': paper.venue,
            'conference': paper.conference,
            'url_add_scib': paper.url_add_sclib,
            'bibtex': paper.bibtex,
            'url_scholarbib': paper.url_scholarbib,
            'code': paper.code,
            'eprint_url': paper.eprint_url,
            'num_citations': paper.num_citations,
            'cited_by_url': paper.cited_by_url,
            'url_related_articles': paper.url_related_articles,
            'authors': paper.authors,
            'abstract': paper.abstract,
            'pub_time': paper.pub_date,
            'keywords': paper.keywords,
            'create_time': datetime.datetime.now()
        }

        try:
            with db.mysql_db_new.atomic():

                _, created_paper_info = db.PaperInfo.get_or_create(url=paper.pub_url,
                                                                   defaults=paper_info_data)

                _, created_summary = db.Summaries.get_or_create(pdf_hash=paper.pdf_hash,
                                                                language="中文",
                                                                defaults=summary_data)

                token_cost_all = 0
                # TODO 迁移 向量数据库
                problem_to_ask, problem_tokens = await From_ChunkText_Get_Questions(paper.summary, language='English')
                title, title_zh, basic_info, brief_introduction, token_cost = await get_title_brief_info(
                    first_page=paper.summary,
                    final_res=paper.summary, language="中文")

                token_cost_all += token_cost
                basic_info = re.sub(r'\\+n', '\n', basic_info)
                brief_introduction = re.sub(r'\\+n', '\n', brief_introduction)
                final_sum = re.sub(r'\\+n', '\n', paper.summary)
                title_zh = re.sub(r'\\+n', '\n', title_zh)

                token_cost_all += problem_tokens
                meta_data = json.dumps({
                    "title": paper.title,
                    "title_zh": title_zh,
                    "basic_info": basic_info,
                    "brief_intro": brief_introduction,
                    "summary": final_sum,
                    "problem_to_ask": problem_to_ask
                }, ensure_ascii=False, indent=4)
                pdf_vec, vec_tokens = await embed_text(meta_data)
                token_cost_all += vec_tokens
                summary_data = {
                    'language': "中文",
                    'pdf_hash': paper.pdf_hash,
                    'title': paper.title,
                    'title_zh': title_zh,
                    "basic_info": basic_info,
                    "brief_intro": brief_introduction,
                    'content': final_sum,
                    'create_time': datetime.datetime.now(),
                }
                # 存入 db 和 vec db
                # TODO
                question_obg = db.PaperQuestions.create(
                    pdf_hash=paper.pdf_hash,
                    language='English',
                    question=problem_to_ask,
                    page=0,
                    cost_tokens=problem_tokens
                )
                token_cost_all += problem_tokens
                logger.info(f"insert PaperQuestions, pdf_hash summary:{paper.pdf_hash},cost_tokens={problem_tokens}")

                # TODO 检查是否已经存过，没有才插入
                res = await milvus_PaperDocManager.search_ids_by_pdf_hash(pdf_hash=paper.pdf_hash)
                if res:  # 如果存在，就跳过插入
                    pass
                else:
                    await milvus_PaperDocManager.insert_data(ids=gen_uuid(),
                                                             vecs=pdf_vec,
                                                             pdf_hash=paper.pdf_hash,
                                                             sql_id=question_obg.id)


            logger.info(f'move paper info, summary, title: {paper.title}, pdf_hash: {paper.pdf_hash}')
        except Exception as e:
            logger.error(f"move paper  title: {paper.title}, pdf_hash: {paper.pdf_hash}, error{repr(e)}")


if __name__ == "__main__":
    move_paper_data()
