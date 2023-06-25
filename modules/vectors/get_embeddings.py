import asyncio
import itertools
import os
import re
from typing import Union, Optional

import fitz
import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from loguru import logger
from dotenv import load_dotenv
import time

import api_key_db
from modules.chatmodel.openai_chat import chat_paper_api
from modules.fileactioins.filesplit import find_next_section, split_text


from modules.util import split_list, print_token, save_data_to_json, load_data_from_json
from modules.util import gen_uuid, retry

if os.getenv('ENV') == 'DEV':
    load_dotenv()

ENCODER = tiktoken.get_encoding("gpt2")


def token_str(content: str) -> int:
    return len(ENCODER.encode(content))


def estimate_embedding_token(path: str) -> int:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    full_text = ""
    with fitz.open(path) as doc:  # type: ignore
        page_cnt = 0
        for page in doc:
            page_cnt += 1
            text = page.get_text("text")
            full_text += f"page:{page_cnt}\n" + text
    return token_str(full_text)


async def From_ChunkText_Get_Questions(text: str, language: str) -> tuple:
    """
    从第一页的信息中提取出 Basic Info
    默认全部用英文就可以
    """
    logger.info("start Extract paper basic")
    convo_id = "ChunkText_GetQuestions:" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are extracted 'the most possible questions reader may to ask' and with concise language and keep the same format."
    )
    logger.info(f"input chunk text:{text}")
    logger.info(f"origin text length:{len(text)}")
    content = f"""Original Paper chunked text is as follows: {text}, please extracted the most 5th possible questions reader may to ask in {language}. 
        Remember to:
        - Retain proper nouns in original language.
        - extracted the most 5th possible keywords for the text

        Organize your response using the following markdown structure:

        # Possible questions to ask and answers:
        - Q1: xxx
        - A1: xxx
        
        - Q2: xxx
        - A2: xxx
        ...
        # keywords
        xxx, xxx,...
        """
    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    chat_paper_api.conversation[convo_id] = None
    print_token("extract_paper_basic_info", result)
    logger.info("end extract paper basic")
    res = re.sub(r'\\+n', '\n', result[0])
    return res, result[3]


async def embed_text(text: str, max_attempts: int = 3) -> tuple:
    attempts = 0
    vector = None
    token_cost = 0
    while attempts < max_attempts:
        try:
            alive_key = api_key_db.get_single_alive_key()
            embeddings = OpenAIEmbeddings(openai_api_key=alive_key)
            vector = embeddings.embed_query(text)
            token_cost = token_str(text)
            logger.info(f"{attempts + 1} try,embed text: {text}")
            break
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            attempts += 1
            if attempts < max_attempts:
                logger.info("Retrying...")
                time.sleep(0.5)  # 等待1秒后重试
            else:
                logger.error("Maximum number of attempts reached. Giving up.")
            raise f"{e}"
    return vector, token_cost


async def StructuredResultsVectors(text: str, language: str = "English") -> tuple:
    try:
        possible_text, struct_tokens = await From_ChunkText_Get_Questions(text, language)
        vector, vec_tokens = await embed_text(possible_text)
        total_tokens = struct_tokens + vec_tokens
        logger.info(f"Structured text:{text},tokens: {total_tokens}")
        return possible_text, vector, total_tokens
    except Exception as e:
        logger.error(f"StructureResultsVectors error {e}")
        return None, None, None


from pydantic import BaseModel


class PDFMetaInfoModel(BaseModel):
    """
    存储PDF拆分的基本信息结构
    """
    id: Union[int, None]
    structure_info: Union[str, None]  # 结构化的信息
    text: Union[str, None]  # 原文文本
    vector: Union[list[float], None]  # 向量
    page: Union[int, None]  # 页码
    tokens: Union[int, None]  # 消耗的token


async def process_pdf_infos(pdf_models: list[PDFMetaInfoModel], language: str = "English") -> list[PDFMetaInfoModel]:
    """
    处理一个list中的pdf models
    """
    logger.info(f"process pdf models")
    final_res = []
    for res in pdf_models:
        pdf_res = await process_pdf_info(res, language=language)
        final_res.append(pdf_res)
    return final_res


async def process_pdf_info(pdf_model: PDFMetaInfoModel, language: str = "English") -> Union[PDFMetaInfoModel, None]:
    """
    压缩信息，拆分成n份
    """
    try:
        result = await retry(3,
                             get_pdf_info,
                             pdf_model=pdf_model,
                             language=language
                             )
        return result
    except Exception as e:
        logger.error(f"process pdf info vec error,err: {e}")
        raise e


async def get_pdf_info(pdf_model: PDFMetaInfoModel, language: str = "English") -> PDFMetaInfoModel:
    possible_questions, tokens_ques = await From_ChunkText_Get_Questions(pdf_model.text, language=language)
    vec, token_cost = await embed_text(possible_questions)
    total_token = tokens_ques + token_cost
    pdf_model.structure_info = possible_questions
    pdf_model.vector = vec
    pdf_model.tokens = total_token
    return pdf_model


async def get_embeddings_from_pdf(path: str, max_token: int = 256) -> tuple:
    infos = []
    metadatas = []
    pdf_hash = path.split('/')[-1].split('.')[0]
    structure_path = os.path.join(os.getenv('FILE_PATH'), f"out/{pdf_hash}_structure.json")

    if os.path.exists(structure_path):  # 如果index存在
        flat_results_json = await load_data_from_json(structure_path)
        flat_results = []
        info_token_cost = 0
        for res in flat_results_json:
            pdf_model = PDFMetaInfoModel()
            pdf_model.id = res['id']
            pdf_model.page = res['page']
            pdf_model.structure_info = res['structure_info']
            pdf_model.vector = res['vectors']
            pdf_model.tokens = res['tokens']
            pdf_model.text = res['text']
            flat_results.append(pdf_model)
            info_token_cost += pdf_model.tokens
        logger.info(f"load embeddings from {structure_path}")
        return flat_results, info_token_cost


    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    full_text = ""
    with fitz.open(path) as doc:  # type: ignore
        page_cnt = 0
        for page in doc:
            page_cnt += 1
            text = page.get_text("text")
            full_text += f"pdf_page:{page_cnt}\n" + text
    # replace multiple spaces with single space
    full_text = re.sub(r"\s+", " ", full_text)
    # remove all leading and trailing spaces
    full_text = full_text.strip()
    # 优化全文切分方案：
    split_res = await split_text(full_text, max_token=max_token)
    last_page = 1
    pdf_models = []
    for info in split_res:
        pdf_model = PDFMetaInfoModel()
        try:
            page_cnt = int(re.findall(r"pdf_page:(\d+)", info)[0])
            pdf_model.page = page_cnt
        except:
            pdf_model.page = last_page
            page_cnt = last_page
        if token_str(info) > 100:
            pdf_model.text = info
            pdf_models.append(pdf_model)
        last_page = page_cnt

    # 提取问题，结构化文本，转换成向量
    # 将文本，页数放在类中，然后返回list,再拆分进线程中进行结构化+向量处理

    # 拆分chunks
    if len(pdf_models) < 30:
        num_chunks = len(pdf_models)
    else:
        num_chunks = int(len(pdf_models) / 30)

    split_pdf_models = split_list(pdf_models, num_chunks)

    pdf_info_tasks = [process_pdf_infos(sentence, language='English') for sentence in split_pdf_models]
    results = await asyncio.gather(*pdf_info_tasks)
    flat_results_raw = list(itertools.chain(*results))
    flat_results_raw = [result for result in flat_results_raw if result is not None]  # 过滤掉None
    flat_results = []
    flag = 0
    for res in flat_results_raw:
        res.id = flag
        flag += 1
        flat_results.append(res)

    info_token_cost = sum(pdf_meta.tokens for pdf_meta in flat_results)
    logger.info(f"Info Token cost: {info_token_cost}")
    logger.info(f"Info length: {len(flat_results)}")
    try:
        # 将list中多个数据存储到json到
        flat_results_json = []
        for res in flat_results:
            flat_results_info = {
                "id": res.id,
                "page": res.page,
                "structure_info": res.structure_info,
                "vectors": res.vector,
                "text": res.text,
                "tokens": res.tokens,
            }
            flat_results_json.append(flat_results_info)

        await save_data_to_json(flat_results_json, structure_path)
    except Exception as e:
        logger.error(f"save {pdf_hash}_structure.pkl fail {e}")

    return flat_results, info_token_cost

    # store = FAISS.from_texts(infos, OpenAIEmbeddings(
    #     openai_api_key=key), metadatas=metadatas)
    # try:
    #     faiss.write_index(store.index, f"{os.getenv('FILE_PATH')}/out/{pdf_hash}.index")
    #     store.index = None
    #     logger.info(f"save {pdf_hash}.index success")
    # except Exception as e:
    #     logger.error(f"save {pdf_hash}.index fail {e}")
    #
    # try:
    #     with open(f"{os.getenv('FILE_PATH')}/out/{pdf_hash}.pkl", "wb") as f:
    #         pickle.dump(store, f)
    #         logger.info(f"save {pdf_hash}.pkl success")
    # except Exception as e:
    #     logger.error(f"save {pdf_hash}.pkl fail {e}")
    # return info_token_cost


async def test_embeedings():
    text = """
    Section Name: Introduction\nContent: The paper introduces a new task called dense video
object captioning, which involves detecting, tracking, and captioning trajectories of objects
in a video. The authors propose an end-to-end model for this task, consisting of modules for
spatial localization, tracking, and captioning. They highlight the benefits of training the
model with a mixture of disjoint tasks and leveraging diverse datasets, resulting in
impressive zero-shot performance. The authors also repurpose existing video grounding
datasets for evaluation and domain-specific finetuning. They demonstrate that their model
outperforms state-of-the-art models for spatial grounding on these datasets.\n\nSection
Name: Related Work\nContent: The paper discusses related work in the areas of image
captioning, dense object captioning, object tracking, and video object grounding. It mentions
the use of large-scale transformer models for image and video captioning, as well as the
popularity of dense object captioning in the Visual Genome dataset. The authors also
highlight the challenges of object tracking in the context of their task and describe the
discriminative approach used in video object grounding. They note that their model,
although not specifically trained for grounding, can still be used for this task.\n\nSection
    """
    possible_questions, tokens = await From_ChunkText_Get_Questions(text, language="English")
    res = await embed_text(possible_questions)

    pass


async def test_spilit_pdf():
    path = '../../../uploads/3047b38215263278f07178419489a887.pdf'
    res = await get_embeddings_from_pdf(path, max_token=512)
    pass


if __name__ == '__main__':
    #  测试 转向量
    # asyncio.run(test_embeedings())

    # 拆分文本
    asyncio.run(test_spilit_pdf())
