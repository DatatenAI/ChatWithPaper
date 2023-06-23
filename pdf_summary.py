import asyncio
import json
import os
import re
from pathlib import Path
from typing import Callable

import aiofiles
import fitz
import math
from loguru import logger

# 开发测试
from dotenv import load_dotenv

from modules.chatmodel.openai_chat import chat_paper_api

if os.getenv('ENV') == 'DEV':
    load_dotenv()

from modules.vectors.get_embeddings import embed_text, From_ChunkText_Get_Questions
from modules.fileactioins.filesplit import get_paper_split_res
from modules.util import retry, token_str, gen_uuid, save_to_file, load_from_file, print_token, save_data_to_json, \
    load_data_from_json


async def Extract_Brief_Introduction(text: str, language: str) -> tuple:
    """
    将brief introduction 提取出来
    """
    logger.info("start translating paper summary")
    convo_id = "read_paper_brief_intro:" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are excellently extract the brief introduction with concise language and keep the same format."
    )
    logger.info(f"input title:{text}")
    logger.info(f"origin title length:{len(text)}")
    content = f"""Original Paper summary is as follows: 
    {text}
    please extract brief introduction it into {language}. 
    Remember to:
    - Retain proper nouns in original language.
    - Retain authors in original language.
    - keep the key result.
    - summarize it within 200 words.
    """
    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    chat_paper_api.conversation[convo_id] = None
    print_token("get_paper_summary_translation", result)
    logger.info("end get paper title translation")
    return result[0], result[3]


def truncate_title(title: str):
    """
    让title不要超过一定长度
    """
    if len(title) <= 180:
        return title
    else:
        truncated_title = title[:180] + '...'
        return truncated_title


async def From_BasicInfo_Extract_title(text: str, language: str) -> tuple:
    """
    将title翻译为中文
    """
    logger.info("start translating paper summary")
    convo_id = "translate_paper_title:" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are extracted the title with concise language and keep the same format."
    )
    logger.info(f"input title:{text}")
    logger.info(f"origin title length:{len(text)}")
    content = f"""Original Paper basic info is as follows: {text}, please extracted the title in {language}. 
    Remember to:
    - Retain proper nouns in original language.
    - Retain authors in original language.
    - Do not keep the 'Title:' prefix
    """
    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    chat_paper_api.conversation[convo_id] = None
    print_token("extract_paper_title", result)
    logger.info("end extract paper title")
    return truncate_title(result[0]), result[3]


async def From_FirstPage_Extract_BasicInfo(text: str, language: str) -> tuple:
    """
    从第一页的信息中提取出 Basic Info
    """
    logger.info("start Extract paper basic")
    convo_id = "Extract_paper_basic:" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are extracted the basic information with concise language and keep the same format."
    )
    logger.info(f"input title:{text}")
    logger.info(f"origin title length:{len(text)}")
    content = f"""Original Paper basic info is as follows: {text}, please extracted the basic info in {language}. 
        Remember to:
        - Retain proper nouns in original language.
        - Retain authors in original language.

        Organize your response using the following markdown structure:
        
        # Basic Information:
        - Title: xxx
        - Authors: xxx
        - Affiliation: xxx
        - Keywords: xxx
        - URLs: xxx or xxx , xxx
        """
    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    chat_paper_api.conversation[convo_id] = None
    print_token("extract_paper_basic_info", result)
    logger.info("end extract paper basic")
    return result[0], result[3]


async def get_title_brief_info(first_page: str, final_res: str, language: str = '中文') -> tuple:
    """
    从第一页信息里面 提取 title, brief info
    """
    # TODO 从第一页信息里面提取
    logger.info("start get paper final basic info")

    # 尝试处理三次
    async def process_text(func: Callable[[str, str], object], text: str,
                           language: str) -> tuple:
        res = await retry(3, func, text=text, language=language)
        if isinstance(res, tuple):
            return res
        else:
            raise Exception("No summary found")

    async def process_basic():
        res_basic = await process_text(From_FirstPage_Extract_BasicInfo, first_page, "中文")
        basic_info = res_basic[0]
        token_cost = res_basic[1]
        return basic_info, token_cost

    # 定义并发执行的协程函数
    async def process_title_zh():
        res_title_zh = await process_text(From_BasicInfo_Extract_title, first_page, "中文")
        title_zh = res_title_zh[0]
        token_cost = res_title_zh[1]
        return title_zh, token_cost

    async def process_title():
        res_title = await process_text(From_BasicInfo_Extract_title, first_page, "English")
        title = res_title[0]
        token_cost = res_title[1]
        return title, token_cost

    async def process_brief_intro():
        content = await process_text(Extract_Brief_Introduction, final_res, language)
        brief_intro = content[0]
        token_cost = content[1]
        return brief_intro, token_cost

    # 并行执行协程任务
    basic_task = process_basic()
    title_zh_task = process_title_zh()
    title_task = process_title()
    brief_intro_task = process_brief_intro()
    # 使用 asyncio.gather() 并行运行协程任务
    results = await asyncio.gather(basic_task, title_zh_task, title_task, brief_intro_task)
    # 提取各任务的结果
    basic_info, token_cost_basic = results[0]
    title_zh, token_cost_title_zh = results[1]
    title, token_cost_title = results[2]
    brief_intro, token_cost_intro = results[3]
    # 计算总的 token_cost
    token_cost = token_cost_basic + token_cost_title_zh + token_cost_title + token_cost_intro

    logger.info(
        f"end get paper title brief , title_zh:{title_zh}, title:{title}, brief intro:{brief_intro}, token_cost:{token_cost}")
    if title and title_zh and basic_info and brief_intro:
        return title, title_zh, basic_info, brief_intro, token_cost
    else:
        raise Exception("No summary found")
    pass


async def read_str_files(file_path: str) -> str:
    """
    读取txt文件内容为字符串
    """
    # read file
    if Path(file_path).is_file():
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                data = await f.read()
            return data
        except Exception as e:
            logger.error(f"read {file_path} error {e}")
            raise f"read {file_path} error {e}"
    else:
        return ""


async def save_str_files(file_data: str, file_path: str):
    """
    保存str内容为字符串txt
    """
    # save file title_zh_path
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(file_data)



async def get_the_formatted_summary_from_pdf(
        pdf_file_path: str,
        language: str = "Chinese",
        summary_temp: str = 'default',
) -> tuple:
    logger.info(f"start summary pdf,path:{pdf_file_path},language:{language}")
    base_path, ext = os.path.splitext(pdf_file_path)
    pdf_hash = base_path.replace('\\', '/').split('/')[-1]

    first_page_path = f"{base_path}.firstpage_conclusion.{language}.{summary_temp}.txt"
    format_path = f"{base_path}.formatted.{language}.{summary_temp}.txt"
    title_path = f"{base_path}.title.{language}.{summary_temp}.txt"
    title_zh_path = f"{base_path}.title_zh.{language}.{summary_temp}.txt"
    basic_info_path = f"{base_path}.basic_info.{language}.{summary_temp}.txt"
    brief_intro_path = f"{base_path}.brief.{language}.{summary_temp}.txt"
    token_path = f"{base_path}.tokens.{language}.{summary_temp}.txt"
    pdf_vec_path = f"{os.getenv('FILE_PATH')}/out/{pdf_hash}.json"

    token_cost_all = 0
    if not os.path.isfile(format_path):  # 如果不存在
        logger.info(f"{format_path} formatted txt not exists")
        try:
            complete_sum_res, firstpage_conclusion, token_cost = await get_the_complete_summary(
                pdf_file_path, language=language, summary_temp=summary_temp)  # 信息压缩
        except Exception as e:
            logger.error(f"get the complete summary error: {e}")
            raise Exception(str(e))
        token_cost_all += token_cost
        if complete_sum_res is None:
            logger.error(f"complete summary is None")
            raise Exception("summary is None")
        complete_sum_res = str(complete_sum_res)
        if len(str(complete_sum_res)) < 200:
            raise Exception("summary is None")
        summary_res, token_cost = await get_paper_final_summary(
            text=complete_sum_res,
            language=language)
        # 将格式重新处理

        token_cost_all += token_cost
        if not isinstance(summary_res, str):
            raise Exception("No summary found")

        if not os.path.isfile(first_page_path) or os.path.getsize(first_page_path) < 50:
            raise Exception("No conclusion found")

        # 如果不够稳定的话，可以用chat做完整的替代，这个成本应该不高，但是应该比较慢
        # 在这里将基本信息和总结信息拼接起来：      
        # 我们需要先将基本信息的标题单独提取出来；
        # 从basic_info 中提取中文title, brief intro
        summary_res = re.sub(r'\\+n', '\n', summary_res)
        # TODO 报错处理
        title, title_zh, basic_info, brief_intro, token_cost = await get_title_brief_info(
            first_page=firstpage_conclusion,
            final_res=summary_res, language="中文")
        token_cost_all += token_cost

        basic_info = re.sub(r'\\+n', '\n', basic_info)
        brief_intro = re.sub(r'\\+n', '\n', brief_intro)
        final_res = re.sub(r'\\+n', '\n', summary_res)
        title_zh = re.sub(r'\\+n', '\n', title_zh)


        if Path(pdf_vec_path).is_file():
            pdf_vec_data = await load_data_from_json(pdf_vec_path)
            pdf_vec = pdf_vec_data['vectors']
            vec_tokens = pdf_vec_data['tokens']
            token_cost_all += vec_tokens
            logger.info(f"load pdf vec:{pdf_vec_path}")
        else:
            # 向量化
            # 对这篇文章可能问的问题
            problem_to_ask, problem_tokens = await From_ChunkText_Get_Questions(final_res, language='English')
            token_cost_all += problem_tokens
            meta_data = json.dumps({
                "title": title,
                "title_zh": title_zh,
                "basic_info": basic_info,
                "brief_intro": brief_intro,
                "summary": final_res,
                "problem_to_ask": problem_to_ask
            }, ensure_ascii=False, indent=4)
            pdf_vec, vec_tokens = await embed_text(meta_data)
            token_cost_all += vec_tokens
            # 存入json文件中
            pdf_vec_info = {
                "pdf_hash": pdf_hash,
                "vectors": pdf_vec,
                "problem_to_ask": problem_to_ask,
                "tokens": vec_tokens
            }
            await save_data_to_json(pdf_vec_info, pdf_vec_path)  # 存储单篇文章的向量化内容
        # 在这儿存最终的总结文本信息：

        # save file title_path
        await save_str_files(title, title_path)
        await save_str_files(title_zh, title_zh_path)
        await save_str_files(basic_info, basic_info_path)
        await save_str_files(brief_intro, brief_intro_path)
        await save_str_files(final_res, format_path)
        await save_str_files(str(token_cost_all), token_path)

        return title, title_zh, basic_info, brief_intro, firstpage_conclusion, final_res, pdf_vec, token_cost_all

    else:  # 如果format 文件存在
        logger.info(f"{format_path} formatted txt exists")

        if not os.path.isfile(first_page_path) or os.path.getsize(first_page_path) < 50:
            raise Exception("No conclusion found")

        title = await read_str_files(title_path)
        title_zh = await read_str_files(title_zh_path)
        basic_info = await read_str_files(basic_info_path)
        brief_intro = await read_str_files(brief_intro_path)
        firstpage_conclusion = await read_str_files(first_page_path)
        final_res = await read_str_files(format_path)
        token_cost_all = await read_str_files(token_path)
        token_cost_all = int(0 if token_cost_all == '' else int(token_cost_all))

        if Path(pdf_vec_path).is_file():
            pdf_vec_data = await load_data_from_json(pdf_vec_path)
            pdf_vec = pdf_vec_data['vectors']
            vec_tokens = pdf_vec_data['tokens']
            token_cost_all += vec_tokens
            logger.info(f"load pdf vec:{pdf_vec_path}")
        else:
            # 向量化
            # 对这篇文章可能问的问题
            problem_to_ask, problem_tokens = await From_ChunkText_Get_Questions(final_res, language='English')
            token_cost_all += problem_tokens
            meta_data = json.dumps({
                "title": title,
                "title_zh": title_zh,
                "basic_info": basic_info,
                "brief_intro": brief_intro,
                "summary": final_res,
                "problem_to_ask": problem_to_ask
            }, ensure_ascii=False, indent=4)
            pdf_vec, vec_tokens = await embed_text(meta_data)
            token_cost_all += vec_tokens
            # 存入json文件中
            pdf_vec_info = {
                "pdf_hash": pdf_hash,
                "vectors": pdf_vec,
                "problem_to_ask": problem_to_ask,
                "tokens": vec_tokens
            }
            await save_data_to_json(pdf_vec_info, pdf_vec_path)  # 存储单篇文章的向量化内容


        return title, title_zh, basic_info, brief_intro, firstpage_conclusion, final_res, pdf_vec, token_cost_all


async def get_the_complete_summary(pdf_file_path: str, language: str, summary_temp: str = 'default') -> tuple:
    """
    处理返回全部总结和第一页的总结
    """
    logger.info("start get complete summary")
    base_path, ext = os.path.splitext(pdf_file_path)
    new_path = f"{base_path}.complete.{language}.{summary_temp}.txt"  # 完整的文本内容
    first_page_path = f"{base_path}.firstpage_conclusion.{language}.{summary_temp}.txt"
    result = None
    token_cost_all = 0
    # 开始处理长文本内容。
    if not os.path.isfile(new_path) or os.path.getsize(new_path) < 1000:  # 如果不存在或者小于 1000 字节
        result, first_page_info, token = await rewrite_paper_and_extract_information(
            pdf_file_path, language=language)
        token_cost_all += token
        # save file complete
        await save_str_files(result, new_path)
        await save_str_files(first_page_info, first_page_path)
    elif not os.path.isfile(first_page_path) or os.path.getsize(first_page_path) < 100:
        sentences = await get_paper_split_res(pdf_file_path)  # 将paper内容拆分
        if len(sentences) == 0:
            raise Exception("there is no text in the paper")
        # 当选择16K模型时，则不需要压缩：
        # tasks = [process_information(sentences[0], pdf_file_path, language=language)]
        # result, token_cost = await asyncio.gather(*tasks)
        result = "\n".join(sentences)
        token_cost_all += 0
    # read file
    result = await read_str_files(new_path)
    first_page_info = await read_str_files(first_page_path)
    logger.info(f"end get complete summary,token_cost_all: {token_cost_all}")
    return result, first_page_info, token_cost_all


async def rewrite_paper_and_extract_information(path: str, language: str) -> tuple:
    # 先开始压缩全文信息
    logger.info(f"start rewrite paper and extract,path:{path}")
    sentences = await get_paper_split_res(path, max_token=2048) # 基本翻了4倍
    if len(sentences) == 0:
        raise Exception("there is no text in the paper")
    sentences_length = len(sentences)
    logger.info(f"sentences length: {sentences_length}")
    # tasks = [
    #     process_sentence(sentence, sentences_length) for sentence in sentences
    # ]
    # tasks.append(process_information(sentences[0], path, language=language))
    #
    # results = await asyncio.gather(*tasks)
    sentence_tasks = [process_sentence(sentence, sentences_length) for sentence in sentences]
    information_task = process_information(sentences[0], path, language=language)

    # 使用 asyncio.gather() 并行执行两个任务
    results, informations = await asyncio.gather(asyncio.gather(*sentence_tasks), information_task)
    if information_task is None or results[0] is None:
        logger.error(f"process_information error return None")
        raise f"process_information error return None"

    # 解包获取每个任务的结果

    rewrite_str = ""
    token_cost = 0
    for result in results:
        if isinstance(result, tuple):
            rewrite_str += '\n' + result[0]
            token_cost += result[1]
    token_cost += informations[1]
    logger.info("end rewrite paper and extract")
    return rewrite_str, informations[0], token_cost


async def get_paper_final_summary(text: str,
                                  language: str):
    logger.info("start get paper final summary")

    # 尝试处理三次
    async def process_text(func: Callable[[str, str], object], text: str,
                           language: str) -> tuple:
        res = await retry(3, func, text=text, lang=language)
        if isinstance(res, tuple):
            return res
        else:
            raise Exception("No summary found")

    content = await process_text(get_paper_summary, text, language)

    res = content[0]
    token_cost = content[1]
    logger.info(f"end get paper final summary,res:{res}, token_cost:{token_cost}")
    if res is not None:
        res = str(res)
        res = re.sub(r'\\+n', '\n', res)
        return res, token_cost
    else:
        raise Exception("No summary found")


async def get_paper_summary(text, lang: str) -> tuple:
    """
    汇总后总结的内容
    """
    logger.info("start get paper summary")
    convo_id = "read_paper_summary" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are skilled at summarizing academic papers using concise language."
    )
    logger.info(f"input text:{text}")
    logger.info(f"origin text length:{len(text)}")
    text = truncate_text(text)
    logger.info(f"truncate_text length:{len(text)}")
    print("real_lang:", lang)
    if lang == "中文":
        content = f"""When summarizing the text, focus on providing clear and concise information. Highlight key 
        concepts, techniques, and findings, and ensure the response is well-structured and coherent. Use the following 
        markdown structure, replacing the xxx placeholders with your answer, Use a scholarly response in {lang}, 
        maintaining proper academic language:
        
    Start summarizing the rest of the story (including Methods, Results section.), Output Format as follows:
        
    # 方法:
    - a. 理论背景:
            - xxx
    - b. 技术路线:
            - xxx
            - xxx
            - xxx
            
    # 结果:
    - a. 详细的实验设置:
            - xxx
    - b. 详细的实验结果: 
            - xxx

    # Note:
    - 本总结源自于LLM的总结，请注意数据判别. Power by ChatPaper.org .

    Please analyze the following original text and generate the response based on it:
    Original text:
    {text}
    Remember to:
    - Retain proper nouns in English.
    - Methods shoould be as detailed as possible, and introduce Technical route step by step if necessary. 
    - Results should be as specific as possible, keep specific nouns and values.
    - When output, never output the contents of () and () of Output Format.
    - Ensure that the response is well-structured, coherent, and addresses all sections.
    - Make sure that every noun and number in your summary is already in your Original text. Then, organize the input text better.
    - Unless necessary information, please note that the output does not repeat the previous content and information.
    - Use a scholarly response in {lang}, maintaining proper academic language and make sure the output is easier to read.
    """
    else:
        content = f"""When summarizing the text, focus on providing clear and concise information. Highlight key 
        concepts, techniques, and findings, and ensure the response is well-structured and coherent. Use the following 
        markdown structure, replacing the xxx placeholders with your answer, Use a scholarly response in {lang}, 
        maintaining proper academic language:
        
    Start summarizing the rest of the story (including Methods, Results section.), Output Format as follows:
        
    # Methods:
    - a. Theoretical basis of the study:
            - xxx
    - b. Technical route of the article (step by step):
            - xxx
            - xxx
            - xxx
            
    # Results:
    - a. Experimental settings in detail:
            - xxx
    - Experimental results in detail: 
            - xxx

    # Note:
    - 本总结源自于LLM的总结，请注意数据判别. Power by ChatPaper.org .

    Please analyze the following original text and generate the response based on it:
    Original text:
    {text}
    Remember to:
    - Retain proper nouns in English.    
    - Methods shoould be as detailed as possible, and introduce Technical route step by step if necessary. 
    - Results should be as specific as possible, keep specific nouns and values.
    - When output, never output the contents of () and () of Output Format.
    - Ensure that the response is well-structured, coherent, and addresses all sections.
    - Make sure that every noun and number in your summary is already in Original text. Then, organize the input text better.
    - Unless necessary information, please note that the output does not repeat the previous content and information.
    - Use a scholarly response in {lang}, maintaining proper academic language and make sure the output is easier to read.
    """

    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    chat_paper_api.conversation[convo_id] = None
    print_token("get_paper_summary", result)
    logger.info("end get paper summary")
    return result[0], result[3]


async def translate_summary(text, lang: str) -> tuple:
    logger.info("start translating paper summary")
    convo_id = "read_paper_summary" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are translating the summary with concise language and keep the same format."
    )
    logger.info(f"input text:{text}")
    logger.info(f"origin text length:{len(text)}")
    text = truncate_text(text)
    logger.info(f"truncate_text length:{len(text)}")
    content = f"""Original Summary is as follows: {text}, please translate it into {lang}. 
    Remember to:
    - Retain proper nouns in original language.
    - Retain authors in original language.
    """
    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    chat_paper_api.conversation[convo_id] = None
    print_token("get_paper_summary_translation", result)
    logger.info("end get paper summary translation")
    return result[0], result[3]

async def translate_one_field(text, lang: str, field: str) -> tuple:
    logger.info(f"start translating {field}")
    convo_id = f"translate_paper_{field}_{str(gen_uuid())}"

    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        f"You are a research scientist and you are translating the {field} of a research paper with concise language and keep the same format."
    )
    logger.info(f"input text:{text}")
    logger.info(f"origin text length:{len(text)}")
    text = truncate_text(text)
    logger.info(f"truncate_text length:{len(text)}")
    content = f"""Original {field} is as follows: {text}, please translate it into {lang}. 
    Remember to:
    - Retain proper nouns in original language.
    - Retain authors in original language.
    """
    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    chat_paper_api.conversation[convo_id] = None
    print_token(f"get_paper_{field}_translation", result)
    logger.info(f"end get paper {field} translation")
    return result[0], result[3]

async def translate(texts: dict, lang: str) -> dict:
    translated_texts = {}
    for field, text in texts.items():
        translated_text = await translate_one_field(text, lang, field)
        translated_texts[field] = translated_text[0]

    return translated_texts

def truncate_text(text, max_token=2560, steps=None):
    if steps is None:
        steps = [800, 400, 200]
    split_pos = 0
    while split_pos < len(text):
        for step in steps:
            next_step = min(step, len(text) - split_pos)
            temp_text = text[:split_pos + next_step]
            if token_str(temp_text) <= max_token:
                split_pos += next_step
                break
        else:
            break
    return text[:split_pos]


async def process_information(sentence: str, path: str, language: str):
    """
    处理基本信息
    """
    try:
        result = await retry(
            3,
            conclude_first_page_information,
            path=path,
            text=sentence,
            language=language
        )
        return result
    except Exception as e:
        logger.error(f"process information error,path {path},err: {e}")
        return "", 0


async def conclude_first_page_information(path: str, text: str,
                                          language: str,
                                          summary_temp: str = 'default') -> tuple:
    """
    总结 title, 第一页的结论
    """
    logger.info(f"start conclude firstpage conclusion information,path:{path}")
    base_path, ext = os.path.splitext(path)
    con_path = f"{base_path}.firstpage_conclusion.{language}.{summary_temp}.txt"
    # title_path = f"{base_path}.title.txt"
    if os.path.isfile(con_path) and os.path.getsize(con_path) > 50:
        # read file
        async with aiofiles.open(con_path, "r", encoding="utf-8") as f:
            text = await f.read()
            text = re.sub(r'\\+n', '\n', text)
            return text, 0

    convo_id = "read_paper_title:" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are skilled at summarizing academic papers using concise language."
    )
    logger.info(f"input text:{text}")
    logger.info(f"origin text length:{len(text)}")

    chat_paper_api.add_to_conversation(
        message=
        "This is the first page of a research paper, and I can help me to read and summarize some questions",
        role="system",
        convo_id=convo_id)
    if language == "中文":
        content = f"""
        This is the first page of a research paper, and I need your help to read and summarize some questions.
        Original text:
        {text}
        Please provide a concise and academic response. Avoid copying and pasting the abstract; instead, expand upon it as needed. Help me complete the following tasks:

            1. Identify the title of the paper (with Chinese translation)
            2. List all authors' names (in English)
            3. Indicate the first author's affiliation (with Chinese translation)
            4. Highlight the keywords of this article (in English)
            5. Provide links to the paper and GitHub code (if available; if not, use "GitHub: None")
            6. You should first summarize this work in "one" sentence! The language should be rigorous, in the style of a popular science writer, including what problems, what methods were used, were solved and what results were achieved. (you should output as {language}!)
            7. Whole research background, output as {language}!
            
    Organize your response using the following markdown structure:

    # Basic Information:

    - Title: xxx
    - Authors: xxx
    - Affiliation: xxx
    - Keywords: xxx
    - URLs: xxx or xxx , xxx

    # 论文简要 :

    - xxx 

    # 背景信息:

    - 论文背景: xxx
    - 过去方案: xxx (Introduce past methods and their problems!)        
    - 论文的Motivation: xxx (Motivation: How does the author move from background knowledge to the research in this paper)

    Remember to:
    - Ensure that the response strictly follows the provided format and does not include any additional content. 
    - Make sure your output is comprehensive and accurate!
    - Retain proper nouns (items, authors) in English, all other output is in {language}.
    - Motivation needs to retain the logic of the Original text.
    - When output, never output the contents of () and () of Output Format.
    - Avoid copying and pasting! Unless necessary information, please note that the new output content does not repeat the previous output content and information.
    - Replace the xxx placeholders with the corresponding information, and maintain line breaks as shown.
    """
    else:
        content = f"""
        This is the first page of a research paper, and I need your help to read and summarize some questions.
        Original text:
        {text}
        Please provide a concise and academic response. Avoid copying and pasting the abstract; instead, expand upon it as needed. Help me complete the following tasks:
            1. Identify the title of the paper (with Chinese translation)
            2. List all authors' names (in English)
            3. Indicate the first author's affiliation (with Chinese translation)
            4. Highlight the keywords of this article (in English)
            5. Provide links to the paper and GitHub code (if available; if not, use "GitHub: None")
            6. You should first summarize this work in "one" sentence! The language should be rigorous, in the style of a popular science writer, including what problems, what methods were used, were solved and what results were achieved. (you should output as {language}!)
            7. Whole research background, output as {language}!
            
    Organize your response using the following markdown structure:

    # Basic Information:

    - Title: xxx
    - Authors: xxx
    - Affiliation: xxx
    - Keywords: xxx
    - URLs: xxx or xxx , xxx

    # Brief introduction :

    - xxx 

    # Background:

    - BackGround: xxx
    - Past methods: xxx (Introduce past methods and their problems!)        
    - Motivation: xxx (Motivation: How does the author move from background knowledge to the research in this paper)

    Remember to:
    - Ensure that the response strictly follows the provided format and does not include any additional content. 
    - Make sure your output is comprehensive and accurate!
    - Retain proper nouns (items, authors) in English, all other output is in {language}.
    - Motivation needs to retain the logic of the Original text.
    - When output, never output the contents of () and () of Output Format.
    - Avoid copying and pasting! Unless necessary information, please note that the new output content does not repeat the previous output content and information.
    - Replace the xxx placeholders with the corresponding information, and maintain line breaks as shown.
    """

    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    print_token("conclude_first_page_information", result)
    chat_paper_api.conversation[convo_id] = None
    res = re.sub(r'\\+n', '\n', str(result[0]))
    # save file
    async with aiofiles.open(con_path, "w", encoding="utf-8") as f:
        await f.write(res)
        logger.info(f"write {con_path}")
    logger.info(f"end conclude first page conclusion information")
    return res, result[3]


async def process_sentence(sentence: str, n: int):
    """
    压缩信息，拆分成n份
    """
    try:
        result = await retry(3,
                             get_condensed_text,
                             text=sentence,
                             n=n)
        return result
    except Exception as e:
        logger.error(f"process sentence error,number:{n},err: {e}")
        return e


async def get_condensed_text(
        text,
        n,
        condensed_length: int = 3000,
        language: str = "English"):
    """压缩信息"""
    text_hash = hash(text)
    logger.info(
        f"{text_hash} get condensed text,n:{n},condensed length:{condensed_length},language:{language}")
    if token_str(text) < 100:
        return text, 0
    convo_id = "core_summary" + gen_uuid()
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt="You are an assistant skilled in extracting the core points of articles for further "
                      "summarization, while preserving the original meaning."
    )
    words_num = math.ceil(condensed_length / n)
    # 需要提取核心的内容，比如设置，方法，过程，数据，结论等。
    content = f"""Based on the current text content {text}, you should determine which section the text belongs to "Methods, Experimental settings, and Experimental details" or others.
    If there is a corresponding section, you should output a specific description; if not, a section is ignored.
    If the text contains the Method section, you need to summarize the method in detail, step by step. 
    If the text contains experimental setting, you need to summarize the experimental setting in detail.
    If the text contains experimental results, you need to summarize the experimental performance according to current text.
    Remember:                                                                    
    - You must keep concise and clear, and output as English.
    - You should maintain the key data, nouns, settings, and other specific valuable information in the original text, and retain the original logic and correspondence.
    - Do not output any specific data when the current text does not exist!
    - In short, make sure your output is comprehensive and accurate!
    - Output as following format:
    Section Name:
        Content.
        
    """

    result = await chat_paper_api.ask(prompt=content,
                                      role="user",
                                      convo_id=convo_id)
    chat_paper_api.conversation[convo_id] = None
    print_token(text_hash, result)
    logger.info(f"{text_hash} end get condensed text")
    return str(result[0]), result[3]


#############################################################################
async def test_translate_summary():
    text = """
    Recommender systems play a vital role in various online services.
However, the insulated nature of training and deploying separately
within a specific domain limits their access to open-world knowl-
edge. Recently, the emergence of large language models (LLMs)
has shown promise in bridging this gap by encoding extensive
world knowledge and demonstrating reasoning capability. Never-
theless, previous attempts to directly use LLMs as recommenders
have not achieved satisfactory results. In this work, we propose
an Open-World Knowledge Augmented Recommendation Frame-
work with Large Language Models, dubbed KAR, to acquire two
types of external knowledge from LLMs — the reasoning knowl-
edge on user preferences and the factual knowledge on items. We
introduce factorization prompting to elicit accurate reasoning on
user preferences. The generated reasoning and factual knowledge
are effectively transformed and condensed into augmented vec-
tors by a hybrid-expert adaptor in order to be compatible with the
recommendation task. The obtained vectors can then be directly
used to enhance the performance of any recommendation model.
We also ensure efficient inference by preprocessing and prestoring
the knowledge from the LLM. Extensive experiments show that
KAR significantly outperforms the state-of-the-art baselines and is
compatible with a wide range of recommendation algorithms.
    """
    res = await translate_summary(text=text, lang="中文")
    print(res)

async def test_translate_one_field():
    # basic info
    text_basic_info = """
    # 基本信息:
- 标题: 密集视频对象字幕生成与不连续监督
- 作者: Xingyi Zhou, Anurag Arnab, Chen Sun, Cordelia Schmid
- 机构: Google Research
- 关键词: 密集视频对象字幕生成, 空间定位, 跟踪, 字幕
- 链接: [论文](https://arxiv.org/ab...
    """
    res = await translate_one_field(text=text_basic_info, lang="English", field="basic info")
    print(res)

    # brief intro
    text_brief_intro = """
    这篇论文介绍了一项名为密集视频对象字幕的新任务，旨在在视频中检测、跟踪和字幕化对象的轨迹。作者提出了一个端到端的模型，包括空间定位、跟踪和字幕化模块。他们通过使用混合的不相交任务和多样化的数据集训练模型，实现了令人印象深刻的零样本性能。此外，作者还重新利用现有的视频定位数据集进行评估和领域特定的微调，证明了他们的模型在这些数据集上的空间定位方面优于最先进的模型。论文在图像字幕和图像中的密集对象字幕方面建立在先前的技术基础上，模型包括区域提议、端到端跟踪和字幕化模块。区域提议模块为每个帧生成无类别的区域提议，然后由跟踪模块将对象分组成轨迹，最后通过自回归语言解码器生成字幕。总的来说，该研究提出的模型在密集视频对象字幕任务上取得了显著的成果。
    """
    res = await translate_one_field(text=text_brief_intro, lang="English", field="basic info")
    print(res)

    # content
    text_content = """
    # 方法:
- a. 理论背景:
    - 本文介绍了一项名为密集视频对象字幕的新任务，涉及在视频中检测、跟踪和字幕化对象的轨迹。作者提出了一个端到端的模型，包括空间定位、跟踪和字幕化模块。他们强调了使用混合的不相交任务和多样化的数据集训练模型的好处，从而实现了令人印象深刻的零样本性能。作者还重新利用现有的视频定位数据集进行评估和领域特定的微调。他们证明了他们的模型在这些数据集上的空间定位方面优于最先进的模型。

- b. 技术路线:
    - 本文在图像字幕和图像中的密集对象字幕方面建立在先前的技术基础上。他们的模型包括区域提议、端到端跟踪和字幕化模块。区域提议模块为每个帧生成无类别的区域提议，然后由跟踪模块将对象分组成轨迹。得到的轨迹被输入到自回归语言解码器中进行字幕生成。作者描述了他们如何使用混合的不相交任务和数据集有效地训练他们的模型，包括COCO、Visual Genome和Spoken Moments in Time。他们还提到了重新利用现有的视频定位数据集进行评估和领域特定的微调。

# 结果:
- a. 详细的实验设置:
    - 本文没有提供关于实验设置的具体细节。

- b. 详细的实验结果: 
    - 本文没有提供关于实验细节的具体细节。

# Note:
- 本总结源自于LLM的总结，请注意数据判别. Power by ChatPaper. End.
    """
    res = await translate_one_field(text=text_content, lang="English", field="basic info")
    print(res)


async def test_Extract_title():
    text = """
# Basic Information:

- Title: Introduction of an exclusive, highly linear, and matrix-effectless analytical method based on dispersive micro solid phase extraction using MIL-88B(Fe) followed by dispersive liquid–liquid microextraction specialized for the analysis of pesticides in celery and tomato juices without dilution
- (论文题目：基于MIL-88B(Fe)的一种独特的、高线性和无基质效应的分散式固相微萃取与分散式液-液微萃取联用的分析方法, 以特定于芹菜和番茄汁中农药分析为例)
- Authors: Sakha Pezhhanfar, Mir Ali Farajzadeh, Navid Mohsen Daraei, Negin Taghipour BaghaliNobar, Seyed Abolfazl Hosseini-Yazdi, Mohammad Reza Afshar Mogaddam
- Affiliation: Department of Analytical Chemistry, Faculty of Chemistry, University of Tabriz, Tabriz, Iran
- Keywords: MIL-88B(Fe), Celery juice, Tomato juice, Gas chromatography, Sample preparation
- URLs: https://doi.org/10.1016/j.microc.2022.107967, GitHub: None

    """
    res, tokens = await From_BasicInfo_Extract_title(text=text, lang="中文")
    print(res)

    res, tokens = await From_BasicInfo_Extract_title(text=text, lang="English")
    print(res)


async def test_extract_brief_info():
    text = """
    # Basic Information: 

- Title: LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS (低秩适应大型语言模型)
- Authors: Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- Affiliation: Microsoft Corporation (微软公司)
- Keywords: Natural Language Processing, Large-scale Pre-training, Adaption, Low-Rank Adaptation, GPT-3
- Urls: https://arxiv.org/abs/2106.09681, https://github.com/microsoft/LoRA.

# Background:
   - a. 主题和特征：
     - GPT-3模型的适应模块和参数共享方法的缺点及其解决方案。
   - b. 历史发展：
     - 为了提高模型性能并减少计算时间，研究人员提出了各种技术策略，如适配器层、输入层激活优化等。
   - c. 过去的方法：
     - 过去的方法主要是使用适配器和参数共享。
   - d. 过去研究的缺点：
     - 过去的方法在某些情况下存在迟滞、可用序列长度降低等缺点。
   - e. 当前需要解决的问题：
     - 当前需要解决如何在满足模型性能的前提下降低计算时间及内存需求。

# Methods:
   - a. 研究的理论基础：
     - LoRA方法基于过参数化模型具有低内在维度的假设。
   - b. 文章的技术路线（分步描述）：
     - 1. 固定预训练模型权重，将可训练秩分解矩阵引入Transformer架构的每一层。
     - 2. 对训练期间的B和A进行训练，而不更新W0。
     - 3.最终将W0和∆W = BA相加以得出h = W0x + ∆Wx。

# Conclusion:
   - a. 工作意义：
     - LoRA方法介绍了一种新的高效方法，在保持模型性能不变的前提下，减少了模型的计算时间和内存需求。
   - b. 创新、表现和工作量：
     - 在RoBERTa、DeBERTa、GPT-2和GPT-3等多个模型上，与适配器相比，LoRA方法具有更快的训练速度、更少的可训练参数以及没有额外的推理延迟。
   - c. 研究结论（列举要点）：
     - LoRA方法显著降低了可训练参数和GPU内存的要求，同时保持或提高了模型质量。并且，在处理自然语言处理中，实证研究了排名缺陷对模型性能的影响。

机器学习模型的开发需要不断地改进，研究人员正在寻找提高模型性能的方法，常常需要在保证质量的同时尽量减少计算成本。在处理自然语言时，为了适应不同场景的任务要求，需要确定一些最佳的超参数。适应模块、优化输入层激活等方法被提出以提升模型性能，但这些方法往往存在计算时间长、内存要求大等问题。然而，使用适配器层也存在一些问题，特别是适配层需要进行序列处理，导致在无模型并行的情况下，模型推理的等待时间增大。

在这种情况下，我们提出了一种新的方法，称为 Low-Rank Adaptation （LoRA），它采用过参数化模型和低秩分解的思想，引入训练期间的可训练秩分解矩阵，并将其冻结，使每个任务需训练的参数大大减少。我们在RoBERTa、DeBERTa、GPT-2和GPT-3等多个模型上进行了实验，在减少了可训练参数和GPU内存的要求的同时，保持或提高了模型质量。

LoRA方法的实施包括用于与PyTorch模型集成的软件包以及提供RoBERTa、DeBERTa和GPT-2的实现和模型检查点。这种方法能够在自然语言处理的任务中更高效地处理序列输入，不仅减少了计算成本，而且提高了模型性能和训练效率。我们的方法介绍了一种新的思想，可以在解决现有问题的同时，为下一步研究提供思路。
    """
    res, tokens = await Extract_Brief_Introduction(text=text, language="中文")
    print(res)


async def test_get_title_brief_info():
    basic_info = """
    # Basic Information:

- Title: Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation（论文笔记之Conv-TasNet：超越理想的时频幅度掩码语音分离）
- Authors: Yi Luo and Nima Mesgarani
- Affiliation: Columbia University, USA（美国哥伦比亚大学）
- Keywords: speech separation, deep learning, Conv-TasNet, time-domain, end-to-end
- URLs: Paper: https://ieeexplore.ieee.org/document/8462045, GitHub: https://github.com/JusperLee/Conv-TasNet-pytorch
    """

    summ = """
    # Basic Information:

- Title: Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation（论文笔记之Conv-TasNet：超越理想的时频幅度掩码语音分离）
- Authors: Yi Luo and Nima Mesgarani
- Affiliation: Columbia University, USA（美国哥伦比亚大学）
- Keywords: speech separation, deep learning, Conv-TasNet, time-domain, end-to-end
- URLs: Paper: https://ieeexplore.ieee.org/document/8462045, GitHub: https://github.com/JusperLee/Conv-TasNet-pytorch

# Summary:
- a. Research background of this article:
  - 传统的时频掩蔽方法具有一些缺陷，如相位和幅度的解耦问题、时频表示次优性和存在延迟等限制，因此需要深入研究解决方案。

- b. Past methods, their problems, and motivation:
  - 过去的研究集中于时频掩蔽方法、使用数据驱动特征提取和设计内置特征提取和分离模块的架构。这些方法在实现语音分离方面表现出了一定的成果，但是表现的稳定性和广泛性需进一步验证。

- c. Research methodology proposed in this paper:
  - 本文作者提出了一种名为Conv-TasNet的全卷积时域音频分离网络，使用线性编码器来生成语音波形的表示形式，并利用卷积分离模块构建掩码，进而实现说话人的分离。

- d. Task and performance achieved by the methods in this paper:
  - 与传统的时频掩蔽方法相比，该方法在分离两个和三个说话人的情况下，表现明显优于以往的方法，并超越了几个理想的时频幅度掩码。此外，Conv-TasNet具有较小的模型尺寸和短的最小延迟，适用于离线和实时语音分离的场景。

# Background:
- a. Subject and characteristics:
  - 本文研究了语音分离算法，其中波形表示期望包含两个或多个重叠语音信号。

- b. Historical development:
  - 传统的时频掩码方法存在相位和幅度解耦问题、时频表示次优性和时间延迟等问题，这也启发了使用深度学习方法来解决语音分离问题。

- c. Past methods:
  - 过去的研究主要集中于时频掩蔽方法、使用数据驱动特征提取和设计内置特征提取和分离模块的架构。

- d. Past research shortcomings:
  - 过去的研究表现的稳定性和广泛性需要进一步的验证。

- e. Current issues to address:
  - 需要提高分离的稳定性和准确性，为语音处理技术的实际应用提供更好的支持。

# Methods:
- a. Theoretical basis of the study:
  - 本文主要基于深度学习的方法来解决语音分离问题，其中的Conv-TasNet网络结构主要包括全卷积时域分离器、线性编码器、分离层、交替可分离卷积层、批量归一化层、平均池化层和波形层。

- b. Technical route of the article (step by step):
  - Conv-TasNet使用卷积编码器对多人讲话的语音信号进行编码，并将编码后的信息再次进行卷积处理得到掩码，并利用掩码将语音信号分离。具体而言，Conv-TasNet网络结构主要包括全卷积时域分离器、线性编码器、分离层、交替可分离卷积层、批量归一化层、平稳池化层和波形层。

# Conclusion:
- a. Significance of the work:
  - 本文提出的Conv-TasNet算法是一种新的实现语音分离的深度学习框架，解决了传统方法的几个缺陷，为实现真实世界语音处理技术的语音分离系统迈出了重要一步。

- b. Innovation, performance, and workload:
  - 与现有方法不同，Conv-TasNet通过全卷积时域分离器和线性编码器的结合，同时处理相位和幅度信息，具有更高的准确性和稳定性。此外，Conv-TasNet的较小模型尺寸和短的最小延迟，使其具有更低的计算负载，能够适应离线和实时语音分离的场景。

- c. Research conclusions (list points):
  - Conv-TasNet算法是一种新的实现语音分离的深度学习框架。
  - Conv-TasNet同时处理相位和幅度信息，具有更高的准确性和稳定性。
  - Conv-TasNet的较小模型尺寸和短的最小延迟，使其具有更低的计算负载，尤其适用于离线和实时语音分离的场景。
  - 未来需要进一步验证Conv-TasNet算法的泛化性能和处理复杂环境的能力。
    """

    res = await get_title_brief_info(basic_info, summ, "中文")
    print(res)


async def test_get_the_formatted_summary_from_pdf():
    pdf_path = '../uploads/0bf316e9c1daea38a8250c2201e42dfc.pdf'
    language = '中文'
    summary_temp = 'default'
    if os.path.exists(pdf_path):
        print("file exists")
    else:
        return None
    res = await get_the_formatted_summary_from_pdf(pdf_path, language, summary_temp=summary_temp)
    print(res)


async def test_conclude_first_page_information():
    text = """
    FAIR: A Causal Framework for Accurately
Inferring Judgments Reversals
Minghua He1(�), Nanfei Gu2, Yuntao Shi1, Qionghui Zhang3, and Yaying
Chen1(�) 1 College of Computer Science and Technology, Jilin University, Changchun, China 2 School of Law, Jilin University, Changchun, China 3 Gould School of Law, University of Southern California, Los Angeles, USA
Abstract. Artificial intelligence researchers have made significant ad- vances in legal intelligence in recent years. However, the existing studies have not focused on the important value embedded in judgments rever- sals, which limits the improvement of the efficiency of legal intelligence.
In this paper, we propose a causal Framework for Accurately Inferring case Reversals (FAIR), which models the problem of judgments rever- sals based on real Chinese judgments. We mine the causes of judgments reversals by causal inference methods and inject the obtained causal re- lationships into the neural network as a priori knowledge. And then, our framework is validated on a challenging dataset as a legal judgment pre- diction task. The experimental results show that our framework can tap the most critical factors in judgments reversal, and the obtained causal relationships can effectively improve the neural network’s performance.
In addition, we discuss the generalization ability of large language models for legal intelligence tasks using ChatGPT as an example. Our experi- ment has found that the generalization ability of large language models still has defects, and mining causal relationships can effectively improve the accuracy and explain ability of model predictions.
Keywords: Legal Intelligence · Causal Inference · Language Processing.
1
Introduction
Legal intelligence is dedicated to assist legal tasks through the application of ar- tificial intelligence. Data resources in the legal field are mainly presented in the form of textual documents, and China has the world’s largest database of judg- ment documents, which can be further explored for its significant value through natural language processing(NLP). In recent years, with the increase of comput- ing power and data scale, deep learning algorithms have developed rapidly and gradually become the mainstream technology of legal intelligence. ChatGPT is a typical large language model(LLM) that has triggered intense discussions, and its generalization ability in the legal field also needs to be studied.
First Author and Second Author contribute equally to this work.
arXiv:2306.11585v1  [cs.CL]  20 Jun 2023 2
M. He et al.
Artificial intelligence researchers have put forth many fruitful efforts in ad- vancing the use of deep learning in legal intelligence. Several works in recent years have contributed very rich legal data resources to the natural language processing community [1,3,21], and these datasets together form the basis of legal intelligence research. Based on these datasets, researchers have designed diverse legal AI tasks based on the practical needs of the legal domain, among which representative tasks include legal judgment prediction (LJP) [17], legal case matching [23], legal entity extraction [3], etc. Based on natural language processing techniques, researchers have developed corresponding solutions for these tasks and applied them in judicial practice.
However, the established work neglects the issue of judgments reversals, which is the area most closely linked to the application of law. According to our statistics, the percentage of revision of judgments reaches 14.63% of all judgments in China, which is a non-negligible part. The problem of judgments reversals is directly related to the direction of application of AI techniques and the effect of models. In the LJP task, extracting the causal relationship in judg- ments reversals as a priori knowledge helps to improve the accuracy as well as interpretability of model prediction.
Although the problem of judgments reversals has important theoretical and practical value, there are major challenges in the research. 1) It is more difficult to model the actual situation of reversals of judgments with high quality. The dif- ficulty of this part of the work is that it is difficult to uncover all the factors that influence the judgment, and it is difficult to quantify and analyze factors such as judges’ subjective will. 2) It is difficult to directly apply the prior knowledge to the improvement of neural networks. How to make neural networks efficiently use prior knowledge from different domains has been one of the challenges of research in artificial intelligence.
In this paper, we propose a causal Framework for Accurately Inferring judg- ments Reversals (FAIR), which mines why revisions occur based on causal infer- ence, which is the process of exploring how one variable T affects another vari- able Y . In the construction of FAIR, first, the causal graph is initially modeled with the help of legal experts by training an encoder to remove the redundant constraints in the graph. Then, the causal effects between different variables are estimated quantitatively using a causal inference algorithm. Finally, the obtained causal knowledge is injected into the neural network model of the downstream task, which can effectively improve the performance of the model.
While the recent rise of Large Language Models (LLMs) has had a huge im- pact on the natural language processing community, we are also interested in the generalization ability of LLMs on legal intelligence tasks. We designed chal- lenging experiments to explore the knowledge exploitation ability and reasoning power of LLMs in the legal domain, and added LLMs as comparisons in the evaluation experiments of the FAIR framework. The experiments reveal some current limitations of LLM and demonstrate that the generalization ability of
LLM can be enhanced by causal knowledge mining and injection.
FAIR 3
Our main contributions are as follows: 1) We propose FAIR, a causal Frame- work for Accurately Inferring judgments Reversals, and better mine the causal relationships in complex legal judgments based on causal inference to uncover the reasons for judgments reversals. 2)The results obtained from performing the
LJP task on a real legal dataset indicate that it is effective to improve the per- formance of neural networks by injecting prior knowledge. 3) We explore the knowledge utilization capability and inference capability of LLM in the legal do- main. By comparing our framework with LLM, we revealed some limitations of
LLM currently existing and proposed ways to improve its generalization ability.
2
Related Work 2.1
Legal Intelligence
Legal Intelligence focuses on applying natural language processing techniques to the legal domain, for which researchers have designed diverse tasks and provided rich data resources. CAIL2018 [21] is a large-scale Chinese legal dataset designed for the LJP task, focusing on LJP in the criminal law domain. LEVEN [22] considers the legal event detection task. FSCS [13] provides multilingual data for the LJP task and studies the legal differences in different regions. LeSICiN [14] designed the law and regulation identification task, using graphs to model the citation network between case documents and legal texts. MSJudge [11] describes a courtroom argument scenario with multi-actor dialogues for the LJP task.
Some work has attempted to provide solutions to the above tasks using natural language processing techniques, and Lawformer [20] has designed a pre-training model for legal text training. EPM [6] considers implicit constraints between events in the LJP task. NSCL [7] attempts to use contrast learning to capture the subtle differences between legal texts in the LJP task. QAjudge [24] uses reinforcement learning to provide interpretable predictions for LJP. However, these works have not taken into account the issue of judgments reversals, which is directly related to the application of the law.
2.2
Causal Inference for Legal Domain
Recent work has attempted to use causal inference to provide more reliable ex- planations and greater robustness for legal intelligence. Liepina [9] introduces a semi-formal causal inference framework to model factual causality arguments in legal cases. Chockler [4] investigates the problem of legal attribution of respon- sibility using causal inference to capture complex causal relationships between multiple parties and events. GCI [10] designs a causal inference framework for unlabeled legal texts, using a graph-based approach to construct causal graphs from factual descriptions. Evan [8] uses causal inference to provide explanations for binary algorithms in legal practice. Law-Match [18] considers the influence of legal provisions in legal case-matching tasks and incorporates them as instru- mental variables in causal graphs. Chen et al [2] investigated the problem of pseudo-correlation error introduced by pre-trained models and eliminated this error by learning the underlying causal knowledge in legal texts.
4
M. He et al.
3
Methodology
Our framework FAIR consists of three main parts, including causal graph mod- eling, estimating causal effects on the modeled causal graph, and injecting causal effects into the neural network. Figure 1 illustrates the structure of FAIR.
Case 
Basic 
Fact
Encoder
Training Encoder
A
B
C
D
Injection Effect to NN 0 1 1 0 0 0.12 0.92 0.2 0.84 3 0.07
Estimation 
Causal Effect
Construct Causal Graph
Fig. 1: Overall structure of FAIR 3.1
Modeling Causal Graph
Preliminary Modeling and Analysis Before conducting a quantitative anal- ysis of causal effects, we need to model the problem based on prior knowledge to ensure the clarity of causal assumptions, and the modeling results are given in the form of a causal graph. We describe the possible causal relationships in the judgment with the help of legal experts as Figure 2(a). However, in Figure 2(a), we cannot directly estimate the causal relationship between "Judgment Basis" and "Case Basic Fact" because there are multiple causal paths between them, and we need to block the paths that are not directly connected. Considering the presence of unobserved confounders in Figure 2(a), we choose the instrumental variable method to block the paths through the confounders, which means that "Case Basic Fact" will be used as an instrumental variable, and it needs to sat- isfy the correlation and exogeneity. To ensure exogeneity, we need to block the direct path from the instrumental variable to the outcome, which means we need to extract the part of the instrumental variable that is relevant to the treatment and not relevant to the outcome, and we do this using a law article prediction task.
Unobserved 
Confounders
Case Basic 
Fact
Judgment 
Basis
Change of 
Judgment (a) Preliminary Causal Graph
Unobserved 
Confounders
Encoded Case 
Basic Fact
Judgment 
Basis
Change of 
Judgment (b) Target Causal Graph
Fig. 2: Preliminary and Target Causal Graph
FAIR 5
Task Definition Given a factual description of the judgment containing n tokens X = {x1, x2, ..., xn} and a set L = {l1, l2, ..., lm} containing m legal entries, we want the model to find a many-to-one mapping F from set X to a subset of L, and the result of the mapping is denoted as an m-dimensional multi- hot vector. This task can be understood as a multi-label classification task.
    """
    path = '../uploads/3047b38215263278f07178419489a887.pdf'
    res = await conclude_first_page_information(path, text, "中文")
    pass


async def test_extract_basic():
    text = """
    # Basic Information:
 - Title: FAIR: A Causal Framework for Accurately Inferring Judgments Reversals (FAIR: 一个用
准确推断判决翻转的因果框架)
- Authors: Minghua He, Nanfei Gu, Yuntao Shi, Qionghui Zhang, Yaying Chen
- Affiliation: College of Computer Science and Technology, Jilin University, Changchun, China
(吉林大学计算机科学与技术学院)
- Keywords: Legal Intelligence, Causal Inference, Language Processing
- URLs: [Paper](https://arxiv.org/abs/2306.11585v1), [GitHub: None]
 # 论文简要 :
 - 本文提出了一个因果框架FAIR,用于准确推断判决翻转。通过因果推断方法挖掘判决翻转的原因,并
获得的因果关系作为先验知识注入神经网络,从而提高模型的性能。
 # 背景信息:
 - 论文背景: 近年来,人工智能研究人员在法律智能方面取得了重要进展。然而,现有研究并未关注判决
转中蕴含的重要价值,这限制了法律智能效率的提高。
- 过去方案: 过去的研究工作忽视了判决翻转的问题,而判决翻转是与法律应用直接相关的领域。判决翻
问题与人工智能技术的应用方向和模型效果直接相关。在法律判决预测任务中,提取判决翻转中的因果
系作为先验知识有助于提高模型预测的准确性和可解释性。
- 论文的Motivation: 本文的动机是解决现有法律智能研究中忽视的判决翻转问题。作者通过因果推断方
挖掘判决翻转的原因,并将获得的因果关系注入神经网络模型,从而提高模型的性能。同时,作者还探
了大型语言模型在法律智能任务中的泛化能力,并发现挖掘因果关系可以有效提高模型预测的准确性和
释能力。
    """
    res = await From_FirstPage_Extract_BasicInfo(text, "中文")
    print(res)


async def test_get_paper_split_res():
    pdf_path = '../uploads/0bf316e9c1daea38a8250c2201e42dfc.pdf'

    # pdf_path = '../uploads/3047b38215263278f07178419489a887.pdf'
    # pdf_path = '../uploads/1c600238d3800bd2d1386ab7943b57ad.pdf'

    res = await get_paper_split_res(pdf_path, max_token=512)
    print(res)


async def test_rewrite_paper_and_extract_information():
    pdf_path = '../uploads/6a2648f178c0c61f0b77e01473f85488.pdf'
    language = '中文'
    summary_temp = 'default'
    if os.path.exists(pdf_path):
        print("file exists")
    else:
        return None
    res = await rewrite_paper_and_extract_information(pdf_path, language)
    print(res)
    pass

async def test_get_the_complete_summary():
    pdf_path = '../uploads/6a2648f178c0c61f0b77e01473f85488.pdf'
    language = '中文'
    summary_temp = 'default'
    if os.path.exists(pdf_path):
        print("file exists")
    else:
        return None
    res = await get_the_complete_summary(pdf_path,language)
    print(res)

async def test_get_condensed_text():
    text = """
    Towards Explainable Evaluation Metrics for Machine Translation
Christoph Leiter1, Piyawat Lertvittayakumjorn2, Marina Fomicheva3,
Wei Zhao4, Yang Gao5, Steffen Eger1 1Bielefeld University, 2Imperial College London, 3University of Sheffield 4Heidelberg Institute for Theoretical Studies, 5Royal Holloway, University of London
Abstract
Unlike classical lexical overlap metrics such as BLEU, most current evaluation metrics for machine translation (for example, COMET or BERTScore) are based on black-box large language models. They often achieve strong correlations with human judgments, but re- cent research indicates that the lower-quality classical metrics remain dominant, one of the potential reasons being that their decision processes are more transparent. To foster more widespread acceptance of novel high-quality metrics, explainability thus becomes crucial. In this concept paper, we identify key properties as well as key goals of explainable machine translation metrics and provide a comprehensive synthesis of recent techniques, relating them to our established goals and properties. In this context, we also discuss the latest state-of-the-art approaches to explainable metrics based on generative models such as Chat-
GPT and GPT4. Finally, we contribute a vision of next-generation approaches, including natural language explanations. We hope that our work can help catalyze and guide future research on explainable evaluation metrics and, mediately, also contribute to better and more transparent machine translation systems.
1
Introduction
The field of evaluation metrics for Natural Language Generation (NLG), especially machine translation (MT) is in a crisis (Marie et al. 2021).1 Despite the development of multiple high- quality evaluation metrics in recent years (Zhao et al. 2019; Zhang et al. 2020a; Rei et al.
2020; Sellam et al. 2020; Yuan et al. 2021), the Natural Language Processing (NLP) community appears hesitant to adopt them for assessing NLG systems (Marie et al. 2021; Gehrmann et al.
2023). Empirical investigations of Marie et al. (2021) indicate that the majority of MT papers relies on surface-level evaluation metrics such as BLEU and METEOR (Papineni et al. 2002;
Banerjee and Lavie 2005), which were created two decades ago, a trend that may allegedly have worsened in recent times. These surface-level metrics cannot (even) measure semantic similarity of their inputs and are thus fundamentally flawed, particularly when it comes to assessing the quality of recent state-of-the-art MT systems (e.g. Peyrard 2019; Freitag et al. 2022), raising concerns about the credibility of the scientific field. We argue that the potential reasons for this neglect of recent high-quality metrics include: (i) non-enforcement by reviewers; (ii) easier comparison to previous research, for example, by copying BLEU-based results from tables of related work (potentially a pitfall in itself); (iii) computational inefficiency to run expensive new metrics at large scale; (iv) lack of trust in and transparency of high-quality black box metrics.
In this work, we concern ourselves with the last-named reason, and address the explainability of such metrics.
1We published an earlier version of this paper (Leiter et al. 2022a) under a different title. Both versions consider the conceptualization of explainable metrics and are overall similar. However, while the old version contains several novel experiments, this new version puts a stronger emphasis on the survey of approaches for the explanation of MT metrics including the latest LLM based approaches. For example, this comprises techniques that return fine-grained error labels or natural language explanations.
1 arXiv:2306.13041v1  [cs.CL]  22 Jun 2023
In recent years, explainability has become a crucial research area in AI due to its potential benefits for users, designers, and developers of AI systems (Samek et al. 2018; Vaughan and
Wallach 2021).2
For users of the AI systems, explanations help them make more informed decisions (es- pecially in high-stake domains) (Sachan et al. 2020; Lertvittayakumjorn et al. 2021), better understand and hence gain trust of the AI systems (Pu and Chen 2006; Toreini et al. 2020), and even learn from the AI systems to accomplish tasks more successfully (Mac Aodha et al. 2018;
Lai et al. 2020). For AI system designers and developers, explanations allow them to identify the problems and weaknesses of the systems (Krause et al. 2016; Han et al. 2020), calibrate the confidence of the systems (Zhang et al. 2020b), and improve them accordingly (Kulesza et al.
2015; Lertvittayakumjorn and Toni 2021).
Explainability is particularly desirable for evaluation metrics.
Sai et al. (2022) suggest that explainable NLG metrics should focus on providing more information than just a single score (such as fluency or adequacy). Celikyilmaz et al. (2020) stress the need for explainable evaluation metrics to spot system quality issues and to achieve a higher trust in the evaluation of NLG systems. Explanations indeed play a vital role in building trust for new evaluation metrics.3 For instance, one potential reason that a metric will likely be better accepted by the research community is if the explanations of the scores align well with human reasoning and faithfully reflect its internal decision process (see §2.2).
By contrast, if faithfulness is given and the explanations are counter-intuitive, users and developers might lower their trust and be alerted to take additional actions, such as trying to improve the metrics using insights from the explanations or looking for alternative metrics that are more trustworthy. Furthermore, explainable metrics can be used for other purposes: for example, when a metric produces a low score for a given input, highlighted words (a widely used method for explanation, see §4) in the input are natural candidates for manual post-editing.
This concept paper aims at providing a systematic overview of the existing efforts in ex- plainable MT evaluation metrics and an outlook for promising future research directions. The main focus of this work lies on explainable evaluation metrics for MT, many of our observations and discussions can likely be adapted to other NLG tasks, however.
We make the following contributions, outlined by the structure of our paper: §2 Background & Terminology: We give an overview of concepts important for explain- able MT evaluation. We also highlight different goals and the audiences that require them.
§4 Taxonomy: We provide a structured overview of previous efforts in explainable MT evaluation. An overview is shown in Figure 4. The process of selecting works for this taxonomy is described in §3 Literature Review & Selection.
§5 Future Work: An analysis of underexplored research directions , and a collection of exemplary methods from other NLP domains to illustrate potential future paths. Here, we also discuss the usage of large language models (LLMs), like ChatGPT (OpenAI 2023b) and GPT4 (OpenAI 2023a).
We provide an overview of Related Work in §6 and a Conclusion in §7.
2As of now, there is no universally accepted definition of explainability in the AI community. In this work, we adapt the definition by Barredo Arrieta et al. (2020), which we discuss in §2.2.
3As an illustrating example, Moosavi et al. (2021) distrust using the metric BERTScore (Zhang et al. 2020a) applied to a novel text generation task, as it assigns a score of 0.74 to a nonsensical output, which could be taken as an unreasonably high value. While BERTScore may indeed be unsuitable for their novel task, the score of 0.74 is meaningless here, as evaluation metrics may have arbitrary ranges. In fact, BERTScore typically has a particularly narrow range, so a score of 0.74 even for bad outputs may not be surprising (BERTScore provides a scaling method that solves these issues to some degree). Explainability techniques would be very helpful in preventing such misunderstandings, e.g. by showing that the features BERTScore attends to align with human judgement.
2
With this work, we aim to solidify the field of explainable MT metrics and provide guidance for researchers, in specific metric developers, who aim to better understand and explain MT metrics, as well as users that want to employ explainability techniques to explain metric outputs.
In the long term, we envision that our work will aid the development of improved MT metrics and thereby help to improve the quality of machine translations. Potentially, explanations can also aid other use cases such as translation selection and semi-automatic labeling (see §2.3). We also hope that our work will more generally inspire explainable NLG metric design.
2
Background & Terminology
In this section, we first introduce and relate definitions and dimensions of MT metrics and explainability, which we use in the later parts of the paper. Further, we collect goals and target audiences of explainable MT metrics.
2.1
Machine Translation Evaluation Metrics
MT metrics grade machine translated content, the hypothesis, based on ground truth data. We differentiate these metrics along several dimensions. A summary is shown in Table 1. We note that metrics can be categorized along many dimensions (e.g. Sai et al. 2022; Celikyilmaz et al.
2020). Here, we focus on a minimal specification that we will leverage later. Some parts of this section refer to explainability (see definitions in §2.2).
Dimension
Description
Input type
Whether source, reference translation or both are used as ground truth for comparison
Granularity
At which level a metric operates: Word-level, sentence-level, document- level
Quality aspect
What a metric measures: Adequacy, fluency, etc.
Learning objective
How a metric is induced: Regression, ranking, etc.
Table 1: A typology of categorizations for evaluation metrics.
Input type
We call a metric reference-based if it requires one or multiple human reference translations to compare with the hypothesis as ground truth, which can be seen as a form of supervision (e.g. Papineni et al. 2002; Zhao et al. 2019; Rei et al. 2022a). Reference-free metrics do not require a reference translation to grade the hypothesis (e.g. Zhao et al. 2020;
Ranasinghe et al. 2020b; Belouadi and Eger 2023). Instead, they directly compare the source to the hypothesis. In the literature, reference-free MT evaluation is sometimes also referred to as “reference-less” (e.g. Mathur et al. 2020) or “quality estimation” (e.g. Zerva et al. 2022). This dimension is important for explainable MT evaluation, as explainability techniques might need to consider which resources are available.
Granularity
Translation quality can be evaluated at different levels of granularity: word-level, sentence-level and document-level. The majority of metrics for MT return a single sentence-level score for each input sentence (e.g. Zhao et al. 2019; Zhang et al. 2020a). Beyond individual sentences, a metric may also score whole documents (multiple sentences) (Jiang et al. 2022;
Zhao et al. 2023). In the MT community, metrics that evaluate translations at the word-level (for example, whether individual words are correct or not) are also common (Turchi et al.
2014; Shenoy et al. 2021). Recently, there has been a tendency to compare metrics to human scores derived from fine-grained error annotations — multidimensional quality metrics (MQM) 3 (Lommel et al. 2014) — as they tend to correspond better to human professional translators (Freitag et al. 2021a). Metrics of higher granularity (e.g. word-level) provide more explanatory value than those of lower granularity, as they provide more details on which parts of the input might be translated incorrectly (e.g. Leiter 2021; Fomicheva et al. 2021, 2022). Most techniques we describe in §4 target the explainability of sentence-level metrics.
    """
    res = await get_condensed_text(text,3)
    print(res)

if __name__ == '__main__':
    # asyncio.run(test_translate_summary())
    asyncio.run(translate_one_field())
    # asyncio.run(test_Extract_title())

    # 测试第一页
    # asyncio.run(test_get_the_complete_summary())

    #
    # asyncio.run(test_rewrite_paper_and_extract_information())

    # 测试压缩信息
    # asyncio.run(test_get_condensed_text())

    # 测试拆分
    # asyncio.run(test_get_paper_split_res())

    # 提取 brief intro
    # asyncio.run(test_extract_brief_info())

    #  test_extract_brief_info
    # asyncio.run(test_get_title_brief_info())

    # 从 first page info 提取 basic info
    # asyncio.run(test_extract_basic())

    #   测试总结first page info
    # asyncio.run(test_conclude_first_page_information())

    # 测试全部总结
    # asyncio.run(test_get_the_formatted_summary_from_pdf())

    pass
