import asyncio
import os
import re
from typing import Callable

import aiofiles
import fitz
import math
from loguru import logger

# 开发测试
from dotenv import load_dotenv
load_dotenv()

import optimize_openai
from util import retry, token_str, gen_uuid




chat_paper_api = optimize_openai.ChatPaperAPI(
    model_name="gpt-3.5-turbo-16k",
    # model_name="gpt-3.5-turbo",
    top_p=1,
    temperature=0.0,
    apiTimeInterval=0.02)


async def Extract_Brief_Introduction(text: str, language: str) -> tuple:
    """
    将brief introduction 提取出来
    """
    logger.info("start translating paper summary")
    convo_id = "read_paper_summary" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are excellently extract the brief introduction with concise language and keep the same format."
    )
    logger.info(f"input title:{text}")
    logger.info(f"origin title length:{len(text)}")
    content = f"""Original Paper summary is as follows: {text}, please extract brief introduction it into {language}. 
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

def truncate_title(title:str):
    """
    让title不要超过一定长度
    """
    if len(title) <= 180:
        return title
    else:
        truncated_title = title[:180] + '...'
        return truncated_title

async def From_BasicInfo_Extract_title(text: str, lang: str) -> tuple:
    """
    将title翻译为中文
    """
    logger.info("start translating paper summary")
    convo_id = "translate_paper_title" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are extracted the title with concise language and keep the same format."
    )
    logger.info(f"input title:{text}")
    logger.info(f"origin title length:{len(text)}")
    content = f"""Original Paper basic info is as follows: {text}, please extracted the title in {lang}. 
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

async def get_title_brief_info(basic_info:str, final_res:str, language:str='中文')->tuple:
    logger.info("start get paper final basic info")
    # 尝试处理三次
    async def process_text(func: Callable[[str, str], object], text: str,
                           language: str) -> tuple:
        res = await retry(3, func, text=text, lang=language)
        if isinstance(res, tuple):
            return res
        else:
            raise Exception("No summary found")

    token_cost = 0
    res_title_zh = await process_text(From_BasicInfo_Extract_title, basic_info, "中文")
    title_zh = res_title_zh[0]
    token_cost = token_cost + res_title_zh[1]
    res_title = await process_text(From_BasicInfo_Extract_title, basic_info, "English")
    title = res_title[0]
    token_cost = token_cost + res_title[1]

    content = await process_text(get_paper_summary, final_res, language)
    brief_intro = content[0]
    token_cost = token_cost + content[1]

    logger.info(f"end get paper title brief , title_zh:{title_zh}, title:{title}, brief intro:{brief_intro}, token_cost:{token_cost}")
    if title and title_zh and brief_intro:
        return title, title_zh, brief_intro, token_cost
    else:
        raise Exception("No summary found")
    pass

async def get_the_formatted_summary_from_pdf(
        pdf_file_path: str,
        language: str = "Chinese",
        summary_temp: str = 'default',
) -> tuple:
    logger.info(f"start summary pdf,path:{pdf_file_path},language:{language}")
    base_path, ext = os.path.splitext(pdf_file_path)
    new_path = f"{base_path}.formatted.{language}.txt"
    token_cost_all = 0
    if not os.path.isfile(new_path):  # 如果不存在
        logger.info(f"{new_path} formatted txt not exists")
        try:
            complete_sum_res, token_cost = await get_the_complete_summary(
                pdf_file_path, language=language, summary_temp=summary_temp)
        except Exception as e:
            logger.error(f"get the complete summary error: {e}")
            raise Exception(str(e))
        token_cost_all += token_cost
        if complete_sum_res is None:
            logger.error(f"complete summary is None")
            raise Exception("summary is None")
        complete_sum_res = str(complete_sum_res)
        if len(str(complete_sum_res)) < 500:
            raise Exception("summary is None")
        summary_res, token_cost = await get_paper_final_summary(
            text=complete_sum_res,
            language=language)
        # 将格式重新处理

        token_cost_all += token_cost
        if not isinstance(summary_res, str):
            raise Exception("No summary found")
        con_path = f"{base_path}.firstpage_conclusion.txt"
        if not os.path.isfile(con_path) or os.path.getsize(con_path) < 50:
            raise Exception("No conclusion found")
        async with aiofiles.open(con_path, "r", encoding="utf-8") as f:
            basic_info = await f.read()
        # 如果不够稳定的话，可以用chat做完整的替代，这个成本应该不高，但是应该比较慢
        # 在这里将基本信息和总结信息拼接起来：      
        # 我们需要先将基本信息的标题单独提取出来；
        # 从basic_info 中提取中文title, brief intro
        summary_res = re.sub(r'\\+n', '\n', summary_res)
        title, title_zh, brief_intro, token_cost = await get_title_brief_info(basic_info=basic_info, final_res=summary_res,language="中文")
        token_cost_all += token_cost

        final_res = f"{basic_info}\n\n{summary_res}"
        final_res = re.sub(r'\\+n', '\n', final_res)

        # 在这儿存最终的总结文本信息：
        async with aiofiles.open(new_path, "w", encoding="utf-8") as f:
            await f.write(final_res)
        return title, title_zh, basic_info, brief_intro, summary_res, token_cost_all

    else:
        logger.info(f"{new_path} formatted txt exists")
        con_path = f"{base_path}.firstpage_conclusion.txt"
        if not os.path.isfile(con_path) or os.path.getsize(con_path) < 50:
            raise Exception("No conclusion found")
        async with aiofiles.open(con_path, "r", encoding="utf-8") as f:
            basic_info = await f.read()
        async with aiofiles.open(new_path, "r", encoding="utf-8") as f:
            res = await f.read()
            res = re.sub(r'\\+n', '\n', res)
        return basic_info, res, token_cost_all


async def get_the_complete_summary(pdf_file_path: str, language: str, summary_temp: str = 'default') -> tuple:
    logger.info("start get complete summary")
    base_path, ext = os.path.splitext(pdf_file_path)
    new_path = f"{base_path}.complete.txt"
    first_page_path = f"{base_path}.firstpage_conclusion.txt"
    result = None
    token_cost_all = 0
    # 开始处理长文本内容。
    if not os.path.isfile(new_path) or os.path.getsize(new_path) < 1000:    # 如果存在並且小于 1000 字节
        result, token = await rewrite_paper_and_extract_information(
            pdf_file_path, language=language)
        token_cost_all += token
        async with aiofiles.open(new_path, "w", encoding="utf-8") as f:
            await f.write(result)
    elif not os.path.isfile(
            first_page_path) or os.path.getsize(first_page_path) < 100:
        sentences = get_paper_split_res(pdf_file_path)
        if len(sentences) == 0:
            raise Exception("there is no text in the paper")
        # 当选择16K模型时，则不需要压缩：
        # tasks = [process_information(sentences[0], pdf_file_path, language=language)]
        # result, token_cost = await asyncio.gather(*tasks)
        result = "\n".join(sentences)
        token_cost_all += 0
    async with aiofiles.open(new_path, "r", encoding="utf-8") as f:
        result = await f.read()
    logger.info(f"end get complete summary,token_cost_all: {token_cost_all}")
    return result, token_cost_all


async def rewrite_paper_and_extract_information(path: str, language: str) -> tuple:
    # 先开始压缩全文信息
    logger.info(f"start rewrite paper and extract,path:{path}")
    sentences = get_paper_split_res(path)
    if len(sentences) == 0:
        raise Exception("there is no text in the paper")
    sentences_length = len(sentences)
    logger.info(f"sentences length: {sentences_length}")
    tasks = [
        process_sentence(sentence, sentences_length) for sentence in sentences
    ]
    tasks.append(process_information(sentences[0], path, language=language))
    results = await asyncio.gather(*tasks)
    rewrite_str = ""
    token_cost = 0
    for result in results:
        if isinstance(result, tuple):
            rewrite_str += result[0]
            token_cost += result[1]
    logger.info("end rewrite paper and extract")
    return rewrite_str, token_cost


def find_next_section(text):
    pattern = r'\n[A-Z]'
    match = re.search(pattern, text)
    if match:
        return match.start()
    else:
        return 0


def replace_newlines(text):
    punctuation = ".?!"
    uppercase_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []
    i = 0

    while i < len(text) - 1:
        if text[i] == '\n':
            if text[i - 1] not in punctuation and text[i + 1] not in uppercase_letters:
                result.append(' ')
            else:
                result.append(text[i])
        else:
            result.append(text[i])
        i += 1
    result.append(text[-1])
    return ''.join(result)


def get_paper_split_res(
        path: str,
        split_page: int = 2,
        extract_last: bool = True,
        max_token: int = 2560) -> list[str]:
    string_steps = [800, 400, 200]

    with fitz.open(path) as doc:
        text = ''.join(page.get_text("text") for page in doc)
    logger.info(f"{len(text.split(' '))} original text words")
    # clip text by reference
    reference_index = text.lower().find("references")
    if reference_index > 0:
        main_text = text[:reference_index]
    else:
        reference_index = text.find("References")
        main_text = text[:reference_index]
    text = main_text
    logger.info(f"{len(text.split(' '))} clip text words")

    # split by words
    res = []
    num_split = len(text) // max_token
    logger.info(f"{num_split} original num_split")
    # split by tokens
    num_split = token_str(text) // max_token + 1
    logger.info(f"{num_split} token num_split")

    # 接下来将text分割成num_split个部分, 且每个部分之间的间隔是按照\n 大写字母来分割的。
    for index in range(num_split):
        # 每个部分都从第一个字符开始
        split_str_index = 0
        # 当切分字符小于最大的字符数时，继续切分
        while split_str_index < len(text):
            # 逐步切分，每次切分的字符数是string_steps中的一个值
            for step in string_steps:
                # 下一次切割的字符数，是step和剩下的字符数中的最小值
                next_str_step = min(step, len(text) - split_str_index)
                # 先加上下一次切割的字符数，看看是否超过了最大的字符数
                temp_text = text[:split_str_index + next_str_step]
                # 如果没有超过，那么就把本次切割字符数加上去
                if token_str(temp_text) <= max_token:
                    # 把切割字符数加上去，跳出循环，进行下一次切割
                    split_str_index += next_str_step
                    break
            # 如果切割字符数超过了最大的字符数，那么就把上一次的切割字符数作为切割的字符数
            else:
                # 到了最大的切割字符，需要补齐一下最近的一个段落索引。
                temp_split_str_index = find_next_section(text[split_str_index:])
                # 先测测加上这个字符数是否超过了最大的字符数
                temp_text = text[:split_str_index + temp_split_str_index]
                # 如果没有超过，那么就把补齐的索引加上
                if token_str(temp_text) <= max_token + 100:
                    # 把切割字符数加上去，跳出循环，进行下一次切割
                    split_str_index += temp_split_str_index
                    # 如果超过了，那么就不加了，直接跳出循环
                break
        res.append(text[:split_str_index])
        # 把text更新成剩下的部分
        text = text[split_str_index:]

        # 这里将每个
    temp_paper_split = []
    for ps in res:
        # print("original_text:", ps)
        if len(ps) > 1:
            format_text = replace_newlines(ps)
            # print("format_text:", format_text)
            temp_paper_split.append(format_text)
    return temp_paper_split


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
    - 本总结源自于LLM的总结，请注意数据判别. Power by ChatPaper. End.

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
    - 本总结源自于LLM的总结，请注意数据判别. Power by ChatPaper. End.

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
        "You are a research scientist and you are translated the summary with concise language and keep the same format."
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
    try:
        result = await retry(
            3,
            conclude_basic_information,
            path=path,
            text=sentence,
            language=language
        )
        return result
    except Exception as e:
        logger.error(f"process information error,path {path},err: {e}")
        return e


async def conclude_basic_information(path: str, text: str, language: str):
    logger.info(f"start conclude basic information,path:{path}")
    base_path, ext = os.path.splitext(path)
    con_path = f"{base_path}.firstpage_conclusion.txt"
    title_path = f"{base_path}.title.txt"
    if os.path.isfile(con_path) and os.path.getsize(con_path) > 50:
        async with aiofiles.open(con_path, "r", encoding="utf-8") as f:
            text = await f.read()
            text = re.sub(r'\\+n', '\n', text)
            return "", 0
    convo_id = "read_paper_title" + str(gen_uuid())
    chat_paper_api.reset(
        convo_id=convo_id,
        system_prompt=
        "You are a research scientist and you are skilled at summarizing academic papers using concise language."
    )
    chat_paper_api.add_to_conversation(
        message=
        "This is the first page of a research paper, and I need your help to read and summarize some questions: "
        + text,
        role="system",
        convo_id=convo_id)
    if language == "中文":
        content = f"""Please provide a concise and academic response. Avoid copying and pasting the abstract; instead, expand upon it as needed. Help me complete the following tasks:

            1. Identify the title of the paper (with Chinese translation)
            2. List all authors' names (in English)
            3. Indicate the first author's affiliation (with Chinese translation)
            4. Highlight the keywords of this article (in English)
            5. Provide links to the paper and GitHub code (if available; if not, use "GitHub: None")
            6. You should first summarize this work in "one" sentence! The language should be rigorous, in the style of a popular science writer, including what problems, what methods were used, were solved and what results were achieved. (you should output as {lang}!)
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
        content = f"""Please provide a concise and academic response. Avoid copying and pasting the abstract; instead, expand upon it as needed. Help me complete the following tasks:

            1. Identify the title of the paper (with Chinese translation)
            2. List all authors' names (in English)
            3. Indicate the first author's affiliation (with Chinese translation)
            4. Highlight the keywords of this article (in English)
            5. Provide links to the paper and GitHub code (if available; if not, use "GitHub: None")
            6. You should first summarize this work in "one" sentence! The language should be rigorous, in the style of a popular science writer, including what problems, what methods were used, were solved and what results were achieved. (you should output as {lang}!)
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
    print_token("conclude_basic_information", result)
    chat_paper_api.conversation[convo_id] = None
    res = re.sub(r'\\+n', '\n', str(result[0]))
    async with aiofiles.open(con_path, "w", encoding="utf-8") as f:
        await f.write(res)
    async with aiofiles.open(title_path, "w", encoding="utf-8") as f:
        await f.write(res)
    logger.info(f"end conclude basic information")
    return "", result[3]


async def process_sentence(sentence: str, n: int):
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


def print_token(tip, result):
    logger.info(
        f"{tip} prompt used: {str(result[1])} tokens. Completion used: {str(result[2])} tokens. Totally used: {str(result[3])} tokens.")



async def test_translate():
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
    pdf_path = '../uploads/3047b38215263278f07178419489a887.pdf'
    language = '中文'
    summary_temp = 'default'

    res = await get_the_formatted_summary_from_pdf(pdf_path, language, summary_temp=summary_temp)
    print(res)

if __name__ == '__main__':
    # asyncio.run(test_translate())
    # asyncio.run(test_Extract_title())
    # asyncio.run(test_extract_brief_info())

    #
    # asyncio.run(test_extract_brief_info())

    # 测试全部总结
    asyncio.run(test_get_the_formatted_summary_from_pdf())