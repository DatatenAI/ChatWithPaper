import asyncio
import os
import re
from typing import Callable

import aiofiles
import fitz
import math
from loguru import logger

import optimize_openai
from util import retry, token_str, gen_uuid


chat_paper_api = optimize_openai.ChatPaperAPI(model_name="gpt-3.5-turbo",
                                              top_p=1,
                                              temperature=0.0,
                                              apiTimeInterval=0.02)


async def get_the_formatted_summary_from_pdf(
        pdf_file_path: str,
        language: str = "Chinese"
) -> tuple:
    logger.info(f"start summary pdf,path:{pdf_file_path},language:{language}")
    base_path, ext = os.path.splitext(pdf_file_path)
    new_path = f"{base_path}.formatted.{language}.txt"
    token_cost_all = 0
    if not os.path.isfile(new_path):
        logger.info(f"{new_path} formatted txt not exists")
        try:
            complete_sum_res, token_cost = await get_the_complete_summary(
                pdf_file_path, language)
        except Exception as e:
            raise Exception(str(e))
        token_cost_all += token_cost
        if complete_sum_res is None:
            raise Exception("summary is None")
        complete_sum_res = str(complete_sum_res)
        if len(str(complete_sum_res)) < 500:
            raise Exception("summary is None")
        summary_res, token_cost = await get_paper_final_summary(
            text=complete_sum_res,
            language=language)
        token_cost_all += token_cost
        if not isinstance(summary_res, str):
            raise Exception("No summary found")
        con_path = f"{base_path}.firstpage_conclusion.txt"
        if not os.path.isfile(con_path) or os.path.getsize(con_path) < 50:
            raise Exception("No conclusion found")
        async with aiofiles.open(con_path, "r", encoding="utf-8") as f:
            basic_info = await f.read()
        final_res = f"{basic_info}\n\n{summary_res}"
        final_res = re.sub(r'\\+n', '\n', final_res)
        async with aiofiles.open(new_path, "w", encoding="utf-8") as f:
            await f.write(final_res)
        return final_res, token_cost_all
    else:
        logger.info(f"{new_path} formatted txt exists")
        async with aiofiles.open(new_path, "r", encoding="utf-8") as f:
            res = await f.read()
            res = re.sub(r'\\+n', '\n', res)
            return res, token_cost_all


async def get_the_complete_summary(pdf_file_path: str, lang: str) -> tuple:
    logger.info("start get complete summary")
    base_path, ext = os.path.splitext(pdf_file_path)
    new_path = f"{base_path}.complete.txt"
    first_page_path = f"{base_path}.firstpage_conclusion.txt"
    result = None
    token_cost_all = 0
    # 开始处理长文本内容。
    if not os.path.isfile(new_path) or os.path.getsize(new_path) < 1000:
        result, token = await rewrite_paper_and_extract_information(
            pdf_file_path, lang=lang)
        token_cost_all += token
        async with aiofiles.open(new_path, "w", encoding="utf-8") as f:
            await f.write(result)
    elif not os.path.isfile(
            first_page_path) or os.path.getsize(first_page_path) < 100:
        sentences = get_paper_split_res(pdf_file_path)
        if len(sentences) == 0:
            raise Exception("there is no text in the paper")
        tasks = [process_information(sentences[0], pdf_file_path, lang=lang)]
        result, token_cost = await asyncio.gather(*tasks)
        token_cost_all += token_cost
    async with aiofiles.open(new_path, "r", encoding="utf-8") as f:
        result = await f.read()
    logger.info(f"end get complete summary,token_cost_all: {token_cost_all}")
    return result, token_cost_all


async def rewrite_paper_and_extract_information(path: str, lang: str) -> tuple:
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
    tasks.append(process_information(sentences[0], path, lang=lang))
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
                if token_str(temp_text) <= max_token+100:
                    # 把切割字符数加上去，跳出循环，进行下一次切割
                    split_str_index += temp_split_str_index                        
                # 如果超过了，那么就不加了，直接跳出循环
                break
        res.append(text[:split_str_index])
        # 把text更新成剩下的部分
        text = text[split_str_index:]  
    return res


async def get_paper_final_summary(text: str,
                                  language: str):
    logger.info("start get paper final summary")

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
    logger.info(f"end get paper final summary,res:{res},token_cost:{token_cost}")
    if res is not None:
        res = str(res)
        res = re.sub(r'\\+n', '\n', res)
        return res, token_cost
    else:
        raise Exception("No summary found")


async def get_paper_summary(text, lang: str) -> tuple:
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
    content = f"""When summarizing the text, focus on providing clear and concise information. Highlight key 
    concepts, techniques, and findings, and ensure the response is well-structured and coherent. Use the following 
    markdown structure, replacing the xxx placeholders with your answer, Use a scholarly response in {lang}, 
    maintaining proper academic language:
    
You should first summarize this work in one sentence, the language should be rigorous, in the style of a popular science writer,
including what methods were used, what problems were solved and what results were achieved. And then start summarizing the rest of the story, Output Format as follows:

# Brief introduction:
   - xxx
      
# Methods:
   - xxx (Theoretical basis of the study)    
   - xxx (Technical route of the article (step by step))
        
# Results:
   - xxx (Experimental settings)        
   - xxx (Experimental results)        

# Conclusion:
   - xxx (Significance of the work)        
   - xxx (Innovation, performance, and workload)        
   - xxx (Research conclusions (list points))        


Please analyze the following original text and generate the response based on it:
Original text:
{text}
Remember to:
- Retain proper nouns in English.
- Do not output vague statements without a specific name or value.
- Methods shoould be as detailed as possible. 
- Motivation needs to retain the logic of the Original text.
- Results should be as specific as possible, keep specific nouns and values.
- When output, never output the contents of () and () of Output Format.
- Ensure that the response is well-structured, coherent, and addresses all sections.
- Make sure that every noun and number in your summary is already in your Original text. Then, organize the input text better.
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
    - Retain authers in original language.
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


async def process_information(sentence: str, path: str, lang: str):
    try:
        result = await retry(
            3,
            conclude_basic_information,
            path=path,
            text=sentence,
            lang=lang
        )
        return result
    except Exception as e:
        logger.error(f"process information error,path {path},err: {e}")
        return e


async def conclude_basic_information(path: str, text: str, lang: str):
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
    content = f"""Please provide a concise and academic response. Avoid copying and pasting the abstract; instead, expand upon it as needed. Help me complete the following tasks:

        1. Identify the title of the paper (with Chinese translation)
        2. List all authors' names (in English)
        3. Indicate the first author's affiliation (with Chinese translation)
        4. Highlight the keywords of this article (in English)
        5. Provide links to the paper and GitHub code (if available; if not, use "GitHub: None")
        
Organize your response using the following markdown structure:

# Basic Information:

- Title: xxx
- Authors: xxx
- Affiliation: xxx
- Keywords: xxx
- URLs: xxx or xxx , xxx

# Background (next, you should output as {lang}):

- xxx (Whole research background).
- xxx (Introduce past methods and their problems)        
- xxx (Motivation: How does the author move from background knowledge to the research in this paper)        

Remember to:
- Ensure that the response strictly follows the provided format and does not include any additional content. 
- Make sure your output is comprehensive and accurate!
- Retain proper nouns in English.
- Motivation needs to retain the logic of the Original text.
- When output, never output the contents of () and () of Output Format.
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
