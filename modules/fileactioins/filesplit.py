import fitz
from loguru import logger
import re

from modules.util import token_str


async def extract_text_from_pdf(path: str) -> str:
    with fitz.open(path) as doc:
        text = ''.join(page.get_text("text") for page in doc)  # Extract text from PDF using PyMuPDF
    logger.info(f"{len(text.split(' '))} original text words")
    return text


def find_references(text: str) -> int:
    # Convert the text to lower case and reverse it
    text = text.lower()[::-1]
    # Search for the word "references" from the end of the document
    ref_index = text.find("secnerefer")
    if ref_index == -1:
        # The word "references" was not found
        return -1
    else:
        # "references" was found, check the following text for a common reference format
        following_text = text[:ref_index][::-1]
        if re.search(r'\n\[\d+\]', following_text):
            # A common reference format was found after "references"
            return len(text) - ref_index
        elif re.search("\S\s\(\d+\)", following_text):
            # A common reference format was found after "references"
            return len(text) - ref_index
        else:
            # A common reference format was not found after "references",
            # continue searching from the current position
            return find_references(text[ref_index+10:])

def clip_text_by_reference(text: str) -> str:
    reference_index = find_references(text)  # Find the index of the "References" section
    if reference_index > 0:
        text = text[:reference_index]  # Clip the text up to the "References" section
    logger.info(f"{len(text.split(' '))} clip text words")
    return text


def find_next_section(text: str):
    pattern = r'\n[A-Z]'
    match = re.search(pattern, text)
    if match:
        return match.start()
    else:
        return 0


def replace_newlines(text: str) -> str:
    # Replace with your actual implementation
    # This function should replace newline characters in the text
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

async def split_text(
        text: str,
        max_token: int = 512) -> list[str]:
    """
    拆分文本
    """
    string_steps = [40, 80, 160, 320]

    # with fitz.open(path) as doc:
    #     text = ''.join(page.get_text("text") for page in doc)
    # logger.info(f"{len(text.split(' '))} original text words")

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

# async def split_text(text: str, max_token: int = 2560) -> list[str]:
#     string_steps = [800, 400, 200]  # Steps to incrementally split the text
#     num_split = token_str(text) // max_token + 1
#     logger.info(f"{num_split} split chunks")
#     res = []
#     # 接下来将text分割成num_split个部分, 且每个部分之间的间隔是按照\n 大写字母来分割的。
#     for index in range(num_split):
#         # 每个部分都从第一个字符开始
#         split_str_index = 0
#         # 当切分字符小于最大的字符数时，继续切分
#         while split_str_index < len(text):
#             # 逐步切分，每次切分的字符数是string_steps中的一个值
#             for step in string_steps:
#                 # 下一次切割的字符数，是step和剩下的字符数中的最小值
#                 next_str_step = min(step, len(text) - split_str_index)
#                 # 先加上下一次切割的字符数，看看是否超过了最大的字符数
#                 temp_text = text[:split_str_index + next_str_step]
#                 # 如果没有超过，那么就把本次切割字符数加上去
#                 if token_str(temp_text) <= max_token:
#                     # 把切割字符数加上去，跳出循环，进行下一次切割
#                     split_str_index += next_str_step
#                     break
#             # 如果切割字符数超过了最大的字符数，那么就把上一次的切割字符数作为切割的字符数
#             else:
#                 # 到了最大的切割字符，需要补齐一下最近的一个段落索引。
#                 temp_split_str_index = find_next_section(text[split_str_index:])
#                 # 先测测加上这个字符数是否超过了最大的字符数
#                 temp_text = text[:split_str_index + temp_split_str_index]
#                 # 如果没有超过，那么就把补齐的索引加上
#                 if token_str(temp_text) <= max_token + 100:
#                     # 把切割字符数加上去，跳出循环，进行下一次切割
#                     split_str_index += temp_split_str_index
#                     # 如果超过了，那么就不加了，直接跳出循环
#                 break
#         res.append(text[:split_str_index])
#         # 把text更新成剩下的部分
#         text = text[split_str_index:]
#
#     return temp_paper_split


async def get_paper_split_res(path: str, max_token: int = 2560):
    """
    将paper拆分的算法
    """
    text = await extract_text_from_pdf(path)
    clipped_text = clip_text_by_reference(text)
    processed_text = replace_newlines(clipped_text)
    split_res = await split_text(processed_text, max_token)
    return split_res
