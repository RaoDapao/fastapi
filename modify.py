import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import prompt_text
from test_bailian import call_stream_with_messages
from tqdm import tqdm
import pandas as pd
from datetime import datetime

os.environ["OPENAI_API_KEY"] = "your_api_key"
CHAIN_TYPE = 'refine'
current_time = datetime.now()  # 获取当前日期和时间对象
time_string = current_time.strftime('%Y_%m_%d_%H_%M_%S')  # 将日期和时间对象格式化为字符串

def get_split_text(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        state_of_the_union = f.read()

        # 定义文本分割器 每块文本大小为5000，不重叠
        text_splitter = CharacterTextSplitter(
            separator="。",
            chunk_size=5000,
            chunk_overlap=0,
            length_function=len,
        )

    text_list = text_splitter.split_text(state_of_the_union)
    return text_list

def get_summarizer(text_list, model_name):
    answer = ''
    prompt_1 = """针对<正文>内容，撰写总结摘要。
要求：
 - 提取正文的主题
 - 理解核心内容，并重新组织语言形成摘要
 - 在摘要内容中，使用序号罗列要点
 - 使用第三人称
格式：
    会议主题：<>
    会议要点：
        1. <>
        2. <>
        ...
正文：
 {text}
摘要内容："""
    prompt_2 = """请根据<现有摘要>和<补充内容>撰写摘要。
    下面是现有摘要：{existing_answer}
    请根据<补充内容>完善现有摘要，形成一份新的摘要。
    请注意，新的摘要也要提供会议主题，并使用序号罗列要点。
    补充内容如下：
    ------------
    {text}
    ------------
    如果上面的补充内容对撰写摘要没有帮助，则直接返回现有摘要。"""
    
    data_list = []
    for i in tqdm(text_list):
        if answer == '':
            text_with_prompt = prompt_1.replace("{text}", i)
        else:
            text_with_prompt = prompt_2.replace("{text}", i).replace("{existing_answer}", answer)
        answer = call_stream_with_messages(text_with_prompt, model_name)
        data_list.append({
            'text_with_prompt': text_with_prompt,
            'answer': answer
        })
        print(answer)
    return data_list

def summarize_to_string(data_list):
    combined_summary = "\n".join([entry['answer'] for entry in data_list])
    return combined_summary

data_path = "data.text"
model_name_list = ["qwen-max", "qwen1.5-110b-chat", "qwen-72b-chat", "qwen1.5-32b-chat", "qwen1.5-14b-chat", "baichuan2-turbo-192k", "baichuan2-turbo"]
all_res = []
for model_name_i in model_name_list[:1]:
    text_list = get_split_text(data_path)
    data_list = get_summarizer(text_list, model_name_i)
    combined_summary = summarize_to_string(data_list)
    all_res.append({
        'model_name': model_name_i,
        'summarizer': combined_summary
    })

all_summaries = "\n\n".join([res['summarizer'] for res in all_res])
print(all_summaries)
