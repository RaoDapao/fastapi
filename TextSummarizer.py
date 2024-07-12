import os
import requests
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
from datetime import datetime

class TextSummarizer:
    def __init__(self, base_url, model_name, api_path):
        self.base_url = base_url
        self.model_name = model_name
        self.api_path = api_path
        self.headers = {
            "Content-Type": "application/json"
        }
        self.current_time = datetime.now()
        self.time_string = self.current_time.strftime('%Y_%m_%d_%H_%M_%S')
    
    def call_api(self, messages):
        url = f"{self.base_url}{self.api_path}"
        data = {
            "model": self.model_name,
            "messages": messages
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()
    
    def get_split_text(self, text, chunk_size=5000, chunk_overlap=0, separator="。"):
        text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        text_list = text_splitter.split_text(text)
        return text_list

    def get_summarizer(self, text_list):
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
            response = self.call_api([
                {"role": "user", "content": text_with_prompt},
                {"role": "system", "content": "不要有重复内容，同时尽可能详细准确，序号标对"}
            ])
            answer = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            data_list.append({
                'text_with_prompt': text_with_prompt,
                'answer': answer
            })
        return data_list[-1]['answer']


if __name__ == "__main__":

    
    input_text = """
习近平：高举中国特色社会主义伟大旗帜 为全面建设社会主义现代化国家而团结奋斗——在中国共产党第二十次全国代表大会上的报告
2022-10-25 21:37 来源： 新华社字号：默认 大 超大|打印|     
新华社北京10月25日电

高举中国特色社会主义伟大旗帜
为全面建设社会主义现代化国家而团结奋斗
——在中国共产党第二十次全国代表大会上的报告
（2022年10月16日）
习近平



10月16日，习近平在中国共产党第二十次全国代表大会上作报告。新华社记者 饶爱民 摄

同志们：


"""
    base_url = "http://192.168.20.180:8001"
    model_name = "qwen2_7b_instruct"
    api_path = "/v1/chat/completions"

    '''目前这三个参数是固定的，不能修改'''
    "base_url 这里是内网ip，本机运行的话，直接localhost:8001即可,mu"
    summarizer = TextSummarizer(base_url, model_name, api_path)
    text_list = summarizer.get_split_text(input_text)
    result = summarizer.get_summarizer(text_list)

    
    with open('summary.txt', 'w', encoding='utf-8') as file:
        file.write(result)

print("摘要已保存到summary.txt文件中")