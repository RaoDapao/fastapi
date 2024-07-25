import requests
import json
import time

url = "http://192.168.20.180:9054/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

input_str = """
习近平：高举中国特色社会主义伟大旗帜 为全面建设社会主义现代化国家而团结奋斗——在中国共产党第二十次全国代表大会上的报告
2022-10-25 21:37 来源： 新华社字号：默认 大 超大|打印|     
新华社北京10月25日电

高举中国特色社会主义伟大旗帜
为全面建设社会主义现代化国家而团结奋斗
——在中国共产党第二十次全国代表大会上的报告
（2022年10月16日）
习近平




同志们！党用伟大奋斗创造了百年伟业，也一定能用新的伟大奋斗创造新的伟业。全党全军全国各族人民要紧密团结在党中央周围，牢记空谈误国、实干兴邦，坚定信心、同心同德，埋头苦干、奋勇前进，为全面建设社会主义现代化国家、全面推进中华民族伟大复兴而团结奋斗！
"""
# Function to split input string into chunks of at most max_length characters
# Function to split input string into chunks of at most max_length characters
# Function to split input string into chunks of at most max_length characters
def split_into_chunks(text, max_length):
    chunks = []
    while len(text) > max_length:
        chunk = text[:max_length]
        last_period = chunk.rfind('。')
        if last_period != -1:
            chunk = chunk[:last_period + 1]
        else:
            chunk = chunk[:max_length]
        chunks.append(chunk)
        text = text[len(chunk):]
    chunks.append(text)
    return chunks

# Function to summarize text chunks
def summarize_chunks(chunks):

    summaries = []

    for chunk in chunks:
        prompt = f"本段文本: {chunk}\n请给出本段总结。非常详细，非常具体，不要说“本段强调了什么”，要以“会议指出”开头，强调人物和地点和时间（如果有）"
        data = {
            "model": "qwen2_7b_instruct",
            "messages": [
                {"role": "user", "content": prompt},
            ]
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_json = response.json()
        summary = response_json["choices"][0]["message"]["content"]
        print(summary,end="\n\n\n\n\n\n")
        summaries.append(summary)

    return ''.join(summaries)

# Main loop to summarize until the final summary is less than or equal to 2500 characters
def iterative_summary(text, max_length=2000):
    chunks = split_into_chunks(text, max_length)
    summary = summarize_chunks(chunks)
    
    while len(summary) > 4000:
        chunks = split_into_chunks(summary, max_length)
        summary = summarize_chunks(chunks)
    
    return summary

start_time = time.time()
final_summary = iterative_summary(input_str)

# Final processing with the full model for the final summary
prompt = f"根据上文内容给出最终总结:\n{final_summary}"
data = {
    "model": "qwen2_7b_instruct",
    "messages": [
        {"role": "user", "content": prompt}
    ]
}
response = requests.post(url, headers=headers, data=json.dumps(data))
response_json = response.json()
final_result = response_json["choices"][0]["message"]["content"]

end_time = time.time()
elapsed_time = end_time - start_time

print("Response Time:", elapsed_time, "seconds")
print(final_result)