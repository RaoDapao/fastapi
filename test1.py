import requests

# 定义 API URL 和请求头
url = "http://192.168.20.180:8001/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

# 定义请求数据
data = {
            "model": "qwen2_7b_instruct",
            "messages": [
                {"role": "user", "content": 'sdasdasdas'},
                {"role": "assistant", "content": "会议指出"},
                {"role": "system", "content": "总结文本摘要"}
            ]
        }

# 发送 POST 请求
response = requests.post(url, headers=headers, json=data)

# 打印响应
print(response.json())