import requests

class ApiClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.endpoint = "/generate-response/"
    
    def generate_response(self, system_setting, user_prompt, max_tokens=32):
        url = self.base_url + self.endpoint
        data = {
            "system_setting": system_setting,
            "user_prompt": user_prompt,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # 如果响应状态码不是200，则引发HTTPError异常
            result = response.json()
            return result
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # HTTP错误
            return None
        except Exception as err:
            print(f"Other error occurred: {err}")  # 其他错误
            return None

# 使用示例
if __name__ == "__main__":
    client = ApiClient("http://192.168.20.180:8000")
    #之后如果需要可以使用不同的端口运行不同的模型，现在只有一个qwen2
    # 定义请求参数
    system_setting = "你是取名字的专家，取男名，王姓"
    user_prompt = '易经 楚辞 诗经里各找4个冷门句子有美好寓意的，谦逊之意，五行需要水支援，缩写成两个字'

    max_tokens = 500
    
    # 调用API并获取响应

    for i in range(100):
        result = client.generate_response(system_setting, user_prompt, max_tokens)
        
        if result:
            print(f"Inference time: {result['inference_time']} s")
            print(f"Response: {result['response']}")
