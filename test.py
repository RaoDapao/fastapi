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
    system_setting = "你是一个高级助理，会整理会议内容，并提炼详细的会议纪要"
    user_prompt = "“江医生:大家好，欢视参加今天的心脏病会诊讨论会。我们的主题是的法国红酒看来关于德国士大夫撒旦发射点发射点犯得上广泛的犯得上广泛大使馆顺丰单号给发的黄金分割大师傅大使馆是豆腐干反对公式规范合法的"
    max_tokens = 1000
    
    # 调用API并获取响应
    result = client.generate_response(system_setting, user_prompt, max_tokens)
    
    if result:
        print(f"Inference time: {result['inference_time']} s")
        print(f"Response: {result['response']}")
