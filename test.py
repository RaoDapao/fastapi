import requests

class ApiClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.endpoint = "/generate-response/"
    
    def generate_response(self, system_setting, user_prompt, max_tokens=1000):
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
    
    # 定义请求参数
    system_setting = "随便扯扯"
    user_prompt = '随便扯扯随便扯扯随便扯扯随便扯扯随便扯扯随便扯扯随便扯扯随便扯扯随便扯扯随便扯扯随便'

    max_tokens = 1000
    
    # 调用API并获取响应

    for i in range(100):
        result = client.generate_response(system_setting, user_prompt, max_tokens)
        
        if result:
            print(f"Inference time: {result['inference_time']} s")
            print(f"Response: {result['response']}")
            print(f"Input Tokens: {result['input_tokens']}")
            print("Memory Usage:")
            print(f"  Total: {result['memory_usage']['total']}")
            print(f"  Available: {result['memory_usage']['available']}")
            print(f"  Percent: {result['memory_usage']['percent']}%")
            print(f"  Used: {result['memory_usage']['used']}")
            print(f"  Free: {result['memory_usage']['free']}")
            print("XPU Memory Usage:")
            print(result['xpu_memory_usage'])
        else:
            print("Failed to get a response from the API.")
