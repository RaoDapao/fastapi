import requests
import json
import os

class ApiClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.endpoint = "/generate-response_1.5b/"
    
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

def print_and_store_result(result, output_file):
    output_data = {
        "inference_time": f"{result['inference_time']:.2f} s",
        "response": result['response'],
        "input_tokens": result['input_tokens'],
    }
    
    with open(output_file, 'a') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        f.write('\n')
    
    print(json.dumps(output_data, ensure_ascii=False, indent=4))

# 使用示例
if __name__ == "__main__":
    client = ApiClient("http://192.168.20.180:8000")
    
    # 定义请求参数
    system_setting = "我"
    user_prompt = '你'
    max_tokens = 206
    output_file = "api_results.json"

    if os.path.exists(output_file):
        os.remove(output_file)  # 删除现有文件，以便每次运行时创建一个新文件
    
    # 调用API并获取响应
    for i in range(1, 8):
        result = client.generate_response(system_setting * 100*i, user_prompt * i, max_tokens * i)
        
        if result:
            print_and_store_result(result, output_file)
        else:
            print("Failed to get a response from the API.")
