import requests
import json
import os

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

def format_memory_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"

def print_and_store_result(result, total_xpu_memory, output_file):
    current_memory = result['xpu_memory_usage'].get('allocated_bytes.all.current', 0)
    peak_memory = result['xpu_memory_usage'].get('allocated_bytes.all.peak', 0)

    output_data = {
        "inference_time": result['inference_time'],
        "response": result['response'],
        "input_tokens": result['input_tokens'],
        "memory_usage": {
            "total": format_memory_size(result['memory_usage']['total']),
            "available": format_memory_size(result['memory_usage']['available']),
            "percent": f"{result['memory_usage']['percent']}%",
            "used": format_memory_size(result['memory_usage']['used']),
            "free": format_memory_size(result['memory_usage']['free'])
        },
        "xpu_memory_usage": {
            "allocated_bytes_all_current": format_memory_size(current_memory),
            "allocated_bytes_all_peak": format_memory_size(peak_memory)
        }
    }
    
    with open(output_file, 'a') as f:
        f.write(json.dumps(output_data, ensure_ascii=False, indent=4) + '\n')
    
    print(json.dumps(output_data, ensure_ascii=False, indent=4))

# 使用示例
if __name__ == "__main__":
    client = ApiClient("http://192.168.20.180:8000")
    
    # 定义请求参数
    system_setting = "扮演老教授，经常给出markdown格式的数学公式来解决问题"
    user_prompt = '什么是随机svd,给出数学的严格证明,证明为啥有效，同时给出python代码来实现'
    max_tokens = 2000
    total_xpu_memory = 16 * 1024 * 1024 * 1024  # 16GB 假设总的XPU显存为16GB
    output_file = "api_results.json"

    if os.path.exists(output_file):
        os.remove(output_file)  # 删除现有文件，以便每次运行时创建一个新文件
    
    # 调用API并获取响应
    for i in range(7):
        result = client.generate_response(system_setting, user_prompt, max_tokens)
        
        if result:
            print_and_store_result(result, total_xpu_memory, output_file)
        else:
            print("Failed to get a response from the API.")
