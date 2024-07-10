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

def format_memory_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"

def print_and_store_result(result, output_file):
    def format_memory_info(memory_info):
        return {
            "total_memory": format_memory_size(memory_info["total_memory"]),
            "available_memory": format_memory_size(memory_info["available_memory"]),
            "allocated_current": format_memory_size(memory_info["memory_stats"].get("allocated_bytes.all.current", 0)),
            "allocated_peak": format_memory_size(memory_info["memory_stats"].get("allocated_bytes.all.peak", 0))
        }
    
    initial_memory = format_memory_info(result['xpu_memory_usage_initial'])
    warmup_memory = format_memory_info(result['xpu_memory_usage_warmup'])
    before_generation_memory = format_memory_info(result['xpu_memory_usage_before_generation'])
    after_generation_memory = format_memory_info(result['xpu_memory_usage_after_generation'])
    final_memory = format_memory_info(result['xpu_memory_usage_final'])

    output_data = {
        "inference_time": f"{result['inference_time']:.2f} s",
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
            "initial": initial_memory,
            "warmup": warmup_memory,
            "before_generation": before_generation_memory,
            "after_generation": after_generation_memory,
            "final": final_memory
        }
    }
    
    with open(output_file, 'a') as f:
        f.write(json.dumps(output_data, ensure_ascii=False, indent=4) + '\n')
    
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
    for i in range(7):
        result = client.generate_response(system_setting*1*i, user_prompt*1*i, max_tokens*i)
        
        if result:
            print_and_store_result(result, output_file)
        else:
            print("Failed to get a response from the API.")
