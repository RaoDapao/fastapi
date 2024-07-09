import torch
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM
import psutil

app = FastAPI()

# Load model and tokenizer at startup
model_path = "/home/qwen_intel/code/models/qwen2"

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True).to("xpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_path)

def get_xpu_memory_info():
    properties = torch.xpu.get_device_properties(0)
    total_memory = properties.total_memory
    memory_stats = torch.xpu.memory_stats()
    available_memory = total_memory - memory_stats['allocated_bytes.all.current']
    return {
        "total_memory": total_memory,
        "available_memory": available_memory,
        "memory_stats": memory_stats
    }

class RequestData(BaseModel):
    system_setting: str
    user_prompt: str
    max_tokens: int = 1024  # Default max tokens to predict

@app.post("/generate-response/")
async def generate_response(data: RequestData):
    # Clear memory before processing the request
    torch.xpu.empty_cache()
    print("Cleared XPU memory cache.")
    
    # Check initial memory usage
    initial_memory_info = get_xpu_memory_info()
    print("Initial XPU memory usage:", initial_memory_info)
    
    messages = [
        {"role": "system", "content": data.system_setting},
        {"role": "user", "content": data.user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("xpu")
    
    # Warmup generation
    model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
    warmup_memory_info = get_xpu_memory_info()
    print("XPU memory usage after warmup:", warmup_memory_info)

    # Actual generation with timing
    st = time.time()
    generation_memory_info_before = get_xpu_memory_info()
    print("XPU memory usage before generation:", generation_memory_info_before)
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
    torch.xpu.synchronize()
    end = time.time()
    
    generation_memory_info_after = get_xpu_memory_info()
    print("XPU memory usage after generation:", generation_memory_info_after)

    generated_ids = generated_ids.cpu()
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    inference_time = end - st

    # Monitor memory usage after generation
    mem_info = psutil.virtual_memory()
    xpu_memory_info_final = get_xpu_memory_info()

    return {
        "inference_time": inference_time,
        "response": response,
        "input_tokens": len(model_inputs.input_ids[0]),
        "memory_usage": {
            "total": mem_info.total,
            "available": mem_info.available,
            "percent": mem_info.percent,
            "used": mem_info.used,
            "free": mem_info.free
        },
        "xpu_memory_usage_initial": initial_memory_info,
        "xpu_memory_usage_warmup": warmup_memory_info,
        "xpu_memory_usage_before_generation": generation_memory_info_before,
        "xpu_memory_usage_after_generation": generation_memory_info_after,
        "xpu_memory_usage_final": xpu_memory_info_final
    }
