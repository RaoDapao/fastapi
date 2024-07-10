import torch
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

app = FastAPI()

# 获取模型加载前的XPU显存使用情况
preload_xpu_memory_info = torch.xpu.memory_stats()

# Load model and tokenizer at startup
model_path = "/home/qwen_intel/code/models/qwen2"

def load_model_and_tokenizer(model_path):
    # Step 1: Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True).to("xpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Step 2: Create custom embedding layer (example: same size as original)
    custom_embedding = torch.nn.Embedding(model.config.vocab_size, model.config.hidden_size).to("xpu")
    
    # Step 3: Replace the model's embedding layer
    model.get_input_embeddings = lambda: custom_embedding
    
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_path)

class RequestData(BaseModel):
    system_setting: str
    user_prompt: str
    max_tokens: int = 1024  # Default max tokens to predict

@app.post("/generate-response_1.5b/")
async def generate_response(data: RequestData):
    messages = [
        {"role": "system", "content": data.system_setting},
        {"role": "user", "content": data.user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("xpu")
    
    # Warmup generation
    model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)

    # Actual generation with timing
    st = time.time()
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
    torch.xpu.synchronize()  # 同步XPU
    end = time.time()

    generated_ids = generated_ids.cpu()
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    inference_time = end - st

    # Monitor memory usage
    mem_info = psutil.virtual_memory()
    xpu_memory_info = torch.xpu.memory_stats()

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
        "xpu_memory_usage": xpu_memory_info,
        "preload_xpu_memory_usage": preload_xpu_memory_info
    }
