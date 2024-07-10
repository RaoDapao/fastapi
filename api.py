import torch
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM
import psutil
import os
import logging

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model and tokenizer at startup
model_path = "/home/qwen_intel/code/models/qwen2"

def load_model_and_tokenizer(model_path):
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        optimize_model=True,
        trust_remote_code=True,
        use_cache=True,
        max_memory={0: "15GB"},  # 为 XPU 设置最大内存使用量
        torch_dtype=torch.float16
    ).to("xpu")
    
    device_name = torch.xpu.get_device_name(0)  # 获取设备型号
    logger.info(f"Model loaded on XPU: {device_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    logger.info("Tokenizer loaded.")
    return model, tokenizer

logger.info("Starting up and loading model/tokenizer...")
model, tokenizer = load_model_and_tokenizer(model_path)
logger.info("Model and tokenizer loaded at startup.")

class RequestData(BaseModel):
    system_setting: str
    user_prompt: str
    max_tokens: int = 1024  # Default max tokens to predict

@app.post("/generate-response_1.5b/")
async def generate_response(data: RequestData):
    logger.info("Received request.")
    
    messages = [
        {"role": "system", "content": data.system_setting},
        {"role": "user", "content": data.user_prompt}
    ]
    logger.info(f"Messages prepared: {messages}")

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logger.info(f"Text after applying chat template: {text}")
    model_inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=2000).to("xpu")
    logger.info(f"Model inputs prepared and moved to XPU: {model_inputs}")
    
    # Warmup generation
    logger.info("Starting warmup generation...")
    model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
    logger.info("Warmup generation done.")

    # Actual generation with timing
    try:
        logger.info("Starting actual generation...")
        
        st = time.time()
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
        torch.xpu.synchronize()
        end = time.time()
        logger.info("Generation done.")

        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        inference_time = end - st

        # Monitor memory usage
        mem_info = psutil.virtual_memory()
        xpu_memory_info = torch.xpu.memory_stats()
        
        logger.info("Response generated.")
        logger.info(f"Inference time: {inference_time}")
        logger.info(f"Memory usage: {mem_info}")
        logger.info(f"XPU memory usage: {xpu_memory_info}")

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
        }

    except Exception as e:
        logger.error(f"Error during actual generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Generation failed: " + str(e))

# Run the app
if __name__ == "__main__":
    logger.info("Starting the app...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("App is running.")
