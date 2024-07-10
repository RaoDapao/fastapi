import torch
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM
import psutil
import os

app = FastAPI()

# Load model and tokenizer at startup
model_path = "/home/qwen_intel/code/models/qwen2"



def load_model_and_tokenizer(model_path):
    print("Loading model and tokenizer...")  # Debug print
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        optimize_model=True,
        trust_remote_code=True,
        use_cache=True, 
        device_map="auto",
        max_memory={0: "15GB"},  # 为 XPU 设置最大内存使用量    
    ).to("xpu")
    print("Model loaded.")  # Debug print
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,  
    )
    print("Tokenizer loaded.")  # Debug print
    return model, tokenizer

print("Starting up and loading model/tokenizer...")  # Debug print
model, tokenizer = load_model_and_tokenizer(model_path)
print("Model and tokenizer loaded at startup.")  # Debug print

class RequestData(BaseModel):
    system_setting: str
    user_prompt: str
    max_tokens: int = 1024  # Default max tokens to predict

@app.post("/generate-response_1.5b/")
async def generate_response(data: RequestData):
    print("Received request.")  # Debug print
    
    messages = [
        {"role": "system", "content": data.system_setting},
        {"role": "user", "content": data.user_prompt}
    ]
    print("Messages prepared: ", messages)  # Debug print

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Text after applying chat template: ", text)  # Debug print
    model_inputs = tokenizer([text], return_tensors="pt",truncation=True, max_length=2000).to("xpu")
    print("Model inputs prepared: ", model_inputs)  # Debug print
    print("Model inputs prepared and moved to XPU.")  # Debug print
    
    # Warmup generation
    try:
        print("Starting warmup generation...")  # Debug print
        model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
        print("Warmup generation done.")  # Debug print
    except Exception as e:
        print("Error during warmup generation: ", str(e))  # Debug print
        raise HTTPException(status_code=500, detail="Warmup generation failed: " + str(e))

    # Actual generation with timing
    try:
        print("Starting actual generation...")  # Debug print
        
        st = time.time()
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
        torch.xpu.synchronize()
        end = time.time()
        print("Generation done.")  # Debug print

        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        inference_time = end - st

        # Monitor memory usage
        mem_info = psutil.virtual_memory()
        xpu_memory_info = torch.xpu.memory_stats()
        
        print("Response generated.")  # Debug print
        print("Inference time: ", inference_time)  # Debug print
        print("Memory usage: ", mem_info)  # Debug print
        print("XPU memory usage: ", xpu_memory_info)  # Debug print

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
        print("Error during actual generation: ", str(e))  # Debug print
        raise HTTPException(status_code=500, detail="Generation failed: " + str(e))

# Run the app
if __name__ == "__main__":
    print("Starting the app...")  # Debug print
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("App is running.")  # Debug print


