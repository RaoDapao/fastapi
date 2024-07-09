import torch
import time
import warnings
import multiprocessing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM

app = FastAPI()

# Load model and tokenizer at startup
model_path = "/home/qwen_intel/code/models/qwen2"

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        optimize_model=True,
        trust_remote_code=True,
        use_cache=True
    ).to("xpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_path)

class RequestData(BaseModel):
    system_setting: str
    user_prompt: str
    max_tokens: int = 1024  # Default max tokens to predict

@app.post("/generate-response/")
async def generate_response(data: RequestData):
    messages = [
        {"role": "system", "content": data.system_setting},
        {"role": "user", "content": data.user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("xpu")

    try:
        torch.cuda.empty_cache()  # 清理显存

        # Warmup generation
        _ = model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)

        # Actual generation with timing
        st = time.time()
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
        torch.xpu.synchronize()
        end = time.time()

        torch.cuda.empty_cache()  # 清理显存

        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        inference_time = end - st

        return {
            "inference_time": inference_time,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during response generation: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    # Properly clean up resources if necessary
    for proc in multiprocessing.active_children():
        proc.terminate()
    multiprocessing.active_children()
    # If using any custom resources, add their cleanup logic here
    # Example: closing database connections, releasing file handles, etc.

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")
warnings.filterwarnings("ignore", message="There appear to be .* leaked semaphore objects")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

