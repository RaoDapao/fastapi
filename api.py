import torch
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM

app = FastAPI()

# Load model and tokenizer at startup
model_path = "/home/qwen_intel/code/models/qwen2"
try:
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True).to("xpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")

class RequestData(BaseModel):
    system_setting: str
    user_prompt: str
    max_tokens: int = 1024  # Default max tokens to predict

@app.post("/generate-response/")
async def generate_response(data: RequestData):
    try:
        messages = [
            {"role": "system", "content": data.system_setting},
            {"role": "user", "content": data.user_prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("xpu")
        
        # Warmup generation
        try:
            model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Warmup generation failed: {str(e)}")
        
        # Actual generation with timing
        try:
            st = time.time()
            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=data.max_tokens)
            torch.xpu.synchronize()
            end = time.time()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Actual generation failed: {str(e)}")

        try:
            generated_ids = generated_ids.cpu()
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            inference_time = end - st
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Decoding response failed: {str(e)}")

        return {
            "inference_time": inference_time,
            "response": response
        }
    except HTTPException as e:
        # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
