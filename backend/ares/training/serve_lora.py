"""
ARES Phase 4: Native PEFT LoRA Inference Server.
Serves the fine-tuned prompt optimizer adapter directly from local GPU VRAM
exposing an OpenAI/Ollama-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Ensure backend root on sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEFAULT_ADAPTER_DIR = backend_dir / "ares" / "training" / "artifacts" / "ares_prompt_optimizer_lora"

app = FastAPI(title="ARES Fine-Tuned LoRA Optimizer Engine", version="1.0.0")

model = None
tokenizer = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "ares-optimizer-lora"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512


@app.get("/healthz")
def healthz():
    return {
        "status": "ready" if model is not None else "loading",
        "model": "ares-optimizer-lora",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading into GPU")

    start_t = time.perf_counter()
    msg_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

    inputs = tokenizer.apply_chat_template(
        msg_dicts,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.2,
            do_sample=(req.temperature or 0.2) > 0.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off prompt tokens to keep only generated response
    gen_tokens = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    return {
        "id": f"chatcmpl-lora-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": len(gen_tokens),
            "total_tokens": input_ids.shape[1] + len(gen_tokens),
        },
        "latency_ms": round(elapsed_ms, 2),
    }


def load_fine_tuned_engine(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_dir: Path = DEFAULT_ADAPTER_DIR,
):
    global model, tokenizer
    logger.info("=== Loading Fine-Tuned ARES Optimizer Model ===")
    logger.info("Base Model: %s", base_model_name)
    logger.info("LoRA Adapter: %s", adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # If adapter_dir doesn't have adapter_model.safetensors directly, check latest checkpoint
    if not (adapter_dir / "adapter_model.safetensors").exists():
        checkpoints = sorted(adapter_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            adapter_dir = checkpoints[-1]
            logger.info("Using latest checkpoint adapter from %s", adapter_dir)

    logger.info("Attaching trained PEFT LoRA adapter weights...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    logger.info("Fine-Tuned LoRA model ready on GPU: %s", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve Fine-Tuned ARES LoRA Model")
    parser.add_argument("--port", type=int, default=11435, help="Server port")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base model")
    parser.add_argument("--adapter-dir", type=str, default=str(DEFAULT_ADAPTER_DIR), help="Adapter directory")
    args = parser.parse_args()

    load_fine_tuned_engine(base_model_name=args.base_model, adapter_dir=Path(args.adapter_dir))
    uvicorn.run(app, host="127.0.0.1", port=args.port)
