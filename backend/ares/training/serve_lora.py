"""
ARES Phase 4: Native PEFT LoRA Inference Server.
Serves the fine-tuned prompt optimizer adapter directly from local GPU VRAM
exposing an OpenAI/Ollama-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Ensure backend root on sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ares.training.model_registry import _SAFE_ORG_ID_RE, validate_org_id  # noqa: E402, F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEFAULT_ADAPTER_DIR = backend_dir / "ares" / "training" / "artifacts" / "ares_prompt_optimizer_lora"

model = None
tokenizer = None
_model_lock = threading.Lock()
_load_state: str = "idle"
_load_error: Optional[str] = None
_loaded_adapters: Dict[str, str] = {}
_active_adapter: str = "default"
_cli_config: Dict[str, Any] = {
    "auto_load": False,
    "base_model": DEFAULT_BASE_MODEL,
    "adapter_dir": DEFAULT_ADAPTER_DIR,
}


def _background_load_model() -> None:
    """Load the base model and LoRA adapter asynchronously so HTTP binds immediately."""
    global _load_state, _load_error
    _load_state = "loading"
    _load_error = None
    try:
        load_fine_tuned_engine(
            base_model_name=str(_cli_config.get("base_model") or DEFAULT_BASE_MODEL),
            adapter_dir=Path(_cli_config.get("adapter_dir") or DEFAULT_ADAPTER_DIR),
        )
        _load_state = "ready"
    except Exception as exc:
        logger.error("Failed background LoRA model load: %s", exc, exc_info=True)
        _load_error = str(exc)
        _load_state = "error"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if _cli_config.get("auto_load") and model is None and _load_state != "loading":
        loader_thread = threading.Thread(
            target=_background_load_model,
            daemon=True,
            name="lora-model-loader",
        )
        loader_thread.start()
    yield


app = FastAPI(title="ARES Fine-Tuned LoRA Optimizer Engine", version="1.1.0", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "ares-optimizer-lora"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
    org_id: Optional[str] = None


class AdapterLoadRequest(BaseModel):
    org_id: str
    adapter_dir: Optional[str] = None


def _safe_org_adapter_dir(org_id: str) -> Optional[Path]:
    try:
        cleaned = validate_org_id(org_id)
    except ValueError:
        return None
    models_root = (backend_dir / "models").resolve()
    candidate = (models_root / cleaned / "compression_adapter").resolve()
    if models_root not in candidate.parents:
        return None
    return candidate


def resolve_and_activate_adapter(org_id: Optional[str]) -> str:
    """Resolve and activate a per-client PEFT adapter under `_model_lock` when available."""
    global _active_adapter
    if model is None or not hasattr(model, "set_adapter"):
        return "default"

    if not org_id or org_id == "default":
        if "default" in _loaded_adapters:
            try:
                model.set_adapter("default")
                _active_adapter = "default"
            except Exception as exc:
                logger.debug("Could not switch back to default adapter: %s", exc)
        return "default"

    candidate_dir = _safe_org_adapter_dir(org_id)
    if candidate_dir is None:
        logger.warning("Rejected invalid org_id '%s' for PEFT adapter lookup", org_id)
        return "default"

    if org_id not in _loaded_adapters:
        if (candidate_dir / "adapter_model.safetensors").exists() and hasattr(model, "load_adapter"):
            try:
                model.load_adapter(str(candidate_dir), adapter_name=org_id)
                _loaded_adapters[org_id] = str(candidate_dir)
                logger.info("Loaded per-client PEFT adapter for '%s' from %s", org_id, candidate_dir)
            except Exception as exc:
                logger.warning("Failed loading per-client PEFT adapter for '%s': %s", org_id, exc)

    if org_id in _loaded_adapters:
        try:
            model.set_adapter(org_id)
            _active_adapter = org_id
            return org_id
        except Exception as exc:
            logger.warning("Failed activating per-client adapter '%s': %s", org_id, exc)

    if "default" in _loaded_adapters:
        try:
            model.set_adapter("default")
            _active_adapter = "default"
        except Exception:
            pass
    return "default"


@app.get("/healthz")
def healthz():
    resolved_status = (
        "ready"
        if model is not None
        else (_load_state if _load_state in ("loading", "error") else "loading")
    )
    return {
        "status": resolved_status,
        "model": "ares-optimizer-lora",
        "active_adapter": _active_adapter,
        "loaded_adapters": sorted(_loaded_adapters.keys()),
        "error": _load_error,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.post("/v1/adapters/load")
def load_client_adapter_endpoint(req: AdapterLoadRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Base LoRA model is not ready yet")
    default_dir = _safe_org_adapter_dir(req.org_id)
    if default_dir is None:
        raise HTTPException(status_code=400, detail=f"Invalid organization ID '{req.org_id}'")
    target_dir = Path(req.adapter_dir).resolve() if req.adapter_dir else default_dir
    if not (target_dir / "adapter_model.safetensors").exists():
        raise HTTPException(
            status_code=404,
            detail=f"PEFT adapter weights not found at {target_dir}",
        )
    with _model_lock:
        if req.org_id not in _loaded_adapters and hasattr(model, "load_adapter"):
            model.load_adapter(str(target_dir), adapter_name=req.org_id)
        _loaded_adapters[req.org_id] = str(target_dir)
    return {
        "status": "loaded",
        "org_id": req.org_id,
        "adapter_dir": str(target_dir),
        "loaded_adapters": sorted(_loaded_adapters.keys()),
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        detail = f"Model load failed: {_load_error}" if _load_state == "error" and _load_error else "Model is still loading into GPU"
        raise HTTPException(status_code=503, detail=detail)

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

    with _model_lock:
        used_adapter = resolve_and_activate_adapter(req.org_id)
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
        "adapter": used_adapter,
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
    global model, tokenizer, _active_adapter
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
    _loaded_adapters["default"] = str(adapter_dir)
    _active_adapter = "default"
    logger.info("Fine-Tuned LoRA model ready on GPU: %s", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve Fine-Tuned ARES LoRA Model")
    parser.add_argument("--port", type=int, default=11435, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base model")
    parser.add_argument("--adapter-dir", "--adapter-path", dest="adapter_dir", type=str, default=str(DEFAULT_ADAPTER_DIR), help="Adapter directory or path")
    parser.add_argument("--workers", type=int, default=1, help="Worker count")
    args, _ = parser.parse_known_args()

    _cli_config["auto_load"] = True
    _cli_config["base_model"] = args.base_model
    _cli_config["adapter_dir"] = Path(args.adapter_dir)
    uvicorn.run(app, host=args.host, port=args.port)
