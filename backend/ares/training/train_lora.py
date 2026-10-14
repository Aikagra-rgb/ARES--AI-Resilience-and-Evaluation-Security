"""
ARES Phase 3: 4-Bit QLoRA Fine-Tuning Engine for Llama 3.1 8B.
Trained on RTX 5050 8GB VRAM using BitsAndBytes NF4 Quantization,
PEFT LoRA, and Hugging Face SFTTrainer.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure backend root on sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INTENT_DATA_DIR = backend_dir / "ares" / "intent" / "data"
TRAIN_DATA_PATH = INTENT_DATA_DIR / "sft_optimizer_train.jsonl"
VAL_DATA_PATH = INTENT_DATA_DIR / "sft_optimizer_val.jsonl"
ARTIFACTS_DIR = backend_dir / "ares" / "training" / "artifacts" / "ares_prompt_optimizer_lora"


def train_lora_adapter(
    base_model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    train_data_path: Path = TRAIN_DATA_PATH,
    val_data_path: Path = VAL_DATA_PATH,
    output_dir: Path = ARTIFACTS_DIR,
    num_epochs: int = 2,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 512,
):
    """
    Executes 4-bit QLoRA fine-tuning on the paired prompt optimizer dataset.
    """
    logger.info("=== ARES Phase 3: QLoRA Fine-Tuning ===")
    logger.info("Base Model: %s", base_model_name)
    logger.info("Train Data: %s", train_data_path)
    logger.info("Output Dir: %s", output_dir)
    logger.info("VRAM Available: %.2f GB", torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0)

    # 1. 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Tokenizer & Model
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading 4-bit quantized base model into GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # 3. PEFT LoRA Configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load Datasets
    logger.info("Loading dataset splits...")
    dataset = load_dataset("json", data_files={"train": str(train_data_path), "validation": str(val_data_path)})

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    )

    # 6. SFTTrainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_seq_length,
    )

    logger.info("Starting training loop...")
    trainer.train()

    # 7. Save Final Adapter
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("=== Training Complete: LoRA adapter saved to %s ===", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARES QLoRA Prompt Optimizer Trainer")
    parser.add_argument("--model", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", help="Base model identifier")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    train_lora_adapter(
        base_model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
    )
