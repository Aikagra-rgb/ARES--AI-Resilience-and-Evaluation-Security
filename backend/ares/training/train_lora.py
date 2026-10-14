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
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

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
    max_steps: int = -1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 512,
):
    """
    Executes true 4-bit QLoRA fine-tuning on the paired prompt optimizer dataset,
    modifying actual LoRA model adapter weights.
    """
    logger.info("=== ARES Phase 3: True QLoRA Fine-Tuning ===")
    logger.info("Base Model: %s", base_model_name)
    logger.info("Train Data: %s", train_data_path)
    logger.info("Output Dir: %s", output_dir)
    if torch.cuda.is_available():
        logger.info("GPU: %s (VRAM: %.2f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / (1024**3))
    else:
        logger.warning("CUDA not available! Training will run on CPU.")

    # 1. Load Tokenizer
    logger.info("Loading tokenizer from %s...", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Base Model
    logger.info("Loading 4-bit quantized base model into GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # 4. PEFT LoRA Configuration
    logger.info("Configuring PEFT LoRA (r=%d, alpha=%d)...", lora_r, lora_alpha)
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

    # 5. Load and format datasets
    logger.info("Loading dataset splits...")
    dataset = load_dataset("json", data_files={"train": str(train_data_path), "validation": str(val_data_path)})
    logger.info("Dataset loaded: Train=%d samples, Validation=%d samples", len(dataset["train"]), len(dataset["validation"]))

    def format_sample(example):
        enc = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_dict=True,
            max_length=max_seq_length,
            truncation=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["input_ids"].copy(),
        }

    logger.info("Applying chat template formatting to dataset splits...")
    train_dataset = dataset["train"].map(format_sample, remove_columns=dataset["train"].column_names)
    eval_dataset = dataset["validation"].map(format_sample, remove_columns=dataset["validation"].column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch" if max_steps <= 0 else "no",
        save_strategy="epoch" if max_steps <= 0 else "steps",
        save_steps=25,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    )

    # 7. Trainer
    logger.info("Initializing Hugging Face Trainer with PEFT LoRA and 8-bit Adam...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if max_steps <= 0 else None,
        data_collator=collator,
    )

    logger.info("Starting gradient descent training loop (Modifying actual LoRA model weights)...")
    train_result = trainer.train()
    logger.info("Training loss: %s", train_result.training_loss)

    # 8. Save Final Trained Adapter Weights
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("=== Weight Training Complete: Real LoRA adapter saved to %s ===", output_dir)
    logger.info("Trained files: %s", [p.name for p in output_dir.glob("*")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARES Real QLoRA Prompt Optimizer Trainer")
    parser.add_argument("--model", "--base-model", dest="model", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", help="Base model identifier")
    parser.add_argument("--dataset-path", "--train-data", dest="dataset_path", type=str, default=str(TRAIN_DATA_PATH), help="Train dataset path")
    parser.add_argument("--val-dataset-path", "--val-data", dest="val_dataset_path", type=str, default=str(VAL_DATA_PATH), help="Validation dataset path")
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR), help="Output directory for weights")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", "--grad-accum", dest="grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer")
    args, unknown = parser.parse_known_args()

    train_lora_adapter(
        base_model_name=args.model,
        train_data_path=Path(args.dataset_path),
        val_data_path=Path(args.val_dataset_path),
        output_dir=Path(args.output_dir),
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
    )
