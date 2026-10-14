"""
ARES Phase 2: SFT Dataset Formatter & Stratified Partitioner.
Converts optimization pairs into Llama 3.1 Instruct Chat template records
and partitions into train (80%), validation (10%), and test (10%) splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INTENT_DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_PAIRS_PATH = INTENT_DATA_DIR / "optimization_pairs_1500.jsonl"

TRAIN_SPLIT_PATH = INTENT_DATA_DIR / "sft_optimizer_train.jsonl"
VAL_SPLIT_PATH = INTENT_DATA_DIR / "sft_optimizer_val.jsonl"
TEST_SPLIT_PATH = INTENT_DATA_DIR / "sft_optimizer_test.jsonl"

SYSTEM_INSTRUCTION = (
    "You are the ARES Prompt Compiler & Optimization Engine.\n"
    "Your objective: Rewrite and compress the user's prompt to minimize token usage "
    "while retaining 100% of its technical intent, domain requirements, and core semantics.\n"
    "Strip pleasantries, conversational padding, greetings, and repetitive phrases, "
    "while strictly preserving technical constraints, types, and schema specifications.\n"
    "Output ONLY the rewritten, compressed prompt text. Do NOT prepend explanations, introductions, or pleasantries."
)


def format_chat_record(pair: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a single prompt pair into the official Llama 3.1 Instruct Chat template schema.
    """
    original = pair["original_prompt"].strip()
    optimized = pair["optimized_prompt"].strip()

    user_content = f"ORIGINAL UNOPTIMIZED PROMPT:\n{original}\n\nREWRITTEN OPTIMIZED PROMPT:"

    return {
        "id": pair["id"],
        "intent": pair["intent"],
        "domain": pair.get("domain", "general"),
        "original_tokens": pair.get("original_tokens", 0),
        "optimized_tokens": pair.get("optimized_tokens", 0),
        "token_savings_pct": pair.get("token_savings_pct", 0.0),
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": optimized},
        ],
    }


def partition_stratified(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partition records with stratified balance across all intent categories.
    """
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_intent.setdefault(r["intent"], []).append(r)

    train_set: List[Dict[str, Any]] = []
    val_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    random.seed(42)  # Deterministic partition
    for intent, items in by_intent.items():
        shuffled = list(items)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set.extend(shuffled[:n_train])
        val_set.extend(shuffled[n_train : n_train + n_val])
        test_set.extend(shuffled[n_train + n_val :])

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def build_sft_splits(
    input_path: Path = INPUT_PAIRS_PATH,
    output_dir: Path = INTENT_DATA_DIR,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Reads pairs, formats into chat templates, and saves train/val/test splits.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input pairs file not found: {input_path}")

    raw_pairs: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_pairs.append(json.loads(line))

    logger.info("Loaded %d raw optimization pairs from %s", len(raw_pairs), input_path)

    chat_records = [format_chat_record(p) for p in raw_pairs]
    train_records, val_records, test_records = partition_stratified(
        chat_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_path = output_dir / "sft_optimizer_train.jsonl"
    val_path = output_dir / "sft_optimizer_val.jsonl"
    test_path = output_dir / "sft_optimizer_test.jsonl"

    def write_jsonl(path: Path, data: List[Dict[str, Any]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Saved %d records to %s", len(data), path)

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)

    print("\n" + "=" * 80)
    print(" SFT DATASET SPLIT PARTITION SUMMARY")
    print("=" * 80)
    print(f"Total Formatted SFT Records: {len(chat_records)}")
    print(f"  - Training Split:   {len(train_records)} records ({train_path.name})")
    print(f"  - Validation Split: {len(val_records)} records ({val_path.name})")
    print(f"  - Test Holdout:     {len(test_records)} records ({test_path.name})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SFT Dataset Splits for Llama 3.1 LoRA")
    parser.add_argument("--input", "--input-path", dest="input", type=str, default=str(INPUT_PAIRS_PATH), help="Path to input pairs JSONL")
    parser.add_argument("--output-dir", type=str, default=str(INTENT_DATA_DIR), help="Output directory for SFT splits")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test split ratio")
    args = parser.parse_args()

    build_sft_splits(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
