"""
ARES Phase 4: Ollama Packaging & Modelfile Exporter.
Creates the Ollama Modelfile and registers ares-optimizer:8b into local Ollama.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BACKEND_DIR / "ares" / "training" / "artifacts"
LORA_DIR = ARTIFACTS_DIR / "ares_prompt_optimizer_lora"
MODELFILE_PATH = ARTIFACTS_DIR / "Modelfile"

MODELFILE_TEMPLATE = """FROM llama3.1:8b
{adapter_clause}
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"

SYSTEM \"\"\"You are the ARES Prompt Compiler & Optimization Engine.
Your objective: Rewrite and compress the user's prompt to minimize token usage while retaining 100% of its technical intent, domain requirements, and core semantics.
Strip pleasantries, conversational padding, greetings, and repetitive phrases, while strictly preserving technical constraints, types, and schema specifications.
Output ONLY the rewritten, compressed prompt text. Do NOT prepend explanations, introductions, or pleasantries.\"\"\"

MESSAGE user \"\"\"ORIGINAL UNOPTIMIZED PROMPT:
Hey there AI! I have been working on this all morning and really need some help. Here is what I am trying to do: For a production Python system, asyncio producer-consumer queue with graceful shutdown and worker timeout handling Please make sure this is accurate, complete, and formatted well. Appreciate it!

REWRITTEN OPTIMIZED PROMPT:\"\"\"
MESSAGE assistant \"\"\"Implement asyncio producer-consumer queue with graceful shutdown and worker timeout handling.\"\"\"

MESSAGE user \"\"\"ORIGINAL UNOPTIMIZED PROMPT:
Hello! Could you please help me with the following? Regarding our legal service, parse patent specification into patent_number, filing_date, inventors, primary_claim_text, classification_cpc_codes Thank you very much for your assistance!

REWRITTEN OPTIMIZED PROMPT:\"\"\"
MESSAGE assistant \"\"\"Extract JSON schema: {{patent_number, filing_date, inventors, primary_claim_text, classification_cpc_codes}} from patent specification.\"\"\"

MESSAGE user \"\"\"ORIGINAL UNOPTIMIZED PROMPT:
From an enterprise architecture and compliance perspective, please address: Regarding our fintech service, Assess the economic implications of central bank digital currencies (CBDCs) on commercial banking fractional reserve models Ensure the output aligns with production best practices.

REWRITTEN OPTIMIZED PROMPT:\"\"\"
MESSAGE assistant \"\"\"Assess the economic implications of central bank digital currencies (CBDCs) on commercial banking fractional reserve models.\"\"\"

MESSAGE user \"\"\"ORIGINAL UNOPTIMIZED PROMPT:
URGENT: I need immediate assistance with this task for a production deadline: Compose an authoritative B2B cold email to Enterprise Chief Information Security Officers announcing our AI prompt security firewall Please provide the answer right away without unnecessary filler.

REWRITTEN OPTIMIZED PROMPT:\"\"\"
MESSAGE assistant \"\"\"Compose an authoritative B2B cold email to Enterprise Chief Information Security Officers announcing our AI prompt security firewall.\"\"\"

MESSAGE user \"\"\"ORIGINAL UNOPTIMIZED PROMPT:
From an enterprise architecture and compliance perspective, please address: For a production healthcare system, You are an AI patient triage guidance assistant. Review reported symptoms and categorize urgency into emergency, urgent care, or routine. Always include medical doctor disclaimer. Never prescribe dosages. Ensure the output aligns with production best practices.

REWRITTEN OPTIMIZED PROMPT:\"\"\"
MESSAGE assistant \"\"\"System Directive:
You are an AI patient triage guidance assistant. Review reported symptoms and categorize urgency into emergency, urgent care, or routine. Always include medical doctor disclaimer. Never prescribe dosages.\"\"\"
"""


def package_and_register_ollama(
    model_name: str = "ares-optimizer:8b",
    adapter_path: Path | None = None,
):
    """
    Generate Modelfile and register the model with Ollama.
    """
    logger.info("=== ARES Phase 4: Ollama Packaging & Registration ===")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter_clause = ""
    if adapter_path and adapter_path.exists():
        logger.info("Including GGUF adapter: %s", adapter_path)
        adapter_clause = f"ADAPTER {adapter_path.resolve()}"

    content = MODELFILE_TEMPLATE.format(adapter_clause=adapter_clause)
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Generated Ollama Modelfile at %s", MODELFILE_PATH)

    # Execute ollama create
    cmd = ["ollama", "create", model_name, "-f", str(MODELFILE_PATH)]
    logger.info("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Ollama Output: %s", proc.stdout.strip())
        logger.info("=== Successfully registered '%s' in Ollama! ===", model_name)
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to register Ollama model: %s\nStderr: %s", exc, exc.stderr)
        raise exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package ARES Optimizer into Ollama")
    parser.add_argument("--name", type=str, default="ares-optimizer:8b", help="Ollama model name")
    parser.add_argument("--adapter", type=str, default=None, help="Path to GGUF adapter file")
    args = parser.parse_args()

    adapter_file = Path(args.adapter) if args.adapter else None
    package_and_register_ollama(model_name=args.name, adapter_path=adapter_file)
