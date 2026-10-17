"""
A/B Testing Framework for ARES Prompt Optimizer
Compares baseline prompts vs fine-tuned LoRA optimizer outputs.
"""
import asyncio
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ares.config import settings
from ares.optimizer.pipeline import UserPromptOptimizer, _cosine_similarity, _estimate_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ABTestRunner:
    """Runs parallel prompt optimization comparisons between baseline and optimized models."""

    def __init__(self):
        self.optimizer = UserPromptOptimizer(settings=settings)

    async def run_ab_test(
        self,
        prompt: str,
        intent: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run single prompt evaluation through optimization pipeline."""
        t0 = datetime.utcnow()
        orig_tokens = _estimate_tokens(prompt)

        result = await self.optimizer.optimize_prompt(
            user_prompt=prompt,
            domain=domain or "general",
            target_fidelity=None,
            allow_harmful=True,
        )

        opt_tokens = result.optimized_tokens
        savings_pct = result.token_savings_pct
        fidelity = result.semantic_fidelity

        return {
            "intent": intent,
            "domain": domain or "general",
            "original_prompt": prompt,
            "optimized_prompt": result.optimized_prompt,
            "original_tokens": orig_tokens,
            "optimized_tokens": opt_tokens,
            "token_savings_pct": round(savings_pct, 4),
            "semantic_fidelity": round(fidelity, 4),
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "timestamp": t0.isoformat(),
        }

async def run_ab_test_suite(
    test_dataset_path: str,
    output_path: str = "backend/ares/training/artifacts/ab_test_results.json",
    sample_limit: int = 50,
):
    """Run A/B test suite across holdout records."""
    runner = ABTestRunner()
    path = Path(test_dataset_path)
    if not path.exists():
        logger.error("Test dataset %s not found", test_dataset_path)
        return

    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    selected = samples[:sample_limit]
    logger.info("Running A/B test on %d samples from %s", len(selected), path)

    results: List[Dict[str, Any]] = []
    for s in selected:
        user_msg = next((m["content"] for m in s["messages"] if m["role"] == "user"), "")
        raw = user_msg.replace("ORIGINAL UNOPTIMIZED PROMPT:\n", "").replace("\n\nREWRITTEN OPTIMIZED PROMPT:", "").strip()
        intent = s.get("intent", "general")
        domain = s.get("domain", "general")

        res = await runner.run_ab_test(prompt=raw, intent=intent, domain=domain)
        results.append(res)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    avg_savings = sum(r["token_savings_pct"] for r in results) / len(results) if results else 0.0
    avg_fid = sum(r["semantic_fidelity"] for r in results) / len(results) if results else 0.0

    print("\n" + "=" * 80)
    print(" ARES A/B TEST BENCHMARK COMPLETE")
    print(f" Samples Evaluated: {len(results)}")
    print(f" Mean Token Savings: {avg_savings * 100:.1f}%")
    print(f" Mean Fidelity:      {avg_fid:.3f} / 1.000")
    print(f" Results Saved:      {out_p}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(
        run_ab_test_suite(
            test_dataset_path="backend/ares/intent/data/sft_optimizer_test.jsonl",
            sample_limit=25,
        )
    )
