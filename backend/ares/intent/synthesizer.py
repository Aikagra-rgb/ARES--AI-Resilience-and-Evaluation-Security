"""
Synthetic Data Expansion Engine for Prompt Intent Classification.
Generates realistic, varied, and conversational prompts using rule-based perturbations
and local Ollama (Llama 3.1 8B) synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Dict, List, Optional

from ares.intent.taxonomy import IntentSample, PromptIntent
from ares.llm.client import LLMClient
from ares.llm.models import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

# Prefixes that users commonly add to prompts
CONVERSATIONAL_PREFIXES = [
    "Hey there! Can you please help me with something? ",
    "Hello AI, I really need some assistance today. ",
    "Could you kindly ",
    "I was hoping you could help me figure out how to ",
    "I have a quick question for you: ",
    "Act as an expert and ",
    "My boss asked me to get this done ASAP, so please ",
    "I am completely stuck on this problem. Could you ",
]

# Suffixes that users commonly add
CONVERSATIONAL_SUFFIXES = [
    " Thanks in advance for your help!",
    " Please explain step by step so I understand.",
    " Make sure the answer is high quality and accurate.",
    " Let me know if you need more details from me.",
    " Be concise and don't include unnecessary fluff.",
]

# Domain context injections
DOMAIN_CONTEXTS: Dict[PromptIntent, List[str]] = {
    PromptIntent.CODING_TECHNICAL: [
        "In our TypeScript backend microservice, ",
        "For an enterprise production system deployed on AWS ECS, ",
        "Inside a high-frequency C++ order execution library, ",
        "Using Python 3.12 with asyncio and pydantic v2, ",
    ],
    PromptIntent.DATA_EXTRACTION_JSON: [
        "From this raw unstructured PDF scan text: ",
        "Given the following financial balance sheet statement: ",
        "Reviewing these customer support feedback tickets: ",
        "Extracting data from this electronic medical health record: ",
    ],
    PromptIntent.ANALYTICAL_QA: [
        "In the context of modern distributed database systems, ",
        "From a regulatory compliance and risk management perspective, ",
        "Considering both economic inflation and central bank monetary policy, ",
        "Evaluating microservices versus monolithic software architectures, ",
    ],
    PromptIntent.CREATIVE_WRITING: [
        "Writing for a high-end luxury lifestyle magazine, ",
        "Targeting junior software engineers with an engaging tech blog post, ",
        "Drafting a keynote announcement for our annual developer summit, ",
        "Crafting a dark mystery narrative set in Victorian London, ",
    ],
    PromptIntent.SYSTEM_DIRECTIVE: [
        "You are the internal compliance monitor for a tier-1 investment bank. ",
        "You are an empathetic, clinical AI triage nurse. ",
        "You are a zero-trust security gateway proxy agent. ",
        "You are an automated code review bot enforcing strict linting standards. ",
    ],
}


class IntentDataSynthesizer:
    """Expands seed datasets into large diversified corpora."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    def generate_rule_based_variations(
        self,
        seed_samples: List[IntentSample],
        target_multiplier: int = 4,
        seed: int = 42,
    ) -> List[IntentSample]:
        """
        Generate diverse samples via rule-based conversational perturbation
        (prefix injection, suffix bloating, domain contextualization).
        Runs in 0ms without requiring an LLM.
        """
        rng = random.Random(seed)
        augmented: List[IntentSample] = list(seed_samples)

        for sample in seed_samples:
            for k in range(target_multiplier):
                prefix = rng.choice(CONVERSATIONAL_PREFIXES) if rng.random() > 0.3 else ""
                suffix = rng.choice(CONVERSATIONAL_SUFFIXES) if rng.random() > 0.3 else ""
                domain_ctx = (
                    rng.choice(DOMAIN_CONTEXTS.get(sample.intent, [""]))
                    if rng.random() > 0.4
                    else ""
                )

                new_prompt = f"{prefix}{domain_ctx}{sample.prompt}{suffix}".strip()
                augmented.append(
                    IntentSample(
                        id=f"aug_{sample.id}_{k+1}",
                        prompt=new_prompt,
                        intent=sample.intent,
                        domain=sample.domain,
                        source="rule_augmented",
                        verbose_ratio=round(len(new_prompt.split()) / max(1, len(sample.prompt.split())), 2),
                    )
                )

        return augmented

    async def generate_llm_synthetic_samples(
        self,
        intent: PromptIntent,
        count: int = 5,
        domain: str = "general",
        model: str = "llama3.1:8b",
        provider: LLMProvider = LLMProvider.OLLAMA,
    ) -> List[IntentSample]:
        """
        Query local Ollama to synthesize novel real-world user prompts
        that exhibit bloated/conversational phrasing for a specific intent.
        """
        if self.llm_client is None:
            self.llm_client = LLMClient()

        prompt_system = (
            f"You are a dataset generator for AI benchmarking. "
            f"Generate realistic, verbose, and conversational prompts written by everyday developers or enterprise users "
            f"whose primary goal is: {intent.value} in domain '{domain}'. "
            f"Make them realistic: include conversational rambling, extra context, and natural filler words. "
            f"Output exactly {count} distinct prompts, one per line, starting with [PROMPT]."
        )

        messages = [
            ChatMessage(role="system", content=prompt_system),
            ChatMessage(
                role="user",
                content=f"Generate {count} distinct realistic user prompts for intent {intent.value}.",
            ),
        ]

        try:
            response = await self.llm_client.complete(
                messages=messages,
                provider=provider.value if isinstance(provider, LLMProvider) else str(provider),
                model=model,
                temperature=0.7,
                max_tokens=1000,
            )
            raw_text = response.content
        except Exception as e:
            logger.warning("LLM synthetic generation failed (%s), returning empty list", e)
            return []

        generated: List[IntentSample] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if line.startswith("[PROMPT]"):
                clean = line.replace("[PROMPT]", "").strip()
                if clean:
                    generated.append(
                        IntentSample(
                            id=f"synth_{intent.value}_{random.randint(1000, 9999)}",
                            prompt=clean,
                            intent=intent,
                            domain=domain,
                            source="ollama_synthetic",
                        )
                    )

        return generated
