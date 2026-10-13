"""
ARES Intent-Driven Prompt Classification & Optimization Module.
"""

from ares.intent.taxonomy import (
    PromptIntent,
    OptimizationDirective,
    IntentSample,
    get_directive_for_intent,
    INTENT_DIRECTIVES,
)
from ares.intent.dataset import IntentDatasetManager, SEED_INTENT_DATASET
from ares.intent.synthesizer import IntentDataSynthesizer
from ares.intent.classifier import (
    IntentClassificationResult,
    BaseIntentClassifier,
    SklearnIntentClassifier,
)

__all__ = [
    "PromptIntent",
    "OptimizationDirective",
    "IntentSample",
    "get_directive_for_intent",
    "INTENT_DIRECTIVES",
    "IntentDatasetManager",
    "SEED_INTENT_DATASET",
    "IntentDataSynthesizer",
    "IntentClassificationResult",
    "BaseIntentClassifier",
    "SklearnIntentClassifier",
]
