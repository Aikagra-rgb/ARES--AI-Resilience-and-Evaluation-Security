"""
Unit and integration tests for ARES Intent Classification & Dataset Pipeline (Week 1).
"""

import pytest
from pathlib import Path

from ares.intent.taxonomy import (
    PromptIntent,
    OptimizationDirective,
    get_directive_for_intent,
    INTENT_DIRECTIVES,
)
from ares.intent.dataset import IntentDatasetManager
from ares.intent.synthesizer import IntentDataSynthesizer
from ares.intent.classifier import SklearnIntentClassifier


def test_intent_taxonomy_has_five_classes():
    """Verify all 5 core enterprise prompt intent categories are defined."""
    expected = {
        "coding_technical",
        "data_extraction_json",
        "analytical_qa",
        "creative_writing",
        "system_directive",
    }
    actual = {intent.value for intent in PromptIntent}
    assert actual == expected

    for intent in PromptIntent:
        directive = get_directive_for_intent(intent)
        assert isinstance(directive, OptimizationDirective)
        assert directive.intent == intent
        assert 0.15 <= directive.target_token_reduction_pct <= 0.50
        assert len(directive.preservation_rules) > 0
        assert len(directive.pruning_rules) > 0
        assert len(directive.structural_enforcements) > 0


def test_intent_dataset_manager_seed_loading_and_splitting(tmp_path: Path):
    """Verify dataset manager loads seed data and splits with stratification."""
    mgr = IntentDatasetManager(data_dir=tmp_path)
    seeds = mgr.get_seed_dataset()
    assert len(seeds) >= 25

    # Check all 5 intents are present in seed data
    intents_present = {s.intent for s in seeds}
    assert len(intents_present) == 5

    train_set, val_set, test_set = mgr.split_train_val_test(seeds, train_ratio=0.8, val_ratio=0.1)
    assert len(train_set) + len(val_set) + len(test_set) == len(seeds)
    assert len(train_set) > len(val_set)

    # Test serialization round-trip
    file_path = mgr.save_dataset_jsonl(train_set, "test_train.jsonl")
    assert file_path.exists()
    loaded = mgr.load_dataset_jsonl("test_train.jsonl")
    assert len(loaded) == len(train_set)
    assert loaded[0].prompt == train_set[0].prompt


def test_intent_synthesizer_rule_based_augmentation():
    """Verify rule-based synthesis augments sample count and injects conversational variance."""
    mgr = IntentDatasetManager()
    seeds = mgr.get_seed_dataset()
    synthesizer = IntentDataSynthesizer()

    augmented = synthesizer.generate_rule_based_variations(seeds, target_multiplier=3)
    assert len(augmented) > len(seeds)
    # Check that augmented prompts exist with varied lengths
    ratios = [s.verbose_ratio for s in augmented if s.source == "rule_augmented"]
    assert len(ratios) > 0
    assert any(r > 1.0 for r in ratios)


def test_sklearn_intent_classifier_fit_predict_and_accuracy(tmp_path: Path):
    """
    Train the scikit-learn classifier on augmented seed data,
    verify latency < 15ms and accuracy >= 85%.
    """
    mgr = IntentDatasetManager(data_dir=tmp_path)
    seeds = mgr.get_seed_dataset()
    synthesizer = IntentDataSynthesizer()
    augmented = synthesizer.generate_rule_based_variations(seeds, target_multiplier=4)

    train_set, _, test_set = mgr.split_train_val_test(augmented, train_ratio=0.8, val_ratio=0.1)

    train_prompts = [s.prompt for s in train_set]
    train_labels = [s.intent for s in train_set]

    test_prompts = [s.prompt for s in test_set]
    test_labels = [s.intent for s in test_set]

    model_file = tmp_path / "test_clf.joblib"
    classifier = SklearnIntentClassifier(model_path=model_file)
    metrics = classifier.fit(train_prompts, train_labels)

    assert metrics["train_accuracy"] >= 0.90
    assert metrics["sample_count"] == len(train_prompts)

    # Evaluation on holdout test set
    eval_metrics = classifier.evaluate(test_prompts, test_labels)
    assert eval_metrics["accuracy"] >= 0.85
    assert eval_metrics["macro_f1"] >= 0.80

    # Test single-prompt prediction & performance
    coding_query = "Could you write a Python function with asyncio to query a Redis cluster?"
    result = classifier.predict(coding_query)
    assert result.intent == PromptIntent.CODING_TECHNICAL
    assert result.confidence >= 0.35
    assert result.probabilities[PromptIntent.CODING_TECHNICAL.value] == max(result.probabilities.values())
    assert result.latency_ms < 15.0  # Ultra-fast CPU inference
    assert result.directive.intent == PromptIntent.CODING_TECHNICAL

    # Test persistence
    loaded_clf = SklearnIntentClassifier(model_path=model_file)
    loaded_clf.load()
    res2 = loaded_clf.predict(coding_query)
    assert res2.intent == PromptIntent.CODING_TECHNICAL


@pytest.mark.asyncio
async def test_qdrant_exemplar_store_indexing_and_retrieval():
    """Verify Qdrant exemplar store indexes pairs and formats in-context demonstrations."""
    from qdrant_client import QdrantClient
    from ares.intent.exemplar_store import QdrantExemplarStore, PromptExemplar, SEED_EXEMPLARS
    from ares.vectordb.store import QdrantStore
    from unittest.mock import AsyncMock

    mock_embedder = AsyncMock()
    # Return deterministic pseudo-vectors based on query hash
    mock_embedder.embed_text = AsyncMock(side_effect=lambda text: [0.1] * 768)

    in_memory_client = QdrantClient(":memory:")
    base_store = QdrantStore(client=in_memory_client)
    base_store.vector_size = 768

    store = QdrantExemplarStore(qdrant_store=base_store, embedder=mock_embedder)
    store.collection_name = "test_prompt_exemplars"

    # Batch index seed exemplars
    count = await store.batch_index_exemplars(SEED_EXEMPLARS)
    assert count == len(SEED_EXEMPLARS)

    # Search for coding exemplar
    hits = await store.search_exemplars(
        query_prompt="Write a Python script for quicksort",
        intent=PromptIntent.CODING_TECHNICAL,
        limit=2,
    )
    assert len(hits) > 0
    assert hits[0].intent == PromptIntent.CODING_TECHNICAL
    assert hits[0].token_savings_pct > 0.0

    # Format few-shot block
    block = store.format_few_shot_block(hits)
    assert "IN-CONTEXT OPTIMIZATION DEMONSTRATIONS" in block
    assert "UNOPTIMIZED:" in block
    assert "OPTIMIZED" in block


def test_scaled_5000_dataset_and_model_loading():
    """Verify that the scaled 5,000 dataset and trained model artifact load and predict accurately."""
    mgr = IntentDatasetManager()
    full_data = mgr.load_dataset_jsonl("scaled_5000_full.jsonl")
    assert len(full_data) >= 5000

    # Ensure all 5 intents have substantial representation
    by_intent = {}
    for sample in full_data:
        by_intent[sample.intent] = by_intent.get(sample.intent, 0) + 1
    assert len(by_intent) == 5
    for intent, count in by_intent.items():
        assert count >= 900, f"Intent {intent} has only {count} samples"

    # Verify scaled model artifact
    model_path = mgr.data_dir / "intent_classifier_scaled_5000.joblib"
    assert model_path.exists()

    clf = SklearnIntentClassifier(model_path=model_path)
    res = clf.predict("Extract invoice table into strict JSON schema with tax and subtotal")
    assert res.intent == PromptIntent.DATA_EXTRACTION_JSON
    assert res.confidence >= 0.50
    assert res.latency_ms < 15.0


