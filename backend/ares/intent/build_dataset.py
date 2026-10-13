"""
Script to build, split, and save the intent dataset, and fit the baseline model.
Can be run via: python -m ares.intent.build_dataset
"""

import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ares.intent.dataset import IntentDatasetManager
from ares.intent.synthesizer import IntentDataSynthesizer
from ares.intent.classifier import SklearnIntentClassifier


def build_and_evaluate(multiplier: int = 5):
    print("=== ARES Intent Dataset Pipeline (Week 1) ===")
    mgr = IntentDatasetManager()
    seeds = mgr.get_seed_dataset()
    print(f"Loaded {len(seeds)} curated seed prompts across 5 intent classes.")

    synthesizer = IntentDataSynthesizer()
    augmented = synthesizer.generate_rule_based_variations(seeds, target_multiplier=multiplier)
    print(f"Augmented dataset to {len(augmented)} samples with conversational variations.")

    train_set, val_set, test_set = mgr.split_train_val_test(augmented, train_ratio=0.8, val_ratio=0.1)
    train_path = mgr.save_dataset_jsonl(train_set, "intent_dataset_train.jsonl")
    val_path = mgr.save_dataset_jsonl(val_set, "intent_dataset_val.jsonl")
    test_path = mgr.save_dataset_jsonl(test_set, "intent_dataset_test.jsonl")
    print(f"Saved datasets to: {train_path.parent}")
    print(f"  • Train: {len(train_set)} samples")
    print(f"  • Validation: {len(val_set)} samples")
    print(f"  • Test: {len(test_set)} samples")

    classifier = SklearnIntentClassifier()
    print("\nFitting baseline SklearnIntentClassifier (TF-IDF + Calibrated Logistic Regression)...")
    train_metrics = classifier.fit([s.prompt for s in train_set], [s.intent for s in train_set])
    eval_metrics = classifier.evaluate([s.prompt for s in test_set], [s.intent for s in test_set])

    print(f"Training Accuracy:  {train_metrics['train_accuracy'] * 100:.2f}%")
    print(f"Holdout Accuracy:   {eval_metrics['accuracy'] * 100:.2f}%")
    print(f"Holdout Macro F1:   {eval_metrics['macro_f1']:.4f}")
    print(f"Holdout Wgt F1:     {eval_metrics['weighted_f1']:.4f}")
    print(f"Model saved to:     {classifier.model_path}")
    print("=== Week 1 Intent Pipeline Initialization Complete ===")


if __name__ == "__main__":
    build_and_evaluate()
