"""
Runtime Loader & Inference Cache for Per-Client Intent & Threat Classifiers.

Loads trained per-organization models from `backend/models/{org_id}/intent_classifier/`
with thread-safe caching so Gateway Level-1.5 and Optimizer Stage-2 run in <2ms.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib

from ares.intent.classifier import IntentClassificationResult
from ares.intent.taxonomy import PromptIntent, get_directive_for_intent
from ares.training.model_registry import MODELS_ROOT_DIR, validate_org_id

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.RLock()
_BUNDLE_CACHE: Dict[str, "PerClientClassifierBundle"] = {}


class PerClientClassifierBundle:
    """Encapsulates one client's trained threat detector and 5-class intent router."""

    def __init__(self, org_id: str, model_dir: Path) -> None:
        self.org_id = org_id
        self.model_dir = model_dir
        self.threat_pipeline = joblib.load(model_dir / "threat_classifier.joblib")
        self.intent_pipeline = joblib.load(model_dir / "intent_classifier.joblib")

    def predict_threat(self, prompt: str) -> Tuple[bool, Optional[str], float, str]:
        """
        Evaluate whether `prompt` matches client-specific attack patterns.
        Returns `(triggered, category, confidence, explanation)`.
        """
        probs = self.threat_pipeline.predict_proba([prompt])[0]
        classes = [str(c) for c in self.threat_pipeline.classes_]
        prob_map = dict(zip(classes, probs))

        benign_prob = float(prob_map.get("benign", 0.0))
        attack_prob = max(0.0, min(1.0, 1.0 - benign_prob))

        attack_candidates = [(cls, float(p)) for cls, p in prob_map.items() if cls != "benign"]
        if attack_candidates:
            top_cat, top_cat_prob = max(attack_candidates, key=lambda item: item[1])
        else:
            top_cat, top_cat_prob = "benign", benign_prob

        effective_conf = max(attack_prob, top_cat_prob)
        if attack_prob >= 0.70 and top_cat != "benign":
            return (
                True,
                top_cat,
                round(effective_conf, 4),
                f"Per-client ML classifier [{self.org_id}] detected '{top_cat}' ({effective_conf * 100:.1f}% confidence).",
            )
        return (
            False,
            top_cat if top_cat != "benign" else "clean",
            round(benign_prob if benign_prob >= attack_prob else effective_conf, 4),
            f"Per-client ML classifier [{self.org_id}]: benign ({benign_prob * 100:.1f}%).",
        )

    def predict_intent(self, prompt: str) -> IntentClassificationResult:
        """Predict 5-class optimization intent using the client's specialized model."""
        t0 = time.perf_counter()
        probs = self.intent_pipeline.predict_proba([prompt])[0]
        classes = [str(c) for c in self.intent_pipeline.classes_]
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
        best_idx = int(probs.argmax())
        top_intent = PromptIntent(classes[best_idx])
        confidence = float(probs[best_idx])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return IntentClassificationResult(
            intent=top_intent,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            directive=get_directive_for_intent(top_intent),
            latency_ms=latency_ms,
            model_version=f"per-client-{self.org_id}-v1",
        )


def get_per_client_classifier(
    org_id: Optional[str],
    models_root: Optional[Path] = None,
) -> Optional[PerClientClassifierBundle]:
    """Retrieve (and cache) the trained classifier bundle for `org_id` if present on disk."""
    if not org_id or org_id == "default":
        return None
    try:
        safe_org = validate_org_id(org_id)
    except ValueError:
        logger.warning("Rejected invalid organization ID '%s' during per-client classifier lookup.", org_id)
        return None

    root = (Path(models_root) if models_root else MODELS_ROOT_DIR).resolve()
    cache_key = f"{root}::{safe_org}"

    with _CACHE_LOCK:
        if cache_key in _BUNDLE_CACHE:
            return _BUNDLE_CACHE[cache_key]

        model_dir = (root / safe_org / "intent_classifier").resolve()
        if root not in model_dir.parents:
            return None
        threat_file = model_dir / "threat_classifier.joblib"
        intent_file = model_dir / "intent_classifier.joblib"
        if not (threat_file.exists() and intent_file.exists()):
            return None

        try:
            bundle = PerClientClassifierBundle(org_id=safe_org, model_dir=model_dir)
            _BUNDLE_CACHE[cache_key] = bundle
            return bundle
        except Exception as exc:
            logger.warning("Could not load per-client classifier for '%s': %s", safe_org, exc)
            return None


def clear_per_client_cache(org_id: Optional[str] = None) -> None:
    """Clear cached classifier bundles (all or for a specific org_id)."""
    with _CACHE_LOCK:
        if org_id is None:
            _BUNDLE_CACHE.clear()
        else:
            keys_to_remove = [k for k in _BUNDLE_CACHE if k.endswith(f"::{org_id}")]
            for k in keys_to_remove:
                _BUNDLE_CACHE.pop(k, None)
