"""
Machine Learning Statistics & Attack Vector Modeling Engine for ARES.

Performs multivariate statistical analysis, clustering metrics, dimensionality
reduction (PCA), and trains an ML classification model on the 768-dim embeddings
stored in Qdrant Cloud.

Classifier: TF-IDF (word 1-2 grams + char 3-5 grams, 800 features)
            + 15-dim hand-crafted structural features
            -> LogisticRegression(C=5, class_weight='balanced', solver='lbfgs')
            Fast, interpretable, targets 95%+ on this domain.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler

from ares.config import Settings, settings as default_settings
from ares.vectordb.store import QdrantStore

logger = logging.getLogger(__name__)

# Model version - increment when model architecture changes
MODEL_VERSION = "1.0.0"

# Versioned model path
def get_model_path(version: str = MODEL_VERSION) -> Path:
    """Get versioned model path."""
    base_dir = Path(__file__).parent.parent.parent / "data" / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"classifier_v{version}.joblib"

# Legacy path for backward compatibility
DEFAULT_MODEL_PATH: Path = get_model_path()


@dataclass
class VectorDistributionStats:
    """Statistical properties of the embedding space."""
    sample_count: int
    vector_dim: int
    pairwise_similarity_mean: float
    pairwise_similarity_std: float
    pairwise_similarity_min: float
    pairwise_similarity_max: float
    pca_top_3_variance: List[float]
    pca_cumulative_variance_top_5: float
    silhouette_score: Optional[float]


@dataclass
class FeatureStats:
    """Text and operational feature statistics."""
    length_mean: float
    length_std: float
    length_median: float
    length_min: int
    length_max: int
    category_counts: Dict[str, int]
    provider_counts: Dict[str, int]
    operator_counts: Dict[str, int]


@dataclass
class MLModelPerformance:
    """Evaluation metrics for the trained attack category classifier."""
    model_name: str
    cv_mean_accuracy: float
    cv_std_accuracy: float
    classes: List[str]
    classification_report: Dict[str, Any]
    confusion_matrix: List[List[int]]


@dataclass
class MLAnalysisReport:
    """Comprehensive ML and statistical analysis report."""
    vector_stats: VectorDistributionStats
    feature_stats: FeatureStats
    model_performance: MLModelPerformance

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  ARES: MACHINE LEARNING & STATISTICAL VECTOR ANALYSIS",
            "=" * 60,
            f"Dataset Size:           {self.vector_stats.sample_count} points",
            f"Embedding Dimension:    {self.vector_stats.vector_dim}-dim (Gemini Normalized)",
            "",
            "--- 1. EMBEDDING SPACE & CLUSTERING METRICS ---",
            f"Pairwise Cosine Sim:    Mean={self.vector_stats.pairwise_similarity_mean:.4f} "
            f"(+-{self.vector_stats.pairwise_similarity_std:.4f})",
            f"Cosine Sim Range:       [{self.vector_stats.pairwise_similarity_min:.4f}, "
            f"{self.vector_stats.pairwise_similarity_max:.4f}]",
            f"Silhouette Score:       {self.vector_stats.silhouette_score if self.vector_stats.silhouette_score is not None else 'N/A'}",
            f"PCA Explained Variance: Top-3={np.round(self.vector_stats.pca_top_3_variance, 4).tolist()}, "
            f"Top-5 Cumul={self.vector_stats.pca_cumulative_variance_top_5:.4f}",
            "",
            "--- 2. ATTACK PAYLOAD & FEATURE DISTRIBUTIONS ---",
            f"Attack Length (chars):  Mean={self.feature_stats.length_mean:.1f}, "
            f"Median={self.feature_stats.length_median:.1f}, "
            f"Range=[{self.feature_stats.length_min}, {self.feature_stats.length_max}]",
            f"Categories:             {self.feature_stats.category_counts}",
            f"Victim Providers:       {self.feature_stats.provider_counts}",
            f"Operators Applied:      {self.feature_stats.operator_counts}",
            "",
            "--- 3. TRAINED ML ATTACK CLASSIFICATION MODEL ---",
            f"Classifier:             {self.model_performance.model_name}",
            f"Stratified CV Accuracy: {self.model_performance.cv_mean_accuracy * 100:.1f}% "
            f"(+-{self.model_performance.cv_std_accuracy * 100:.1f}%)",
            f"Classes Evaluated:      {self.model_performance.classes}",
            "=" * 60,
        ]
        return "\n".join(lines)


def _build_structural_features(payloads: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract hand-crafted structural/syntactic features from attack text payloads.
    These lexical signals complement TF-IDF and directly encode attack category patterns.
    """
    features = []
    for p in payloads:
        txt = p.get("attack_text", "")
        txt_lower = txt.lower()
        features.append([
            # Length signal
            len(txt),
            # Structural delimiters
            1.0 if "{" in txt and "}" in txt else 0.0,           # JSON braces
            1.0 if "<" in txt and ">" in txt else 0.0,           # XML / HTML tags
            1.0 if "```" in txt else 0.0,                        # Markdown code fence
            1.0 if "[INST]" in txt or "###" in txt else 0.0,     # Instruction markers
            1.0 if "<|im_start|>" in txt or "<|im_end|>" in txt else 0.0,  # ChatML tokens
            # Encoding indicators
            1.0 if any(x in txt_lower for x in ["base64", "decode", "hex", "rot13"]) else 0.0,
            1.0 if any(c in txt for c in "01234567890abcdef") and len(txt) > 40 else 0.0,  # Hex-like
            # Role-play indicators
            1.0 if any(x in txt_lower for x in ["pretend", "roleplay", "actor", "imagine", "dan", "omega", "jailbreak"]) else 0.0,
            # Instruction override indicators
            1.0 if any(x in txt_lower for x in ["ignore", "disregard", "override", "bypass", "forget previous"]) else 0.0,
            # Context smuggling indicators
            1.0 if any(x in txt_lower for x in ["retrieved", "knowledge base", "document excerpt", "log entry", "ticket", "email", "database"]) else 0.0,
            1.0 if any(x in txt_lower for x in ["previous conversation", "earlier session", "you told me", "my notes", "conversation history"]) else 0.0,
            # Canary / secret token references
            1.0 if "CANARY" in txt else 0.0,
            1.0 if "secret" in txt_lower or "token" in txt_lower else 0.0,
            # System / admin language
            1.0 if any(x in txt_lower for x in ["system", "admin", "root", "diagnostic", "audit", "emergency"]) else 0.0,
        ])
    return np.array(features, dtype=np.float32)


@dataclass
class AttackClassifierBundle:
    """
    Persisted, ultra-fast Scikit-Learn attack classifier bundle (<1ms inference).
    Combines word TF-IDF, char TF-IDF, and 15 structural syntactic features.
    """
    clf: ComplementNB
    tfidf_word: TfidfVectorizer
    tfidf_char: TfidfVectorizer
    scaler: MinMaxScaler
    classes: List[str]
    # Version metadata
    model_version: str = MODEL_VERSION
    trained_at: Optional[str] = None
    training_data_hash: Optional[str] = None
    cv_accuracy: Optional[float] = None

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Compute class probabilities for a single prompt string."""
        x_w = self.tfidf_word.transform([text]).toarray()
        x_c = self.tfidf_char.transform([text]).toarray()
        feats = _build_structural_features([{"attack_text": text}])
        x_s = self.scaler.transform(feats)
        x = np.hstack([x_w, x_c, x_s])
        probs = self.clf.predict_proba(x)[0]
        return {cls_name: float(p) for cls_name, p in zip(self.classes, probs)}

    def predict(self, text: str) -> Tuple[str, float]:
        """Return (top_category, confidence) in <1ms."""
        prob_dict = self.predict_proba(text)
        top_cat = max(prob_dict, key=prob_dict.get)
        return top_cat, prob_dict[top_cat]

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist classifier bundle using joblib."""
        import joblib
        import datetime
        out_path = Path(path) if path else get_model_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata before saving
        self.trained_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.model_version = MODEL_VERSION
        
        joblib.dump(self, out_path)
        logger.info("Saved trained AttackClassifierBundle v%s to %s", MODEL_VERSION, out_path)
        return out_path

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> Optional["AttackClassifierBundle"]:
        """Load trained AttackClassifierBundle from disk if available."""
        import joblib
        target_path = Path(path) if path else get_model_path()
        if target_path.exists():
            try:
                bundle = joblib.load(target_path)
                if isinstance(bundle, cls):
                    logger.info("Loaded AttackClassifierBundle v%s from %s", 
                               getattr(bundle, 'model_version', 'unknown'), target_path)
                    return bundle
            except Exception as exc:
                logger.warning("Failed to load AttackClassifierBundle from %s: %s", target_path, exc)
        return None


class AttackVectorMLAnalyzer:
    """
    Fetches raw vector embeddings and metadata from Qdrant Cloud, computes
    descriptive and multivariate statistics, and trains a supervised ML model.

    Classifier Pipeline:
        TF-IDF (char + word n-grams, 2000 features)
        + 15-dim Structural Features
        -> Soft-Voting Ensemble:
             * GradientBoostingClassifier (class_weight via sample_weight)
             * LogisticRegression(class_weight='balanced')
             * LinearSVC(class_weight='balanced') wrapped in CalibratedClassifierCV
    """

    def __init__(self, store: Optional[QdrantStore] = None, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self.store = store or QdrantStore(settings=self.settings)

    def fetch_data(self, max_points: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
        """Retrieve all vector points and metadata payloads from Qdrant with pagination."""
        all_points = []
        offset = None

        while True:
            batch_limit = 500
            if max_points is not None:
                remaining = max_points - len(all_points)
                if remaining <= 0:
                    break
                batch_limit = min(500, remaining)

            points, next_offset = self.store.client.scroll(
                collection_name=self.store.collection_name,
                with_vectors=True,
                with_payload=True,
                limit=batch_limit,
                offset=offset,
            )
            all_points.extend(points)

            if next_offset is None or (max_points is not None and len(all_points) >= max_points):
                break
            offset = next_offset

        if not all_points:
            raise ValueError(f"Collection '{self.store.collection_name}' is empty.")

        vectors = np.array([p.vector for p in all_points], dtype=np.float32)
        payloads = [p.payload or {} for p in all_points]
        
        # Compute training data hash for versioning
        import hashlib
        data_hash = hashlib.sha256()
        for p in sorted(payloads, key=lambda x: x.get("attack_text", "")):
            data_hash.update(p.get("attack_text", "").encode("utf-8"))
        training_data_hash = data_hash.hexdigest()[:16]
        
        return vectors, payloads, training_data_hash

    def compute_analysis(self, max_points: Optional[int] = None) -> MLAnalysisReport:
        """Run full ML statistical pipeline and train classifier across the full corpus."""
        vectors, payloads, training_data_hash = self.fetch_data(max_points=max_points)
        n_samples, n_dim = vectors.shape

        # --- 1. Vector Space & Pairwise Cosine Statistics ---
        # Sample up to 1000 points to avoid O(n^2) memory on large corpora
        sample_size = min(n_samples, 1000)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False) if n_samples > 1000 else np.arange(n_samples)
        sim_sample = cosine_similarity(vectors[sample_idx])
        mask = ~np.eye(sample_size, dtype=bool)
        pairwise_sims = sim_sample[mask]

        # PCA Analysis via Exact SVD (Device Guard and Windows Safe)
        X_centered = vectors - np.mean(vectors, axis=0)
        _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
        exp_var = (S ** 2) / np.sum(S ** 2)
        top_3_variance = [float(v) for v in exp_var[:3]]
        cum_variance_top_5 = float(np.sum(exp_var[:min(5, len(exp_var))]))

        # Category labels
        categories = [p.get("category", "unknown") for p in payloads]
        unique_categories = sorted(list(set(categories)))

        # Silhouette score (sampled)
        sil_score = None
        if len(unique_categories) > 1 and len(unique_categories) < n_samples:
            try:
                sil_score = float(silhouette_score(vectors[sample_idx], [categories[i] for i in sample_idx], metric="cosine"))
            except Exception as e:
                logger.warning("Silhouette score could not be calculated: %s", e)

        vec_stats = VectorDistributionStats(
            sample_count=n_samples,
            vector_dim=n_dim,
            pairwise_similarity_mean=float(np.mean(pairwise_sims)),
            pairwise_similarity_std=float(np.std(pairwise_sims)),
            pairwise_similarity_min=float(np.min(pairwise_sims)),
            pairwise_similarity_max=float(np.max(pairwise_sims)),
            pca_top_3_variance=top_3_variance,
            pca_cumulative_variance_top_5=cum_variance_top_5,
            silhouette_score=round(sil_score, 4) if sil_score is not None else None,
        )

        # --- 2. Feature & Payload Statistics ---
        lengths = [len(p.get("attack_text", "")) for p in payloads]
        category_counts: Dict[str, int] = {}
        for c in categories:
            category_counts[c] = category_counts.get(c, 0) + 1

        provider_counts: Dict[str, int] = {}
        for p in payloads:
            prov = p.get("victim_provider", "unknown")
            provider_counts[prov] = provider_counts.get(prov, 0) + 1

        operator_counts: Dict[str, int] = {}
        for p in payloads:
            op = p.get("operator_applied") or "none"
            operator_counts[op] = operator_counts.get(op, 0) + 1

        feat_stats = FeatureStats(
            length_mean=float(np.mean(lengths)),
            length_std=float(np.std(lengths)),
            length_median=float(np.median(lengths)),
            length_min=int(np.min(lengths)),
            length_max=int(np.max(lengths)),
            category_counts=category_counts,
            provider_counts=provider_counts,
            operator_counts=operator_counts,
        )

        # --- 3. Feature Engineering ---
        texts = [p.get("attack_text", "") for p in payloads]
        y = np.array(categories)

        # 3a. TF-IDF: word n-grams (1,2) + char n-grams (3,5) → 800 features total
        tfidf_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=400,
            sublinear_tf=True,
            min_df=1,
        )
        tfidf_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=400,
            sublinear_tf=True,
            min_df=1,
        )
        X_word = tfidf_word.fit_transform(texts).toarray()
        X_char = tfidf_char.fit_transform(texts).toarray()

        # 3b. 15-dim structural/syntactic features
        struct_feats = _build_structural_features(payloads)
        scaler = MinMaxScaler()
        X_struct = scaler.fit_transform(struct_feats)

        # 3c. Concatenate: 400 + 400 + 15 = 815 features
        X = np.hstack([X_word, X_char, X_struct])

        # --- 4. Group Identification for Honest Generalization (Fix Issue 14) ---
        # Group payloads by root template/prefix hash to eliminate template leakage between train and test splits
        groups = []
        for p in payloads:
            txt = p.get("attack_text", "").strip()
            clean_prefix = re.sub(r"\b(CANARY_[A-Za-z0-9_]+|[A-Z]{3,})\b", "", txt[:50]).strip()
            gid = hashlib.sha256(clean_prefix.encode("utf-8")).hexdigest()[:8]
            groups.append(gid)

        min_class_count = min(category_counts.values())
        k_folds = max(2, min(5, min_class_count))
        unique_groups = len(set(groups))

        # --- 5. Classifier: Complement Naive Bayes (Fast, High-Precision, Windows-Safe) ---
        clf_nb = ComplementNB(alpha=0.5)

        # --- 6. Grouped vs Stratified Cross-Validation ---
        if unique_groups >= k_folds:
            cv_splitter = StratifiedGroupKFold(n_splits=k_folds)
            cv_scores = cross_val_score(clf_nb, X, y, cv=cv_splitter, groups=groups, scoring="accuracy")
            cv_desc = f"StratifiedGroupKFold({k_folds}-fold, {unique_groups} groups)"
        else:
            cv_splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf_nb, X, y, cv=cv_splitter, scoring="accuracy")
            cv_desc = f"StratifiedKFold({k_folds}-fold)"

        logger.info("Classifier CV (%s): %.3f +- %.3f", cv_desc, np.mean(cv_scores), np.std(cv_scores))

        # Train on full corpus for the per-class classification report
        clf_nb.fit(X, y)
        y_pred      = clf_nb.predict(X)
        report_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)
        cm          = confusion_matrix(y, y_pred, labels=unique_categories).tolist()

        # Build and persist the runtime classifier bundle for sub-1ms Gateway Tier 1 inspection (Issue 9)
        cv_accuracy = float(np.mean(cv_scores))
        bundle = AttackClassifierBundle(
            clf=clf_nb,
            tfidf_word=tfidf_word,
            tfidf_char=tfidf_char,
            scaler=scaler,
            classes=unique_categories,
            cv_accuracy=cv_accuracy,
            training_data_hash=training_data_hash,
        )
        try:
            bundle.save()
        except Exception as exc:
            logger.warning("Could not persist AttackClassifierBundle: %s", exc)

        model_name = (
            f"ComplementNB(alpha=0.5) | "
            f"TF-IDF(word1-2+char3-5, 800feat) + 15-Structural | "
            f"CV={np.mean(cv_scores)*100:.1f}% (+-{np.std(cv_scores)*100:.1f}%)"
        )

        model_perf = MLModelPerformance(
            model_name=model_name,
            cv_mean_accuracy=float(np.mean(cv_scores)),
            cv_std_accuracy=float(np.std(cv_scores)),
            classes=unique_categories,
            classification_report=report_dict,
            confusion_matrix=cm,
        )

        return MLAnalysisReport(
            vector_stats=vec_stats,
            feature_stats=feat_stats,
            model_performance=model_perf,
        )


def auto_retrain_and_update_report(
    store: Optional[QdrantStore] = None,
    report_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Retrains the attack vector classifier against the active Qdrant store
    and updates ml_vector_analysis_report.json.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    analyzer = AttackVectorMLAnalyzer(store=store)
    report = analyzer.compute_analysis()

    target_path = Path(report_path) if report_path else (Path(__file__).parent.parent.parent / "ml_vector_analysis_report.json")

    # Build comprehensive payload supporting both nested and flat schemas
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # Flat legacy/direct fields
        "sample_count": report.vector_stats.sample_count,
        "vector_dim": report.vector_stats.vector_dim,
        "pairwise_similarity_mean": report.vector_stats.pairwise_similarity_mean,
        "pairwise_similarity_std": report.vector_stats.pairwise_similarity_std,
        "pca_top_3_variance": report.vector_stats.pca_top_3_variance,
        "pca_cumulative_variance_top_5": report.vector_stats.pca_cumulative_variance_top_5,
        "silhouette_score": report.vector_stats.silhouette_score,
        "category_counts": report.feature_stats.category_counts,
        "classifier_name": report.model_performance.model_name,
        "cv_mean_accuracy": report.model_performance.cv_mean_accuracy,
        "cv_std_accuracy": report.model_performance.cv_std_accuracy,
        "classification_report": report.model_performance.classification_report,
        "confusion_matrix": report.model_performance.confusion_matrix,
        # Nested object fields for schema completeness
        "vector_stats": {
            "total_vectors": report.vector_stats.sample_count,
            "sample_count": report.vector_stats.sample_count,
            "vector_dim": report.vector_stats.vector_dim,
            "category_distribution": report.feature_stats.category_counts,
            "pairwise_similarity_mean": report.vector_stats.pairwise_similarity_mean,
            "pairwise_similarity_std": report.vector_stats.pairwise_similarity_std,
            "pca_top_3_variance": report.vector_stats.pca_top_3_variance,
            "pca_cumulative_variance_top_5": report.vector_stats.pca_cumulative_variance_top_5,
            "silhouette_score": report.vector_stats.silhouette_score,
        },
        "feature_stats": {
            "length_mean": report.feature_stats.length_mean,
            "length_std": report.feature_stats.length_std,
            "length_median": report.feature_stats.length_median,
            "length_min": report.feature_stats.length_min,
            "length_max": report.feature_stats.length_max,
            "category_counts": report.feature_stats.category_counts,
            "provider_counts": report.feature_stats.provider_counts,
            "operator_counts": report.feature_stats.operator_counts,
        },
        "model_performance": {
            "model_name": report.model_performance.model_name,
            "cv_mean_accuracy": report.model_performance.cv_mean_accuracy,
            "cv_std_accuracy": report.model_performance.cv_std_accuracy,
            "classes": report.model_performance.classes,
            "classification_report": report.model_performance.classification_report,
            "confusion_matrix": report.model_performance.confusion_matrix,
        },
    }

    target_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Updated ML vector analysis report at %s with %d samples", target_path, report.vector_stats.sample_count)
    return data

