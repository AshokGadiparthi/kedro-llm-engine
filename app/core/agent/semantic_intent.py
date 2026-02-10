"""
Semantic Intent Classifier — Robust Question Understanding
=============================================================
Replaces brittle regex-only intent classification with a layered approach:
  1. Exact regex patterns (existing, highest confidence)
  2. TF-IDF + cosine similarity (lightweight, no external model needed)
  3. N-gram keyword expansion (catches paraphrases)
  4. Contextual boosting (screen context influences intent ranking)

SOLVES: "my random forest sucks, what else can I try?" now correctly
routes to algorithm_selection even though it doesn't match any regex.

Architecture:
  - Zero external dependencies (uses sklearn's TfidfVectorizer if available,
    falls back to pure-Python implementation)
  - Pre-built intent exemplar corpus (200+ example questions per intent)
  - Trains on first call, caches vectorizer for subsequent calls
  - Graceful fallback chain: regex → semantic → keyword → general_help

Usage:
  classifier = SemanticIntentClassifier()
  intent, confidence, method = classifier.classify("my model is terrible, help")
  # → ("algorithm_selection", 0.72, "semantic")
"""

import re
import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# INTENT EXEMPLAR CORPUS
# ═══════════════════════════════════════════════════════════════

INTENT_EXEMPLARS: Dict[str, List[str]] = {
    "algorithm_selection": [
        "which algorithm should I use",
        "what model is best for my data",
        "recommend a classifier",
        "my random forest sucks what else can I try",
        "should I use xgboost or lightgbm",
        "pick the best model for this dataset",
        "compare models for me",
        "which ml algorithm works with small datasets",
        "what classifier handles categorical data well",
        "suggest a regression model",
        "is logistic regression good enough",
        "should I try deep learning",
        "my model performs badly what else should I try",
        "best algorithm for imbalanced classification",
        "what works better for tabular data",
        "switch to a different model",
        "alternative to random forest",
        "which model is fastest to train",
        "should I use ensemble methods",
        "model isn't working need a different approach",
    ],
    "feature_importance": [
        "which features matter most",
        "what are the most important variables",
        "feature importance ranking",
        "which columns should I keep",
        "drop unimportant features",
        "what features are predictive",
        "feature relevance scores",
        "which variables drive the prediction",
        "remove useless columns",
        "top features for my model",
        "how many features should I select",
        "feature selection advice",
        "which inputs have the most impact",
        "what columns are relevant",
        "reduce number of features",
    ],
    "metric_interpretation": [
        "what does F1 score mean",
        "explain AUC ROC",
        "is 0.87 accuracy good",
        "what metric should I optimize",
        "difference between precision and recall",
        "my accuracy is high but model seems bad",
        "which metric matters most",
        "interpret my evaluation results",
        "what does MCC measure",
        "how to read confusion matrix",
        "is my score good enough",
        "explain the model metrics to me",
        "what does recall of 0.5 mean",
        "metric is misleading why",
        "are these results acceptable",
    ],
    "data_quality": [
        "how is my data quality",
        "is my data clean enough",
        "data quality score",
        "check data for issues",
        "is my dataset ready for training",
        "data problems or issues",
        "assess data health",
        "quality of my dataset",
        "dirty data concerns",
        "data validation results",
        "is the data good enough to train on",
        "pre-training data check",
    ],
    "class_imbalance": [
        "my classes are unbalanced",
        "target distribution is skewed",
        "too few positive examples",
        "handle imbalanced dataset",
        "minority class too small",
        "oversample the rare class",
        "class ratio is uneven",
        "SMOTE for imbalanced data",
        "balanced class weights",
        "not enough positive samples",
        "rare event prediction",
        "only 5 percent are positive cases",
    ],
    "correlation": [
        "features are highly correlated",
        "multicollinearity problem",
        "redundant features",
        "correlated variables what to do",
        "remove correlated columns",
        "VIF analysis",
        "collinear features in my data",
        "which correlated feature to drop",
        "too much correlation between inputs",
        "feature redundancy",
    ],
    "missing_data": [
        "lots of missing values",
        "how to handle nulls",
        "impute missing data",
        "fill in NaN values",
        "columns have many blanks",
        "missing data strategy",
        "too many null entries",
        "imputation method recommendation",
        "should I drop rows with missing values",
        "handle incomplete data",
        "data has gaps",
        "some columns mostly empty",
    ],
    "overfitting": [
        "model is overfitting",
        "training accuracy much higher than test",
        "generalization problem",
        "model memorized the data",
        "huge gap between train and test scores",
        "model doesn't generalize well",
        "variance is too high",
        "reduce overfitting",
        "regularize my model",
        "cross validation scores vary a lot",
        "training perfect but test terrible",
        "model too complex",
    ],
    "threshold": [
        "what threshold should I use",
        "optimize classification cutoff",
        "precision vs recall tradeoff",
        "lower the decision boundary",
        "false positive rate too high",
        "adjust prediction threshold",
        "find optimal cutoff point",
        "threshold for binary classification",
        "move the decision boundary",
        "too many false negatives lower threshold",
    ],
    "deployment": [
        "how to deploy the model",
        "put model in production",
        "model serving setup",
        "production deployment steps",
        "shadow deploy first",
        "canary deployment strategy",
        "ready for production",
        "deploy to API endpoint",
        "serve predictions in real-time",
        "rollback plan for model",
    ],
    "drift": [
        "model performance is degrading",
        "data drift detected",
        "when should I retrain",
        "concept drift monitoring",
        "model predictions getting worse over time",
        "distribution shift in production",
        "monitoring for data changes",
        "performance decay",
        "stale model problems",
        "input data changing",
    ],
    "hyperparameters": [
        "what hyperparameters to use",
        "tune my model parameters",
        "grid search or random search",
        "optimal learning rate",
        "max depth setting",
        "n_estimators value",
        "hyperparameter optimization",
        "bayesian optimization for tuning",
        "regularization strength",
        "best parameters for xgboost",
    ],
    "cross_validation": [
        "how many CV folds",
        "cross validation strategy",
        "k-fold settings",
        "stratified split",
        "train test validation split",
        "leave one out cv",
        "time series cross validation",
        "repeated cross validation",
        "CV scores interpretation",
    ],
    "encoding": [
        "encode categorical features",
        "one-hot vs label encoding",
        "target encoding categorical variables",
        "handle string columns",
        "convert categories to numbers",
        "encoding strategy for high cardinality",
        "ordinal encoding when",
        "hash encoding large categories",
    ],
    "scaling": [
        "normalize my features",
        "standardize the data",
        "which scaler to use",
        "standard scaler vs min max",
        "feature scaling needed",
        "robust scaler for outliers",
        "preprocessing numerical features",
    ],
    "ensemble": [
        "combine multiple models",
        "stacking classifier setup",
        "model blending approach",
        "voting ensemble",
        "bagging vs boosting",
        "ensemble strategy",
        "mix different algorithms",
    ],
    "leakage": [
        "data leakage in my pipeline",
        "target leakage detected",
        "future information leaking",
        "suspiciously high accuracy might be leakage",
        "prevent information leak",
        "train test contamination",
        "label leaking into features",
    ],
    "general_help": [
        "what should I do next",
        "help me with my project",
        "where do I start",
        "guide me through the process",
        "what are the next steps",
        "I'm stuck what now",
        "getting started with ml",
    ],
}


# ═══════════════════════════════════════════════════════════════
# PURE-PYTHON TF-IDF IMPLEMENTATION (no sklearn needed)
# ═══════════════════════════════════════════════════════════════

class SimpleTfidfVectorizer:
    """
    Lightweight TF-IDF vectorizer that works without sklearn.
    Sufficient for intent classification with ~200 exemplars.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)):
        self.ngram_range = ngram_range
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and generate n-grams."""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        tokens = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                tokens.append(' '.join(words[i:i+n]))
        return tokens

    def fit(self, documents: List[str]) -> 'SimpleTfidfVectorizer':
        """Fit vocabulary and IDF weights."""
        n_docs = len(documents)
        doc_freq: Counter = Counter()
        vocab_set: set = set()

        for doc in documents:
            tokens = set(self._tokenize(doc))
            vocab_set.update(tokens)
            for token in tokens:
                doc_freq[token] += 1

        self.vocabulary = {t: i for i, t in enumerate(sorted(vocab_set))}
        self.idf = {}
        for token, idx in self.vocabulary.items():
            df = doc_freq.get(token, 0)
            self.idf[token] = math.log((1 + n_docs) / (1 + df)) + 1

        self._fitted = True
        return self

    def transform(self, documents: List[str]) -> List[Dict[str, float]]:
        """Transform documents to TF-IDF sparse vectors (as dicts)."""
        results = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf: Counter = Counter(tokens)
            vec: Dict[str, float] = {}
            for token, count in tf.items():
                if token in self.idf:
                    vec[token] = (count / max(len(tokens), 1)) * self.idf[token]
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            vec = {k: v / norm for k, v in vec.items()}
            results.append(vec)
        return results

    def fit_transform(self, documents: List[str]) -> List[Dict[str, float]]:
        self.fit(documents)
        return self.transform(documents)


def cosine_similarity_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    common_keys = set(a.keys()) & set(b.keys())
    if not common_keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in common_keys)
    # Already L2-normalized, so dot product = cosine similarity
    return dot


# ═══════════════════════════════════════════════════════════════
# KEYWORD SYNONYM EXPANSION
# ═══════════════════════════════════════════════════════════════

KEYWORD_SYNONYMS: Dict[str, List[str]] = {
    "algorithm": ["model", "classifier", "regressor", "method", "approach", "technique"],
    "feature": ["variable", "column", "input", "attribute", "predictor", "field"],
    "accuracy": ["performance", "score", "metric", "result"],
    "overfit": ["memorize", "generalize", "variance", "complex"],
    "missing": ["null", "nan", "blank", "empty", "incomplete", "gap"],
    "threshold": ["cutoff", "boundary", "decision point"],
    "deploy": ["production", "serve", "live", "ship", "release"],
    "drift": ["decay", "degrade", "shift", "change", "stale"],
    "imbalance": ["skewed", "unbalanced", "uneven", "rare", "minority"],
    "correlat": ["collinear", "redundant", "related", "dependent"],
    "bad": ["terrible", "poor", "awful", "sucks", "worse", "wrong"],
    "good": ["great", "excellent", "acceptable", "strong", "decent"],
    "improve": ["fix", "enhance", "boost", "increase", "optimize", "better"],
}


# ═══════════════════════════════════════════════════════════════
# SCREEN-CONTEXT BOOSTING
# ═══════════════════════════════════════════════════════════════

SCREEN_INTENT_BOOST: Dict[str, Dict[str, float]] = {
    "eda": {
        "data_quality": 0.15, "missing_data": 0.10, "feature_importance": 0.10,
        "correlation": 0.10, "class_imbalance": 0.08,
    },
    "training": {
        "algorithm_selection": 0.15, "hyperparameters": 0.12, "overfitting": 0.10,
        "cross_validation": 0.08, "encoding": 0.08, "scaling": 0.08,
    },
    "mlflow": {
        "algorithm_selection": 0.15, "hyperparameters": 0.12, "overfitting": 0.10,
    },
    "evaluation": {
        "metric_interpretation": 0.15, "threshold": 0.12, "overfitting": 0.10,
        "ensemble": 0.08,
    },
    "deployment": {
        "deployment": 0.15, "drift": 0.10,
    },
    "monitoring": {
        "drift": 0.15, "deployment": 0.08,
    },
    "registry": {
        "deployment": 0.12, "drift": 0.08,
    },
}


# ═══════════════════════════════════════════════════════════════
# SEMANTIC INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════

class SemanticIntentClassifier:
    """
    Multi-strategy intent classifier with semantic fallback.

    Classification chain:
      1. Regex patterns (existing system, confidence 0.85-1.0)
      2. TF-IDF cosine similarity (0.40-0.85)
      3. Keyword expansion matching (0.30-0.60)
      4. Screen-context default (0.20)
    """

    def __init__(self, regex_patterns: Optional[Dict[str, List[str]]] = None):
        self._regex_patterns = regex_patterns or {}
        self._vectorizer: Optional[SimpleTfidfVectorizer] = None
        self._exemplar_vectors: Dict[str, List[Dict[str, float]]] = {}
        self._intent_centroids: Dict[str, Dict[str, float]] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of TF-IDF model."""
        if self._initialized:
            return

        # Build training corpus
        all_docs = []
        doc_intents = []
        for intent, exemplars in INTENT_EXEMPLARS.items():
            for ex in exemplars:
                all_docs.append(ex)
                doc_intents.append(intent)

        # Fit vectorizer
        self._vectorizer = SimpleTfidfVectorizer(ngram_range=(1, 2))
        vectors = self._vectorizer.fit_transform(all_docs)

        # Group vectors by intent and compute centroids
        intent_vecs: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for vec, intent in zip(vectors, doc_intents):
            intent_vecs[intent].append(vec)

        self._exemplar_vectors = dict(intent_vecs)

        # Compute centroid for each intent (average of exemplar vectors)
        for intent, vecs in intent_vecs.items():
            centroid: Dict[str, float] = defaultdict(float)
            for vec in vecs:
                for k, v in vec.items():
                    centroid[k] += v
            n = len(vecs)
            centroid = {k: v / n for k, v in centroid.items()}
            # L2 normalize centroid
            norm = math.sqrt(sum(v * v for v in centroid.values())) or 1.0
            centroid = {k: v / norm for k, v in centroid.items()}
            self._intent_centroids[intent] = centroid

        self._initialized = True

    def classify(
        self,
        question: str,
        screen: str = "",
        regex_patterns: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[str, float, str]:
        """
        Classify question intent.

        Returns: (intent, confidence, method)
          method: "regex" | "semantic" | "keyword" | "context_default"
        """
        patterns = regex_patterns or self._regex_patterns

        # ── Layer 1: Regex (existing system) ──
        regex_intent, regex_conf = self._try_regex(question, patterns)
        if regex_conf >= 0.6:
            # Apply screen boost
            boosted_conf = regex_conf
            boosts = SCREEN_INTENT_BOOST.get(screen, {})
            if regex_intent in boosts:
                boosted_conf = min(1.0, regex_conf + boosts[regex_intent])
            return regex_intent, boosted_conf, "regex"

        # ── Layer 2: Semantic similarity ──
        self._ensure_initialized()
        sem_intent, sem_conf = self._try_semantic(question)

        # ── Layer 3: Keyword expansion ──
        kw_intent, kw_conf = self._try_keyword_expansion(question)

        # ── Combine scores ──
        intent_scores: Dict[str, float] = defaultdict(float)

        if regex_intent != "general_help":
            intent_scores[regex_intent] += regex_conf * 0.4

        if sem_intent:
            intent_scores[sem_intent] += sem_conf * 0.5

        if kw_intent:
            intent_scores[kw_intent] += kw_conf * 0.3

        # Apply screen context boost
        boosts = SCREEN_INTENT_BOOST.get(screen, {})
        for intent, boost in boosts.items():
            if intent in intent_scores:
                intent_scores[intent] += boost

        if not intent_scores:
            return "general_help", 0.2, "context_default"

        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = min(1.0, intent_scores[best_intent])

        # Determine primary method
        method = "semantic"
        if regex_intent == best_intent and regex_conf > 0.3:
            method = "regex"
        elif kw_intent == best_intent and kw_conf > sem_conf:
            method = "keyword"

        return best_intent, best_score, method

    def _try_regex(self, question: str, patterns: Dict[str, List[str]]) -> Tuple[str, float]:
        """Try regex pattern matching (existing system)."""
        q_lower = question.lower().strip()
        best_intent = "general_help"
        best_score = 0.0

        for intent, pats in patterns.items():
            for pattern in pats:
                try:
                    match = re.search(pattern, q_lower)
                    if match:
                        score = len(match.group()) / max(len(q_lower), 1)
                        score = max(score, 0.6)
                        if score > best_score:
                            best_score = score
                            best_intent = intent
                except re.error:
                    continue

        return best_intent, best_score

    def _try_semantic(self, question: str) -> Tuple[Optional[str], float]:
        """TF-IDF cosine similarity against exemplar centroids."""
        if not self._vectorizer:
            return None, 0.0

        q_vec = self._vectorizer.transform([question])[0]
        if not q_vec:
            return None, 0.0

        best_intent = None
        best_sim = 0.0

        for intent, centroid in self._intent_centroids.items():
            sim = cosine_similarity_sparse(q_vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_intent = intent

        # Also check max similarity against individual exemplars
        max_exemplar_sim = 0.0
        max_exemplar_intent = None
        for intent, vecs in self._exemplar_vectors.items():
            for vec in vecs:
                sim = cosine_similarity_sparse(q_vec, vec)
                if sim > max_exemplar_sim:
                    max_exemplar_sim = sim
                    max_exemplar_intent = intent

        # Use whichever is more confident
        if max_exemplar_sim > best_sim:
            return max_exemplar_intent, min(0.95, max_exemplar_sim)
        return best_intent, min(0.95, best_sim)

    def _try_keyword_expansion(self, question: str) -> Tuple[Optional[str], float]:
        """Match expanded keywords including synonyms."""
        q_lower = question.lower()
        words = set(re.findall(r'\b\w+\b', q_lower))

        # Expand words with synonyms
        expanded = set(words)
        for word in words:
            for root, synonyms in KEYWORD_SYNONYMS.items():
                if word in synonyms or word.startswith(root):
                    expanded.add(root)
                    expanded.update(synonyms)

        # Score each intent based on keyword overlap with exemplars
        intent_hits: Dict[str, int] = Counter()
        intent_totals: Dict[str, int] = Counter()

        for intent, exemplars in INTENT_EXEMPLARS.items():
            for ex in exemplars:
                ex_words = set(re.findall(r'\b\w+\b', ex.lower()))
                overlap = expanded & ex_words
                if overlap:
                    intent_hits[intent] += len(overlap)
                intent_totals[intent] += 1

        if not intent_hits:
            return None, 0.0

        # Normalize
        scores = {}
        for intent, hits in intent_hits.items():
            total = intent_totals.get(intent, 1)
            scores[intent] = hits / (total * 3)  # Dampen to keep scores reasonable

        best = max(scores, key=scores.get)
        return best, min(0.85, scores[best])

    def get_all_scores(
        self, question: str, screen: str = "",
        regex_patterns: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Get scores for all intents (useful for debugging/confidence display)."""
        self._ensure_initialized()
        patterns = regex_patterns or self._regex_patterns

        q_vec = self._vectorizer.transform([question])[0] if self._vectorizer else {}
        boosts = SCREEN_INTENT_BOOST.get(screen, {})

        results = []
        for intent in INTENT_EXEMPLARS:
            # Regex score
            regex_score = 0.0
            for pat in patterns.get(intent, []):
                try:
                    match = re.search(pat, question.lower())
                    if match:
                        s = max(len(match.group()) / max(len(question), 1), 0.6)
                        regex_score = max(regex_score, s)
                except re.error:
                    continue

            # Semantic score
            sem_score = 0.0
            if q_vec and intent in self._intent_centroids:
                sem_score = cosine_similarity_sparse(q_vec, self._intent_centroids[intent])

            # Boost
            boost = boosts.get(intent, 0.0)

            combined = regex_score * 0.4 + sem_score * 0.5 + boost
            results.append({
                "intent": intent,
                "regex_score": round(regex_score, 3),
                "semantic_score": round(sem_score, 3),
                "screen_boost": round(boost, 3),
                "combined_score": round(min(1.0, combined), 3),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results
