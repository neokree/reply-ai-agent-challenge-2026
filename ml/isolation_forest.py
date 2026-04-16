# ml/isolation_forest.py
import numpy as np
from sklearn.ensemble import IsolationForest

class IsolationForestScorer:
    """Wrapper for scikit-learn Isolation Forest with normalized scores."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self._fitted = False

    def fit(self, features: np.ndarray) -> None:
        """Fit the model on training features (normal behavior)."""
        self.model.fit(features)
        self._fitted = True

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Score features for anomaly.

        Returns:
            Array of scores 0.0-1.0 (higher = more anomalous)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # decision_function returns negative for anomalies, positive for normal
        # We want: high score (close to 1) for anomalies, low score (close to 0) for normal
        raw_scores = self.model.decision_function(features)

        # Use sigmoid-like normalization around the decision boundary (0)
        # Shift so that 0 (boundary) maps to 0.5, very negative maps to 1, very positive maps to 0
        # Use tanh-like scaling: score = 0.5 - 0.5 * tanh(raw_score)
        # This makes negative scores -> high (approaching 1), positive scores -> low (approaching 0)

        # For simplicity, just invert and use sigmoid
        # Score = 1 / (1 + exp(raw_score))
        # This naturally makes negative raw_scores -> scores close to 1
        normalized = 1.0 / (1.0 + np.exp(raw_scores))

        return normalized
