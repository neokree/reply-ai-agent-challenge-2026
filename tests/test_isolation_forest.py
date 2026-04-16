# tests/test_isolation_forest.py
import pytest
import numpy as np
from ml.isolation_forest import IsolationForestScorer

def test_isolation_forest_fit_and_score():
    # Normal training data
    train_features = np.array([
        [0.1, 0.1, 0.1],
        [0.15, 0.12, 0.08],
        [0.2, 0.1, 0.15],
        [0.1, 0.2, 0.1],
    ])

    scorer = IsolationForestScorer()
    scorer.fit(train_features)

    # Normal point should have low score
    normal = np.array([[0.12, 0.11, 0.1]])
    scores = scorer.score(normal)
    assert scores[0] < 0.5

    # Anomaly should have high score
    anomaly = np.array([[0.9, 0.9, 0.9]])
    scores = scorer.score(anomaly)
    assert scores[0] > 0.5

def test_isolation_forest_returns_normalized_scores():
    train = np.random.rand(100, 5) * 0.3
    scorer = IsolationForestScorer()
    scorer.fit(train)

    test = np.random.rand(20, 5)
    scores = scorer.score(test)

    assert all(0 <= s <= 1 for s in scores)
