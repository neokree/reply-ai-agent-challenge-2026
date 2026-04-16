# tests/test_graph_analyzer.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from ml.graph_analyzer import GraphAnalyzer

@pytest.fixture
def sample_transactions():
    base_time = datetime(2087, 3, 15, 10, 0)
    return pd.DataFrame({
        "transaction_id": ["tx1", "tx2", "tx3", "tx4", "tx5"],
        "sender_id": ["A", "B", "C", "A", "B"],
        "recipient_id": ["B", "C", "A", "D", "D"],
        "amount": [1000, 980, 960, 100, 200],
        "timestamp": [
            base_time,
            base_time + timedelta(minutes=30),
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=2),
            base_time + timedelta(hours=3),
        ],
    })

def test_graph_analyzer_builds_graph(sample_transactions):
    analyzer = GraphAnalyzer()
    analyzer.build_graph(sample_transactions)

    assert analyzer.graph is not None
    assert len(analyzer.graph.nodes()) == 4  # A, B, C, D
    assert len(analyzer.graph.edges()) == 5

def test_graph_analyzer_detects_circular_flow(sample_transactions):
    analyzer = GraphAnalyzer()
    analyzer.build_graph(sample_transactions)

    # A -> B -> C -> A is a cycle
    cycles = analyzer.detect_circular_flows()
    assert len(cycles) > 0

def test_graph_analyzer_detects_mule(sample_transactions):
    # D receives from both A and B -> potential mule
    analyzer = GraphAnalyzer()
    analyzer.build_graph(sample_transactions)

    mules = analyzer.detect_mule_accounts(in_threshold=2, out_threshold=0)
    assert "D" in mules

def test_graph_analyzer_scores_transactions(sample_transactions):
    analyzer = GraphAnalyzer()
    analyzer.build_graph(sample_transactions)

    scores = analyzer.score_transactions(sample_transactions)
    assert len(scores) == len(sample_transactions)
    assert all(0 <= s <= 1 for s in scores.values())
