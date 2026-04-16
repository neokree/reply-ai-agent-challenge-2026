# tests/test_time.py
import pytest
from datetime import datetime
from scoring.time import time_scorer

@pytest.fixture
def profile():
    return {
        "typical_hours": [9, 10, 11, 12, 14, 15, 16, 17],
        "weekend_active": False,
    }

def test_time_scorer_normal_hour_returns_zero(profile):
    # Tuesday at 10:00 - normal hour
    ts = datetime(2087, 1, 6, 10, 0, 0)  # Tuesday
    score = time_scorer(ts, profile)
    assert score == 0.0

def test_time_scorer_night_hour_returns_high_score(profile):
    # Tuesday at 3:00 AM - far from typical
    ts = datetime(2087, 1, 6, 3, 0, 0)
    score = time_scorer(ts, profile)
    assert score > 0.5

def test_time_scorer_weekend_boost_when_not_active(profile):
    # Saturday at 10:00 - typical hour but weekend
    ts = datetime(2087, 1, 4, 10, 0, 0)  # Saturday
    score = time_scorer(ts, profile)
    assert score == 0.3  # weekend boost only

def test_time_scorer_no_weekend_boost_when_active(profile):
    profile["weekend_active"] = True
    ts = datetime(2087, 1, 4, 10, 0, 0)  # Saturday
    score = time_scorer(ts, profile)
    assert score == 0.0
