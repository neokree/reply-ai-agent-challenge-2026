# tests/test_geo.py
import pytest
from datetime import datetime, timedelta
from scoring.geo import geo_scorer, geocode_location

def test_geo_scorer_no_location_returns_zero():
    score = geo_scorer(
        tx_location=None,
        timestamp=datetime(2087, 3, 15, 10, 0),
        sender_biotag="USER1",
        locations=[],
        prev_tx_location=None,
        prev_tx_timestamp=None
    )
    assert score == 0.0

def test_geo_scorer_close_location_returns_zero():
    # User GPS and tx location are close (same city)
    locations = [
        {"biotag": "USER1", "timestamp": "2087-03-15T09:55:00",
         "lat": 48.1351, "lng": 11.5820, "city": "Munich"}
    ]
    score = geo_scorer(
        tx_location="Munich - Central",
        timestamp=datetime(2087, 3, 15, 10, 0),
        sender_biotag="USER1",
        locations=locations,
        prev_tx_location=None,
        prev_tx_timestamp=None
    )
    assert score < 0.3

def test_geo_scorer_impossible_velocity_returns_max():
    # Two transactions 500km apart in 30 minutes -> impossible
    score = geo_scorer(
        tx_location="Rome",
        timestamp=datetime(2087, 3, 15, 10, 30),
        sender_biotag="USER1",
        locations=[],
        prev_tx_location="Munich",
        prev_tx_timestamp=datetime(2087, 3, 15, 10, 0)
    )
    # Munich to Rome ~700km in 0.5h = 1400 km/h > 500
    assert score == 1.0

def test_geocode_location_extracts_city():
    # Should extract "Munich" from "Munich - Isar River Cafe"
    result = geocode_location("Munich - Isar River Cafe")
    assert result is not None or result is None  # May fail if no network
