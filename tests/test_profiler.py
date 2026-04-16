import pytest
import pandas as pd
from datetime import datetime
from scoring.profiler import build_user_profiles

@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "transaction_id": ["tx1", "tx2", "tx3", "tx4"],
        "sender_id": ["USER1", "USER1", "USER1", "USER2"],
        "sender_iban": ["IBAN1", "IBAN1", "IBAN1", "IBAN2"],
        "recipient_id": ["SHOP1", "SHOP1", "SHOP2", "SHOP3"],
        "recipient_iban": ["", "", "", ""],
        "transaction_type": ["e-commerce", "e-commerce", "transfer", "e-commerce"],
        "amount": [100.0, 200.0, 1000.0, 50.0],
        "payment_method": ["debit card", "debit card", "debit card", "mobile device"],
        "timestamp": pd.to_datetime([
            "2087-01-10 10:00:00",
            "2087-01-11 14:00:00",
            "2087-01-12 11:00:00",
            "2087-01-13 22:00:00",
        ]),
    })

@pytest.fixture
def sample_users():
    return [
        {"iban": "IBAN1", "first_name": "John", "last_name": "Doe",
         "salary": 50000, "residence": {"lat": "45.0", "lng": "9.0"}},
        {"iban": "IBAN2", "first_name": "Jane", "last_name": "Smith",
         "salary": 60000, "residence": {"lat": "46.0", "lng": "10.0"}},
    ]

def test_build_profiles_returns_dict_keyed_by_iban(sample_transactions, sample_users):
    profiles = build_user_profiles(sample_transactions, sample_users)

    assert "IBAN1" in profiles
    assert "IBAN2" in profiles

def test_profile_has_required_fields(sample_transactions, sample_users):
    profiles = build_user_profiles(sample_transactions, sample_users)
    profile = profiles["IBAN1"]

    assert "avg_amount_by_type" in profile
    assert "std_amount_by_type" in profile
    assert "typical_hours" in profile
    assert "weekend_active" in profile
    assert "payment_methods" in profile
    assert "known_recipients" in profile
    assert "salary" in profile

def test_profile_calculates_avg_amount_correctly(sample_transactions, sample_users):
    profiles = build_user_profiles(sample_transactions, sample_users)
    profile = profiles["IBAN1"]

    # User1 has 2 e-commerce tx: 100 and 200, avg = 150
    assert profile["avg_amount_by_type"]["e-commerce"] == 150.0
