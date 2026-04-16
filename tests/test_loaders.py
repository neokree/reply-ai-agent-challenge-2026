# tests/test_loaders.py
import pytest
from utils.loaders import load_data

def test_load_data_returns_dict_with_expected_keys():
    data = load_data("The Truman Show")

    assert "train_transactions" in data
    assert "eval_transactions" in data
    assert "users" in data
    assert "train_sms" in data
    assert "train_mails" in data
    assert "train_locations" in data
    assert "eval_sms" in data
    assert "eval_mails" in data
    assert "eval_locations" in data

def test_load_data_transactions_have_required_columns():
    data = load_data("The Truman Show")

    required_cols = ["transaction_id", "sender_id", "recipient_id",
                     "transaction_type", "amount", "timestamp"]
    for col in required_cols:
        assert col in data["train_transactions"].columns

def test_load_data_invalid_dataset_raises():
    with pytest.raises(FileNotFoundError):
        load_data("Nonexistent Dataset")
