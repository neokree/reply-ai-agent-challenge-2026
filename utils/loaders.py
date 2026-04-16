# utils/loaders.py
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_data(dataset_name: str) -> dict:
    """
    Load all data files for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "The Truman Show")

    Returns:
        Dictionary with all loaded data
    """
    base_path = Path(__file__).parent.parent
    train_path = base_path / "training-dataset" / f"{dataset_name} - train"
    eval_path = base_path / "evaluation-dataset" / f"{dataset_name} - validation"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {eval_path}")

    # Load transactions
    train_tx = pd.read_csv(train_path / "transactions.csv")
    eval_tx = pd.read_csv(eval_path / "transactions.csv")

    # Parse timestamps
    train_tx["timestamp"] = pd.to_datetime(train_tx["timestamp"])
    eval_tx["timestamp"] = pd.to_datetime(eval_tx["timestamp"])

    # Load JSON files
    with open(train_path / "users.json") as f:
        users = json.load(f)

    with open(train_path / "sms.json") as f:
        train_sms = json.load(f)
    with open(eval_path / "sms.json") as f:
        eval_sms = json.load(f)

    with open(train_path / "mails.json") as f:
        train_mails = json.load(f)
    with open(eval_path / "mails.json") as f:
        eval_mails = json.load(f)

    with open(train_path / "locations.json") as f:
        train_locations = json.load(f)
    with open(eval_path / "locations.json") as f:
        eval_locations = json.load(f)

    # Check for audio (only Deus Ex has it)
    audio_path = train_path / "audio"
    has_audio = audio_path.exists() and any(audio_path.iterdir())

    return {
        "dataset_name": dataset_name,
        "train_transactions": train_tx,
        "eval_transactions": eval_tx,
        "users": users,
        "train_sms": train_sms,
        "eval_sms": eval_sms,
        "train_mails": train_mails,
        "eval_mails": eval_mails,
        "train_locations": train_locations,
        "eval_locations": eval_locations,
        "has_audio": has_audio,
        "audio_path": str(audio_path) if has_audio else None,
    }
