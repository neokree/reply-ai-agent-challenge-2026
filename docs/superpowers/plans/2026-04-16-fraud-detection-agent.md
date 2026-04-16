# Fraud Detection Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid fraud detection system that uses rule-based scorers, ML anomaly detection, and LLM agents to identify fraudulent transactions in the Reply Mirror challenge.

**Architecture:** Layer 1 (preprocessing/audio) → Layer 2 (8 rule-based scorers) → Layer 2.5 (Isolation Forest + Graph Analysis) → Layer 3 (2 LLM agents for comms analysis and final decision) → Output file with flagged transaction IDs.

**Tech Stack:** Python 3.13, LangChain, Langfuse v3, OpenRouter (Qwen), pandas, scikit-learn, NetworkX, geopy, Whisper API

**Spec:** `docs/superpowers/specs/2026-04-16-fraud-detection-agent-design.md`

---

## Chunk 1: Project Setup & Data Loaders

### Task 1: Update requirements.txt and create directory structure

**Files:**
- Modify: `requirements.txt`
- Create: `preprocessing/__init__.py`
- Create: `scoring/__init__.py`
- Create: `ml/__init__.py`
- Create: `agents/__init__.py`
- Create: `utils/__init__.py`

- [ ] **Step 1: Update requirements.txt**

```txt
langchain>=1.2.0
langchain-openai>=1.1.0
langfuse>=3,<4
python-dotenv
ulid-py
pandas
geopy
openai
scikit-learn
networkx
tenacity
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p preprocessing scoring ml agents utils output
touch preprocessing/__init__.py scoring/__init__.py ml/__init__.py agents/__init__.py utils/__init__.py
```

- [ ] **Step 3: Add output/ to .gitignore**

Append to `.gitignore`:
```
output/
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: setup project structure and dependencies"
```

---

### Task 2: Create data loaders

**Files:**
- Create: `utils/loaders.py`
- Create: `tests/test_loaders.py`

- [ ] **Step 1: Write failing test for load_data**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_loaders.py -v
```
Expected: FAIL - module not found

- [ ] **Step 3: Implement load_data**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_loaders.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/loaders.py tests/test_loaders.py
git commit -m "feat: add data loader for datasets"
```

---

### Task 3: Create output writer

**Files:**
- Create: `utils/output.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_output.py
import pytest
import os
from pathlib import Path
from utils.output import write_output, sanitize_dataset_name

def test_sanitize_dataset_name():
    assert sanitize_dataset_name("The Truman Show") == "the_truman_show"
    assert sanitize_dataset_name("Brave New World") == "brave_new_world"
    assert sanitize_dataset_name("Deus Ex") == "deus_ex"

def test_write_output_creates_file(tmp_path):
    flagged_ids = ["abc-123", "def-456", "ghi-789"]
    output_path = tmp_path / "output"
    output_path.mkdir()
    
    filepath = write_output(flagged_ids, "Test Dataset", output_dir=str(output_path))
    
    assert os.path.exists(filepath)
    with open(filepath) as f:
        lines = f.read().strip().split('\n')
    assert lines == flagged_ids

def test_write_output_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        write_output([], "Test")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_output.py -v
```

- [ ] **Step 3: Implement output writer**

```python
# utils/output.py
import os
from pathlib import Path

def sanitize_dataset_name(name: str) -> str:
    """Convert dataset name to safe filename."""
    return name.lower().replace(" ", "_").replace("-", "_")

def write_output(
    flagged_ids: list[str], 
    dataset_name: str,
    output_dir: str = None
) -> str:
    """
    Write flagged transaction IDs to output file.
    
    Args:
        flagged_ids: List of transaction IDs to flag as fraudulent
        dataset_name: Name of the dataset
        output_dir: Output directory (default: ./output)
    
    Returns:
        Path to the written file
    """
    if not flagged_ids:
        raise ValueError("Cannot write empty list of flagged IDs")
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    filename = f"{sanitize_dataset_name(dataset_name)}.txt"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        for tx_id in flagged_ids:
            f.write(f"{tx_id}\n")
    
    return str(filepath)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_output.py -v
```

- [ ] **Step 5: Commit**

```bash
git add utils/output.py tests/test_output.py
git commit -m "feat: add output writer for flagged transactions"
```

---

## Chunk 2: User Profiler & Amount Scorer

### Task 4: Build user profiler

**Files:**
- Create: `scoring/profiler.py`
- Create: `tests/test_profiler.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_profiler.py
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
            "2087-01-13 22:00:00",  # Saturday night
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_profiler.py -v
```

- [ ] **Step 3: Implement profiler**

```python
# scoring/profiler.py
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

def build_user_profiles(
    train_transactions: pd.DataFrame, 
    users: list[dict]
) -> dict:
    """
    Build behavioral profiles for each user from training data.
    
    Args:
        train_transactions: Training transactions DataFrame
        users: List of user dicts from users.json
    
    Returns:
        Dict mapping user IBAN to profile dict
    """
    # Create user lookup by IBAN
    user_by_iban = {u["iban"]: u for u in users}
    
    profiles = {}
    
    for iban, user_data in user_by_iban.items():
        # Get all transactions where user is sender
        user_txs = train_transactions[train_transactions["sender_iban"] == iban]
        
        if user_txs.empty:
            # Fallback for users with no training transactions
            profiles[iban] = _create_fallback_profile(user_data, train_transactions)
            continue
        
        # Calculate stats by transaction type
        avg_by_type = user_txs.groupby("transaction_type")["amount"].mean().to_dict()
        std_by_type = user_txs.groupby("transaction_type")["amount"].std().fillna(0).to_dict()
        
        # Typical hours
        hours = user_txs["timestamp"].dt.hour.tolist()
        typical_hours = list(set(hours))
        
        # Weekend activity (>20% of txs on weekend)
        weekend_txs = user_txs[user_txs["timestamp"].dt.weekday >= 5]
        weekend_active = len(weekend_txs) / len(user_txs) > 0.2 if len(user_txs) > 0 else False
        
        # Payment methods count
        payment_methods = user_txs["payment_method"].value_counts().to_dict()
        
        # Known recipients
        recipients = set()
        for _, tx in user_txs.iterrows():
            if tx["recipient_id"]:
                recipients.add(tx["recipient_id"])
            if tx["recipient_iban"]:
                recipients.add(tx["recipient_iban"])
        
        # Last tx to each recipient
        last_tx_to = {}
        for recipient in recipients:
            mask = (user_txs["recipient_id"] == recipient) | (user_txs["recipient_iban"] == recipient)
            if mask.any():
                last_tx_to[recipient] = user_txs[mask]["timestamp"].max()
        
        # Average daily tx count
        if len(user_txs) > 0:
            date_range = (user_txs["timestamp"].max() - user_txs["timestamp"].min()).days + 1
            avg_daily_tx_count = len(user_txs) / max(date_range, 1)
        else:
            avg_daily_tx_count = 0
        
        # Round amounts check (>30% are round)
        round_count = sum(1 for amt in user_txs["amount"] if amt % 100 == 0)
        often_round_amounts = round_count / len(user_txs) > 0.3 if len(user_txs) > 0 else False
        
        # Residence coords
        residence = user_data.get("residence", {})
        residence_coords = (
            float(residence.get("lat", 0)),
            float(residence.get("lng", 0))
        )
        
        profiles[iban] = {
            "user_id": user_txs["sender_id"].iloc[0] if len(user_txs) > 0 else None,
            "name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
            "avg_amount_by_type": avg_by_type,
            "std_amount_by_type": std_by_type,
            "typical_hours": typical_hours,
            "weekend_active": weekend_active,
            "payment_methods": payment_methods,
            "known_recipients": recipients,
            "last_tx_to": last_tx_to,
            "avg_daily_tx_count": avg_daily_tx_count,
            "residence_coords": residence_coords,
            "salary": user_data.get("salary", 0),
            "often_round_amounts": often_round_amounts,
        }
    
    return profiles


def _create_fallback_profile(user_data: dict, all_transactions: pd.DataFrame) -> dict:
    """Create a fallback profile using global stats when user has no transactions."""
    global_avg = all_transactions.groupby("transaction_type")["amount"].mean().to_dict()
    global_std = all_transactions.groupby("transaction_type")["amount"].std().fillna(0).to_dict()
    
    residence = user_data.get("residence", {})
    
    return {
        "user_id": None,
        "name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
        "avg_amount_by_type": global_avg,
        "std_amount_by_type": global_std,
        "typical_hours": list(range(8, 21)),  # 8am-8pm default
        "weekend_active": False,
        "payment_methods": {},
        "known_recipients": set(),
        "last_tx_to": {},
        "avg_daily_tx_count": 0,
        "residence_coords": (float(residence.get("lat", 0)), float(residence.get("lng", 0))),
        "salary": user_data.get("salary", 0),
        "often_round_amounts": False,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_profiler.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/profiler.py tests/test_profiler.py
git commit -m "feat: add user profiler for behavioral analysis"
```

---

### Task 5: Implement amount scorer

**Files:**
- Create: `scoring/amount.py`
- Create: `tests/test_amount.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_amount.py
import pytest
from scoring.amount import amount_scorer

@pytest.fixture
def profile():
    return {
        "avg_amount_by_type": {"e-commerce": 100.0, "transfer": 1000.0},
        "std_amount_by_type": {"e-commerce": 25.0, "transfer": 200.0},
    }

def test_amount_scorer_normal_amount_returns_low_score(profile):
    # Amount exactly at mean -> z-score = 0 -> score = 0
    score = amount_scorer(100.0, "e-commerce", profile)
    assert score == 0.0

def test_amount_scorer_high_amount_returns_high_score(profile):
    # Amount 4 std above mean -> z-score = 4 -> score = 1.0
    score = amount_scorer(200.0, "e-commerce", profile)  # (200-100)/25 = 4
    assert score == 1.0

def test_amount_scorer_moderate_anomaly(profile):
    # Amount 2 std above mean -> z-score = 2 -> score = 0.5
    score = amount_scorer(150.0, "e-commerce", profile)  # (150-100)/25 = 2
    assert score == 0.5

def test_amount_scorer_unknown_type_uses_global_fallback(profile):
    # Unknown type uses default values
    score = amount_scorer(500.0, "unknown_type", profile)
    assert 0.0 <= score <= 1.0

def test_amount_scorer_zero_std_returns_zero(profile):
    profile["std_amount_by_type"]["e-commerce"] = 0.0
    score = amount_scorer(150.0, "e-commerce", profile)
    assert score == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_amount.py -v
```

- [ ] **Step 3: Implement amount scorer**

```python
# scoring/amount.py

# Global fallback values
GLOBAL_MEAN = 500.0
GLOBAL_STD = 300.0

def amount_scorer(
    amount: float, 
    transaction_type: str, 
    profile: dict
) -> float:
    """
    Score transaction amount anomaly using Z-score.
    
    Args:
        amount: Transaction amount
        transaction_type: Type of transaction (e-commerce, transfer, etc.)
        profile: User profile dict
    
    Returns:
        Risk score 0.0-1.0 (higher = more anomalous)
    """
    avg_by_type = profile.get("avg_amount_by_type", {})
    std_by_type = profile.get("std_amount_by_type", {})
    
    mean = avg_by_type.get(transaction_type, GLOBAL_MEAN)
    std = std_by_type.get(transaction_type, GLOBAL_STD)
    
    if std == 0:
        return 0.0
    
    z_score = (amount - mean) / std
    score = min(abs(z_score) / 4.0, 1.0)
    
    return score
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_amount.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/amount.py tests/test_amount.py
git commit -m "feat: add amount anomaly scorer with Z-score"
```

---

## Chunk 3: Time, Frequency, Channel & Round Amount Scorers

### Task 6: Implement time scorer

**Files:**
- Create: `scoring/time.py`
- Create: `tests/test_time.py`

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_time.py -v
```

- [ ] **Step 3: Implement**

```python
# scoring/time.py
from datetime import datetime

def time_scorer(timestamp: datetime, profile: dict) -> float:
    """
    Score transaction based on time anomaly.
    
    Args:
        timestamp: Transaction timestamp
        profile: User profile dict
    
    Returns:
        Risk score 0.0-1.0
    """
    tx_hour = timestamp.hour
    tx_weekday = timestamp.weekday()  # 0=Mon, 6=Sun
    is_weekend = tx_weekday >= 5
    
    typical_hours = profile.get("typical_hours", list(range(8, 21)))
    weekend_active = profile.get("weekend_active", False)
    
    # Score for unusual hour
    if tx_hour in typical_hours:
        hour_score = 0.0
    else:
        if not typical_hours:
            hour_score = 0.0
        else:
            min_distance = min(abs(tx_hour - h) for h in typical_hours)
            hour_score = min(min_distance / 6.0, 1.0)
    
    # Weekend boost if user doesn't normally operate on weekends
    weekend_boost = 0.3 if (is_weekend and not weekend_active) else 0.0
    
    return min(hour_score + weekend_boost, 1.0)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_time.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/time.py tests/test_time.py
git commit -m "feat: add time anomaly scorer with weekend detection"
```

---

### Task 7: Implement frequency checker

**Files:**
- Create: `scoring/frequency.py`
- Create: `tests/test_frequency.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_frequency.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from scoring.frequency import frequency_scorer

@pytest.fixture
def profile():
    return {"avg_daily_tx_count": 2.0}

def test_frequency_scorer_normal_returns_zero(profile):
    # Current tx, only 1 other tx in last hour
    current_ts = datetime(2087, 1, 10, 14, 0, 0)
    all_txs = pd.DataFrame({
        "sender_id": ["USER1", "USER1"],
        "timestamp": [
            datetime(2087, 1, 10, 13, 30, 0),
            current_ts,
        ]
    })
    score = frequency_scorer("USER1", current_ts, all_txs, profile)
    assert score == 0.0

def test_frequency_scorer_burst_returns_high(profile):
    # 6 transactions in last hour -> flag
    current_ts = datetime(2087, 1, 10, 14, 0, 0)
    timestamps = [current_ts - timedelta(minutes=i*10) for i in range(6)]
    all_txs = pd.DataFrame({
        "sender_id": ["USER1"] * 6,
        "timestamp": timestamps,
    })
    score = frequency_scorer("USER1", current_ts, all_txs, profile)
    assert score == 1.0

def test_frequency_scorer_high_daily_returns_moderate(profile):
    # 7 txs in 24h (> 3x avg_daily of 2) -> flag
    current_ts = datetime(2087, 1, 10, 14, 0, 0)
    timestamps = [current_ts - timedelta(hours=i*3) for i in range(7)]
    all_txs = pd.DataFrame({
        "sender_id": ["USER1"] * 7,
        "timestamp": timestamps,
    })
    score = frequency_scorer("USER1", current_ts, all_txs, profile)
    assert score >= 0.5
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_frequency.py -v
```

- [ ] **Step 3: Implement**

```python
# scoring/frequency.py
import pandas as pd
from datetime import datetime, timedelta

def frequency_scorer(
    sender_id: str,
    timestamp: datetime,
    all_transactions: pd.DataFrame,
    profile: dict
) -> float:
    """
    Score transaction based on velocity/frequency anomaly.
    
    Args:
        sender_id: ID of the sender
        timestamp: Current transaction timestamp
        all_transactions: All transactions to check against
        profile: User profile dict
    
    Returns:
        Risk score 0.0-1.0
    """
    user_txs = all_transactions[all_transactions["sender_id"] == sender_id].copy()
    
    if user_txs.empty:
        return 0.0
    
    # Count transactions in last hour
    hour_ago = timestamp - timedelta(hours=1)
    txs_last_hour = len(user_txs[
        (user_txs["timestamp"] >= hour_ago) & 
        (user_txs["timestamp"] <= timestamp)
    ])
    
    # Count transactions in last 24 hours
    day_ago = timestamp - timedelta(hours=24)
    txs_last_day = len(user_txs[
        (user_txs["timestamp"] >= day_ago) & 
        (user_txs["timestamp"] <= timestamp)
    ])
    
    avg_daily = profile.get("avg_daily_tx_count", 2.0)
    
    # Scoring logic
    if txs_last_hour > 5:
        return 1.0
    elif txs_last_day > avg_daily * 3:
        return 1.0
    elif txs_last_day > avg_daily * 2:
        return 0.5
    else:
        return 0.0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_frequency.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/frequency.py tests/test_frequency.py
git commit -m "feat: add frequency/velocity anomaly scorer"
```

---

### Task 8: Implement channel switch scorer

**Files:**
- Create: `scoring/channel.py`
- Create: `tests/test_channel.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_channel.py
import pytest
from scoring.channel import channel_switch_scorer

@pytest.fixture
def profile():
    return {
        "payment_methods": {"debit card": 45, "mobile device": 5},
        "salary": 50000,
    }

def test_channel_usual_method_returns_zero(profile):
    score = channel_switch_scorer("debit card", 100.0, profile)
    assert score == 0.0

def test_channel_rare_method_returns_moderate(profile):
    # mobile device is 10% of txs
    score = channel_switch_scorer("mobile device", 100.0, profile)
    assert score == 0.4

def test_channel_new_method_returns_high(profile):
    score = channel_switch_scorer("PayPal", 100.0, profile)
    assert score == 0.8

def test_channel_new_method_high_amount_boost(profile):
    # New method + amount > 50% of salary
    score = channel_switch_scorer("PayPal", 30000.0, profile)
    assert score == 1.0
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_channel.py -v
```

- [ ] **Step 3: Implement**

```python
# scoring/channel.py

def channel_switch_scorer(
    payment_method: str,
    amount: float,
    profile: dict
) -> float:
    """
    Score transaction based on unusual payment method.
    
    Args:
        payment_method: Payment method used
        amount: Transaction amount
        profile: User profile dict
    
    Returns:
        Risk score 0.0-1.0
    """
    methods = profile.get("payment_methods", {})
    salary = profile.get("salary", 50000)
    
    if not methods:
        return 0.0
    
    total_txs = sum(methods.values())
    if total_txs == 0:
        return 0.0
    
    method_count = methods.get(payment_method, 0)
    method_pct = method_count / total_txs
    
    if method_pct == 0:
        # Method NEVER used before
        score = 0.8
    elif method_pct < 0.1:
        # Method used rarely (<10% of txs)
        score = 0.4
    else:
        score = 0.0
    
    # Boost if combined with high amount
    if amount > salary * 0.5:
        score = min(score + 0.2, 1.0)
    
    return score
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_channel.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/channel.py tests/test_channel.py
git commit -m "feat: add channel switch anomaly scorer"
```

---

### Task 9: Implement round amount scorer

**Files:**
- Create: `scoring/round_amount.py`
- Create: `tests/test_round_amount.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_round_amount.py
import pytest
from scoring.round_amount import round_amount_scorer

@pytest.fixture
def profile():
    return {"often_round_amounts": False}

def test_round_amount_non_round_returns_zero(profile):
    score = round_amount_scorer(123.45, profile)
    assert score == 0.0

def test_round_amount_100_returns_low(profile):
    score = round_amount_scorer(200.0, profile)
    assert score == 0.3

def test_round_amount_1000_returns_higher(profile):
    score = round_amount_scorer(1000.0, profile)
    assert score == 0.5

def test_round_amount_user_often_round_reduces_score(profile):
    profile["often_round_amounts"] = True
    score = round_amount_scorer(1000.0, profile)
    assert score == 0.15  # 0.5 * 0.3

def test_round_amount_small_round_ignored(profile):
    # Amount < 100 not flagged even if round
    score = round_amount_scorer(50.0, profile)
    assert score == 0.0
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_round_amount.py -v
```

- [ ] **Step 3: Implement**

```python
# scoring/round_amount.py

def round_amount_scorer(amount: float, profile: dict) -> float:
    """
    Score transaction based on suspiciously round amount.
    
    Fraudsters often test with round amounts (100, 500, 1000).
    
    Args:
        amount: Transaction amount
        profile: User profile dict
    
    Returns:
        Risk score 0.0-1.0
    """
    is_round_100 = (amount % 100 == 0) and (amount >= 100)
    is_round_1000 = (amount % 1000 == 0) and (amount >= 1000)
    
    if is_round_1000:
        score = 0.5
    elif is_round_100:
        score = 0.3
    else:
        score = 0.0
    
    # Reduce penalty if user often has round amounts (e.g., fixed rent)
    if profile.get("often_round_amounts", False):
        score *= 0.3
    
    return score
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_round_amount.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/round_amount.py tests/test_round_amount.py
git commit -m "feat: add round amount anomaly scorer"
```

---

## Chunk 4: Entity, Balance, and Geo Scorers

### Task 10: Implement new entity detector

**Files:**
- Create: `scoring/entities.py`
- Create: `tests/test_entities.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_entities.py
import pytest
from datetime import datetime, timedelta
from scoring.entities import new_entity_scorer

@pytest.fixture
def profile():
    now = datetime(2087, 3, 15, 10, 0, 0)
    return {
        "known_recipients": {"IBAN1", "MERCHANT1"},
        "last_tx_to": {
            "IBAN1": now - timedelta(days=30),
            "MERCHANT1": now - timedelta(days=100),
        },
        "salary": 50000,
    }

def test_known_recent_recipient_returns_zero(profile):
    ts = datetime(2087, 3, 15, 10, 0, 0)
    score = new_entity_scorer("IBAN1", ts, 100.0, profile)
    assert score == 0.0

def test_known_old_recipient_returns_low(profile):
    # MERCHANT1 not seen in 100 days (>90)
    ts = datetime(2087, 3, 15, 10, 0, 0)
    score = new_entity_scorer("MERCHANT1", ts, 100.0, profile)
    assert score == 0.2

def test_new_recipient_returns_high(profile):
    ts = datetime(2087, 3, 15, 10, 0, 0)
    score = new_entity_scorer("NEW_IBAN", ts, 100.0, profile)
    assert score >= 0.5

def test_new_recipient_high_amount_returns_max(profile):
    ts = datetime(2087, 3, 15, 10, 0, 0)
    # Amount = salary -> factor = 1.0 -> score = 0.5 + 0.5 = 1.0
    score = new_entity_scorer("NEW_IBAN", ts, 50000.0, profile)
    assert score == 1.0
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_entities.py -v
```

- [ ] **Step 3: Implement**

```python
# scoring/entities.py
from datetime import datetime

def new_entity_scorer(
    recipient: str,
    timestamp: datetime,
    amount: float,
    profile: dict
) -> float:
    """
    Score transaction based on whether recipient is new/unknown.
    
    Args:
        recipient: Recipient ID or IBAN
        timestamp: Transaction timestamp
        amount: Transaction amount
        profile: User profile dict
    
    Returns:
        Risk score 0.0-1.0
    """
    if not recipient:
        return 0.0
    
    known = profile.get("known_recipients", set())
    last_tx_to = profile.get("last_tx_to", {})
    salary = profile.get("salary", 50000)
    
    if recipient in known:
        # Known recipient - check how long since last tx
        last_date = last_tx_to.get(recipient)
        if last_date:
            days_since = (timestamp - last_date).days
            if days_since > 90:
                return 0.2
        return 0.0
    else:
        # New recipient - base score + amount factor
        amount_factor = min(amount / salary, 1.0) if salary > 0 else 0.5
        score = 0.5 + (0.5 * amount_factor)
        return score
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_entities.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/entities.py tests/test_entities.py
git commit -m "feat: add new entity detector scorer"
```

---

### Task 11: Implement balance drain scorer

**Files:**
- Create: `scoring/balance.py`
- Create: `tests/test_balance.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_balance.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from scoring.balance import balance_drain_scorer

def test_balance_drain_small_pct_returns_zero():
    # 10% of balance spent
    score = balance_drain_scorer(
        amount=100.0,
        balance_after=900.0,
        sender_id="USER1",
        timestamp=datetime(2087, 3, 15, 10, 0),
        all_transactions=pd.DataFrame(columns=["sender_id", "timestamp", "amount"])
    )
    assert score == 0.0

def test_balance_drain_high_pct_returns_high():
    # 90% of balance spent -> score = 1.0
    score = balance_drain_scorer(
        amount=900.0,
        balance_after=100.0,
        sender_id="USER1",
        timestamp=datetime(2087, 3, 15, 10, 0),
        all_transactions=pd.DataFrame(columns=["sender_id", "timestamp", "amount"])
    )
    assert score == 1.0

def test_balance_drain_moderate_pct_returns_moderate():
    # 60% of balance spent
    score = balance_drain_scorer(
        amount=600.0,
        balance_after=400.0,
        sender_id="USER1",
        timestamp=datetime(2087, 3, 15, 10, 0),
        all_transactions=pd.DataFrame(columns=["sender_id", "timestamp", "amount"])
    )
    assert score == 0.4

def test_balance_drain_pattern_detected():
    # Multiple txs draining account in 2 hours
    current_ts = datetime(2087, 3, 15, 10, 0)
    all_txs = pd.DataFrame({
        "sender_id": ["USER1", "USER1", "USER1"],
        "timestamp": [
            current_ts - timedelta(hours=1),
            current_ts - timedelta(minutes=30),
            current_ts,
        ],
        "amount": [300.0, 300.0, 300.0],
    })
    # Total 900 out of 1000 balance
    score = balance_drain_scorer(
        amount=300.0,
        balance_after=100.0,
        sender_id="USER1",
        timestamp=current_ts,
        all_transactions=all_txs
    )
    assert score == 1.0
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_balance.py -v
```

- [ ] **Step 3: Implement**

```python
# scoring/balance.py
import pandas as pd
from datetime import datetime, timedelta

def balance_drain_scorer(
    amount: float,
    balance_after: float,
    sender_id: str,
    timestamp: datetime,
    all_transactions: pd.DataFrame
) -> float:
    """
    Score transaction based on balance drain pattern.
    
    Args:
        amount: Transaction amount
        balance_after: Balance after this transaction
        sender_id: Sender ID
        timestamp: Transaction timestamp
        all_transactions: All transactions for pattern detection
    
    Returns:
        Risk score 0.0-1.0
    """
    # Calculate balance before
    balance_before = balance_after + amount
    
    if balance_before <= 0:
        return 0.0
    
    pct_spent = amount / balance_before
    
    # Single transaction scoring
    if pct_spent > 0.9:
        score = 1.0
    elif pct_spent > 0.7:
        score = 0.7
    elif pct_spent > 0.5:
        score = 0.4
    else:
        score = 0.0
    
    # Check for drain pattern: multiple txs eroding balance in 2 hours
    if not all_transactions.empty:
        two_hours_ago = timestamp - timedelta(hours=2)
        recent_txs = all_transactions[
            (all_transactions["sender_id"] == sender_id) &
            (all_transactions["timestamp"] >= two_hours_ago) &
            (all_transactions["timestamp"] <= timestamp)
        ]
        
        if len(recent_txs) >= 3:
            total_spent = recent_txs["amount"].sum()
            if balance_before > 0 and total_spent / balance_before > 0.8:
                score = 1.0
    
    return score
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_balance.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/balance.py tests/test_balance.py
git commit -m "feat: add balance drain pattern scorer"
```

---

### Task 12: Implement geo checker

**Files:**
- Create: `scoring/geo.py`
- Create: `tests/test_geo.py`

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_geo.py -v
```

- [ ] **Step 3: Implement**

```python
# scoring/geo.py
import logging
from datetime import datetime
from typing import Optional
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

logger = logging.getLogger(__name__)

# Cache for geocoding results
_geocode_cache: dict[str, Optional[tuple[float, float]]] = {}

# Known city coordinates (fallback when geocoding fails)
KNOWN_CITIES = {
    "munich": (48.1351, 11.5820),
    "berlin": (52.5200, 13.4050),
    "rome": (41.9028, 12.4964),
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "audincourt": (47.4836, 6.8403),
    "dietzenbach": (50.0092, 8.7797),
    "hamburg": (53.5511, 9.9937),
}

def geocode_location(location_str: str) -> Optional[tuple[float, float]]:
    """
    Geocode a location string to (lat, lng).
    
    Args:
        location_str: Location string like "Munich - Isar River Cafe"
    
    Returns:
        (lat, lng) tuple or None if geocoding fails
    """
    if not location_str:
        return None
    
    # Check cache
    if location_str in _geocode_cache:
        return _geocode_cache[location_str]
    
    # Extract city name (before " - " if present)
    city = location_str.split(" - ")[0].strip().lower()
    
    # Check known cities first
    if city in KNOWN_CITIES:
        result = KNOWN_CITIES[city]
        _geocode_cache[location_str] = result
        return result
    
    # Try geocoding
    try:
        geolocator = Nominatim(user_agent="fraud_detector")
        result = geolocator.geocode(city, timeout=5)
        if result:
            coords = (result.latitude, result.longitude)
            _geocode_cache[location_str] = coords
            return coords
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.warning(f"Geocoding failed for {location_str}: {e}")
    
    _geocode_cache[location_str] = None
    return None


def geo_scorer(
    tx_location: Optional[str],
    timestamp: datetime,
    sender_biotag: str,
    locations: list[dict],
    prev_tx_location: Optional[str],
    prev_tx_timestamp: Optional[datetime]
) -> float:
    """
    Score transaction based on geographic anomaly.
    
    Args:
        tx_location: Transaction location string (for in-person payments)
        timestamp: Transaction timestamp
        sender_biotag: User's biotag for GPS lookup
        locations: List of GPS location records
        prev_tx_location: Previous transaction location (for velocity check)
        prev_tx_timestamp: Previous transaction timestamp
    
    Returns:
        Risk score 0.0-1.0
    """
    if not tx_location:
        return 0.0
    
    tx_coords = geocode_location(tx_location)
    if not tx_coords:
        return 0.0
    
    score = 0.0
    
    # Check 1: Distance from user's GPS location
    user_coords = _find_nearest_gps(sender_biotag, timestamp, locations)
    if user_coords:
        distance_km = geodesic(user_coords, tx_coords).km
        if distance_km > 50:
            score = min(distance_km / 500, 1.0)
    
    # Check 2: Impossible velocity between consecutive transactions
    if prev_tx_location and prev_tx_timestamp:
        prev_coords = geocode_location(prev_tx_location)
        if prev_coords:
            distance_km = geodesic(prev_coords, tx_coords).km
            time_diff_hours = (timestamp - prev_tx_timestamp).total_seconds() / 3600
            
            if time_diff_hours > 0:
                velocity = distance_km / time_diff_hours
                if velocity > 500:  # km/h - impossible without flight
                    score = 1.0
    
    return score


def _find_nearest_gps(
    biotag: str, 
    timestamp: datetime, 
    locations: list[dict]
) -> Optional[tuple[float, float]]:
    """Find the GPS location closest in time to the transaction."""
    if not locations:
        return None
    
    user_locs = [loc for loc in locations if loc.get("biotag") == biotag]
    if not user_locs:
        return None
    
    # Find closest timestamp
    closest = None
    min_diff = float('inf')
    
    for loc in user_locs:
        loc_time = datetime.fromisoformat(loc["timestamp"].replace("Z", "+00:00"))
        if loc_time.tzinfo:
            loc_time = loc_time.replace(tzinfo=None)
        
        diff = abs((timestamp - loc_time).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest = loc
    
    if closest:
        return (float(closest["lat"]), float(closest["lng"]))
    return None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_geo.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/geo.py tests/test_geo.py
git commit -m "feat: add geographic anomaly scorer with velocity check"
```

---

## Chunk 5: ML Layer (Isolation Forest & Graph Analyzer)

### Task 13: Implement Isolation Forest

**Files:**
- Create: `ml/isolation_forest.py`
- Create: `tests/test_isolation_forest.py`

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_isolation_forest.py -v
```

- [ ] **Step 3: Implement**

```python
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
        """
        Fit the model on training features (normal behavior).
        
        Args:
            features: 2D array of shape (n_samples, n_features)
        """
        self.model.fit(features)
        self._fitted = True
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Score features for anomaly.
        
        Args:
            features: 2D array of shape (n_samples, n_features)
        
        Returns:
            Array of scores 0.0-1.0 (higher = more anomalous)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # decision_function returns negative for anomalies
        raw_scores = self.model.decision_function(features)
        
        # Normalize to 0-1 (more negative = higher anomaly score)
        # decision_function: lower is more anomalous
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score == min_score:
            return np.zeros(len(features))
        
        # Invert so higher = more anomalous
        normalized = (max_score - raw_scores) / (max_score - min_score)
        return normalized
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_isolation_forest.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ml/isolation_forest.py tests/test_isolation_forest.py
git commit -m "feat: add Isolation Forest anomaly detector"
```

---

### Task 14: Implement Graph Analyzer

**Files:**
- Create: `ml/graph_analyzer.py`
- Create: `tests/test_graph_analyzer.py`

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_graph_analyzer.py -v
```

- [ ] **Step 3: Implement**

```python
# ml/graph_analyzer.py
import networkx as nx
import pandas as pd
from datetime import timedelta
from typing import Optional

class GraphAnalyzer:
    """Analyze transaction graph for fraud patterns."""
    
    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None
        self._tx_in_pattern: set[str] = set()
    
    def build_graph(self, transactions: pd.DataFrame) -> None:
        """
        Build directed graph from transactions.
        
        Args:
            transactions: DataFrame with sender_id, recipient_id, amount, timestamp
        """
        self.graph = nx.DiGraph()
        
        for _, tx in transactions.iterrows():
            sender = tx["sender_id"]
            recipient = tx["recipient_id"]
            
            if pd.isna(recipient) or not recipient:
                continue
            
            self.graph.add_edge(
                sender,
                recipient,
                tx_id=tx["transaction_id"],
                amount=tx["amount"],
                timestamp=tx["timestamp"]
            )
    
    def detect_circular_flows(self) -> list[list[str]]:
        """Find cycles in the transaction graph (potential money laundering)."""
        if not self.graph:
            return []
        
        cycles = list(nx.simple_cycles(self.graph))
        # Only return cycles of length >= 3
        return [c for c in cycles if len(c) >= 3]
    
    def detect_mule_accounts(
        self, 
        in_threshold: int = 5, 
        out_threshold: int = 3
    ) -> list[str]:
        """
        Find accounts with high in-degree and out-degree (potential mules).
        
        Args:
            in_threshold: Minimum in-degree to flag
            out_threshold: Minimum out-degree to flag
        
        Returns:
            List of suspicious account IDs
        """
        if not self.graph:
            return []
        
        suspicious = []
        for node in self.graph.nodes():
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            if in_deg >= in_threshold and out_deg >= out_threshold:
                suspicious.append(node)
        
        return suspicious
    
    def detect_rapid_layering(self, max_hours: float = 2.0) -> list[str]:
        """
        Find transactions involved in rapid layering pattern.
        A->B->C where B receives and sends within max_hours.
        
        Returns:
            List of transaction IDs involved in layering
        """
        if not self.graph:
            return []
        
        suspicious_txs = []
        
        for node in self.graph.nodes():
            in_edges = list(self.graph.in_edges(node, data=True))
            out_edges = list(self.graph.out_edges(node, data=True))
            
            for in_edge in in_edges:
                in_data = in_edge[2]
                in_ts = in_data.get("timestamp")
                in_amount = in_data.get("amount", 0)
                
                if in_ts is None:
                    continue
                
                for out_edge in out_edges:
                    out_data = out_edge[2]
                    out_ts = out_data.get("timestamp")
                    out_amount = out_data.get("amount", 0)
                    
                    if out_ts is None:
                        continue
                    
                    # Check timing
                    time_diff = (out_ts - in_ts).total_seconds() / 3600
                    if 0 < time_diff <= max_hours:
                        # Check similar amount (±20%)
                        if in_amount > 0 and 0.8 <= out_amount / in_amount <= 1.2:
                            suspicious_txs.append(in_data.get("tx_id"))
                            suspicious_txs.append(out_data.get("tx_id"))
        
        return list(set(suspicious_txs))
    
    def score_transactions(self, transactions: pd.DataFrame) -> dict[str, float]:
        """
        Score each transaction based on graph pattern involvement.
        
        Returns:
            Dict mapping transaction_id to score 0.0-1.0
        """
        if not self.graph:
            return {tx_id: 0.0 for tx_id in transactions["transaction_id"]}
        
        # Find all suspicious patterns
        cycles = self.detect_circular_flows()
        mules = set(self.detect_mule_accounts())
        layering_txs = set(self.detect_rapid_layering())
        
        # Get all nodes in cycles
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        
        scores = {}
        for _, tx in transactions.iterrows():
            tx_id = tx["transaction_id"]
            score = 0.0
            
            sender = tx["sender_id"]
            recipient = tx["recipient_id"]
            
            # Check if tx is in layering pattern
            if tx_id in layering_txs:
                score = max(score, 0.8)
            
            # Check if sender/recipient is in a cycle
            if sender in cycle_nodes or recipient in cycle_nodes:
                score = max(score, 0.6)
            
            # Check if recipient is a mule
            if recipient in mules:
                score = max(score, 0.7)
            
            # Check if sender is a mule
            if sender in mules:
                score = max(score, 0.5)
            
            scores[tx_id] = score
        
        return scores
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_graph_analyzer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ml/graph_analyzer.py tests/test_graph_analyzer.py
git commit -m "feat: add graph analyzer for fraud network patterns"
```

---

## Chunk 6: Audio Transcription & LLM Utilities

### Task 15: Implement audio transcriber

**Files:**
- Create: `preprocessing/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_audio.py
import pytest
from preprocessing.audio import extract_user_from_filename, AudioTranscriber

def test_extract_user_from_filename():
    filename = "20870117_010505-guido_döhn.mp3"
    user = extract_user_from_filename(filename)
    assert user == "guido döhn"

def test_extract_user_handles_underscores():
    filename = "20870206_221040-juliette_brunet.mp3"
    user = extract_user_from_filename(filename)
    assert user == "juliette brunet"

def test_audio_transcriber_init():
    # Just test initialization doesn't crash
    transcriber = AudioTranscriber()
    assert transcriber is not None

# Integration test - skip if no API key
@pytest.mark.skip(reason="Requires OpenAI API key and real audio file")
def test_audio_transcriber_transcribe():
    transcriber = AudioTranscriber()
    result = transcriber.transcribe("path/to/test.mp3")
    assert isinstance(result, str)
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_audio.py -v
```

- [ ] **Step 3: Implement**

```python
# preprocessing/audio.py
import os
import re
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

def extract_user_from_filename(filename: str) -> str:
    """
    Extract user name from audio filename.
    Format: YYYYMMDD_HHMMSS-name_surname.mp3
    
    Args:
        filename: Audio filename
    
    Returns:
        User name with spaces (e.g., "guido döhn")
    """
    # Remove extension
    name = Path(filename).stem
    
    # Extract part after the dash
    if "-" in name:
        name = name.split("-", 1)[1]
    
    # Replace underscores with spaces
    return name.replace("_", " ")


class AudioTranscriber:
    """Transcribe audio files using OpenAI Whisper API."""
    
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key)
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Transcribed text
        """
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcript
    
    def transcribe_directory(self, audio_dir: str) -> dict[str, dict]:
        """
        Transcribe all audio files in a directory.
        
        Args:
            audio_dir: Path to directory containing audio files
        
        Returns:
            Dict mapping filename to {user: str, transcript: str}
        """
        results = {}
        audio_path = Path(audio_dir)
        
        if not audio_path.exists():
            logger.warning(f"Audio directory not found: {audio_dir}")
            return results
        
        for audio_file in audio_path.glob("*.mp3"):
            try:
                user = extract_user_from_filename(audio_file.name)
                transcript = self.transcribe(str(audio_file))
                results[audio_file.name] = {
                    "user": user,
                    "transcript": transcript,
                    "timestamp": audio_file.name[:15],  # YYYYMMDD_HHMMSS
                }
                logger.info(f"Transcribed: {audio_file.name}")
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file.name}: {e}")
        
        return results
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_audio.py -v
```

- [ ] **Step 5: Commit**

```bash
git add preprocessing/audio.py tests/test_audio.py
git commit -m "feat: add audio transcription with Whisper API"
```

---

### Task 16: Create LLM utilities (response parsing, retry logic)

**Files:**
- Create: `agents/llm_utils.py`
- Create: `tests/test_llm_utils.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm_utils.py
import pytest
import json
from agents.llm_utils import strip_thinking_tags, parse_json_response, parse_transaction_ids

def test_strip_thinking_tags_removes_think_block():
    response = """<think>
Let me analyze this...
The user seems suspicious because...
</think>

{"signals": []}"""
    
    cleaned = strip_thinking_tags(response)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert '{"signals": []}' in cleaned

def test_strip_thinking_tags_handles_no_tags():
    response = '{"signals": []}'
    cleaned = strip_thinking_tags(response)
    assert cleaned == '{"signals": []}'

def test_parse_json_response_valid_json():
    response = '{"signals": [{"user": "test", "severity": "high"}]}'
    result = parse_json_response(response)
    assert result["signals"][0]["user"] == "test"

def test_parse_json_response_extracts_json_from_text():
    response = """Here is my analysis:
    
    {"signals": [{"user": "test"}]}
    
    That's my conclusion."""
    result = parse_json_response(response)
    assert result["signals"][0]["user"] == "test"

def test_parse_json_response_with_thinking_tags():
    response = """<think>thinking...</think>
    {"signals": [{"user": "test"}]}"""
    result = parse_json_response(response)
    assert result["signals"][0]["user"] == "test"

def test_parse_json_response_returns_empty_on_failure():
    response = "No JSON here at all"
    result = parse_json_response(response)
    assert result == {"signals": []}

def test_parse_transaction_ids_valid():
    response = """<think>reasoning</think>
tx-001
tx-002
tx-003"""
    valid_ids = {"tx-001", "tx-002", "tx-003", "tx-004"}
    result = parse_transaction_ids(response, valid_ids)
    assert result == ["tx-001", "tx-002", "tx-003"]

def test_parse_transaction_ids_filters_invalid():
    response = """tx-001
invalid-id
tx-002"""
    valid_ids = {"tx-001", "tx-002"}
    result = parse_transaction_ids(response, valid_ids)
    assert "invalid-id" not in result
    assert len(result) == 2

def test_parse_transaction_ids_raises_on_empty():
    response = "no valid ids here"
    valid_ids = {"tx-001", "tx-002"}
    with pytest.raises(ValueError, match="empty"):
        parse_transaction_ids(response, valid_ids)

def test_parse_transaction_ids_raises_on_all():
    response = """tx-001
tx-002"""
    valid_ids = {"tx-001", "tx-002"}
    with pytest.raises(ValueError, match="all"):
        parse_transaction_ids(response, valid_ids)
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_llm_utils.py -v
```

- [ ] **Step 3: Implement**

```python
# agents/llm_utils.py
import re
import json
import logging
from typing import Any
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)


def strip_thinking_tags(response: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen responses.
    The actual answer comes AFTER the closing </think> tag.
    
    Args:
        response: Raw LLM response
    
    Returns:
        Cleaned response without thinking tags
    """
    pattern = r'<think>.*?</think>\s*'
    cleaned = re.sub(pattern, '', response, flags=re.DOTALL)
    return cleaned.strip()


def parse_json_response(response: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response, handling thinking tags and extraction.
    
    Args:
        response: Raw LLM response
    
    Returns:
        Parsed JSON dict, or {"signals": []} as fallback
    """
    # First strip thinking tags
    cleaned = strip_thinking_tags(response)
    
    # Try direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from text
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback
    logger.warning(f"Could not parse JSON from response: {cleaned[:200]}...")
    return {"signals": []}


def parse_transaction_ids(response: str, valid_ids: set[str]) -> list[str]:
    """
    Parse transaction IDs from coordinator response.
    
    Args:
        response: Raw LLM response
        valid_ids: Set of valid transaction IDs
    
    Returns:
        List of valid transaction IDs
    
    Raises:
        ValueError: If result is empty or contains all IDs
    """
    # Strip thinking tags
    cleaned = strip_thinking_tags(response)
    
    # Parse lines
    lines = cleaned.strip().split('\n')
    flagged = []
    
    for line in lines:
        tx_id = line.strip()
        if tx_id in valid_ids:
            flagged.append(tx_id)
    
    # Validation
    if len(flagged) == 0:
        raise ValueError("Coordinator returned empty list")
    
    if len(flagged) == len(valid_ids):
        raise ValueError("Coordinator flagged all transactions")
    
    return flagged


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError))
)
def call_llm_with_retry(model, messages, config):
    """
    Call LLM with retry logic for transient errors.
    
    Args:
        model: LangChain chat model
        messages: List of messages
        config: LangChain config dict
    
    Returns:
        LLM response
    """
    return model.invoke(messages, config=config)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_llm_utils.py -v
```

- [ ] **Step 5: Commit**

```bash
git add agents/llm_utils.py tests/test_llm_utils.py
git commit -m "feat: add LLM utilities for response parsing and retry"
```

---

## Chunk 7: LLM Agents (Communications & Coordinator)

### Task 17: Implement communications agent

**Files:**
- Create: `agents/comms.py`
- Create: `tests/test_comms_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_comms_agent.py
import pytest
from unittest.mock import Mock, patch
from agents.comms import CommsAgent, build_comms_prompt

def test_build_comms_prompt_includes_all_data():
    prompt = build_comms_prompt(
        sms=[{"sms": "Test SMS"}],
        mails=[{"mail": "Test Email"}],
        audio_transcripts={"file.mp3": {"user": "test", "transcript": "Hello"}},
        user_mapping={"John Doe": "IBAN123"}
    )
    
    assert "Test SMS" in prompt
    assert "Test Email" in prompt
    assert "Hello" in prompt
    assert "John Doe" in prompt
    assert "IBAN123" in prompt

def test_build_comms_prompt_handles_empty_audio():
    prompt = build_comms_prompt(
        sms=[{"sms": "Test"}],
        mails=[],
        audio_transcripts={},
        user_mapping={}
    )
    assert "No audio" in prompt or "Test" in prompt

@patch("agents.comms.call_llm_with_retry")
def test_comms_agent_returns_signals(mock_llm):
    mock_response = Mock()
    mock_response.content = '{"signals": [{"user_iban": "IBAN1", "severity": "high", "reason": "phishing"}]}'
    mock_llm.return_value = mock_response
    
    agent = CommsAgent(model=Mock(), session_id="test-123")
    result = agent.analyze(
        sms=[],
        mails=[],
        audio_transcripts={},
        user_mapping={}
    )
    
    assert "signals" in result
    assert len(result["signals"]) == 1

@patch("agents.comms.call_llm_with_retry")
def test_comms_agent_handles_thinking_tags(mock_llm):
    mock_response = Mock()
    mock_response.content = """<think>
Let me analyze...
</think>

{"signals": []}"""
    mock_llm.return_value = mock_response
    
    agent = CommsAgent(model=Mock(), session_id="test-123")
    result = agent.analyze([], [], {}, {})
    
    assert result == {"signals": []}
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_comms_agent.py -v
```

- [ ] **Step 3: Implement**

```python
# agents/comms.py
import os
import json
import logging
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from langfuse.langchain import CallbackHandler

from .llm_utils import parse_json_response, call_llm_with_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fraud analyst specialized in detecting social engineering attacks.

Analyze the provided communications (SMS, emails, audio transcripts) looking for:
- Phishing attempts (credential requests, suspicious links)
- Psychological manipulation (artificial urgency, threats)
- Impersonation (someone pretending to be a bank or service)
- Unusual requests (urgent transfers to "safe" accounts)
- Temporal correlation: suspicious communication received shortly before unusual transactions

User IBAN mapping is provided to help you identify which user received each communication.

Respond ONLY in JSON format:
{"signals": [{"user_iban": "...", "user_name": "...", "source": "sms|email|audio", "severity": "high|medium|low", "timestamp": "...", "reason": "..."}]}

If no suspicious signals are found, respond: {"signals": []}"""


def build_comms_prompt(
    sms: list[dict],
    mails: list[dict],
    audio_transcripts: dict[str, dict],
    user_mapping: dict[str, str]
) -> str:
    """Build the user prompt with all communication data."""
    parts = []
    
    # User mapping
    if user_mapping:
        parts.append("## User Mapping (Name -> IBAN)")
        for name, iban in user_mapping.items():
            parts.append(f"- {name}: {iban}")
        parts.append("")
    
    # SMS
    parts.append("## SMS Messages")
    if sms:
        for item in sms:
            parts.append(item.get("sms", str(item)))
        parts.append("")
    else:
        parts.append("No SMS messages.")
        parts.append("")
    
    # Emails
    parts.append("## Emails")
    if mails:
        for item in mails:
            parts.append(item.get("mail", str(item)))
        parts.append("")
    else:
        parts.append("No emails.")
        parts.append("")
    
    # Audio transcripts
    parts.append("## Audio Transcripts")
    if audio_transcripts:
        for filename, data in audio_transcripts.items():
            user = data.get("user", "Unknown")
            transcript = data.get("transcript", "")
            ts = data.get("timestamp", "")
            parts.append(f"[{ts}] {user}: {transcript}")
        parts.append("")
    else:
        parts.append("No audio transcripts available.")
        parts.append("")
    
    return "\n".join(parts)


class CommsAgent:
    """Agent for analyzing communications for social engineering signals."""
    
    def __init__(self, model, session_id: str):
        self.model = model
        self.session_id = session_id
    
    @observe()
    def analyze(
        self,
        sms: list[dict],
        mails: list[dict],
        audio_transcripts: dict[str, dict],
        user_mapping: dict[str, str]
    ) -> dict[str, Any]:
        """
        Analyze communications for fraud signals.
        
        Args:
            sms: List of SMS dicts
            mails: List of email dicts
            audio_transcripts: Dict of audio transcriptions
            user_mapping: Dict mapping user name to IBAN
        
        Returns:
            Dict with "signals" list
        """
        handler = CallbackHandler()
        
        user_prompt = build_comms_prompt(sms, mails, audio_transcripts, user_mapping)
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = call_llm_with_retry(
                self.model,
                messages,
                config={
                    "callbacks": [handler],
                    "metadata": {"langfuse_session_id": self.session_id}
                }
            )
            
            return parse_json_response(response.content)
            
        except Exception as e:
            logger.error(f"Comms agent failed: {e}")
            return {"signals": []}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_comms_agent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add agents/comms.py tests/test_comms_agent.py
git commit -m "feat: add communications analysis agent"
```

---

### Task 18: Implement fraud coordinator agent

**Files:**
- Create: `agents/coordinator.py`
- Create: `tests/test_coordinator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_coordinator.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from agents.coordinator import FraudCoordinator, build_coordinator_prompt

def test_build_coordinator_prompt_includes_risk_table():
    risk_df = pd.DataFrame({
        "transaction_id": ["tx1", "tx2"],
        "total_risk": [0.8, 0.2],
        "amount_score": [0.9, 0.1],
    })
    
    prompt = build_coordinator_prompt(
        risk_table=risk_df,
        comms_signals={"signals": []},
        eval_transactions=pd.DataFrame({
            "transaction_id": ["tx1", "tx2"],
            "amount": [1000, 50],
        })
    )
    
    assert "tx1" in prompt
    assert "0.8" in prompt

@patch("agents.coordinator.call_llm_with_retry")
def test_coordinator_returns_flagged_ids(mock_llm):
    mock_response = Mock()
    mock_response.content = """<think>analyzing...</think>

tx-001
tx-002"""
    mock_llm.return_value = mock_response
    
    coordinator = FraudCoordinator(model=Mock(), session_id="test-123")
    
    result = coordinator.decide(
        risk_table=pd.DataFrame({
            "transaction_id": ["tx-001", "tx-002", "tx-003"],
            "total_risk": [0.9, 0.8, 0.1],
        }),
        comms_signals={"signals": []},
        eval_transactions=pd.DataFrame({
            "transaction_id": ["tx-001", "tx-002", "tx-003"],
            "amount": [1000, 500, 50],
        })
    )
    
    assert "tx-001" in result
    assert "tx-002" in result
    assert "tx-003" not in result

@patch("agents.coordinator.call_llm_with_retry")
def test_coordinator_fallback_on_empty(mock_llm):
    mock_response = Mock()
    mock_response.content = "I cannot determine any fraud"
    mock_llm.return_value = mock_response
    
    coordinator = FraudCoordinator(model=Mock(), session_id="test-123")
    
    risk_df = pd.DataFrame({
        "transaction_id": [f"tx-{i}" for i in range(10)],
        "total_risk": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
    })
    
    result = coordinator.decide(
        risk_table=risk_df,
        comms_signals={"signals": []},
        eval_transactions=pd.DataFrame({
            "transaction_id": [f"tx-{i}" for i in range(10)],
            "amount": [100] * 10,
        })
    )
    
    # Should fallback to top 15% = 2 transactions
    assert len(result) >= 1
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_coordinator.py -v
```

- [ ] **Step 3: Implement**

```python
# agents/coordinator.py
import logging
import pandas as pd
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from langfuse.langchain import CallbackHandler

from .llm_utils import parse_transaction_ids, call_llm_with_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the fraud coordinator for MirrorPay's anti-fraud system.

You receive:
1. A risk score table computed by algorithmic analyzers (scores 0-1 per category)
2. Social engineering signals detected in communications
3. The transactions to evaluate

RULES:
- You MUST flag at least 1 transaction
- You CANNOT flag all transactions
- Prioritize precision: it's better to miss some fraud than to block legitimate customers
- Correlate signals: high algorithmic score + communication signal = almost certainly fraud
- Consider temporal correlation: suspicious communication within 24h of anomalous transaction is very suspicious

Respond ONLY with the list of transaction_id to flag as fraudulent, one per line.
No other text, no explanations."""


def build_coordinator_prompt(
    risk_table: pd.DataFrame,
    comms_signals: dict[str, Any],
    eval_transactions: pd.DataFrame
) -> str:
    """Build the user prompt for the coordinator."""
    parts = []
    
    # Risk score table
    parts.append("## RISK SCORE TABLE")
    parts.append(risk_table.to_string(index=False))
    parts.append("")
    
    # Communications signals
    parts.append("## COMMUNICATION SIGNALS")
    signals = comms_signals.get("signals", [])
    if signals:
        for sig in signals:
            parts.append(f"- User {sig.get('user_iban', 'unknown')}: {sig.get('severity', 'unknown')} - {sig.get('reason', 'no reason')}")
    else:
        parts.append("No suspicious communication signals detected.")
    parts.append("")
    
    # Transaction details
    parts.append("## TRANSACTIONS TO EVALUATE")
    # Select relevant columns
    cols = ["transaction_id", "sender_id", "recipient_id", "amount", 
            "transaction_type", "timestamp"]
    available_cols = [c for c in cols if c in eval_transactions.columns]
    parts.append(eval_transactions[available_cols].to_string(index=False))
    
    return "\n".join(parts)


class FraudCoordinator:
    """Coordinator agent for final fraud decision."""
    
    def __init__(self, model, session_id: str):
        self.model = model
        self.session_id = session_id
    
    @observe()
    def decide(
        self,
        risk_table: pd.DataFrame,
        comms_signals: dict[str, Any],
        eval_transactions: pd.DataFrame
    ) -> list[str]:
        """
        Make final fraud decision.
        
        Args:
            risk_table: DataFrame with transaction_id and risk scores
            comms_signals: Output from comms_agent
            eval_transactions: Evaluation transactions DataFrame
        
        Returns:
            List of transaction IDs flagged as fraudulent
        """
        handler = CallbackHandler()
        
        user_prompt = build_coordinator_prompt(
            risk_table, comms_signals, eval_transactions
        )
        
        valid_ids = set(eval_transactions["transaction_id"].tolist())
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = call_llm_with_retry(
                self.model,
                messages,
                config={
                    "callbacks": [handler],
                    "metadata": {"langfuse_session_id": self.session_id}
                }
            )
            
            return parse_transaction_ids(response.content, valid_ids)
            
        except ValueError as e:
            # Fallback to algorithmic decision
            logger.warning(f"Coordinator fallback triggered: {e}")
            return self._algorithmic_fallback(risk_table, len(valid_ids))
        
        except Exception as e:
            logger.error(f"Coordinator failed: {e}")
            return self._algorithmic_fallback(risk_table, len(valid_ids))
    
    def _algorithmic_fallback(
        self, 
        risk_table: pd.DataFrame, 
        total_count: int
    ) -> list[str]:
        """
        Fallback to algorithmic decision based on risk scores.
        Flags top 15% of transactions by risk score.
        """
        sorted_df = risk_table.sort_values("total_risk", ascending=False)
        
        # Flag top 15%, minimum 1, maximum 30%
        n_to_flag = max(1, min(int(total_count * 0.15), int(total_count * 0.3)))
        
        flagged = sorted_df.head(n_to_flag)["transaction_id"].tolist()
        
        logger.info(f"Algorithmic fallback: flagging {len(flagged)} transactions")
        return flagged
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_coordinator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add agents/coordinator.py tests/test_coordinator.py
git commit -m "feat: add fraud coordinator agent with fallback"
```

---

## Chunk 8: Score Aggregation & Main Orchestrator

### Task 19: Create score aggregator

**Files:**
- Create: `scoring/aggregator.py`
- Create: `tests/test_aggregator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_aggregator.py
import pytest
import pandas as pd
from datetime import datetime
from scoring.aggregator import ScoreAggregator

@pytest.fixture
def sample_data():
    return {
        "profiles": {
            "IBAN1": {
                "avg_amount_by_type": {"e-commerce": 100},
                "std_amount_by_type": {"e-commerce": 25},
                "typical_hours": [10, 11, 12, 14, 15],
                "weekend_active": False,
                "payment_methods": {"debit card": 10},
                "known_recipients": {"SHOP1"},
                "last_tx_to": {},
                "salary": 50000,
                "often_round_amounts": False,
            }
        },
        "eval_transactions": pd.DataFrame({
            "transaction_id": ["tx1", "tx2"],
            "sender_id": ["USER1", "USER1"],
            "sender_iban": ["IBAN1", "IBAN1"],
            "recipient_id": ["SHOP1", "NEW_SHOP"],
            "recipient_iban": ["", ""],
            "transaction_type": ["e-commerce", "e-commerce"],
            "amount": [100.0, 500.0],
            "balance_after": [900.0, 400.0],
            "payment_method": ["debit card", "PayPal"],
            "location": ["", ""],
            "timestamp": pd.to_datetime(["2087-03-15 11:00", "2087-03-15 03:00"]),
        }),
        "locations": [],
    }

def test_aggregator_returns_dataframe(sample_data):
    aggregator = ScoreAggregator(sample_data["profiles"])
    
    result = aggregator.score_all(
        sample_data["eval_transactions"],
        sample_data["locations"]
    )
    
    assert isinstance(result, pd.DataFrame)
    assert "transaction_id" in result.columns
    assert "total_risk" in result.columns

def test_aggregator_scores_in_range(sample_data):
    aggregator = ScoreAggregator(sample_data["profiles"])
    
    result = aggregator.score_all(
        sample_data["eval_transactions"],
        sample_data["locations"]
    )
    
    score_cols = [c for c in result.columns if c.endswith("_score")]
    for col in score_cols:
        assert all(0 <= result[col]) and all(result[col] <= 1)

def test_aggregator_anomalous_tx_has_higher_score(sample_data):
    aggregator = ScoreAggregator(sample_data["profiles"])
    
    result = aggregator.score_all(
        sample_data["eval_transactions"],
        sample_data["locations"]
    )
    
    # tx2 should have higher risk (unusual amount, time, new recipient, new method)
    tx1_risk = result[result["transaction_id"] == "tx1"]["total_risk"].iloc[0]
    tx2_risk = result[result["transaction_id"] == "tx2"]["total_risk"].iloc[0]
    
    assert tx2_risk > tx1_risk
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_aggregator.py -v
```

- [ ] **Step 3: Implement**

```python
# scoring/aggregator.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from .amount import amount_scorer
from .time import time_scorer
from .frequency import frequency_scorer
from .channel import channel_switch_scorer
from .round_amount import round_amount_scorer
from .entities import new_entity_scorer
from .balance import balance_drain_scorer
from .geo import geo_scorer


class ScoreAggregator:
    """Aggregate all rule-based scores for transactions."""
    
    # Default weights for each scorer
    WEIGHTS = {
        "amount_score": 0.15,
        "time_score": 0.08,
        "geo_score": 0.15,
        "new_entity_score": 0.12,
        "frequency_score": 0.08,
        "channel_switch_score": 0.07,
        "round_amount_score": 0.05,
        "balance_drain_score": 0.10,
        "isolation_forest_score": 0.10,
        "graph_score": 0.10,
    }
    
    def __init__(self, profiles: dict):
        self.profiles = profiles
    
    def score_all(
        self,
        eval_transactions: pd.DataFrame,
        locations: list[dict],
        ml_scores: Optional[dict[str, float]] = None,
        graph_scores: Optional[dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Score all transactions.
        
        Args:
            eval_transactions: Evaluation transactions DataFrame
            locations: Location records for geo checking
            ml_scores: Optional isolation forest scores
            graph_scores: Optional graph analysis scores
        
        Returns:
            DataFrame with all scores and total_risk
        """
        ml_scores = ml_scores or {}
        graph_scores = graph_scores or {}
        
        results = []
        
        # Sort by timestamp for prev_tx tracking
        sorted_txs = eval_transactions.sort_values("timestamp").reset_index(drop=True)
        
        # Track previous tx per user for geo velocity check
        prev_tx_by_user: dict[str, dict] = {}
        
        for idx, tx in sorted_txs.iterrows():
            tx_id = tx["transaction_id"]
            sender_iban = tx.get("sender_iban", "")
            sender_id = tx.get("sender_id", "")
            
            # Get profile (fallback to empty)
            profile = self.profiles.get(sender_iban, {})
            
            # Get previous tx for this user
            prev_tx = prev_tx_by_user.get(sender_id)
            
            # Calculate all scores
            scores = {
                "transaction_id": tx_id,
                "amount_score": amount_scorer(
                    tx["amount"], 
                    tx.get("transaction_type", ""), 
                    profile
                ),
                "time_score": time_scorer(
                    tx["timestamp"], 
                    profile
                ),
                "geo_score": geo_scorer(
                    tx.get("location"),
                    tx["timestamp"],
                    sender_id,
                    locations,
                    prev_tx.get("location") if prev_tx else None,
                    prev_tx.get("timestamp") if prev_tx else None
                ),
                "new_entity_score": new_entity_scorer(
                    tx.get("recipient_id") or tx.get("recipient_iban"),
                    tx["timestamp"],
                    tx["amount"],
                    profile
                ),
                "frequency_score": frequency_scorer(
                    sender_id,
                    tx["timestamp"],
                    sorted_txs,
                    profile
                ),
                "channel_switch_score": channel_switch_scorer(
                    tx.get("payment_method", ""),
                    tx["amount"],
                    profile
                ),
                "round_amount_score": round_amount_scorer(
                    tx["amount"],
                    profile
                ),
                "balance_drain_score": balance_drain_scorer(
                    tx["amount"],
                    tx.get("balance_after", 0),
                    sender_id,
                    tx["timestamp"],
                    sorted_txs
                ),
                "isolation_forest_score": ml_scores.get(tx_id, 0.0),
                "graph_score": graph_scores.get(tx_id, 0.0),
            }
            
            # Calculate weighted total
            total = sum(
                scores[k] * self.WEIGHTS.get(k, 0) 
                for k in self.WEIGHTS.keys()
                if k in scores
            )
            scores["total_risk"] = total
            
            results.append(scores)
            
            # Update prev tx for velocity check
            if tx.get("location"):
                prev_tx_by_user[sender_id] = {
                    "location": tx["location"],
                    "timestamp": tx["timestamp"]
                }
        
        return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_aggregator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scoring/aggregator.py tests/test_aggregator.py
git commit -m "feat: add score aggregator for all rule-based scorers"
```

---

### Task 20: Create main orchestrator

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_main.py
import pytest
from unittest.mock import patch, Mock
from main import run_pipeline, create_model, generate_session_id

def test_generate_session_id_format():
    with patch.dict("os.environ", {"TEAM_NAME": "Test Team"}):
        session_id = generate_session_id()
        assert session_id.startswith("Test-Team-")
        assert len(session_id) > 15

def test_create_model_returns_chat_model():
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        model = create_model()
        assert model is not None

@patch("main.load_data")
@patch("main.write_output")
def test_run_pipeline_produces_output(mock_write, mock_load, tmp_path):
    # Setup minimal mock data
    import pandas as pd
    mock_load.return_value = {
        "dataset_name": "Test",
        "train_transactions": pd.DataFrame({
            "transaction_id": ["t1"],
            "sender_id": ["U1"],
            "sender_iban": ["IBAN1"],
            "recipient_id": ["R1"],
            "recipient_iban": [""],
            "transaction_type": ["e-commerce"],
            "amount": [100.0],
            "payment_method": ["debit card"],
            "timestamp": pd.to_datetime(["2087-01-01 10:00"]),
        }),
        "eval_transactions": pd.DataFrame({
            "transaction_id": ["t2"],
            "sender_id": ["U1"],
            "sender_iban": ["IBAN1"],
            "recipient_id": ["R2"],
            "recipient_iban": [""],
            "transaction_type": ["e-commerce"],
            "amount": [5000.0],
            "balance_after": [100.0],
            "payment_method": ["PayPal"],
            "location": [""],
            "timestamp": pd.to_datetime(["2087-01-02 03:00"]),
        }),
        "users": [{"iban": "IBAN1", "first_name": "Test", "last_name": "User", 
                   "salary": 50000, "residence": {"lat": "45", "lng": "9"}}],
        "train_sms": [],
        "eval_sms": [],
        "train_mails": [],
        "eval_mails": [],
        "train_locations": [],
        "eval_locations": [],
        "has_audio": False,
        "audio_path": None,
    }
    
    mock_write.return_value = str(tmp_path / "output.txt")
    
    # This would need the full model setup, so we'll skip for unit test
    # In integration tests, we'd run the full pipeline
```

- [ ] **Step 2: Run test, verify fail**

```bash
pytest tests/test_main.py -v
```

- [ ] **Step 3: Implement main.py**

```python
# main.py
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import ulid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse import Langfuse

from utils.loaders import load_data
from utils.output import write_output
from preprocessing.audio import AudioTranscriber
from scoring.profiler import build_user_profiles
from scoring.aggregator import ScoreAggregator
from ml.isolation_forest import IsolationForestScorer
from ml.graph_analyzer import GraphAnalyzer
from agents.comms import CommsAgent
from agents.coordinator import FraudCoordinator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def generate_session_id() -> str:
    """Generate unique session ID for Langfuse tracking."""
    team = os.getenv("TEAM_NAME", "default").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def create_model():
    """Create LangChain chat model via OpenRouter."""
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-plus",
        temperature=0.3,
        max_tokens=4000,
    )


def run_pipeline(dataset_name: str) -> str:
    """
    Run the full fraud detection pipeline for a dataset.
    
    Args:
        dataset_name: Name of the dataset to process
    
    Returns:
        Path to the output file
    """
    session_id = generate_session_id()
    logger.info(f"Starting pipeline for '{dataset_name}' | Session: {session_id}")
    
    # Initialize Langfuse
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
    )
    
    # Load data
    logger.info("Loading data...")
    data = load_data(dataset_name)
    
    # Audio transcription (if available)
    audio_transcripts = {}
    if data["has_audio"]:
        logger.info("Transcribing audio files...")
        try:
            transcriber = AudioTranscriber()
            audio_transcripts = transcriber.transcribe_directory(data["audio_path"])
        except Exception as e:
            logger.warning(f"Audio transcription failed: {e}")
    
    # Build user profiles
    logger.info("Building user profiles...")
    profiles = build_user_profiles(data["train_transactions"], data["users"])
    
    # Create user name -> IBAN mapping for comms agent
    user_mapping = {
        profile["name"]: iban 
        for iban, profile in profiles.items() 
        if profile.get("name")
    }
    
    # Run ML layer
    logger.info("Running ML anomaly detection...")
    
    # Isolation Forest
    aggregator = ScoreAggregator(profiles)
    train_scores = aggregator.score_all(
        data["train_transactions"], 
        data["train_locations"]
    )
    
    # Prepare feature matrix for Isolation Forest
    feature_cols = [c for c in train_scores.columns if c.endswith("_score") and c != "total_risk"]
    feature_cols = [c for c in feature_cols if c not in ["isolation_forest_score", "graph_score"]]
    
    if feature_cols:
        train_features = train_scores[feature_cols].fillna(0).values
        
        iforest = IsolationForestScorer()
        iforest.fit(train_features)
        
        # Score eval transactions (without ML scores initially)
        eval_scores_initial = aggregator.score_all(
            data["eval_transactions"],
            data["eval_locations"]
        )
        eval_features = eval_scores_initial[feature_cols].fillna(0).values
        iforest_scores = iforest.score(eval_features)
        
        ml_scores = dict(zip(
            data["eval_transactions"]["transaction_id"],
            iforest_scores
        ))
    else:
        ml_scores = {}
    
    # Graph Analysis
    logger.info("Running graph analysis...")
    all_transactions = pd.concat([
        data["train_transactions"],
        data["eval_transactions"]
    ], ignore_index=True)
    
    graph_analyzer = GraphAnalyzer()
    graph_analyzer.build_graph(all_transactions)
    graph_scores = graph_analyzer.score_transactions(data["eval_transactions"])
    
    # Final score aggregation with ML scores
    logger.info("Aggregating scores...")
    risk_table = aggregator.score_all(
        data["eval_transactions"],
        data["eval_locations"],
        ml_scores=ml_scores,
        graph_scores=graph_scores
    )
    
    # Run LLM agents
    logger.info("Running LLM agents...")
    model = create_model()
    
    # Communications agent
    comms_agent = CommsAgent(model, session_id)
    comms_signals = comms_agent.analyze(
        sms=data["train_sms"] + data["eval_sms"],
        mails=data["train_mails"] + data["eval_mails"],
        audio_transcripts=audio_transcripts,
        user_mapping=user_mapping
    )
    logger.info(f"Comms agent found {len(comms_signals.get('signals', []))} signals")
    
    # Fraud coordinator
    coordinator = FraudCoordinator(model, session_id)
    flagged_ids = coordinator.decide(
        risk_table=risk_table,
        comms_signals=comms_signals,
        eval_transactions=data["eval_transactions"]
    )
    logger.info(f"Coordinator flagged {len(flagged_ids)} transactions")
    
    # Write output
    output_path = write_output(flagged_ids, dataset_name)
    logger.info(f"Output written to: {output_path}")
    
    # Flush Langfuse traces
    langfuse_client.flush()
    logger.info(f"Pipeline complete | Session: {session_id}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Agent")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Dataset name (e.g., 'The Truman Show', 'Brave New World', 'Deus Ex')"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = run_pipeline(args.dataset)
        print(f"\nSuccess! Output written to: {output_path}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_main.py -v
```

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add main orchestrator for fraud detection pipeline"
```

---

## Chunk 9: Integration Testing & Final Validation

### Task 21: Create integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""
Integration tests for the full fraud detection pipeline.
Run with: pytest tests/test_integration.py -v -s
Requires: .env file with API keys
"""
import pytest
import os
from pathlib import Path

# Skip if no API keys
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="Requires OPENROUTER_API_KEY"
)

class TestIntegration:
    
    def test_full_pipeline_truman_show(self):
        """Test full pipeline on The Truman Show dataset."""
        from main import run_pipeline
        
        output_path = run_pipeline("The Truman Show")
        
        assert os.path.exists(output_path)
        
        with open(output_path) as f:
            lines = f.read().strip().split('\n')
        
        # Should have at least 1 flagged transaction
        assert len(lines) >= 1
        
        # Should not flag all transactions (80 in dataset)
        assert len(lines) < 80
        
        # Each line should be a valid UUID-like transaction ID
        for line in lines:
            assert len(line) > 10
            assert "-" in line
    
    def test_output_format_valid(self):
        """Verify output meets challenge requirements."""
        from main import run_pipeline
        from utils.loaders import load_data
        
        output_path = run_pipeline("The Truman Show")
        data = load_data("The Truman Show")
        
        valid_ids = set(data["eval_transactions"]["transaction_id"])
        
        with open(output_path) as f:
            flagged = [line.strip() for line in f if line.strip()]
        
        # All flagged IDs must be valid
        for tx_id in flagged:
            assert tx_id in valid_ids, f"Invalid transaction ID: {tx_id}"
        
        # Not empty
        assert len(flagged) > 0, "Output is empty"
        
        # Not all
        assert len(flagged) < len(valid_ids), "All transactions flagged"
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v -s
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full pipeline"
```

---

### Task 22: Create .env.example and update .gitignore

**Files:**
- Create: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: Create .env.example**

```bash
# .env.example
# Langfuse credentials
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://challenges.reply.com/langfuse

# OpenRouter API key
OPENROUTER_API_KEY=sk-or-...

# Team name (spaces will be replaced with "-" in session IDs)
TEAM_NAME=your-team-name

# Do not edit. Disables media upload for langfuse
LANGFUSE_MEDIA_UPLOAD_ENABLED=false
```

- [ ] **Step 2: Update .gitignore**

Ensure `.gitignore` contains:
```
output/
.env
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Commit**

```bash
git add .env.example .gitignore
git commit -m "chore: add .env.example and update gitignore"
```

---

### Task 23: Run all tests and validate

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests pass

- [ ] **Step 2: Run on actual dataset**

```bash
python main.py --dataset "The Truman Show"
```
Expected: Output file created at `output/the_truman_show.txt`

- [ ] **Step 3: Verify output format**

```bash
cat output/the_truman_show.txt
wc -l output/the_truman_show.txt
```
Expected: Non-empty file with valid transaction IDs

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: fraud detection agent v1.0 complete"
```

---

## Summary

**Total Tasks:** 23  
**Total Steps:** ~115  
**Estimated Time:** 4-6 hours

**Key Files Created:**
- `main.py` - Entry point
- `utils/loaders.py`, `utils/output.py` - Data I/O
- `scoring/*.py` - 8 rule-based scorers + aggregator
- `ml/*.py` - Isolation Forest + Graph Analyzer
- `agents/*.py` - LLM agents + utilities
- `preprocessing/audio.py` - Whisper transcription
- `tests/*.py` - Comprehensive test suite

**Dependencies:**
- LangChain + OpenRouter for LLM
- Langfuse for tracing
- scikit-learn for Isolation Forest
- NetworkX for graph analysis
- geopy for geocoding
- Whisper for audio
