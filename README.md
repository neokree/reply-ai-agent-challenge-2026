# Reply Mirror Challenge 2026 — Fraud Detection Agent

> **Team:** Fabio, Ale, Gabri

A hybrid fraud detection system for the Reply Mirror Challenge. Combines rule-based scoring, ML anomaly detection, and LLM agents to identify fraudulent transactions.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in your credentials
cp .env.example .env

# Run on a dataset
python main.py --dataset "The Truman Show"
# → output/the_truman_show.txt
```

Supported datasets: `"The Truman Show"`, `"Brave New World"`, `"Deus Ex"`

---

## Architecture

```
Input datasets (training + evaluation)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Preprocessing                            │
│  • Load CSV transactions + JSON (users, SMS, mails) │
│  • Audio transcription via Whisper (Deus Ex only)   │
│  • Build behavioral profiles from training data      │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2 — Rule-Based Scorers (8 scorers)           │
│  Amount · Time · Frequency · Channel switch         │
│  Round amount · New entity · Balance drain · Geo    │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2.5 — ML Anomaly Detection                   │
│  • Isolation Forest (trained on rule score vectors) │
│  • Graph analysis (cycles, mules, layering)         │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  Layer 3 — LLM Agents (Qwen via OpenRouter)         │
│  • CommsAgent: social engineering signal detection  │
│  • FraudCoordinator: final fraud decision           │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
              output/<dataset>.txt
         (one flagged transaction ID per line)
```

---

## Pipeline Steps

`main.py::run_pipeline()` executes these steps in order:

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `utils/loaders.py` | Load all dataset files (CSVs + JSONs) |
| 2 | `preprocessing/audio.py` | Transcribe MP3 files via Whisper API |
| 3 | `scoring/profiler.py` | Build per-user behavioral baseline from training data |
| 4 | `scoring/aggregator.py` | Score training transactions for Isolation Forest fitting |
| 5 | `ml/isolation_forest.py` | Fit IF on training scores, score eval transactions |
| 6 | `ml/graph_analyzer.py` | Detect cycles, mule accounts, rapid layering in tx graph |
| 7 | `scoring/aggregator.py` | Final weighted score aggregation (all scorers + ML) |
| 8 | `agents/comms.py` | LLM analyzes SMS/email/audio for social engineering |
| 9 | `agents/coordinator.py` | LLM makes final decision; fallback to top 15% by risk |
| 10 | `utils/output.py` | Write flagged IDs to `output/` |

---

## Rule-Based Scorers

Each scorer returns a risk score `0.0–1.0`. All scores are combined in `scoring/aggregator.py` with these weights:

| Scorer | Weight | Signal |
|--------|--------|--------|
| `amount.py` | 0.15 | Z-score of transaction amount vs user's historical average by type |
| `geo.py` | 0.15 | Impossible velocity (>500 km/h) between consecutive transactions; distance from GPS |
| `entities.py` | 0.12 | New/unknown recipient (scaled by amount/salary ratio) |
| `balance.py` | 0.10 | Single tx drains >90% of balance; or 3+ txs drain 80% within 2 hours |
| `isolation_forest` | 0.10 | ML anomaly score from Isolation Forest |
| `graph` | 0.10 | Graph pattern involvement (cycles, mules, layering) |
| `time.py` | 0.08 | Transaction at unusual hour; weekend activity for non-weekend users |
| `frequency.py` | 0.08 | >5 transactions/hour or >3× daily average |
| `channel.py` | 0.07 | Payment method never used before (0.8) or rarely used (<10%, score 0.4) |
| `round_amount.py` | 0.05 | Round amounts ≥$1000 (0.5) or ≥$100 (0.3), unless user habitually uses round amounts |

---

## LLM Agents

### CommsAgent (`agents/comms.py`)
Analyzes all SMS, emails, and audio transcripts for social engineering signals:
- Phishing, impersonation, artificial urgency
- Unusual transfer requests
- Returns `{"signals": [{"user_iban", "severity", "reason", ...}]}`

### FraudCoordinator (`agents/coordinator.py`)
Receives the full risk table + comms signals and outputs the final list of fraudulent transaction IDs.

**Fallback:** If the LLM returns an empty or all-transactions response, falls back to flagging the top 15% by `total_risk` score (minimum 1 transaction).

Both agents use **Qwen3-Plus** via OpenRouter and strip `<think>...</think>` reasoning blocks from responses.

---

## ML Layer

### Isolation Forest (`ml/isolation_forest.py`)
- Trained on the 8-dimensional rule-score vector from training data
- Normalized to `[0, 1]` where higher = more anomalous

### Graph Analyzer (`ml/graph_analyzer.py`)
Builds a directed transaction graph (NetworkX) and detects:
- **Circular flows** — cycles of 3+ nodes (money laundering)
- **Mule accounts** — high in-degree and out-degree nodes
- **Rapid layering** — A→B→C within 2 hours with similar amounts (±20%)

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
# Langfuse tracing (challenge submission tracking)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
LANGFUSE_MEDIA_UPLOAD_ENABLED=false

# OpenRouter (LLM access)
OPENROUTER_API_KEY=sk-or-...

# Your team name — used as prefix in Langfuse session IDs
TEAM_NAME=your-team-name
```

---

## Project Structure

```
├── main.py                  # Entry point — run_pipeline()
├── requirements.txt
├── .env.example
│
├── utils/
│   ├── loaders.py           # load_data(dataset_name) → dict
│   └── output.py            # write_output(flagged_ids, dataset_name)
│
├── preprocessing/
│   └── audio.py             # AudioTranscriber (Whisper API)
│
├── scoring/
│   ├── profiler.py          # build_user_profiles(train_txs, users)
│   ├── aggregator.py        # ScoreAggregator — orchestrates all scorers
│   ├── amount.py            # amount_scorer()
│   ├── time.py              # time_scorer()
│   ├── frequency.py         # frequency_scorer()
│   ├── channel.py           # channel_switch_scorer()
│   ├── round_amount.py      # round_amount_scorer()
│   ├── entities.py          # new_entity_scorer()
│   ├── balance.py           # balance_drain_scorer()
│   └── geo.py               # geo_scorer(), geocode_location()
│
├── ml/
│   ├── isolation_forest.py  # IsolationForestScorer
│   └── graph_analyzer.py    # GraphAnalyzer
│
├── agents/
│   ├── llm_utils.py         # strip_thinking_tags, parse_json_response, retry
│   ├── comms.py             # CommsAgent
│   └── coordinator.py       # FraudCoordinator
│
├── tests/                   # 73 tests, 1 skipped
│   ├── test_loaders.py
│   ├── test_output.py
│   ├── test_profiler.py
│   ├── test_amount.py
│   ├── test_time.py
│   ├── test_frequency.py
│   ├── test_channel.py
│   ├── test_round_amount.py
│   ├── test_entities.py
│   ├── test_balance.py
│   ├── test_geo.py
│   ├── test_isolation_forest.py
│   ├── test_graph_analyzer.py
│   ├── test_audio.py
│   ├── test_llm_utils.py
│   ├── test_comms_agent.py
│   ├── test_coordinator.py
│   ├── test_aggregator.py
│   ├── test_main.py
│   └── test_integration.py  # Requires OPENROUTER_API_KEY
│
├── training-dataset/
│   └── <Dataset Name> - train/
│       ├── transactions.csv
│       ├── users.json
│       ├── sms.json
│       ├── mails.json
│       ├── locations.json
│       └── audio/           # Deus Ex only
│
└── evaluation-dataset/
    └── <Dataset Name> - validation/
        └── ...              # Same structure as training
```

---

## Running Tests

```bash
# Unit tests (no API key needed)
pytest tests/ --ignore=tests/test_integration.py -v

# Integration tests (needs OPENROUTER_API_KEY in .env)
pytest tests/test_integration.py -v -s
```

---

## Output Format

The output file contains one transaction ID per line:

```
01JQKM3N2P4R5T6V7W8X9Y0Z1A
01JQKM3N2P4R5T6V7W8X9Y0Z1B
01JQKM3N2P4R5T6V7W8X9Y0Z1C
```

File is written to `output/<dataset_name_snake_case>.txt`.
