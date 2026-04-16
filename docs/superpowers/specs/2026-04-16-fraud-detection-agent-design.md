# Fraud Detection Agent — Design Specification

**Date:** 2026-04-16  
**Challenge:** Reply AI Agent Challenge 2026 — Reply Mirror  
**Status:** Draft

---

## 1. Overview

Sistema ibrido algoritmi + AI per rilevare transazioni fraudolente nel contesto del challenge Reply Mirror. Il sistema processa un dataset alla volta e produce un file di output con gli ID delle transazioni sospette.

### Goals

- Rilevare frodi con alta accuracy, minimizzando falsi positivi
- Rispettare il vincolo "agent-based solutions only"
- Contenere i costi API (scoring secondario del challenge)
- Struttura estensibile per i livelli 4-5 futuri

### Non-Goals

- Supporto real-time streaming
- Training di modelli ML custom
- UI/dashboard

---

## 2. Architecture

```
main.py --dataset "The Truman Show"
    │
    ▼
load_data()
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — PREPROCESSING (no LLM)                           │
├─────────────────────────────────────────────────────────────┤
│  [Se audio presente - solo Deus Ex]                         │
│  audio_transcriber()  → Whisper API → dict[filename, text]  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — ALGORITMI DETERMINISTICI (no LLM)                │
├─────────────────────────────────────────────────────────────┤
│  build_user_profiles()  → stats da training                 │
│  amount_scorer()        → Z-score importi                   │
│  time_scorer()          → flag orari anomali + weekend      │
│  geo_checker()          → velocità impossibili (>500 km/h)  │
│  new_entity_detector()  → recipient nuovi + days_since      │
│  frequency_checker()    → troppe tx in finestra breve       │
│  channel_switch_scorer()→ cambio metodo pagamento           │
│  round_amount_scorer()  → importi tondi sospetti            │
│  balance_drain_scorer() → % saldo spesa in una tx           │
│                                                             │
│  Output: DataFrame risk scores per transaction_id           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2.5 — ML ANOMALY DETECTION (no LLM)                  │
├─────────────────────────────────────────────────────────────┤
│  isolation_forest()     → anomaly score su feature vector   │
│  graph_analyzer()       → GNN per pattern reti di frode     │
│                         → money laundering, smurfing, muli  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — AI AGENTS (2 LLM calls via LangChain)            │
├─────────────────────────────────────────────────────────────┤
│  @observe()                                                 │
│  comms_agent()        → analizza SMS + mail + trascrizioni  │
│                       → output: segnali social engineering  │
│                                                             │
│  @observe()                                                 │
│  fraud_coordinator()  → riceve score + segnali comms        │
│                       → output: lista transaction_id fraud  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
write_output()         → output/{dataset_name}.txt
langfuse_client.flush()
```

---

## 3. File Structure

```
reply-ai-agent-challenge-2026/
├── main.py                    # Entry point + orchestrazione
├── preprocessing/
│   └── audio.py               # audio_transcriber() con Whisper
│                              # Output: dict[filename, transcript_text]
│                              # Estrae user name da filename per join
├── scoring/
│   ├── profiler.py            # build_user_profiles()
│   ├── amount.py              # amount_scorer()
│   ├── time.py                # time_scorer()
│   ├── geo.py                 # geo_checker()
│   ├── entities.py            # new_entity_detector()
│   ├── frequency.py           # frequency_checker()
│   ├── channel.py             # channel_switch_scorer()
│   ├── round_amount.py        # round_amount_scorer()
│   └── balance.py             # balance_drain_scorer()
├── ml/
│   ├── isolation_forest.py    # Anomaly detection
│   └── graph_analyzer.py      # GNN / NetworkX patterns
├── agents/
│   ├── comms.py               # comms_agent() — LLM
│   └── coordinator.py         # fraud_coordinator() — LLM
├── utils/
│   ├── loaders.py             # load_data()
│   └── output.py              # write_output()
├── output/                    # gitignored
├── .env
└── requirements.txt
```

---

## 4. Data Flow

### Input per Dataset

| File | Contenuto | Usato da |
|------|-----------|----------|
| `transactions.csv` | Transazioni con importo, tipo, timestamp, IBAN, location | Tutti gli scorer |
| `users.json` | Profilo utenti (salary, job, residence, IBAN) | profiler, coordinator |
| `locations.json` | GPS tracking (biotag, timestamp, lat/lng) | geo_checker |
| `sms.json` | Thread SMS | comms_agent |
| `mails.json` | Thread email | comms_agent |
| `audio/` | MP3 (solo Deus Ex), naming: `YYYYMMDD_HHMMSS-nome_cognome.mp3` | audio_transcriber → comms_agent |

### Output

File ASCII `output/{dataset_name}.txt`:
```
abc-123-def-456
ghi-789-jkl-012
...
```
Un `transaction_id` per riga, nessun header.

**Vincoli output (da problem statement):**
- Non vuoto (almeno 1 tx)
- Non tutte le tx
- Almeno 15% delle frodi reali identificate

---

## 5. Component Specifications

### 5.1 build_user_profiles()

**Input:** `train_transactions`, `users`

**Output:**
```python
{
  "FR85H4824371990132980420818": {  # user IBAN as key
    "user_id": "RGNR-LNAA-7FF-AUD-0",
    "name": "Alain Regnier",
    "avg_amount_by_type": {"transfer": 1200.0, "e-commerce": 85.0},
    "std_amount_by_type": {"transfer": 400.0, "e-commerce": 30.0},
    "typical_hours": [9, 10, 11, 12, 14, 15, 16, 17, 18],
    "weekend_active": False,           # True se >20% delle tx sono nel weekend
    "payment_methods": {"debit card": 45, "mobile device": 12},
    "known_recipients": {"IBAN1", "IBAN2", "MERCHANT-ID"},
    "last_tx_to": {"IBAN1": datetime, "IBAN2": datetime},  # ultimo tx per recipient
    "avg_daily_tx_count": 2.3,
    "residence_coords": (47.4836, 6.8403),
    "salary": 34100,
    "often_round_amounts": False,      # True se >30% delle tx sono importi tondi
  }
}
```

### 5.2 amount_scorer()

**Logic:**
```python
z_score = (tx.amount - mean) / std
score = min(abs(z_score) / 4, 1.0)
```

**Output:** `float` 0.0-1.0

### 5.3 time_scorer()

**Logic:**
```python
tx_hour = tx.timestamp.hour
tx_weekday = tx.timestamp.weekday()  # 0=Mon, 6=Sun
is_weekend = tx_weekday >= 5

typical_hours = user_profile["typical_hours"]
typical_weekend_activity = user_profile.get("weekend_active", False)

# Score per ora anomala
if tx_hour in typical_hours:
    hour_score = 0.0
else:
    min_distance = min(abs(tx_hour - h) for h in typical_hours)
    hour_score = min(min_distance / 6, 1.0)

# Boost se weekend e utente non opera di solito nel weekend
weekend_boost = 0.3 if (is_weekend and not typical_weekend_activity) else 0.0

score = min(hour_score + weekend_boost, 1.0)
```

**Output:** `float` 0.0-1.0

### 5.4 geo_checker()

**Transaction location source:**
- Il campo `location` in `transactions.csv` contiene stringhe come "Munich - Isar River Cafe"
- Per le tx in-person: geocoding via geopy o lookup su tabella città note
- Per trasferimenti/e-commerce: `location` è vuoto, skip geo check

**Logic:**
1. Trova posizione GPS dell'utente più vicina al timestamp tx (da `locations.json` usando `biotag`)
2. Se tx è in-person e ha `location`: geocode → coordinate
3. Calcola distanza GPS utente vs tx location
4. Se distanza > 50km → flag
5. Check velocità tra tx consecutive con location: se > 500 km/h → impossibile

**Output:** `float` 0.0-1.0

### 5.5 new_entity_detector()

**Logic:**
```python
known = user_profile["known_recipients"]
recipient = tx.recipient_id or tx.recipient_iban

if recipient in known:
    # Già visto: calcola days_since_last_tx
    last_tx_date = user_profile["last_tx_to"].get(recipient)
    if last_tx_date:
        days_since = (tx.timestamp - last_tx_date).days
        # Se non visto da 90+ giorni, lieve sospetto
        score = 0.2 if days_since > 90 else 0.0
    else:
        score = 0.0
else:
    # Mai visto: score base 0.5, boost con importo relativo
    amount_factor = min(tx.amount / user_profile["salary"], 1.0)
    score = 0.5 + (0.5 * amount_factor)  # range 0.5-1.0
```

**Output:** `float` 0.0-1.0

### 5.6 frequency_checker()

**Logic:**
- Conta tx dello stesso utente in finestre 1h e 24h
- Se > 5 tx/ora o > 3x media giornaliera → flag

**Output:** `float` 0.0-1.0 (per singola tx, come gli altri scorer)

### 5.7 channel_switch_scorer()

**Logic:**
```python
usual_methods = user_profile["payment_methods"]  # dict: method -> count
total_txs = sum(usual_methods.values())
method_pct = usual_methods.get(tx.payment_method, 0) / total_txs

if method_pct == 0:
    # Metodo MAI usato prima
    score = 0.8
elif method_pct < 0.1:
    # Metodo usato raramente (<10% delle tx)
    score = 0.4
else:
    score = 0.0

# Boost se combinato con importo alto
if tx.amount > user_profile["salary"] * 0.5:
    score = min(score + 0.2, 1.0)
```

**Output:** `float` 0.0-1.0

### 5.8 round_amount_scorer()

**Logic:**
```python
# Fraudster spesso testano con cifre tonde
is_round_100 = (tx.amount % 100 == 0)
is_round_1000 = (tx.amount % 1000 == 0)

if is_round_1000 and tx.amount >= 1000:
    score = 0.5
elif is_round_100 and tx.amount >= 100:
    score = 0.3
else:
    score = 0.0

# Non penalizzare se l'utente ha storico di importi tondi (es. affitto fisso)
if user_profile.get("often_round_amounts", False):
    score *= 0.3  # Riduci penalità
```

**Output:** `float` 0.0-1.0

### 5.9 balance_drain_scorer()

**Logic:**
```python
# Percentuale del saldo spesa in questa singola tx
balance_before = tx.balance_after + tx.amount
pct_spent = tx.amount / balance_before if balance_before > 0 else 0

if pct_spent > 0.9:
    # Quasi tutto il saldo in una tx
    score = 1.0
elif pct_spent > 0.7:
    score = 0.7
elif pct_spent > 0.5:
    score = 0.4
else:
    score = 0.0

# Check pattern drain: multiple tx che erodono verso 0
recent_txs = get_user_txs_last_hours(tx.sender_id, tx.timestamp, hours=2)
if len(recent_txs) >= 3:
    total_recent_spent = sum(t.amount for t in recent_txs)
    if total_recent_spent / balance_before > 0.8:
        score = 1.0  # Pattern drain attivo
```

**Helper function:**
```python
def get_user_txs_last_hours(sender_id: str, current_ts: datetime, hours: int) -> list:
    """
    Ritorna le transazioni dello stesso utente nelle ultime N ore.
    Deve essere chiamato con accesso al DataFrame delle transazioni eval.
    """
    cutoff = current_ts - timedelta(hours=hours)
    return [tx for tx in eval_transactions 
            if tx.sender_id == sender_id 
            and cutoff <= tx.timestamp < current_ts]
```

**Output:** `float` 0.0-1.0

---

### 5.10 Error Handling

**Empty training data per user:**
- Se un utente non ha tx nel training, usa fallback globali:
  - `avg_amount_by_type`: media globale di tutte le tx training
  - `std_amount_by_type`: deviazione standard globale
  - `typical_hours`: [8-20] (orario lavorativo default)
  - `weekend_active`: False
  - `often_round_amounts`: False

**Missing location data:**
- Se `tx.location` è vuoto (non in-person): `geo_score = 0.0`
- Se `locations.json` non ha dati per l'utente: `geo_score = 0.0`
- Se geocoding fallisce (API error, location ambigua): log warning, `geo_score = 0.0`

**Geocoding specifics:**
```python
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

geolocator = Nominatim(user_agent="fraud_detector")

def geocode_location(location_str: str) -> tuple[float, float] | None:
    """
    Geocode location string to (lat, lng).
    Returns None on failure.
    """
    try:
        # Extract city from strings like "Munich - Isar River Cafe"
        city = location_str.split(" - ")[0].strip()
        result = geolocator.geocode(city, timeout=5)
        if result:
            return (result.latitude, result.longitude)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logging.warning(f"Geocoding failed for {location_str}: {e}")
    return None
```

---

## 5B. ML Anomaly Detection Layer

### 5B.1 isolation_forest()

**Purpose:** Rilevare anomalie in spazio multidimensionale che i singoli scorer potrebbero non catturare.

**Input:** Feature vector per ogni transazione:
```python
features = [
    amount_zscore,
    time_score,
    geo_score,
    new_entity_score,
    frequency_score,
    channel_switch_score,
    round_amount_score,
    balance_drain_score,
    velocity_1h,
    velocity_24h,
    pct_of_balance_spent,
    hour_of_day,
    is_weekend,
]
```

**Logic:**
```python
from sklearn.ensemble import IsolationForest

# Fit su training data (comportamento normale)
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(train_features)

# Predict su eval data
# -1 = anomaly, 1 = normal
predictions = clf.predict(eval_features)
anomaly_scores = clf.decision_function(eval_features)  # più negativo = più anomalo

# Normalizza a 0-1
normalized = (anomaly_scores.max() - anomaly_scores) / (anomaly_scores.max() - anomaly_scores.min())
```

**Output:** `float` 0.0-1.0 per ogni tx

### 5B.2 graph_analyzer()

**Purpose:** Rilevare pattern di frode organizzata (money laundering, smurfing, conti mulo) invisibili a livello di singola transazione.

**Graph structure:**
- **Nodi:** Account (sender_id, recipient_id, IBAN)
- **Archi:** Transazioni (amount, timestamp, type)
- **Node features:** User profile stats, aggregated tx behavior
- **Edge features:** Amount, frequency, time patterns

**Patterns da rilevare:**
1. **Smurfing:** Molte piccole tx sotto soglia reporting che sommano a importo grande
2. **Layering:** Catena A→B→C→D dove fondi passano rapidamente attraverso intermediari
3. **Conti mulo:** Account che riceve da molti e trasferisce a pochi (o viceversa)
4. **Circular flows:** A→B→C→A con importi simili

**Implementation:**
```python
import networkx as nx
# Opzionale per produzione: torch_geometric per GNN

def build_transaction_graph(transactions):
    G = nx.DiGraph()
    for tx in transactions:
        G.add_edge(
            tx.sender_id, 
            tx.recipient_id,
            amount=tx.amount,
            timestamp=tx.timestamp,
            tx_id=tx.id
        )
    return G

def detect_mule_accounts(G):
    """Account con alto in-degree e alto out-degree = potenziale mulo"""
    suspicious = []
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        if in_deg > 5 and out_deg > 3:
            suspicious.append(node)
    return suspicious

def detect_circular_flows(G):
    """Trova cicli nel grafo delle transazioni"""
    cycles = list(nx.simple_cycles(G))
    return [c for c in cycles if len(c) >= 3]

def detect_rapid_layering(G, max_hours=2):
    """
    Catene di tx rapide attraverso intermediari.
    Pattern: A→B→C dove B riceve e trasferisce entro max_hours.
    """
    suspicious_txs = []
    
    for node in G.nodes():
        # Trova tx in entrata e uscita per questo nodo
        in_edges = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))
        
        for in_edge in in_edges:
            in_ts = in_edge[2]['timestamp']
            in_amount = in_edge[2]['amount']
            
            for out_edge in out_edges:
                out_ts = out_edge[2]['timestamp']
                out_amount = out_edge[2]['amount']
                
                # Check: uscita entro max_hours dall'entrata
                time_diff = (out_ts - in_ts).total_seconds() / 3600
                if 0 < time_diff <= max_hours:
                    # Check: importo simile (±20%)
                    if 0.8 <= out_amount / in_amount <= 1.2:
                        suspicious_txs.append(in_edge[2]['tx_id'])
                        suspicious_txs.append(out_edge[2]['tx_id'])
    
    return list(set(suspicious_txs))
```

**Data source:** Il grafo viene costruito da **training + eval transactions** combinati per rilevare pattern cross-dataset.

**Output:** 
- Lista di `account_id` sospetti (muli, intermediari)
- Lista di `transaction_id` coinvolti in pattern sospetti
- Score 0.0-1.0 per ogni tx basato su coinvolgimento in pattern

**Nota:** Per dataset grandi in produzione, considerare PyTorch Geometric con modelli GCN/GAT pre-trainati.

---

### 5C. Score Aggregation

Tutti gli scorer + ML layer vengono applicati a ogni transazione eval:

```python
def aggregate_scores(tx, profiles, locations, ml_scores, graph_scores):
    scores = {
        "transaction_id": tx.id,
        # Rule-based scorers
        "amount_score": amount_scorer(tx, profiles),
        "time_score": time_scorer(tx, profiles),
        "geo_score": geo_checker(tx, locations, profiles),
        "new_entity_score": new_entity_detector(tx, profiles),
        "frequency_score": frequency_checker(tx, profiles),
        "channel_switch_score": channel_switch_scorer(tx, profiles),
        "round_amount_score": round_amount_scorer(tx, profiles),
        "balance_drain_score": balance_drain_scorer(tx, profiles),
        # ML scores
        "isolation_forest_score": ml_scores.get(tx.id, 0.0),
        "graph_score": graph_scores.get(tx.id, 0.0),
    }
    
    # Weighted average - pesi tunable
    weights = {
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
    scores["total_risk"] = sum(
        scores[k] * weights[k] for k in weights
    )
    return scores
```

**Output DataFrame:**
| tx_id | amount | time | geo | new_entity | freq | channel | round | drain | iforest | graph | total_risk |
|-------|--------|------|-----|------------|------|---------|-------|-------|---------|-------|------------|

---

## 6. AI Agents

### 6.1 comms_agent()

**Model:** `qwen/qwen3-plus` via OpenRouter

**Input:**
- SMS training + eval
- Mail training + eval  
- Audio transcriptions (se presenti)
- User name → IBAN mapping

**Context limit handling:**
Per dataset grandi, se il contenuto supera 100K tokens:
1. Raggruppa comunicazioni per utente
2. Processa in batch (max 50K tokens per call)
3. Aggrega i risultati

```python
MAX_TOKENS_PER_CALL = 50000  # ~200K chars

def chunk_communications(sms, mails, audio):
    """Split communications into chunks that fit context window."""
    chunks = []
    current_chunk = {"sms": [], "mails": [], "audio": []}
    current_size = 0
    
    for item in sms + mails + audio:
        item_size = len(str(item))
        if current_size + item_size > MAX_TOKENS_PER_CALL * 4:
            chunks.append(current_chunk)
            current_chunk = {"sms": [], "mails": [], "audio": []}
            current_size = 0
        # Add to appropriate list
        current_chunk[item['type']].append(item)
        current_size += item_size
    
    if current_size > 0:
        chunks.append(current_chunk)
    return chunks
```

**Task:** Identificare segnali di social engineering:
- Phishing
- Urgenza manipolatoria
- Richieste credenziali/PIN
- Link sospetti
- Impersonificazione banca/servizi
- **Correlazione temporale:** comunicazione sospetta ricevuta <24h prima di tx anomala

**Output format:**
```json
{
  "signals": [
    {
      "user_iban": "FR85H4824371990132980420818",
      "user_name": "Alain Regnier",
      "source": "sms|email|audio",
      "severity": "high|medium|low",
      "timestamp": "2087-03-15T14:30:00",
      "reason": "descrizione del segnale"
    }
  ]
}
```

**Nota:** Il prompt include la mappatura nome→IBAN estratta da `users.json` per permettere il join con i profili.

**Correlazione temporale:** Il `fraud_coordinator` riceverà i timestamp dei segnali per correlarli con le tx:
- Segnale <2h prima di tx anomala → boost significativo al risk
- Segnale <24h prima → boost moderato
- Segnale >24h prima → considerato ma peso minore

**Prompt template:**
```
Sei un analista antifrode specializzato in social engineering.

Analizza le comunicazioni degli utenti cercando segnali di:
- Phishing (richieste credenziali, link sospetti)
- Manipolazione psicologica (urgenza artificiosa, minacce)
- Impersonificazione (qualcuno si spaccia per banca/servizio)
- Richieste anomale (trasferimenti "urgenti" a conti "sicuri")

SMS: {sms_data}
Email: {mail_data}
Trascrizioni audio: {audio_data}

Rispondi SOLO in JSON con il formato:
{"signals": [{"user": "...", "source": "...", "severity": "...", "reason": "..."}]}

Se non trovi segnali sospetti, rispondi: {"signals": []}
```

### 6.2 fraud_coordinator()

**Model:** `qwen/qwen3-plus` via OpenRouter

**Input:**
1. Risk score table (output algoritmi)
2. Segnali comms_agent
3. Lista transazioni eval con dettagli
4. Profili utenti (contesto)

**Task:** Decisione finale su quali tx flaggare, correlando:
- Score algoritmici alti
- Segnali social engineering
- Contesto utente (salary, comportamento normale)

**Constraints:**
- Output non vuoto
- Non flaggare tutte le tx
- Bilanciare precision/recall

**Output format:** Lista transaction_id, uno per riga, niente altro.

**Prompt template:**
```
Sei il coordinatore del sistema antifrode MirrorPay.

Ricevi:
1. Una tabella di risk score calcolati da algoritmi (0-1 per categoria)
2. Segnali di social engineering rilevati nelle comunicazioni
3. Le transazioni da valutare

REGOLE:
- Devi flaggare almeno 1 transazione
- Non puoi flaggare tutte le transazioni
- Privilegia precision: meglio perdere qualche frode che bloccare clienti legittimi
- Correla i segnali: score alto + segnale comms = quasi certamente frode

RISK SCORE TABLE:
{risk_table}

SEGNALI COMUNICAZIONI:
{comms_signals}

TRANSAZIONI EVAL:
{eval_transactions}

Rispondi SOLO con la lista di transaction_id da flaggare, uno per riga.
Nessun altro testo, nessuna spiegazione.
```

### 6.3 LLM Output Parsing & Error Handling

**Qwen thinking tag handling:**

Qwen models with thinking enabled wrap their reasoning in `<think>...</think>` tags before the actual response. We must strip this before parsing.

```python
import re

def strip_thinking_tags(response: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen responses.
    The actual answer comes AFTER the closing </think> tag.
    """
    # Pattern matches <think> followed by any content until </think>
    pattern = r'<think>.*?</think>\s*'
    cleaned = re.sub(pattern, '', response, flags=re.DOTALL)
    return cleaned.strip()
```

**comms_agent output parsing:**
```python
def parse_comms_response(response: str) -> dict:
    # First strip thinking tags
    cleaned = strip_thinking_tags(response)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: search for JSON inside the text
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Last fallback: no signals found
        return {"signals": []}

**fraud_coordinator output parsing:**
```python
def parse_coordinator_response(response: str, valid_tx_ids: set) -> list[str]:
    # First strip thinking tags
    cleaned = strip_thinking_tags(response)
    
    lines = cleaned.strip().split('\n')
    flagged = []
    for line in lines:
        tx_id = line.strip()
        if tx_id in valid_tx_ids:
            flagged.append(tx_id)
    
    # Output validation
    if len(flagged) == 0:
        raise ValueError("Coordinator returned empty list")
    if len(flagged) == len(valid_tx_ids):
        raise ValueError("Coordinator flagged all transactions")
    
    return flagged
```

**Fallback strategy:** Se il parsing fallisce o output invalido, usa i risk score algoritmici:
- **Empty output fallback:** Flag top 15% delle tx per `total_risk` score
- **All flagged fallback:** Flag solo top 30% delle tx per `total_risk` score
- Queste soglie garantiscono output valido secondo i vincoli del challenge

**LLM API retry logic:**
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError))
)
def call_llm_with_retry(model, messages, config):
    return model.invoke(messages, config=config)
```

Gestione errori:
- **429 Rate Limit:** Exponential backoff (2s, 4s, 8s)
- **503 Service Unavailable:** Retry con backoff
- **Timeout:** 60s per call, poi retry
- **Max 3 attempts:** Se tutti falliscono, usa fallback algoritmico

---

## 7. Langfuse Integration

Pattern identico all'esempio ufficiale:

```python
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

@observe()
def comms_agent(data, model, session_id):
    handler = CallbackHandler()
    response = model.invoke(
        messages,
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id}
        }
    )
    return parse_response(response.content)
```

**Session ID format:** `{TEAM_NAME}-{ULID}`

---

## 8. Configuration

### .env
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
OPENROUTER_API_KEY=sk-or-...
TEAM_NAME=your-team-name
LANGFUSE_MEDIA_UPLOAD_ENABLED=false
```

### requirements.txt
```
langchain>=1.2.0
langchain-openai>=1.1.0
langfuse>=3,<4
python-dotenv
ulid-py
pandas
geopy
openai
scikit-learn        # Isolation Forest
networkx            # Graph analysis base
tenacity            # Retry logic for LLM calls
# torch-geometric   # Optional: GNN per produzione su larga scala
```

---

## 9. Usage

```bash
# Attiva venv
source .venv/bin/activate

# Esegui per singolo dataset
python main.py --dataset "The Truman Show"
python main.py --dataset "Brave New World"
python main.py --dataset "Deus Ex"

# Output generati in output/
ls output/
# the_truman_show.txt
# brave_new_world.txt
# deus_ex.txt
```

---

## 10. Cost Estimate

| Componente | Calls/dataset | Tokens stimati | Costo |
|------------|---------------|----------------|-------|
| Whisper (solo Deus Ex) | 48 file × ~1.5 min | N/A | ~$0.50 |
| comms_agent | 1 | ~3000 in + 500 out | ~$0.01 |
| fraud_coordinator | 1 | ~2000 in + 200 out | ~$0.01 |
| **Totale per dataset** | | | **~$0.02-0.52** |
| **Totale 3 dataset** | | | **~$0.56** |

Budget $40 → ~70+ run completi possibili.

---

## 11. Open Questions

1. **Soglie scorer:** I valori (Z > 4, distanza > 50km, velocità > 500 km/h) sono ragionevoli ma potrebbero richiedere tuning dopo i primi test
2. ~~**Pesi weighted sum:** Come combinare gli score?~~ → Risolto: weighted average con pesi in 5.7
3. **Audio format:** Whisper supporta MP3 direttamente → Sì, confermato (documentazione OpenAI)

---

## 12. Success Criteria

- [ ] Output valido per tutti e 3 i dataset (non vuoto, non tutto)
- [ ] Almeno 15% frodi identificate (soglia minima challenge)
- [ ] Costo totale < $5 per run completo
- [ ] Trace Langfuse visibili su dashboard con session_id corretto
