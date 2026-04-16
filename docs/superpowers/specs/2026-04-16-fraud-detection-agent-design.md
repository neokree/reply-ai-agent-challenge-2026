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
│  audio_transcriber()  → Whisper API → trascrizioni .txt     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — ALGORITMI DETERMINISTICI (no LLM)                │
├─────────────────────────────────────────────────────────────┤
│  build_user_profiles()  → stats da training                 │
│  amount_scorer()        → Z-score importi                   │
│  time_scorer()          → flag orari anomali                │
│  geo_checker()          → velocità impossibili              │
│  new_entity_detector()  → recipient/merchant nuovi          │
│  frequency_checker()    → troppe tx in finestra breve       │
│                                                             │
│  Output: DataFrame risk scores per transaction_id           │
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
├── scoring/
│   ├── profiler.py            # build_user_profiles()
│   ├── amount.py              # amount_scorer()
│   ├── time.py                # time_scorer()
│   ├── geo.py                 # geo_checker()
│   ├── entities.py            # new_entity_detector()
│   └── frequency.py           # frequency_checker()
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
| `audio/` | MP3 (solo Deus Ex) | audio_transcriber → comms_agent |

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
    "payment_methods": {"debit card": 45, "mobile device": 12},
    "known_recipients": {"IBAN1", "IBAN2", "MERCHANT-ID"},
    "avg_daily_tx_count": 2.3,
    "residence_coords": (47.4836, 6.8403),
    "salary": 34100
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
- Estrae ora della tx
- Confronta con ore tipiche dell'utente da training
- Score = distanza minima normalizzata

**Output:** `float` 0.0-1.0

### 5.4 geo_checker()

**Logic:**
1. Trova posizione GPS dell'utente più vicina al timestamp tx
2. Calcola distanza da location della tx (se in-person)
3. Se distanza > 50km → flag
4. Check velocità tra tx consecutive: se > 500 km/h → impossibile

**Output:** `float` 0.0-1.0

### 5.5 new_entity_detector()

**Logic:**
- Verifica se recipient_id/recipient_iban è nel set known_recipients
- Se nuovo: score base 0.5, aumenta con importo relativo a salary

**Output:** `float` 0.0-1.0

### 5.6 frequency_checker()

**Logic:**
- Conta tx dello stesso utente in finestre 1h e 24h
- Se > 5 tx/ora o > 3x media giornaliera → flag

**Output:** `dict[transaction_id, float]`

---

## 6. AI Agents

### 6.1 comms_agent()

**Model:** `qwen/qwen3-plus` via OpenRouter

**Input:**
- SMS training + eval
- Mail training + eval  
- Audio transcriptions (se presenti)

**Task:** Identificare segnali di social engineering:
- Phishing
- Urgenza manipolatoria
- Richieste credenziali/PIN
- Link sospetti
- Impersonificazione banca/servizi

**Output format:**
```json
{
  "signals": [
    {
      "user": "Alain Regnier",
      "source": "sms|email|audio",
      "severity": "high|medium|low",
      "reason": "descrizione del segnale"
    }
  ]
}
```

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
2. **Pesi weighted sum:** Come combinare gli score? Semplice media o pesi diversi per categoria?
3. **Audio format:** Whisper supporta MP3 direttamente? Verificare prima dell'implementazione

---

## 12. Success Criteria

- [ ] Output valido per tutti e 3 i dataset (non vuoto, non tutto)
- [ ] Almeno 15% frodi identificate (soglia minima challenge)
- [ ] Costo totale < $5 per run completo
- [ ] Trace Langfuse visibili su dashboard con session_id corretto
