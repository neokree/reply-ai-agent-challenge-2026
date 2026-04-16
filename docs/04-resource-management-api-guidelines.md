# Resource Management: Token Usage and Cost Tracking with Langfuse

> Source: https://cdn.reply.com/documents/challenges/02_26/api_guidelines.html

For the challenge, **all costs are tracked exclusively via Langfuse session IDs**.

This tutorial shows how to integrate Langfuse with LangChain to automatically track token usage, costs, and performance.

---

## Why Resource Management Matters

When building production AI agent systems, understanding resource usage is crucial:

- **Cost control** — LLM API calls cost money; you need to track spending
- **Performance optimization** — Token usage affects response times and costs
- **Budget planning** — Predict costs before scaling your system
- **Debugging** — Token metrics help identify inefficient patterns

---

## What Are Tokens?

Tokens are the units that language models process. Roughly:

- 1 token ≈ 4 characters of English text
- 1 token ≈ 0.75 words
- 1000 tokens ≈ 750 words

When you call an agent, it uses:

- **Input tokens** — Your question + system prompt + conversation history
- **Output tokens** — The agent's response
- **Cache tokens** — Optional caching for faster/cheaper repeated queries

---

## Prerequisites

- **Python 3.13** (suggested) — avoid Python 3.14 due to compatibility issues
- **OpenRouter API key** (get one free at https://openrouter.ai)
- **Langfuse credentials** — `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`
- Completed Tutorial 01 (basic agent creation)

> ⚠️ **Langfuse SDK Version**: Use **Langfuse SDK v3** (`>=3,<4`). v4 is not fully supported.

---

## Quick Setup Checklist

1. Install Python 3.13: verify with `python3 --version`
2. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Get an OpenRouter API key at openrouter.ai → Keys → Create Key
4. Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your-api-key-here
LANGFUSE_PUBLIC_KEY=pk-your-public-key-here
LANGFUSE_SECRET_KEY=sk-your-secret-key-here
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=your-team-name
LANGFUSE_MEDIA_UPLOAD_ENABLED=false
```

---

## Installation

```bash
pip install langchain langchain-openai "langfuse>=3,<4" python-dotenv ulid-py --quiet
```

---

## Setup Model

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

model_id = "gpt-4o-mini"

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model_id,
    temperature=0.7,
    max_tokens=1000,
)
```

---

## How Langfuse Works with LangChain

The integration combines two mechanisms:

1. **`@observe()` decorator** — Wraps a function to automatically create a Langfuse trace on each call. All Langfuse operations inside the decorated function are nested under that trace.
2. **`CallbackHandler()`** — Created *inside* the `@observe()` function; automatically attaches to the current trace and captures LangChain-specific metrics (tokens, costs, latency).
3. **Session tracking** — Multiple calls can be grouped under the same `session_id` by passing `config={"metadata": {"langfuse_session_id": session_id}}` to LangChain calls.

### What Gets Tracked Automatically

The `CallbackHandler` captures:
- Inputs and outputs
- Token usage (input, output, and cache tokens)
- Costs (automatically calculated based on model pricing)
- Latency
- Metadata (model parameters, temperature, etc.)

---

## Session ID Format

```
{TEAM_NAME}-{ULID}
```

> **Important**: `session_id` must not contain blank spaces. Normalize `TEAM_NAME` by replacing spaces with `-` when building the ID.

---

## Initialize Langfuse and Helper Functions

```python
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    team = os.getenv("TEAM_NAME", "tutorial").replace(" ", "-")
    return f"{team}-{ulid.new().str}"

def invoke_langchain(model, prompt, langfuse_handler, session_id):
    """Invoke LangChain with the given prompt and Langfuse handler."""
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(
        messages,
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return response.content

@observe()
def run_llm_call(session_id, model, prompt):
    """Run a single LangChain invocation and track it in Langfuse."""
    langfuse_handler = CallbackHandler()
    response = invoke_langchain(model, prompt, langfuse_handler, session_id)
    return response
```

---

## Run a Single Traced Call

```python
session_id = generate_session_id()
print(f"Session ID: {session_id}\n")

response = run_llm_call(session_id, model, "What is the square root of 144?")
print(f"Response: {response}")

langfuse_client.flush()
```

> Always call `langfuse_client.flush()` after your calls to ensure all traces are sent.

---

## Track Multiple Calls with Session Grouping

Since every call to `run_llm_call()` shares the same `session_id`, all traces are grouped together automatically — no manual accumulation needed.

```python
questions = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is the difference between AI and ML?"
]

session_id = generate_session_id()

for i, question in enumerate(questions, 1):
    response = run_llm_call(session_id, model, question)
    print(f"Call {i}: {question[:40]}...")
    print(f"  Response: {response[:80]}...\n")

langfuse_client.flush()
```

---

## How the Tracing Flow Works

```
@observe() decorated function
    ↓
Creates Langfuse trace → pass metadata.langfuse_session_id
    ↓
CallbackHandler() attaches to current trace
    ↓
model.invoke(messages, config={"callbacks": [handler]})
    ↓
CallbackHandler captures: tokens, costs, latency, I/O
    ↓
langfuse_client.flush() → sends to Langfuse
    ↓
Langfuse dashboard (platform page) → view sessions, traces and observations
```

---

## Viewing Your Traces

View tracing details on the **Langfuse dashboard** available in the platform page. The dashboard is associated with your team — all traces from your team members will be visible there.

> Note: The dashboard is **not updated in real time**. There may be a delay of a few minutes before the latest traces appear.

---

## Best Practices

- **Always set session IDs** — Essential for the challenge; groups all costs under one session
- **Use `@observe()` + `CallbackHandler`** — Wrap LLM-calling code so Langfuse captures everything automatically
- **Flush after calls** — Call `langfuse_client.flush()` to ensure traces are sent
- **Generate unique session IDs** — Use ULID to avoid collisions
- **Monitor regularly** — Check the Langfuse dashboard to review costs after each session
- **Optimize prompts** — Shorter prompts = fewer input tokens = lower costs
- **Start small, scale up** — Begin with smaller models and only switch to larger ones if needed
- **Multi-agent strategy** — Use larger models for critical decisions, smaller models for simpler tasks
