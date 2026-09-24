# Multi-Agent Customer Support System (LangGraph)

A customer-support system that behaves like a real company: a **reception desk**
classifies each request, routes it to the right **specialist**, and a
**sentiment guardrail** halts the machine and calls a human whenever the
customer is angry.

Built for the *Generative AI — Multi Agent* assignment (`HW_07_multi agent_2.pdf`).

---

## Target architecture

```
                    ┌──────────────────────┐
   user message ──▶ │  1. Triage Agent     │  classify: BILLING / TECHNICAL / GENERAL
                    │     (structured out) │
                    └──────┬───────┬───────┘
                           │       │
          ┌────────────────┘       └────────────────┐
          ▼                                         ▼
 ┌─────────────────────┐                  ┌─────────────────────┐
 │ 2. Billing          │                  │ 3. Technical        │
 │    Specialist       │                  │    Support          │
 │ • check_subscription│                  │ • search_knowledge  │
 │ • process_refund    │                  │   _base (RAG)       │
 │ off-topic → back to │                  │ no source → says    │
 │ triage              │                  │ "I don't know"      │
 └──────────┬──────────┘                  └──────────┬──────────┘
            └───────────────┬─────────────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ 4. Sentiment         │  negative → interrupt()  ──▶ human manager
                 │    Guardrail         │  neutral/positive → reply to user
                 └──────────────────────┘
```

---

## Build plan

| Step | Scope | Status |
| ---- | ----- | ------ |
| **1** | Project skeleton, `SupportState`, domain schemas, abstract interfaces, config + provider factory, mock data & RAG corpus | ✅ done |
| **2** | Triage agent using `with_structured_output`, plus an offline keyword classifier and a provider-verification script | ✅ done |
| **3** | Tools: subscription repository, refund gateway, RAG retriever (TF-IDF + embeddings) | ✅ done |
| **4** | Billing, Technical and General specialist nodes | ✅ done |
| **5** | Sentiment guardrail, checkpointer, both interrupt forms, `update_state`, graph assembly | ✅ done |
| **6** | Deliverable notebook: 3 scenarios + graph image | ✅ done |

---

## Layout

```
src/support_system/
├── domain/              # pure business types — no LangChain, no API key needed
│   ├── enums.py         #   Department / Sentiment / NextStep
│   ├── state.py         #   SupportState TypedDict (assignment Step 1)
│   └── schemas.py       #   Pydantic contracts for LLM output + tool results
├── interfaces/          # Protocols every concrete component implements
│   ├── llm.py           #   ChatModelProvider
│   ├── classification.py#   IntentClassifier / SentimentAnalyzer
│   ├── retrieval.py     #   KnowledgeRetriever
│   ├── billing.py       #   SubscriptionRepository / RefundGateway
│   ├── nodes.py         #   SupportNode
│   └── human.py         #   HumanReviewer
├── agents/              # graph nodes
│   ├── base.py          #   BaseSupportNode: naming, transcript lines, audit log
│   ├── triage.py        #   Node 1 — routing + loop guard
│   ├── billing.py       #   Node 2 — plan → execute → phrase, bounce-back rule
│   ├── technical.py     #   Node 3 — RAG, refuses to hallucinate
│   ├── general.py       #   greetings and everything else
│   └── guardrail.py     #   Node 4 — sentiment gate + human review
├── tools/definitions.py # the spec's 3 tools as LangChain StructuredTools
├── prompts/             # prompt text, kept out of the agent classes
├── infrastructure/      # concrete implementations of the interfaces
│   ├── llm/factory.py   #   provider registry + reachability probe
│   ├── classification/  #   intent + sentiment, LLM and offline variants
│   ├── retrieval/       #   TF-IDF and embedding retrievers
│   ├── billing/         #   JSON subscription store, mock refund gateway
│   ├── composition/     #   answer phrasing, LLM and template variants
│   ├── planning/        #   billing action planners
│   └── human/           #   scripted / console / auto reviewers
├── graph/
│   ├── builder.py       # nodes, edges, checkpointer, interrupt_before
│   └── application.py   # composition root — the only module wiring concretes
└── config/
    ├── settings.py      # the only module that reads os.environ
    └── dotenv.py        # tiny .env loader (no extra dependency)

notebooks/customer_support_langgraph.ipynb   # ← the deliverable
docs/images/support_graph.{mmd,txt,png}      # the generated graph diagram

data/
├── knowledge_base/      # 6 markdown articles — the RAG corpus
└── mock/                # subscriptions.json, transactions.json
tests/                   # runs offline, no API key required
```

Dependency direction is strictly inward:
`graph → agents → interfaces → domain`, with `infrastructure` plugging into
`interfaces` from the outside.

---

## How SOLID is applied

| Principle | Where |
| --------- | ----- |
| **S**ingle responsibility | `settings.py` is the only place reading `os.environ`; each node does one job; refunds and subscription lookups are separate classes. |
| **O**pen/closed | `@register_provider` adds an LLM vendor without editing the factory; a new department = one enum member + one specialist, the router is untouched. |
| **L**iskov substitution | Any `KnowledgeRetriever` (keyword today, FAISS later) drops into the Technical agent unchanged. |
| **I**nterface segregation | `IntentClassifier` and `SentimentAnalyzer` are separate Protocols; so are `SubscriptionRepository` and `RefundGateway`. No component depends on a method it never calls. |
| **D**ependency inversion | Agents depend on Protocols in `interfaces/`, never on `langchain_openai` or a JSON file. Implementations are injected at construction time. |

---

## Setup

> **Step-by-step run guide (فارسی + English): [`docs/RUN.md`](docs/RUN.md)**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then paste your API key into .env
```

Configuration precedence is *explicit argument → environment variable → default*:

```python
from support_system.config import Settings

settings = Settings.from_env()                        # read .env / environment
settings = Settings.from_env(provider="anthropic")    # or override in code
print(settings.describe())                            # never prints the key
```

Supported providers: `openai` (including any OpenAI-compatible gateway via
`SUPPORT_BASE_URL`), `anthropic`, `google`.

### Verify the provider before running anything

```bash
python scripts/verify_provider.py       # config -> endpoint -> structured output
python scripts/verify_provider.py --list-models
```

The third check is the important one: the triage agent depends on
`with_structured_output`, and not every OpenAI-compatible gateway implements
JSON-schema / function calling. If that check fails, swap
`LLMIntentClassifier` for `KeywordIntentClassifier` — they implement the same
Protocol, so nothing else changes.

## Tests

```bash
pytest -q                              # 193 tests, offline, no API key
pytest -q tests/test_spec_compliance.py  # 35 checks, one per clause of the PDF
```

`tests/test_spec_compliance.py` is organised by the assignment's own headings
(Node 1–4, Step 1–3, the three scenarios, the delivery format), so a failure
names the requirement it breaks.

## Running it

```bash
python scripts/demo.py --offline      # the three scenarios, in the terminal, no API key
python scripts/chat.py --offline      # interactive: type your own messages
jupyter lab notebooks/customer_support_langgraph.ipynb
python scripts/render_graph.py        # regenerate the graph image
```

### Live vs. offline

`build_application()` makes one tiny probe call at start-up and falls back to
deterministic components if the model is unreachable — **visibly**, reporting why.

This matters more than it looks. Every component here degrades gracefully on a failed
model call, which is right in production but means a misconfigured endpoint produces a
system that runs to completion and answers *everything* wrongly. Holding an API key is
not evidence that the model works; the probe is. In offline mode answers come from the
tools and the knowledge base instead of a model — unpolished, but still true — so the
notebook runs for a grader with no key.
