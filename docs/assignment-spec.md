# Assignment requirements (translated from `HW_07_multi agent_2.pdf`)

Reference checklist used to verify the implementation. Original document is in
Persian; this is a working translation, not a substitute for the PDF.

## 1. Goal

Go beyond a single-prompt chatbot and build a **multi-agent** customer support
system that works like a real company: a receptionist inspects requests and
hands them to the relevant specialists (billing, technical). The system must
also be smart enough to stop and escalate to a human operator when the user is
angry, instead of replying automatically.

## 2. Architecture

### Node 1 — Triage Agent (The Triage Agent)
* **Role:** reception desk.
* **Task:** analyse the incoming message and classify it into one of three
  categories:
  * `BILLING` — payment, subscription and refund matters.
  * `TECHNICAL` — technical problems, bugs, product questions.
  * `GENERAL` — general questions and greetings.
* **Output:** a *structured* decision naming the next node.

### Node 2 — Billing Specialist
* **Role:** handle sensitive financial requests.
* **Tools:**
  * `check_subscription_status(user_id)` — is the subscription active or expired.
  * `process_refund(transaction_id)` — mock refund approval.
* **Behaviour:** must not answer technical questions; on an unrelated request it
  must send the request **back to triage**.

### Node 3 — Technical Support
* **Role:** solve software problems.
* **Tools:**
  * `search_knowledge_base(query)` — a **RAG** search over a help file or
    technical documentation.
* **Behaviour:** if the answer is not in the documents it must **not
  hallucinate** and must state that it does not know.

### Node 4 — Sentiment Guardrail
* **Role:** quality control and supervision.
* **Task:** before the agents' final answer reaches the user, check the tone of
  the user's message.
* **Logic:**
  * negative/angry (e.g. "your service is terrible!") → **stop** graph execution
    and wait for human intervention;
  * neutral/positive → show the generated answer to the user.

## 3. Technical implementation steps

### Step 1 — define the state
Define the graph state with `TypedDict` so information can move between agents:

```python
from typing import TypedDict, List, Annotated
import operator

class SupportState(TypedDict):
    messages: Annotated[List[str], operator.add]  # Chat History
    user_id: str
    sentiment: str      # "Positive", "Neutral", "Negative"
    department: str     # "Billing", "Technical", "Triage"
    next_step: str      # Control flow variable
```

### Step 2 — build the triage agent
Use LangChain's `with_structured_output` so the triage agent's output is always
JSON identifying the user's intent.

### Step 3 — human-in-the-loop
The main part of the project. Use LangGraph's **checkpointer**.
* When the sentiment guardrail detects anger, halt execution with
  `graph.interrupt()`.
* Write a simple function playing the role of the **support manager**: it can
  inspect the state and approve or reject the agent's answer (or substitute a
  manual reply).

## 4. Required delivery scenarios

The notebook must demonstrate all three:

1. **Technical happy path**
   * user: "How can I reset my password?"
   * flow: triage → technical specialist → RAG search
   * final answer: "To change your password click the link below…"
2. **Sensitive financial operation**
   * user: "My subscription is not working. My id is 12345."
   * flow: triage → billing specialist → `check_subscription` tool
   * final answer: "Your subscription has expired."
3. **Human escalation**
   * user: "You stole my money! This service is useless! I want to talk to a manager."
   * flow: triage decides it is *billing*, but the sentiment guardrail detects
     anger → system halts.
   * your action: use `update_state` to insert a manual reply from the manager:
     "I am the senior manager. I apologise, I will personally follow up on your
     issue."

## 5. Delivery format

Complete implementation as an `.ipynb` file, together with the generated graph
image produced by `draw_mermaid_png()`.
