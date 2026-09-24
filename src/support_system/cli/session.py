"""
Interactive chat session.

The REPL logic lives here rather than in the script so it can be unit-tested:
:meth:`ChatSession.handle` takes a line of text and returns a
:class:`ChatTurn`, with no ``input()`` or ``print()`` anywhere near it. The
script in ``scripts/chat.py`` is only the keyboard-and-screen wrapper.

Conversation model
------------------
One ``thread_id`` per conversation. Because ``messages`` is an append-only
channel, every turn on the same thread adds to the transcript while the control
fields (``next_step``, ``triage_attempts``, ``escalated``) reset -- so each
message is routed afresh but the agents still see the history. ``/new`` starts
a clean thread.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

TurnKind = Literal["reply", "info", "escalated", "quit", "error"]

HELP = """\
Commands
  /help              show this
  /new               start a fresh conversation (clears the history)
  /user <id>         set the account id used by the billing tools
                     known ids: 12345 (expired), 67890 (active), 11111 (trial),
                                22222 (cancelled)
  /state             dump the full graph state of this conversation
  /history           print the transcript so far
  /tools             list the tools and their data
  /kb <query>        query the knowledge base directly (no agents involved)
  /quit              exit

Try
  How can I reset my password?                 -> technical + RAG
  The app crashes with error E-204             -> technical + RAG
  How do I connect the app to my smart fridge? -> technical, nothing in the docs,
                                                  so it admits it does not know
  My subscription is not working               -> billing + check_subscription_status
  I want a refund for TXN-1001                 -> billing + process_refund
  I want a refund for TXN-1002                 -> refund refused by policy
  How do I reset my password? (with /user set) -> billing bounces it back to triage
  You stole my money! I want a manager         -> guardrail halts, asks you for a reply
"""


@dataclass
class ChatTurn:
    """Everything the caller needs in order to render one exchange."""

    kind: TurnKind
    text: str
    state: dict[str, Any] | None = None
    #: Payload the paused node handed to ``interrupt()``, dynamic mode only.
    interrupt_payload: Any = None
    #: Short trace line: department, sentiment, tools used.
    trace: str = ""


@dataclass
class ChatSession:
    """Drives one interactive conversation against a built application."""

    app: Any
    user_id: str = "12345"
    thread_prefix: str = "chat"
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    thread_id: str = ""

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = self._next_thread()

    def _next_thread(self) -> str:
        return f"{self.thread_prefix}-{next(self._counter)}"

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def handle(self, line: str) -> ChatTurn:
        """Process one line typed by the user."""
        line = line.strip()
        if not line:
            return ChatTurn("info", "")
        if line.startswith("/"):
            return self._command(line)
        return self._message(line)

    # ------------------------------------------------------------------ #
    def _message(self, message: str) -> ChatTurn:
        """Send a customer message through the graph."""
        # tool_calls is an append-only channel, so remember where this turn
        # starts; otherwise the trace would replay every earlier turn's tools.
        tools_before = len(self._current_tool_calls())
        try:
            state = self.app.run(message, user_id=self.user_id, thread_id=self.thread_id)
        except Exception as exc:  # noqa: BLE001 - the REPL must survive anything
            return ChatTurn("error", f"{type(exc).__name__}: {exc}")

        if self.app.is_interrupted(self.thread_id):
            payload = None
            if self.app.hitl_mode == "dynamic":
                payload = self.app.pending_interrupt(self.thread_id)
            return ChatTurn(
                "escalated",
                "The guardrail detected an angry customer, so the graph has HALTED "
                "and no automated reply was sent. You are now the support manager.",
                state=dict(state),
                interrupt_payload=payload,
                trace=self._trace(state, tools_before),
            )

        return ChatTurn(
            "reply",
            state.get("final_response") or "(no reply produced)",
            state=dict(state),
            trace=self._trace(state, tools_before),
        )

    def _current_tool_calls(self) -> list:
        """Tool-call log of this thread so far; empty for a brand-new thread."""
        try:
            return list(self.app.state(self.thread_id).get("tool_calls", []) or [])
        except Exception:  # noqa: BLE001 - an unknown thread simply has no history
            return []

    def resolve_escalation(self, manager_reply: str) -> ChatTurn:
        """Finish a halted conversation with the manager's answer."""
        reply = manager_reply.strip()
        if not reply:
            return ChatTurn("error", "A manager reply is required to continue.")

        tools_before = len(self._current_tool_calls())
        if self.app.hitl_mode == "dynamic":
            state = self.app.resume_with(reply, thread_id=self.thread_id)
        else:
            self.app.inject_manager_reply(reply, thread_id=self.thread_id)
            state = self.app.resume(self.thread_id)

        return ChatTurn(
            "reply",
            state.get("final_response") or "(no reply produced)",
            state=dict(state),
            trace=self._trace(state, tools_before),
        )

    @property
    def awaiting_manager(self) -> bool:
        """True while the graph is paused for a human."""
        return self.app.is_interrupted(self.thread_id)

    # ------------------------------------------------------------------ #
    def _command(self, line: str) -> ChatTurn:
        command, _, argument = line[1:].partition(" ")
        command, argument = command.lower(), argument.strip()

        if command in {"quit", "exit", "q"}:
            return ChatTurn("quit", "Bye.")

        if command in {"help", "h", "?"}:
            return ChatTurn("info", HELP)

        if command == "new":
            self.thread_id = self._next_thread()
            return ChatTurn("info", f"Started a new conversation ({self.thread_id}).")

        if command == "user":
            if not argument:
                return ChatTurn("info", f"Current account id: {self.user_id}")
            self.user_id = argument
            status = self.app.repository.get_status(argument)
            return ChatTurn("info", f"Account id set to {argument}. {status.summary()}")

        if command == "state":
            state = self.app.state(self.thread_id)
            lines = [f"  {key:16} {value!r}"[:140]
                     for key, value in sorted(state.items()) if key != "messages"]
            return ChatTurn("info", "Graph state:\n" + "\n".join(lines))

        if command == "history":
            messages = self.app.state(self.thread_id).get("messages", [])
            if not messages:
                return ChatTurn("info", "(nothing said yet)")
            return ChatTurn("info", "\n".join(f"  {m}" for m in messages))

        if command == "tools":
            from ..tools import make_billing_tools, make_technical_tools

            tools = make_billing_tools(self.app.repository, self.app.gateway)
            tools += make_technical_tools(self.app.retriever)
            return ChatTurn(
                "info",
                "Tools:\n" + "\n".join(f"  {t.name}({', '.join(t.args)})" for t in tools),
            )

        if command == "kb":
            if not argument:
                return ChatTurn("info", "Usage: /kb <query>")
            result = self.app.retriever.search(argument)
            if not result.has_grounding:
                best = result.snippets[0].score if result.snippets else 0.0
                return ChatTurn(
                    "info",
                    f"No grounded match (best score {best:.3f}). "
                    "The technical agent would say it does not know.",
                )
            found = "\n".join(
                f"  [{s.score:.3f}] {s.source}: {s.content.splitlines()[0][:90]}"
                for s in result.snippets
            )
            return ChatTurn("info", f"Grounded matches:\n{found}")

        return ChatTurn("error", f"Unknown command '/{command}'. Try /help.")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _trace(state: dict[str, Any], tools_before: int = 0) -> str:
        """One-line summary of how *this turn's* answer was produced."""
        tools = (state.get("tool_calls") or [])[tools_before:]
        return (
            f"{state.get('department', '?')} | {state.get('sentiment', '?')}"
            f" | tools: {'; '.join(tools) if tools else 'none'}"
        )
