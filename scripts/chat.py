#!/usr/bin/env python
"""
Talk to the support system. Type your own messages.

    python scripts/chat.py                # uses the model if it is reachable
    python scripts/chat.py --offline      # deterministic, no API calls, no cost
    python scripts/chat.py --dynamic      # use the graph.interrupt() form
    python scripts/chat.py --verbose      # show each node's decision as it runs

Type /help inside the session for the command list.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_system.cli import ChatSession  # noqa: E402
from support_system.config import Settings  # noqa: E402
from support_system.graph import build_application  # noqa: E402

# ANSI colours, disabled automatically when the output is piped to a file.
COLOUR = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOUR else text

BOT = lambda t: _c("36", t)      # cyan
DIM = lambda t: _c("90", t)      # grey
WARN = lambda t: _c("33", t)     # yellow
ERR = lambda t: _c("31", t)      # red


def render(turn) -> None:
    """Print one exchange."""
    if turn.kind == "reply":
        print(f"\n{BOT('bot>')} {turn.text}")
        print(DIM(f"      [{turn.trace}]\n"))
    elif turn.kind == "escalated":
        print(f"\n{WARN('*** ' + turn.text + ' ***')}")
        print(DIM(f"      [{turn.trace}]"))
        if turn.interrupt_payload:
            print(DIM(f"      draft withheld: "
                      f"{str(turn.interrupt_payload.get('agent_draft', ''))[:100]}"))
    elif turn.kind == "error":
        print(f"\n{ERR('!')} {turn.text}\n")
    elif turn.text:
        print(f"\n{turn.text}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="force the deterministic components (no API calls)")
    parser.add_argument("--dynamic", action="store_true",
                        help="pause with graph.interrupt() instead of update_state")
    parser.add_argument("--user", default="12345", help="starting account id")
    parser.add_argument("--verbose", action="store_true",
                        help="log each node's decision")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format=DIM("%(levelname)-7s %(name)s | %(message)s"),
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = Settings.from_env()
    app = build_application(
        settings,
        force_offline=args.offline,
        hitl_mode="dynamic" if args.dynamic else "static",
    )
    session = ChatSession(app, user_id=args.user)

    print()
    if app.offline:
        print(WARN(f"MODE: OFFLINE — {app.offline_reason}"))
        print(DIM("      Replies come from the tools and the knowledge base, not a model,"))
        print(DIM("      so they are unpolished but true. Routing and tools are identical."))
    else:
        print(f"MODE: LIVE — {settings.provider}/{settings.model}")
    print(DIM(f"      account id: {session.user_id}   hitl: {app.hitl_mode}"))
    print(DIM("      /help for commands, /quit to exit\n"))

    while True:
        try:
            prompt = "manager> " if session.awaiting_manager else "you> "
            line = input(_c("32", prompt))
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        # While the graph is halted, whatever is typed is the manager's reply.
        if session.awaiting_manager:
            if line.strip().lower() in {"/quit", "/exit", "/q"}:
                print("Bye.")
                return 0
            render(session.resolve_escalation(line))
            continue

        turn = session.handle(line)
        if turn.kind == "quit":
            print(turn.text)
            return 0
        render(turn)


if __name__ == "__main__":
    raise SystemExit(main())
